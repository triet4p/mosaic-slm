"""Deterministic sequential Coordinator baseline using the locked message contract."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mosaic.agents.demand import DemandAgent
from mosaic.agents.inventory import InventoryAgent
from mosaic.agents.messages import (
    AgentRole,
    AnalystRecommendation,
    AnalystRequest,
    CoordinatorDecision,
    ToolEvidence,
    ensure_tool_allowed,
    is_tool_allowed,
)
from mosaic.agents.supplier_risk import SupplierRiskAgent
from mosaic.simulator.actions import (
    AcceptBackorder,
    ActionExecutor,
    AdjustSafetyStock,
    BusinessAction,
    NoAction,
    PlaceOrder,
    SwitchSupplier,
    TransferStock,
)
from mosaic.tools.api import ToolRunner

_DEFAULT_MAX_ANALYST_CALLS = 6
_DEFAULT_MAX_REVISIONS = 2

_DEMAND_INSTRUCTION = "Assess the latest demand observations and short-horizon risk."
_INVENTORY_INSTRUCTION = "Assess inventory position and propose justified stock transfers."

CoordinationOutcome = Literal["decided", "bounded_stop"]


class CoordinationResult(BaseModel):
    """Sequential coordination output for one simulator day.

    Contains every request/recommendation exchanged, the final
    ``CoordinatorDecision`` (only executable output), and the control
    trace so a later sequential runner can record attribution without
    any parallel contract.
    """

    model_config = ConfigDict(extra="forbid")

    episode_seed: int
    day: int
    requests: list[AnalystRequest]
    recommendations: list[AnalystRecommendation]
    decision: CoordinatorDecision
    outcome: CoordinationOutcome
    analyst_calls: int = Field(ge=0)
    revision_count: int = Field(ge=0)
    trace: list[str]


@dataclass(frozen=True, slots=True)
class _SupplierCandidate:
    sku_id: str
    warehouse_id: str
    quantity: int
    shortfall: int


@dataclass(frozen=True, slots=True)
class _ProvenancedAction:
    action: BusinessAction
    recommendation_id: str
    role: AgentRole
    quote_cost: float | None
    quote_kind: str | None


class CoordinatorAgent:
    """Boring sequential coordination policy over the three deterministic analysts.

    Call order is always demand, then inventory, then supplier-risk when an
    observable need exists. Only ``CoordinatorDecision.final_actions`` may be
    executed; analyst proposals are never executed directly. The policy never
    mutates the live simulator or its config: feasibility is checked on a
    deep copy with the real ``ActionExecutor``. No LLM, tokens, memory tools,
    or parallel calls are used.
    """

    def __init__(
        self,
        *,
        max_analyst_calls: int = _DEFAULT_MAX_ANALYST_CALLS,
        max_revisions: int = _DEFAULT_MAX_REVISIONS,
    ) -> None:
        if max_analyst_calls < 1:
            raise ValueError("max_analyst_calls must be at least 1")
        if max_revisions < 0:
            raise ValueError("max_revisions must be non-negative")
        self._max_analyst_calls = max_analyst_calls
        self._max_revisions = max_revisions

    @property
    def max_analyst_calls(self) -> int:
        """Total analyst ``recommend()`` budget for one coordinated day."""
        return self._max_analyst_calls

    @property
    def max_revisions(self) -> int:
        """Total re-ask budget for one coordinated day."""
        return self._max_revisions

    def coordinate(
        self,
        *,
        episode_seed: int,
        day: int,
        tools: ToolRunner,
        request_prefix: str | None = None,
    ) -> CoordinationResult:
        """Coordinate one day sequentially and return an executable decision."""
        simulator = tools.context.simulator
        current_day = simulator.state.day
        if episode_seed != simulator.config.seed:
            raise ValueError("Request episode_seed does not match the active simulator")
        if day != current_day and not (current_day == 0 and day == 1):
            raise ValueError("Request day does not match the active simulator day")

        prefix = request_prefix or f"coord-{episode_seed}-d{day}"
        requests: list[AnalystRequest] = []
        recommendations: list[AnalystRecommendation] = []
        valid: list[tuple[AnalystRequest, AnalystRecommendation]] = []
        coordinator_evidence: list[ToolEvidence] = []
        trace: list[str] = []
        analyst_calls = 0
        revision_count = 0

        # 1. Demand (always required, never revised: advisory NO_ACTION plus signals).
        if analyst_calls >= self._max_analyst_calls:
            return self._bounded_stop(
                episode_seed=episode_seed,
                day=day,
                prefix=prefix,
                requests=requests,
                recommendations=recommendations,
                valid=valid,
                coordinator_evidence=coordinator_evidence,
                trace=trace,
                analyst_calls=analyst_calls,
                revision_count=revision_count,
                reason=(
                    "BOUNDED STOP: analyst-call budget exhausted before the required "
                    "demand assessment; no business action is taken."
                ),
            )
        demand_request = AnalystRequest(
            message_id=f"{prefix}-demand",
            episode_seed=episode_seed,
            day=day,
            to_role=AgentRole.DEMAND,
            instruction=_DEMAND_INSTRUCTION,
        )
        requests.append(demand_request)
        try:
            demand_rec = DemandAgent().recommend(demand_request, tools)
            analyst_calls += 1
            recommendations.append(demand_rec)
            error = self.validate_recommendation(
                demand_request, demand_rec, expected_seed=episode_seed, expected_day=day
            )
            if error is None:
                valid.append((demand_request, demand_rec))
                trace.append(
                    f"call demand {demand_request.message_id} -> {demand_rec.message_id} valid"
                )
            else:
                trace.append(
                    f"call demand {demand_request.message_id} -> "
                    f"{demand_rec.message_id} invalid: {error}"
                )
        except (ValueError, ValidationError) as exc:
            analyst_calls += 1
            trace.append(f"call demand {demand_request.message_id} failed: {exc}")

        # 2. Inventory (always required, never revised: full deterministic screen).
        if analyst_calls >= self._max_analyst_calls:
            return self._bounded_stop(
                episode_seed=episode_seed,
                day=day,
                prefix=prefix,
                requests=requests,
                recommendations=recommendations,
                valid=valid,
                coordinator_evidence=coordinator_evidence,
                trace=trace,
                analyst_calls=analyst_calls,
                revision_count=revision_count,
                reason=(
                    "BOUNDED STOP: analyst-call budget exhausted before the required "
                    "inventory assessment; no business action is taken."
                ),
            )
        inventory_request = AnalystRequest(
            message_id=f"{prefix}-inventory",
            episode_seed=episode_seed,
            day=day,
            to_role=AgentRole.INVENTORY,
            instruction=_INVENTORY_INSTRUCTION,
        )
        requests.append(inventory_request)
        try:
            inventory_rec = InventoryAgent().recommend(inventory_request, tools)
            analyst_calls += 1
            recommendations.append(inventory_rec)
            error = self.validate_recommendation(
                inventory_request,
                inventory_rec,
                expected_seed=episode_seed,
                expected_day=day,
            )
            if error is None:
                valid.append((inventory_request, inventory_rec))
                trace.append(
                    f"call inventory {inventory_request.message_id} -> "
                    f"{inventory_rec.message_id} valid"
                )
            else:
                trace.append(
                    f"call inventory {inventory_request.message_id} -> "
                    f"{inventory_rec.message_id} invalid: {error}"
                )
        except (ValueError, ValidationError) as exc:
            analyst_calls += 1
            trace.append(f"call inventory {inventory_request.message_id} failed: {exc}")

        # 3. Observable information request for supplier-risk context.
        candidates, inventory_evidence, inventory_note = self._derive_supplier_candidates(tools)
        if inventory_evidence is not None:
            coordinator_evidence.append(inventory_evidence)
        trace.append(inventory_note)

        if not candidates:
            trace.append(
                "supplier-risk skipped: no observable safety-stock shortfall in "
                "get_inventory_state; no explicit order context exists."
            )
        elif analyst_calls >= self._max_analyst_calls:
            trace.append(
                "supplier-risk skipped: analyst-call budget exhausted before the "
                "conditional order assessment; deciding from demand/inventory only."
            )
        else:
            attempt = 0
            while attempt < len(candidates):
                if attempt > 0 and revision_count >= self._max_revisions:
                    trace.append(
                        "supplier-risk revision budget exhausted; no further order "
                        "candidates are tried."
                    )
                    break
                if analyst_calls >= self._max_analyst_calls:
                    trace.append(
                        "supplier-risk stopped: analyst-call budget exhausted during "
                        "order assessment."
                    )
                    break
                candidate = candidates[attempt]
                request_id = f"{prefix}-supplier-{attempt}"
                supplier_request = AnalystRequest(
                    message_id=request_id,
                    episode_seed=episode_seed,
                    day=day,
                    to_role=AgentRole.SUPPLIER_RISK,
                    instruction=self._supplier_instruction(candidate),
                )
                requests.append(supplier_request)
                is_revision = attempt > 0
                try:
                    supplier_rec_result = SupplierRiskAgent().recommend(supplier_request, tools)
                    analyst_calls += 1
                    if is_revision:
                        revision_count += 1
                    recommendations.append(supplier_rec_result)
                    error = self.validate_recommendation(
                        supplier_request,
                        supplier_rec_result,
                        expected_seed=episode_seed,
                        expected_day=day,
                    )
                    if error is not None:
                        trace.append(
                            f"call supplier-risk {request_id} -> "
                            f"{supplier_rec_result.message_id} invalid: {error}"
                        )
                        attempt += 1
                        continue
                    valid.append((supplier_request, supplier_rec_result))
                    if self._is_no_action(supplier_rec_result.proposed_actions):
                        trace.append(
                            f"call supplier-risk {request_id} -> "
                            f"{supplier_rec_result.message_id} valid NO_ACTION "
                            f"for {candidate.sku_id}/{candidate.warehouse_id} "
                            f"quantity {candidate.quantity}"
                        )
                        attempt += 1
                        continue
                    trace.append(
                        f"call supplier-risk {request_id} -> "
                        f"{supplier_rec_result.message_id} valid order "
                        f"for {candidate.sku_id}/{candidate.warehouse_id} "
                        f"quantity {candidate.quantity}"
                    )
                    break
                except (ValueError, ValidationError) as exc:
                    analyst_calls += 1
                    if is_revision:
                        revision_count += 1
                    trace.append(f"call supplier-risk {request_id} failed: {exc}")
                    attempt += 1
                    continue

        if not valid:
            return self._bounded_stop(
                episode_seed=episode_seed,
                day=day,
                prefix=prefix,
                requests=requests,
                recommendations=recommendations,
                valid=valid,
                coordinator_evidence=coordinator_evidence,
                trace=trace,
                analyst_calls=analyst_calls,
                revision_count=revision_count,
                reason=(
                    "BOUNDED STOP: no valid correlated analyst recommendation is "
                    "available; no business action is taken."
                ),
            )

        return self._synthesize(
            episode_seed=episode_seed,
            day=day,
            prefix=prefix,
            tools=tools,
            requests=requests,
            recommendations=recommendations,
            valid=valid,
            coordinator_evidence=coordinator_evidence,
            trace=trace,
            analyst_calls=analyst_calls,
            revision_count=revision_count,
        )

    @staticmethod
    def validate_recommendation(
        request: AnalystRequest,
        recommendation: AnalystRecommendation,
        *,
        expected_seed: int,
        expected_day: int,
    ) -> str | None:
        """Return ``None`` when a recommendation is correlated and authoritative."""
        if recommendation.request_id != request.message_id:
            return (
                f"request_id {recommendation.request_id!r} does not match request "
                f"{request.message_id!r}"
            )
        if request.episode_seed != expected_seed or recommendation.episode_seed != expected_seed:
            return "episode_seed does not match the coordinated episode (stale)"
        if request.day != expected_day or recommendation.day != expected_day:
            return "day does not match the coordinated day (stale)"
        if recommendation.role != request.to_role:
            return (
                f"role {recommendation.role.value!r} does not match requested "
                f"{request.to_role.value!r}"
            )
        for item in recommendation.evidence:
            if not is_tool_allowed(recommendation.role, item.tool):
                return (
                    f"evidence tool {item.tool!r} is outside "
                    f"{recommendation.role.value!r} permissions"
                )
        if not recommendation.proposed_actions:
            return "recommendation carries no proposed actions"
        return None

    @staticmethod
    def _is_no_action(actions: Sequence[BusinessAction]) -> bool:
        return len(actions) == 1 and isinstance(actions[0], NoAction)

    def _derive_supplier_candidates(
        self, tools: ToolRunner
    ) -> tuple[list[_SupplierCandidate], ToolEvidence | None, str]:
        """Derive explicit order contexts from currently observable inventory."""
        ensure_tool_allowed(AgentRole.COORDINATOR, "get_inventory_state")
        response = tools.run("get_inventory_state", {})
        success = response.get("success") is True
        raw_result = response.get("result")
        result: dict[str, object] | None = None
        if isinstance(raw_result, dict):
            result = cast(dict[str, object], raw_result)
        elif not success:
            error = response.get("error")
            result = {"error": error} if error is not None else None
        evidence = ToolEvidence(
            tool="get_inventory_state",
            arguments={},
            success=success,
            result=result,
        )
        if not success or result is None:
            return (
                [],
                evidence,
                "information request get_inventory_state failed; supplier-risk has no "
                "observable order context.",
            )
        raw_states = result.get("states")
        if not isinstance(raw_states, list):
            return (
                [],
                evidence,
                "information request get_inventory_state returned an invalid result; "
                "supplier-risk has no observable order context.",
            )
        candidates: list[_SupplierCandidate] = []
        for raw_row in cast(Sequence[object], raw_states):
            if not isinstance(raw_row, dict):
                continue
            row = cast(dict[str, object], raw_row)
            sku_id = row.get("sku_id")
            warehouse_id = row.get("warehouse_id")
            available = row.get("available")
            safety_stock = row.get("safety_stock")
            if (
                not isinstance(sku_id, str)
                or not isinstance(warehouse_id, str)
                or not isinstance(available, int)
                or isinstance(available, bool)
                or not isinstance(safety_stock, int)
                or isinstance(safety_stock, bool)
            ):
                continue
            shortfall = safety_stock - available
            if shortfall > 0:
                candidates.append(
                    _SupplierCandidate(
                        sku_id=sku_id,
                        warehouse_id=warehouse_id,
                        quantity=shortfall,
                        shortfall=shortfall,
                    )
                )
        candidates.sort(key=lambda item: (-item.shortfall, item.sku_id, item.warehouse_id))
        if not candidates:
            return (
                [],
                evidence,
                "information request get_inventory_state shows no safety-stock "
                "shortfall; supplier-risk has no observable order context.",
            )
        summary = ", ".join(
            f"{item.sku_id}/{item.warehouse_id} qty {item.quantity}" for item in candidates[:3]
        )
        return (
            candidates,
            evidence,
            f"information request get_inventory_state found {len(candidates)} "
            f"shortfall candidate(s): {summary}.",
        )

    @staticmethod
    def _supplier_instruction(candidate: _SupplierCandidate) -> str:
        return (
            "Assess this order: sku_id="
            f"{candidate.sku_id}; warehouse_id={candidate.warehouse_id}; "
            f"quantity={candidate.quantity}"
        )

    def _synthesize(
        self,
        *,
        episode_seed: int,
        day: int,
        prefix: str,
        tools: ToolRunner,
        requests: list[AnalystRequest],
        recommendations: list[AnalystRecommendation],
        valid: list[tuple[AnalystRequest, AnalystRecommendation]],
        coordinator_evidence: list[ToolEvidence],
        trace: list[str],
        analyst_calls: int,
        revision_count: int,
    ) -> CoordinationResult:
        provenanced: list[_ProvenancedAction] = []
        for _, recommendation in valid:
            for action in recommendation.proposed_actions:
                if isinstance(action, NoAction):
                    continue
                quote_cost: float | None = None
                quote_kind: str | None = None
                if isinstance(action, TransferStock):
                    quote_cost = self._transfer_quote(recommendation, action)
                    quote_kind = "transfer_cost" if quote_cost is not None else None
                elif isinstance(action, PlaceOrder):
                    quote_cost = self._order_quote(recommendation, action)
                    quote_kind = "order_total" if quote_cost is not None else None
                provenanced.append(
                    _ProvenancedAction(
                        action=action,
                        recommendation_id=recommendation.message_id,
                        role=recommendation.role,
                        quote_cost=quote_cost,
                        quote_kind=quote_kind,
                    )
                )
        conflict_notes: list[str] = []
        if provenanced:
            provenanced, conflict_notes = self._resolve_same_destination_conflicts(provenanced)
        trace.extend(conflict_notes)

        accepted: list[_ProvenancedAction] = []
        rejected: list[tuple[_ProvenancedAction, str]] = []
        if provenanced:
            ordered = sorted(
                provenanced,
                key=lambda item: (
                    self._conflict_key(item.action),
                    self._type_rank(item.action),
                    self._action_tiebreak(item.action),
                    item.recommendation_id,
                ),
            )
            validation_copy = deepcopy(tools.context.simulator)
            executor = ActionExecutor(validation_copy)
            for item in ordered:
                try:
                    executor.apply(item.action)
                    accepted.append(item)
                except ValueError as exc:
                    rejected.append((item, str(exc)))
            trace.append(
                f"feasibility trial on a validation copy: accepted {len(accepted)}, "
                f"rejected {len(rejected)} with current observable state."
            )
            for item, reason in rejected:
                trace.append(
                    f"drop {self._describe_action(item.action)} from "
                    f"{item.recommendation_id}: {reason}"
                )
        else:
            trace.append("no analyst proposed a concrete business action.")

        if accepted:
            final_actions: list[BusinessAction] = [item.action for item in accepted]
        else:
            final_actions = [NoAction()]

        valid_ids = [recommendation.message_id for _, recommendation in valid]
        rationale_parts = [
            "Deterministic Coordinator baseline (non-LLM, sequential, read-only; "
            "no token usage or model inference).",
            f"Episode {episode_seed} day {day} coordinated demand, inventory, and "
            "supplier-risk sequentially where an observable need exists.",
            "Valid correlated recommendations: "
            + "; ".join(
                f"{recommendation.message_id} ({recommendation.role.value}, "
                f"request {request.message_id})"
                for request, recommendation in valid
            )
            + ".",
        ]
        if accepted:
            for item in accepted:
                quote_text = (
                    f"{item.quote_kind} {item.quote_cost:.2f}"
                    if item.quote_cost is not None and item.quote_kind is not None
                    else "no tool quote reused"
                )
                rationale_parts.append(
                    f"Accept {self._describe_action(item.action)} from "
                    f"{item.recommendation_id} ({item.role.value}); {quote_text}; "
                    "feasible on the current observable state including earlier "
                    "accepted actions."
                )
        else:
            rationale_parts.append(
                "No feasible business action remains after observable-state "
                "validation and interaction accounting; NO_ACTION is the truthful "
                "outcome."
            )
        for item, reason in rejected:
            rationale_parts.append(
                f"Exclude {self._describe_action(item.action)} from "
                f"{item.recommendation_id}: {reason}."
            )
        rationale_parts.append(
            f"Analyst calls {analyst_calls}/{self._max_analyst_calls}, revisions "
            f"{revision_count}/{self._max_revisions}."
        )
        rationale = " ".join(rationale_parts)

        decision = CoordinatorDecision(
            message_id=f"{prefix}-decision",
            episode_seed=episode_seed,
            day=day,
            recommendation_ids=valid_ids,
            rationale=rationale,
            final_actions=final_actions,
            evidence=list(coordinator_evidence),
        )
        trace.append(
            f"decision {decision.message_id} with {len(final_actions)} final action(s) "
            "from recommendations " + ", ".join(valid_ids) + "."
        )
        return CoordinationResult(
            episode_seed=episode_seed,
            day=day,
            requests=list(requests),
            recommendations=list(recommendations),
            decision=decision,
            outcome="decided",
            analyst_calls=analyst_calls,
            revision_count=revision_count,
            trace=list(trace),
        )

    def _bounded_stop(
        self,
        *,
        episode_seed: int,
        day: int,
        prefix: str,
        requests: list[AnalystRequest],
        recommendations: list[AnalystRecommendation],
        valid: list[tuple[AnalystRequest, AnalystRecommendation]],
        coordinator_evidence: list[ToolEvidence],
        trace: list[str],
        analyst_calls: int,
        revision_count: int,
        reason: str,
    ) -> CoordinationResult:
        valid_ids = [recommendation.message_id for _, recommendation in valid]
        if not valid_ids and recommendations:
            valid_ids = [recommendations[0].message_id]
        elif not valid_ids and requests:
            valid_ids = [requests[0].message_id]
        elif not valid_ids:
            valid_ids = [f"{prefix}-demand:demand"]
        trace.append(reason)
        rationale = (
            "Deterministic Coordinator baseline (non-LLM, sequential, read-only; "
            "no token usage or model inference). "
            f"{reason} Episode {episode_seed} day {day}. Analyst calls "
            f"{analyst_calls}/{self._max_analyst_calls}, revisions "
            f"{revision_count}/{self._max_revisions}. No business action is claimed."
        )
        decision = CoordinatorDecision(
            message_id=f"{prefix}-decision",
            episode_seed=episode_seed,
            day=day,
            recommendation_ids=valid_ids,
            rationale=rationale,
            final_actions=[NoAction()],
            evidence=list(coordinator_evidence),
        )
        return CoordinationResult(
            episode_seed=episode_seed,
            day=day,
            requests=list(requests),
            recommendations=list(recommendations),
            decision=decision,
            outcome="bounded_stop",
            analyst_calls=analyst_calls,
            revision_count=revision_count,
            trace=list(trace),
        )

    @staticmethod
    def _transfer_quote(
        recommendation: AnalystRecommendation, action: TransferStock
    ) -> float | None:
        for item in recommendation.evidence:
            if item.tool != "calculate_transfer_cost" or not item.success:
                continue
            if item.result is None:
                continue
            args = item.arguments
            if (
                args.get("source_warehouse") != action.source_warehouse
                or args.get("destination_warehouse") != action.destination_warehouse
                or args.get("sku_id") != action.sku_id
                or args.get("quantity") != action.quantity
            ):
                continue
            cost = item.result.get("transfer_cost")
            if (
                isinstance(cost, (int, float))
                and not isinstance(cost, bool)
                and isfinite(cost)
                and cost >= 0.0
            ):
                return float(cost)
        return None

    @staticmethod
    def _order_quote(recommendation: AnalystRecommendation, action: PlaceOrder) -> float | None:
        for item in recommendation.evidence:
            if item.tool != "calculate_order_cost" or not item.success:
                continue
            if item.result is None:
                continue
            args = item.arguments
            if (
                args.get("supplier_id") != action.supplier_id
                or args.get("sku_id") != action.sku_id
                or args.get("quantity") != action.quantity
            ):
                continue
            total = item.result.get("total_cost")
            if (
                isinstance(total, (int, float))
                and not isinstance(total, bool)
                and isfinite(total)
                and total >= 0.0
            ):
                return float(total)
        return None

    @staticmethod
    def _conflict_key(action: BusinessAction) -> tuple[str, ...]:
        if isinstance(action, TransferStock):
            return ("sku-dest", action.sku_id, action.destination_warehouse)
        if isinstance(action, PlaceOrder):
            return ("sku-dest", action.sku_id, action.warehouse_id)
        if isinstance(action, SwitchSupplier):
            return ("sku-switch", action.sku_id)
        if isinstance(action, AdjustSafetyStock):
            return ("sku-wh", action.sku_id, action.warehouse_id)
        if isinstance(action, AcceptBackorder):
            return ("sku-wh", action.sku_id, action.warehouse_id)
        return ("no-action",)

    @staticmethod
    def _type_rank(action: BusinessAction) -> int:
        if isinstance(action, TransferStock):
            return 0
        if isinstance(action, PlaceOrder):
            return 1
        return 2

    @staticmethod
    def _action_tiebreak(action: BusinessAction) -> tuple[str, ...]:
        if isinstance(action, TransferStock):
            return (
                action.source_warehouse,
                action.destination_warehouse,
                action.sku_id,
                str(action.quantity),
            )
        if isinstance(action, PlaceOrder):
            return (
                action.supplier_id,
                action.warehouse_id,
                action.sku_id,
                str(action.quantity),
            )
        if isinstance(action, SwitchSupplier):
            return (action.old_supplier, action.new_supplier, action.sku_id)
        if isinstance(action, AdjustSafetyStock):
            return (action.warehouse_id, action.sku_id, str(action.new_level))
        if isinstance(action, AcceptBackorder):
            return (action.warehouse_id, action.sku_id, str(action.quantity))
        return ()

    def _resolve_same_destination_conflicts(
        self, candidates: list[_ProvenancedAction]
    ) -> tuple[list[_ProvenancedAction], list[str]]:
        groups: dict[tuple[str, ...], list[_ProvenancedAction]] = {}
        for item in candidates:
            groups.setdefault(self._conflict_key(item.action), []).append(item)
        kept: list[_ProvenancedAction] = []
        notes: list[str] = []
        for key in sorted(groups):
            group = groups[key]
            if len(group) == 1:
                kept.append(group[0])
                continue
            ranked = sorted(
                group,
                key=lambda item: (
                    item.quote_cost if item.quote_cost is not None else float("inf"),
                    self._type_rank(item.action),
                    self._action_tiebreak(item.action),
                    item.recommendation_id,
                ),
            )
            winner = ranked[0]
            kept.append(winner)
            losers = ", ".join(
                f"{self._describe_action(item.action)} from {item.recommendation_id}"
                + (
                    f" ({item.quote_kind} {item.quote_cost:.2f})"
                    if item.quote_cost is not None and item.quote_kind is not None
                    else " (no tool quote)"
                )
                for item in ranked[1:]
            )
            winner_quote = (
                f"{winner.quote_kind} {winner.quote_cost:.2f}"
                if winner.quote_cost is not None and winner.quote_kind is not None
                else "no tool quote"
            )
            notes.append(
                f"conflict {key[1]}/{key[2]}: keep "
                f"{self._describe_action(winner.action)} from "
                f"{winner.recommendation_id} ({winner_quote}); drop {losers} to "
                "avoid double-supplying the same destination."
            )
        return kept, notes

    @staticmethod
    def _describe_action(action: BusinessAction) -> str:
        if isinstance(action, TransferStock):
            return (
                f"TRANSFER_STOCK {action.sku_id} {action.source_warehouse}->"
                f"{action.destination_warehouse} qty {action.quantity}"
            )
        if isinstance(action, PlaceOrder):
            return (
                f"PLACE_ORDER {action.sku_id} to {action.warehouse_id} from "
                f"{action.supplier_id} qty {action.quantity}"
            )
        if isinstance(action, SwitchSupplier):
            return f"SWITCH_SUPPLIER {action.sku_id} {action.old_supplier}->{action.new_supplier}"
        if isinstance(action, AdjustSafetyStock):
            return (
                f"ADJUST_SAFETY_STOCK {action.sku_id} at {action.warehouse_id} "
                f"to {action.new_level}"
            )
        if isinstance(action, AcceptBackorder):
            return (
                f"ACCEPT_BACKORDER {action.sku_id} at {action.warehouse_id} qty {action.quantity}"
            )
        return "NO_ACTION"
