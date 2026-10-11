"""Read-only deterministic Supplier-Risk analyst using the published message contract."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import cast

from mosaic.agents.messages import (
    AgentRole,
    AnalystRecommendation,
    AnalystRequest,
    ToolEvidence,
    ensure_tool_allowed,
)
from mosaic.simulator.actions import NoAction, PlaceOrder
from mosaic.tools.api import ToolRunner

_REQUEST_FIELD = re.compile(
    r"(?<![A-Za-z0-9_])(?P<field>sku_id|warehouse_id|quantity|supplier_id)"
    r"\s*=\s*(?P<value>[A-Za-z0-9_.-]+)(?![A-Za-z0-9_,.-])"
)


@dataclass(frozen=True, slots=True)
class _RequestContext:
    sku_id: str | None
    warehouse_id: str | None
    quantity: int | None
    supplier_id: str | None
    duplicate_fields: frozenset[str]
    invalid_quantity: bool

    @property
    def proposal_problem(self) -> str | None:
        if self.duplicate_fields:
            names = ", ".join(sorted(self.duplicate_fields))
            return f"request context is ambiguous for {names}"
        if self.invalid_quantity:
            return "the explicitly requested quantity is not a positive whole number"
        missing = [
            name
            for name, value in (
                ("sku_id", self.sku_id),
                ("warehouse_id", self.warehouse_id),
                ("quantity", self.quantity),
            )
            if value is None
        ]
        if missing:
            return "request context is missing " + ", ".join(missing)
        return None


@dataclass(frozen=True, slots=True)
class _SupplierStatus:
    supplier_id: str
    supported_skus: frozenset[str]
    base_lead_time: int
    current_lead_time: int
    reliability_score: float
    capacity: int
    disruption_types: frozenset[str]

    @property
    def has_cancellation_disruption(self) -> bool:
        return "SUPPLIER_CANCELLATION" in self.disruption_types

    @property
    def has_delay_disruption(self) -> bool:
        return (
            "SUPPLIER_DELAY" in self.disruption_types
            or self.current_lead_time > self.base_lead_time
        )


@dataclass(frozen=True, slots=True)
class _SupplierOption:
    status: _SupplierStatus
    quoted_total_cost: float


class SupplierRiskAgent:
    """Build a correlated supplier-risk recommendation from read-only tool evidence.

    Order proposals require explicit `sku_id`, `warehouse_id`, and positive
    `quantity` key-value tokens in the request instruction. The agent never
    infers a quantity or reads future outcomes to make a supplier choice.
    """

    def recommend(self, request: AnalystRequest, tools: ToolRunner) -> AnalystRecommendation:
        """Return a deterministic, read-only supplier assessment for the active day."""
        if request.to_role is not AgentRole.SUPPLIER_RISK:
            raise ValueError("SupplierRiskAgent requires a request addressed to supplier_risk")

        simulator = tools.context.simulator
        current_day = simulator.state.day
        if request.episode_seed != simulator.config.seed:
            raise ValueError("Request episode_seed does not match the active simulator")
        if request.day != current_day and not (current_day == 0 and request.day == 1):
            raise ValueError("Request day does not match the active simulator day")

        context = self._parse_request_context(request.instruction)
        status_evidence, status_result = self._call_tool(tools, "get_supplier_status", {})
        order_arguments: dict[str, object] = {}
        shipment_arguments: dict[str, object] = {}
        if context.sku_id is not None:
            order_arguments["sku_id"] = context.sku_id
            shipment_arguments["sku_id"] = context.sku_id
        if context.warehouse_id is not None:
            order_arguments["warehouse_id"] = context.warehouse_id
        order_evidence, order_result = self._call_tool(tools, "get_open_orders", order_arguments)
        shipment_evidence, shipment_result = self._call_tool(
            tools, "get_shipments", shipment_arguments
        )
        evidence = [status_evidence, order_evidence, shipment_evidence]

        inventory_result: dict[str, object] | None = None
        if (
            context.sku_id is not None
            and context.warehouse_id is not None
            and not {"sku_id", "warehouse_id"} & context.duplicate_fields
        ):
            inventory_evidence, inventory_result = self._call_tool(
                tools,
                "get_inventory_state",
                {"sku_id": context.sku_id, "warehouse_id": context.warehouse_id},
            )
            evidence.append(inventory_evidence)

        suppliers = self._read_suppliers(status_result)
        suppliers.sort(key=lambda supplier: supplier.supplier_id)
        summary_parts = [
            "Deterministic Supplier-Risk baseline (read-only tools).",
            self._supplier_status_summary(suppliers, status_evidence.success),
            self._open_order_summary(order_result, order_evidence.success, current_day),
            self._shipment_summary(shipment_result, shipment_evidence.success, current_day),
        ]

        if (
            not status_evidence.success
            or not order_evidence.success
            or not shipment_evidence.success
        ):
            summary_parts.append(
                "A required supplier/order/shipment read failed; no supplier action is proposed."
            )
            return self._recommendation(request, evidence, NoAction(), " ".join(summary_parts))
        if context.proposal_problem is not None:
            summary_parts.append(
                f"NoAction: {context.proposal_problem}; an order quantity is never inferred."
            )
            return self._recommendation(request, evidence, NoAction(), " ".join(summary_parts))
        if not self._inventory_target_exists(inventory_result, context):
            summary_parts.append(
                "NoAction: the requested SKU/warehouse pair is not confirmed by "
                "get_inventory_state; no order is proposed."
            )
            return self._recommendation(request, evidence, NoAction(), " ".join(summary_parts))

        assert context.sku_id is not None
        assert context.warehouse_id is not None
        assert context.quantity is not None
        if not suppliers:
            summary_parts.append(
                "NoAction: get_supplier_status returned no usable supplier records."
            )
            return self._recommendation(request, evidence, NoAction(), " ".join(summary_parts))

        requested_supplier = next(
            (supplier for supplier in suppliers if supplier.supplier_id == context.supplier_id),
            None,
        )
        if context.supplier_id is not None and requested_supplier is None:
            summary_parts.append(
                f"Requested supplier {context.supplier_id} is not present "
                "in current supplier status."
            )
        elif requested_supplier is not None:
            if context.sku_id not in requested_supplier.supported_skus:
                summary_parts.append(
                    f"Requested supplier {context.supplier_id} does not list "
                    f"{context.sku_id} as supported."
                )
            elif context.quantity > requested_supplier.capacity:
                summary_parts.append(
                    f"Requested supplier {context.supplier_id} capacity "
                    f"{requested_supplier.capacity} is below requested quantity {context.quantity}."
                )
            elif requested_supplier.has_cancellation_disruption:
                summary_parts.append(
                    f"Requested supplier {context.supplier_id} has an active "
                    "SUPPLIER_CANCELLATION disruption."
                )

        eligible = [
            supplier
            for supplier in suppliers
            if context.sku_id in supplier.supported_skus
            and context.quantity <= supplier.capacity
            and not supplier.has_cancellation_disruption
        ]
        options: list[_SupplierOption] = []
        for supplier in eligible:
            cost_evidence, cost_result = self._call_tool(
                tools,
                "calculate_order_cost",
                {
                    "supplier_id": supplier.supplier_id,
                    "sku_id": context.sku_id,
                    "quantity": context.quantity,
                },
            )
            evidence.append(cost_evidence)
            quoted_total_cost = self._read_quoted_total(cost_result)
            if cost_evidence.success and quoted_total_cost is not None:
                options.append(_SupplierOption(supplier, quoted_total_cost))
            else:
                summary_parts.append(
                    f"Order-cost quote unavailable for {supplier.supplier_id}; "
                    "that supplier is not proposed."
                )

        if not options:
            summary_parts.append(
                "NoAction: no supplier has confirmed SKU support, sufficient per-order "
                "capacity, and an available tool-calculated order-cost quote without an "
                "active cancellation disruption."
            )
            return self._recommendation(request, evidence, NoAction(), " ".join(summary_parts))

        selected = self._select_supplier(options, requested_supplier)
        supplier = selected.status
        is_alternate = (
            context.supplier_id is not None and supplier.supplier_id != context.supplier_id
        )
        if is_alternate:
            summary_parts.append(
                f"Selected alternate supplier {supplier.supplier_id} from current status: "
                f"lead time {supplier.current_lead_time} days, reliability "
                f"{supplier.reliability_score:.2f}, capacity {supplier.capacity}; "
                f"ToolRunner order-cost quote {selected.quoted_total_cost:.2f}."
            )
        else:
            summary_parts.append(
                f"Selected supplier {supplier.supplier_id}: current lead time "
                f"{supplier.current_lead_time} days (base {supplier.base_lead_time}), "
                f"reliability {supplier.reliability_score:.2f}, capacity {supplier.capacity}; "
                f"ToolRunner order-cost quote {selected.quoted_total_cost:.2f}."
            )
        if supplier.has_delay_disruption:
            summary_parts.append(
                f"{supplier.supplier_id} has an active/current lead-time delay signal; "
                "the observed status is disclosed for Coordinator review."
            )
        summary_parts.append(
            f"Propose PLACE_ORDER for the explicitly requested {context.quantity} units of "
            f"{context.sku_id} to {context.warehouse_id}; quantity is not inferred. "
            "No business action was executed."
        )
        return self._recommendation(
            request,
            evidence,
            PlaceOrder(
                supplier_id=supplier.supplier_id,
                warehouse_id=context.warehouse_id,
                sku_id=context.sku_id,
                quantity=context.quantity,
            ),
            " ".join(summary_parts),
        )

    @staticmethod
    def _parse_request_context(instruction: str) -> _RequestContext:
        values: dict[str, str] = {}
        duplicate_fields: set[str] = set()
        for match in _REQUEST_FIELD.finditer(instruction):
            field = match.group("field")
            if field in values:
                duplicate_fields.add(field)
            else:
                values[field] = match.group("value")

        quantity: int | None = None
        invalid_quantity = False
        if "quantity" in values:
            try:
                quantity = int(values["quantity"])
            except ValueError:
                invalid_quantity = True
            if quantity is not None and quantity <= 0:
                invalid_quantity = True
                quantity = None

        return _RequestContext(
            sku_id=values.get("sku_id"),
            warehouse_id=values.get("warehouse_id"),
            quantity=quantity,
            supplier_id=values.get("supplier_id"),
            duplicate_fields=frozenset(duplicate_fields),
            invalid_quantity=invalid_quantity,
        )

    @staticmethod
    def _call_tool(
        tools: ToolRunner, name: str, arguments: dict[str, object]
    ) -> tuple[ToolEvidence, dict[str, object] | None]:
        ensure_tool_allowed(AgentRole.SUPPLIER_RISK, name)
        response = tools.run(name, arguments)
        success = response.get("success") is True
        raw_result = response.get("result")
        result: dict[str, object] | None = None
        if isinstance(raw_result, dict):
            result = cast(dict[str, object], raw_result)
        elif not success:
            error = response.get("error")
            result = {"error": error} if error is not None else None
        return (
            ToolEvidence(
                tool=name,
                arguments=arguments,
                success=success,
                result=result,
            ),
            result,
        )

    @staticmethod
    def _read_suppliers(result: dict[str, object] | None) -> list[_SupplierStatus]:
        if result is None:
            return []
        raw_suppliers = result.get("suppliers")
        if not isinstance(raw_suppliers, list):
            return []
        suppliers: list[_SupplierStatus] = []
        for raw in cast(list[object], raw_suppliers):
            if not isinstance(raw, dict):
                continue
            row = cast(dict[str, object], raw)
            supplier_id = row.get("supplier_id")
            supported_skus = row.get("supported_skus")
            if not isinstance(supported_skus, list):
                continue
            raw_skus = cast(list[object], supported_skus)
            if not all(isinstance(sku, str) for sku in raw_skus):
                continue
            base_lead_time = row.get("base_lead_time")
            current_lead_time = row.get("current_lead_time")
            reliability = row.get("reliability_score")
            capacity = row.get("capacity")
            if (
                not isinstance(supplier_id, str)
                or not isinstance(base_lead_time, int)
                or isinstance(base_lead_time, bool)
                or not isinstance(current_lead_time, int)
                or isinstance(current_lead_time, bool)
                or not isinstance(reliability, (int, float))
                or isinstance(reliability, bool)
                or not isfinite(reliability)
                or not isinstance(capacity, int)
                or isinstance(capacity, bool)
            ):
                continue
            raw_events = row.get("active_disruption_events", [])
            disruption_types: set[str] = set()
            if isinstance(raw_events, list):
                for raw_event in cast(list[object], raw_events):
                    if not isinstance(raw_event, dict):
                        continue
                    event = cast(dict[str, object], raw_event)
                    event_type = event.get("type")
                    if isinstance(event_type, str) and event_type in {
                        "SUPPLIER_DELAY",
                        "SUPPLIER_CANCELLATION",
                    }:
                        disruption_types.add(event_type)
            suppliers.append(
                _SupplierStatus(
                    supplier_id=supplier_id,
                    supported_skus=frozenset(cast(list[str], supported_skus)),
                    base_lead_time=base_lead_time,
                    current_lead_time=current_lead_time,
                    reliability_score=float(reliability),
                    capacity=capacity,
                    disruption_types=frozenset(disruption_types),
                )
            )
        return suppliers

    @staticmethod
    def _supplier_status_summary(
        suppliers: Sequence[_SupplierStatus], status_available: bool
    ) -> str:
        if not status_available:
            return "Supplier status is unavailable from get_supplier_status."
        if not suppliers:
            return "No usable supplier status rows were returned."
        rows: list[str] = []
        for supplier in suppliers:
            disruptions = ",".join(sorted(supplier.disruption_types)) or "none"
            rows.append(
                f"{supplier.supplier_id}: current/base lead time "
                f"{supplier.current_lead_time}/{supplier.base_lead_time} days, "
                f"reliability {supplier.reliability_score:.2f}, capacity {supplier.capacity}, "
                f"active disruptions [{disruptions}]"
            )
        return "Current supplier status: " + "; ".join(rows) + "."

    @staticmethod
    def _open_order_summary(
        result: dict[str, object] | None, available: bool, current_day: int
    ) -> str:
        if not available or result is None:
            return "Open-order evidence is unavailable."
        raw_orders = result.get("orders")
        if not isinstance(raw_orders, list):
            return "Open-order result is invalid."
        count = 0
        for raw_order in cast(list[object], raw_orders):
            if not isinstance(raw_order, dict):
                continue
            order = cast(dict[str, object], raw_order)
            placed_at = order.get("placed_at")
            if (
                order.get("status") in {"OPEN", "PARTIAL"}
                and isinstance(placed_at, int)
                and not isinstance(placed_at, bool)
                and placed_at <= current_day
            ):
                count += 1
        return f"Current matching open/partial orders: {count}."

    @staticmethod
    def _shipment_summary(
        result: dict[str, object] | None, available: bool, current_day: int
    ) -> str:
        if not available or result is None:
            return "Shipment evidence is unavailable."
        raw_shipments = result.get("shipments")
        if not isinstance(raw_shipments, list):
            return "Shipment result is invalid."
        counts: dict[str, int] = {}
        for raw_shipment in cast(list[object], raw_shipments):
            if not isinstance(raw_shipment, dict):
                continue
            shipment = cast(dict[str, object], raw_shipment)
            dispatch_time = shipment.get("dispatch_time")
            status = shipment.get("status")
            if (
                isinstance(dispatch_time, int)
                and not isinstance(dispatch_time, bool)
                and dispatch_time <= current_day
                and isinstance(status, str)
            ):
                counts[status] = counts.get(status, 0) + 1
        if not counts:
            return "Current matching shipment records: none."
        counts_text = ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))
        return f"Current matching shipment statuses: {counts_text}."

    @staticmethod
    def _inventory_target_exists(
        result: dict[str, object] | None, context: _RequestContext
    ) -> bool:
        if result is None or not isinstance(result.get("states"), list):
            return False
        for raw_state in cast(list[object], result["states"]):
            if not isinstance(raw_state, dict):
                continue
            state = cast(dict[str, object], raw_state)
            if state.get("sku_id") == context.sku_id and state.get("warehouse_id") == (
                context.warehouse_id
            ):
                return True
        return False

    @staticmethod
    def _read_quoted_total(result: dict[str, object] | None) -> float | None:
        if result is None:
            return None
        total = result.get("total_cost")
        if (
            not isinstance(total, (int, float))
            or isinstance(total, bool)
            or not isfinite(total)
            or total < 0.0
        ):
            return None
        return float(total)

    @staticmethod
    def _select_supplier(
        options: Sequence[_SupplierOption], requested: _SupplierStatus | None
    ) -> _SupplierOption:
        if requested is not None:
            preferred = next(
                (
                    option
                    for option in options
                    if option.status.supplier_id == requested.supplier_id
                ),
                None,
            )
            if preferred is not None:
                safer_alternatives = [
                    option
                    for option in options
                    if option.status.supplier_id != requested.supplier_id
                    and SupplierRiskAgent._is_safer_alternative(option.status, requested)
                ]
                if safer_alternatives:
                    return min(safer_alternatives, key=SupplierRiskAgent._option_rank)
                return preferred
        return min(options, key=SupplierRiskAgent._option_rank)

    @staticmethod
    def _is_safer_alternative(candidate: _SupplierStatus, requested: _SupplierStatus) -> bool:
        if candidate.has_cancellation_disruption:
            return False
        if candidate.has_delay_disruption and not requested.has_delay_disruption:
            return False
        no_worse_on_observed_metrics = (
            candidate.current_lead_time <= requested.current_lead_time
            and candidate.reliability_score >= requested.reliability_score
        )
        strictly_better = (
            candidate.current_lead_time < requested.current_lead_time
            or candidate.reliability_score > requested.reliability_score
            or (requested.has_delay_disruption and not candidate.has_delay_disruption)
        )
        return no_worse_on_observed_metrics and strictly_better

    @staticmethod
    def _option_rank(option: _SupplierOption) -> tuple[int, int, int, float, float, str]:
        status = option.status
        return (
            int(status.has_cancellation_disruption),
            int(status.has_delay_disruption),
            status.current_lead_time,
            -status.reliability_score,
            option.quoted_total_cost,
            status.supplier_id,
        )

    @staticmethod
    def _recommendation(
        request: AnalystRequest,
        evidence: list[ToolEvidence],
        action: PlaceOrder | NoAction,
        summary: str,
    ) -> AnalystRecommendation:
        return AnalystRecommendation(
            message_id=f"{request.message_id}:supplier-risk",
            request_id=request.message_id,
            episode_seed=request.episode_seed,
            day=request.day,
            role=AgentRole.SUPPLIER_RISK,
            summary=summary,
            proposed_actions=[action],
            evidence=evidence,
        )
