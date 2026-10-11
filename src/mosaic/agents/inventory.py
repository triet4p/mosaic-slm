"""Read-only deterministic Inventory analyst using the published message contract."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, floor, isfinite
from typing import cast

from mosaic.agents.messages import (
    AgentRole,
    AnalystRecommendation,
    AnalystRequest,
    ToolEvidence,
    ensure_tool_allowed,
)
from mosaic.simulator.actions import BusinessAction, NoAction, TransferStock
from mosaic.tools.api import ToolRunner

_HISTORY_DAYS = 30
_FORECAST_HORIZON = 14
_FORECAST_WINDOW = 5
_SAFETY_FACTOR = 1.5

_InventoryKey = tuple[str, str]
_RouteKey = tuple[str, str]
_ReorderKey = tuple[str, str, int]


@dataclass(frozen=True, slots=True)
class _InventoryPosition:
    available: int
    safety_stock: int
    stockout_cost: float


@dataclass(frozen=True, slots=True)
class _TransferCandidate:
    action: TransferStock
    transfer_cost: float
    one_day_stockout_exposure: float
    reorder_point: float
    inventory_position: int

    @property
    def net_value(self) -> float:
        return self.one_day_stockout_exposure - self.transfer_cost


class InventoryAgent:
    """Build a deterministic inventory recommendation from read-only tool evidence.

    The analyst has no action executor and uses only observed demand, inventory,
    inbound-order, shipment, reorder-point, and transfer-cost tools. It returns a
    conservative one-day stockout-cost screening estimate, not a simulated outcome.
    """

    def recommend(self, request: AnalystRequest, tools: ToolRunner) -> AnalystRecommendation:
        """Return a correlated, read-only recommendation for the active episode day."""
        if request.to_role is not AgentRole.INVENTORY:
            raise ValueError("InventoryAgent requires a request addressed to the inventory role")

        simulator = tools.context.simulator
        current_day = simulator.state.day
        if request.episode_seed != simulator.config.seed:
            raise ValueError("Request episode_seed does not match the active simulator")
        if request.day != current_day and not (current_day == 0 and request.day == 1):
            raise ValueError("Request day does not match the active simulator day")

        evidence: list[ToolEvidence] = []
        inventory_evidence, inventory_result = self._call_tool(tools, "get_inventory_state", {})
        evidence.append(inventory_evidence)
        if not inventory_evidence.success:
            return self._recommendation(
                request,
                evidence,
                "Inventory position is unavailable from the inventory tool; "
                "no transfer is recommended.",
                [NoAction()],
            )
        inventory = self._read_inventory(inventory_result)
        if not inventory:
            return self._recommendation(
                request,
                evidence,
                "No inventory states are available to assess; no transfer is recommended.",
                [NoAction()],
            )

        history_evidence, history_result = self._call_tool(
            tools, "get_demand_history", {"days": _HISTORY_DAYS}
        )
        evidence.append(history_evidence)
        if not history_evidence.success:
            return self._recommendation(
                request,
                evidence,
                "Observed demand history is unavailable, so stockout risk cannot be assessed; "
                "no transfer is recommended.",
                [NoAction()],
            )
        demand_pairs = self._read_demand_pairs(history_result, current_day)
        if not demand_pairs:
            return self._recommendation(
                request,
                evidence,
                f"No demand has been observed by simulator day {current_day}; reorder risk "
                "cannot be assessed from a forecast, so no transfer is recommended.",
                [NoAction()],
            )

        daily_demand: dict[str, int] = {}
        failed_forecasts: set[str] = set()
        for sku_id, region_id in demand_pairs:
            forecast_evidence, forecast_result = self._call_tool(
                tools,
                "forecast_demand",
                {
                    "sku_id": sku_id,
                    "region_id": region_id,
                    "horizon": _FORECAST_HORIZON,
                    "window": _FORECAST_WINDOW,
                },
            )
            evidence.append(forecast_evidence)
            if not forecast_evidence.success:
                failed_forecasts.add(sku_id)
                continue
            forecast = self._read_forecast(forecast_result, current_day)
            daily_demand[sku_id] = daily_demand.get(sku_id, 0) + forecast[0]

        for sku_id in failed_forecasts:
            daily_demand.pop(sku_id, None)
        available_skus = {sku_id for sku_id, _ in inventory}
        daily_demand = {
            sku_id: quantity
            for sku_id, quantity in daily_demand.items()
            if quantity > 0 and sku_id in available_skus
        }
        if not daily_demand:
            reason = "No positive SKU-level forecast is available for an observed inventory state."
            if failed_forecasts:
                reason += f" Forecast tools failed for {len(failed_forecasts)} SKU(s)."
            return self._recommendation(
                request, evidence, f"{reason} No transfer is recommended.", [NoAction()]
            )

        open_order_evidence, open_order_result = self._call_tool(tools, "get_open_orders", {})
        evidence.append(open_order_evidence)
        shipment_evidence, shipment_result = self._call_tool(tools, "get_shipments", {})
        evidence.append(shipment_evidence)
        if not open_order_evidence.success or not shipment_evidence.success:
            return self._recommendation(
                request,
                evidence,
                "Open-order or in-flight shipment data is unavailable, so inbound stock cannot "
                "be accounted for safely; no transfer is recommended.",
                [NoAction()],
            )
        inbound = self._read_inbound(open_order_result, shipment_result, current_day)

        states_by_sku: dict[str, set[str]] = {}
        for sku_id, warehouse_id in inventory:
            states_by_sku.setdefault(sku_id, set()).add(warehouse_id)
        warehouses = self._warehouses(inventory)
        route_times: dict[_RouteKey, int] = {}
        route_failures = 0
        long_routes = 0
        for source in warehouses:
            for destination in warehouses:
                if source == destination:
                    continue
                route_sku = next(
                    (
                        sku_id
                        for sku_id in sorted(daily_demand)
                        if source in states_by_sku.get(sku_id, set())
                        and destination in states_by_sku.get(sku_id, set())
                    ),
                    None,
                )
                if route_sku is None:
                    continue
                route_evidence, route_result = self._call_tool(
                    tools,
                    "calculate_transfer_cost",
                    {
                        "source_warehouse": source,
                        "destination_warehouse": destination,
                        "sku_id": route_sku,
                        "quantity": 1,
                    },
                )
                evidence.append(route_evidence)
                if not route_evidence.success:
                    route_failures += 1
                    continue
                _, travel_time = self._read_transfer_quote(route_result)
                if travel_time > _FORECAST_HORIZON:
                    long_routes += 1
                    continue
                route_times[(source, destination)] = travel_time

        reorder_points: dict[_ReorderKey, float | None] = {}
        candidates_by_sku: dict[str, list[_TransferCandidate]] = {}
        risk_destinations = 0
        safety_stock_satisfied = 0
        source_shortages = 0
        cost_rejections = 0
        cost_rejection_details: list[str] = []
        quote_failures = 0
        for (source, destination), travel_time in sorted(route_times.items()):
            for sku_id in sorted(daily_demand):
                source_key = (sku_id, source)
                destination_key = (sku_id, destination)
                source_state = inventory.get(source_key)
                destination_state = inventory.get(destination_key)
                if source_state is None or destination_state is None:
                    continue

                source_reorder = self._reorder_point(
                    tools,
                    evidence,
                    reorder_points,
                    sku_id,
                    source,
                    daily_demand[sku_id],
                    travel_time,
                )
                destination_reorder = self._reorder_point(
                    tools,
                    evidence,
                    reorder_points,
                    sku_id,
                    destination,
                    daily_demand[sku_id],
                    travel_time,
                )
                if source_reorder is None or destination_reorder is None:
                    continue

                arrival_day = current_day + travel_time
                source_inbound = self._inbound_by(inbound, source_key, current_day, arrival_day)
                destination_inbound = self._inbound_by(
                    inbound, destination_key, current_day, arrival_day
                )
                source_position = source_state.available + source_inbound
                destination_position = destination_state.available + destination_inbound
                if destination_position >= destination_reorder:
                    continue
                risk_destinations += 1
                quantity_needed = ceil(destination_state.safety_stock - destination_position)
                if quantity_needed <= 0:
                    safety_stock_satisfied += 1
                    continue
                if source_position <= source_reorder:
                    source_shortages += 1
                    continue

                available_surplus = floor(source_position - source_reorder)
                quantity = min(
                    quantity_needed,
                    available_surplus,
                    max(0, source_state.available),
                )
                if quantity <= 0:
                    source_shortages += 1
                    continue

                quote_evidence, quote_result = self._call_tool(
                    tools,
                    "calculate_transfer_cost",
                    {
                        "source_warehouse": source,
                        "destination_warehouse": destination,
                        "sku_id": sku_id,
                        "quantity": quantity,
                    },
                )
                evidence.append(quote_evidence)
                if not quote_evidence.success:
                    quote_failures += 1
                    continue
                transfer_cost, quoted_travel_time = self._read_transfer_quote(quote_result)
                if quoted_travel_time != travel_time:
                    quote_failures += 1
                    continue

                one_day_exposure = round(quantity * destination_state.stockout_cost, 2)
                if transfer_cost >= one_day_exposure:
                    cost_rejections += 1
                    cost_rejection_details.append(
                        f"{sku_id} {source}->{destination} quantity {quantity}: quoted cost "
                        f"{transfer_cost:.2f} >= one-day stockout exposure "
                        f"{one_day_exposure:.2f}"
                    )
                    continue
                candidate = _TransferCandidate(
                    action=TransferStock(
                        source_warehouse=source,
                        destination_warehouse=destination,
                        sku_id=sku_id,
                        quantity=quantity,
                    ),
                    transfer_cost=transfer_cost,
                    one_day_stockout_exposure=one_day_exposure,
                    reorder_point=destination_reorder,
                    inventory_position=destination_position,
                )
                candidates_by_sku.setdefault(sku_id, []).append(candidate)

        proposals: list[_TransferCandidate] = []
        for sku_id in sorted(candidates_by_sku):
            best = min(
                candidates_by_sku[sku_id],
                key=lambda candidate: (
                    -candidate.net_value,
                    candidate.action.destination_warehouse,
                    candidate.action.source_warehouse,
                ),
            )
            proposals.append(best)

        if proposals:
            actions: list[BusinessAction] = [candidate.action for candidate in proposals]
            lines = [
                f"{candidate.action.sku_id}: transfer {candidate.action.quantity} units "
                f"{candidate.action.source_warehouse} -> "
                f"{candidate.action.destination_warehouse}; inventory position "
                f"{candidate.inventory_position} is below tool-calculated reorder point "
                f"{candidate.reorder_point:g}; transfer cost "
                f"{candidate.transfer_cost:.2f} is below the estimated one-day "
                f"stockout exposure {candidate.one_day_stockout_exposure:.2f}."
                for candidate in proposals
            ]
            summary = "Deterministic Inventory baseline (read-only tools): " + " ".join(lines)
            if failed_forecasts:
                summary += (
                    f" {len(failed_forecasts)} SKU forecast(s) were unavailable and not assessed."
                )
            summary += " These are proposals only; no business action was executed."
            return self._recommendation(request, evidence, summary, actions)

        if risk_destinations == 0:
            summary = (
                "No below-reorder inventory position was found on an available transfer route "
                "after inbound orders and shipments due by route arrival were counted"
            )
            if route_failures:
                summary += (
                    f"; {route_failures} candidate route(s) are unavailable and excluded from "
                    "this assessment"
                )
            summary += "; no transfer is recommended."
        else:
            reasons: list[str] = []
            if cost_rejections:
                reason = (
                    f"{cost_rejections} feasible transfer quote(s) met or exceeded the "
                    "estimated one-day avoidable stockout cost"
                )
                if cost_rejection_details:
                    reason += ": " + "; ".join(cost_rejection_details[:3])
                reasons.append(reason)
            if safety_stock_satisfied:
                reasons.append(
                    "destination position remains at or above the tool-reported safety stock"
                )
            if route_failures:
                reasons.append(f"{route_failures} candidate route(s) are unavailable")
            if long_routes:
                reasons.append(
                    f"{long_routes} route(s) exceed the forecast tool's "
                    f"{_FORECAST_HORIZON}-day projection"
                )
            if quote_failures:
                reasons.append("a final transfer-cost quote was unavailable or inconsistent")
            if not reasons:
                reasons.append("no economically justified transfer is available")
            summary = (
                "Inventory position was below a tool-calculated reorder point, but "
                + "; ".join(reasons)
                + "; no transfer is recommended."
            )
        if failed_forecasts:
            summary += (
                f" {len(failed_forecasts)} SKU forecast(s) were unavailable and not assessed."
            )
        return self._recommendation(request, evidence, summary, [NoAction()])

    @staticmethod
    def _call_tool(
        tools: ToolRunner, name: str, arguments: dict[str, object]
    ) -> tuple[ToolEvidence, dict[str, object] | None]:
        ensure_tool_allowed(AgentRole.INVENTORY, name)
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
    def _read_inventory(
        result: dict[str, object] | None,
    ) -> dict[_InventoryKey, _InventoryPosition]:
        if result is None or not isinstance(result.get("states"), list):
            raise ValueError("get_inventory_state returned an invalid result")
        inventory: dict[_InventoryKey, _InventoryPosition] = {}
        for raw_row in cast(Sequence[object], result["states"]):
            if not isinstance(raw_row, dict):
                raise ValueError("get_inventory_state returned an invalid state row")
            row = cast(dict[str, object], raw_row)
            sku_id = row.get("sku_id")
            warehouse_id = row.get("warehouse_id")
            available = row.get("available")
            safety_stock = row.get("safety_stock")
            stockout_cost = row.get("stockout_cost")
            if (
                not isinstance(sku_id, str)
                or not isinstance(warehouse_id, str)
                or not isinstance(available, int)
                or isinstance(available, bool)
                or not isinstance(safety_stock, int)
                or isinstance(safety_stock, bool)
                or safety_stock < 0
                or not isinstance(stockout_cost, (int, float))
                or isinstance(stockout_cost, bool)
                or not isfinite(stockout_cost)
                or stockout_cost < 0.0
            ):
                raise ValueError("get_inventory_state returned an invalid state row")
            key = (sku_id, warehouse_id)
            if key in inventory:
                raise ValueError("get_inventory_state returned a duplicate SKU/warehouse state")
            inventory[key] = _InventoryPosition(available, safety_stock, float(stockout_cost))
        return inventory

    @staticmethod
    def _read_demand_pairs(
        result: dict[str, object] | None, current_day: int
    ) -> list[tuple[str, str]]:
        if result is None or not isinstance(result.get("history"), list):
            raise ValueError("get_demand_history returned an invalid result")
        pairs: set[tuple[str, str]] = set()
        for raw_row in cast(Sequence[object], result["history"]):
            if not isinstance(raw_row, dict):
                raise ValueError("get_demand_history returned an invalid history row")
            row = cast(dict[str, object], raw_row)
            day = row.get("day")
            sku_id = row.get("sku_id")
            region_id = row.get("region_id")
            quantity = row.get("quantity")
            if (
                not isinstance(day, int)
                or isinstance(day, bool)
                or not isinstance(sku_id, str)
                or not isinstance(region_id, str)
                or not isinstance(quantity, int)
                or isinstance(quantity, bool)
                or quantity < 0
            ):
                raise ValueError("get_demand_history returned an invalid history row")
            if day > current_day:
                raise ValueError("get_demand_history returned unobserved future demand")
            pairs.add((sku_id, region_id))
        return sorted(pairs)

    @staticmethod
    def _read_forecast(result: dict[str, object] | None, current_day: int) -> list[int]:
        if result is None or not isinstance(result.get("forecast"), list):
            raise ValueError("forecast_demand returned an invalid result")
        forecast: list[int] = []
        for offset, raw_point in enumerate(cast(Sequence[object], result["forecast"]), start=1):
            if not isinstance(raw_point, dict):
                raise ValueError("forecast_demand returned an invalid forecast point")
            point = cast(dict[str, object], raw_point)
            day = point.get("day")
            quantity = point.get("quantity")
            if (
                not isinstance(day, int)
                or isinstance(day, bool)
                or day != current_day + offset
                or not isinstance(quantity, int)
                or isinstance(quantity, bool)
                or quantity < 0
            ):
                raise ValueError("forecast_demand returned an invalid forecast point")
            forecast.append(quantity)
        if len(forecast) != _FORECAST_HORIZON:
            raise ValueError("forecast_demand returned an incomplete projection")
        return forecast

    @staticmethod
    def _read_transfer_quote(result: dict[str, object] | None) -> tuple[float, int]:
        if result is None:
            raise ValueError("calculate_transfer_cost returned an invalid result")
        cost = result.get("transfer_cost")
        travel_time = result.get("travel_time_days")
        if (
            not isinstance(cost, (int, float))
            or isinstance(cost, bool)
            or not isfinite(cost)
            or cost < 0.0
            or not isinstance(travel_time, int)
            or isinstance(travel_time, bool)
            or travel_time < 1
        ):
            raise ValueError("calculate_transfer_cost returned an invalid quote")
        return float(cost), travel_time

    @staticmethod
    def _read_inbound(
        open_orders: dict[str, object] | None,
        shipments: dict[str, object] | None,
        current_day: int,
    ) -> dict[_InventoryKey, list[tuple[int, int]]]:
        if open_orders is None or not isinstance(open_orders.get("orders"), list):
            raise ValueError("get_open_orders returned an invalid result")
        if shipments is None or not isinstance(shipments.get("shipments"), list):
            raise ValueError("get_shipments returned an invalid result")
        inbound: dict[_InventoryKey, list[tuple[int, int]]] = {}
        for raw_order in cast(Sequence[object], open_orders["orders"]):
            if not isinstance(raw_order, dict):
                raise ValueError("get_open_orders returned an invalid order")
            order = cast(dict[str, object], raw_order)
            status = order.get("status")
            if status not in {"OPEN", "PARTIAL"}:
                continue
            sku_id = order.get("sku_id")
            warehouse_id = order.get("warehouse_id")
            quantity = order.get("quantity")
            expected = order.get("expected_arrival")
            actual = order.get("actual_arrival")
            if (
                not isinstance(sku_id, str)
                or not isinstance(warehouse_id, str)
                or not isinstance(quantity, int)
                or isinstance(quantity, bool)
                or quantity <= 0
                or not isinstance(expected, int)
                or isinstance(expected, bool)
                or (
                    actual is not None and (not isinstance(actual, int) or isinstance(actual, bool))
                )
            ):
                raise ValueError("get_open_orders returned an invalid order")
            if actual is not None and actual <= current_day:
                continue
            if expected > current_day:
                inbound_quantity = quantity if status == "OPEN" else 0
                if inbound_quantity > 0:
                    inbound.setdefault((sku_id, warehouse_id), []).append(
                        (expected, inbound_quantity)
                    )

        for raw_shipment in cast(Sequence[object], shipments["shipments"]):
            if not isinstance(raw_shipment, dict):
                raise ValueError("get_shipments returned an invalid shipment")
            shipment = cast(dict[str, object], raw_shipment)
            status = shipment.get("status")
            if status not in {"IN_TRANSIT", "DELAYED"}:
                continue
            sku_id = shipment.get("sku_id")
            destination = shipment.get("destination")
            quantity = shipment.get("quantity")
            expected = shipment.get("expected_arrival")
            actual = shipment.get("actual_arrival")
            if (
                not isinstance(sku_id, str)
                or not isinstance(destination, str)
                or not isinstance(quantity, int)
                or isinstance(quantity, bool)
                or quantity <= 0
                or not isinstance(expected, int)
                or isinstance(expected, bool)
                or (
                    actual is not None and (not isinstance(actual, int) or isinstance(actual, bool))
                )
            ):
                raise ValueError("get_shipments returned an invalid shipment")
            if actual is not None and actual <= current_day:
                continue
            inbound.setdefault((sku_id, destination), []).append((expected, quantity))
        return inbound

    @staticmethod
    def _warehouses(inventory: dict[_InventoryKey, _InventoryPosition]) -> list[str]:
        return sorted({warehouse_id for _, warehouse_id in inventory})

    @staticmethod
    def _inbound_by(
        inbound: dict[_InventoryKey, list[tuple[int, int]]],
        key: _InventoryKey,
        current_day: int,
        arrival_day: int,
    ) -> int:
        return sum(
            quantity for eta, quantity in inbound.get(key, []) if current_day < eta <= arrival_day
        )

    @classmethod
    def _reorder_point(
        cls,
        tools: ToolRunner,
        evidence: list[ToolEvidence],
        cache: dict[_ReorderKey, float | None],
        sku_id: str,
        warehouse_id: str,
        daily_demand: int,
        lead_time_days: int,
    ) -> float | None:
        key = (sku_id, warehouse_id, lead_time_days)
        if key in cache:
            return cache[key]
        reorder_evidence, reorder_result = cls._call_tool(
            tools,
            "calculate_reorder_point",
            {
                "sku_id": sku_id,
                "warehouse_id": warehouse_id,
                "daily_demand": daily_demand,
                "lead_time_days": lead_time_days,
                "safety_factor": _SAFETY_FACTOR,
            },
        )
        evidence.append(reorder_evidence)
        if not reorder_evidence.success:
            cache[key] = None
            return None
        if reorder_result is None:
            raise ValueError("calculate_reorder_point returned an invalid result")
        value = reorder_result.get("reorder_point")
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not isfinite(value)
            or value < 0.0
        ):
            raise ValueError("calculate_reorder_point returned an invalid value")
        cache[key] = float(value)
        return cache[key]

    @staticmethod
    def _recommendation(
        request: AnalystRequest,
        evidence: list[ToolEvidence],
        summary: str,
        actions: list[BusinessAction],
    ) -> AnalystRecommendation:
        return AnalystRecommendation(
            message_id=f"{request.message_id}:inventory",
            request_id=request.message_id,
            episode_seed=request.episode_seed,
            day=request.day,
            role=AgentRole.INVENTORY,
            summary=summary,
            proposed_actions=actions,
            evidence=evidence,
        )
