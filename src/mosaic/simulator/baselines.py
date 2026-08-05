"""Heuristic baselines (Sprint 1, task 11).

Các policy phi-LLM bắt buộc theo mục 12.2 INITIALIZATION.md, dùng làm
baseline để biết agent có tạo thêm giá trị hay không và để normalize NTC:

- DoNothingPolicy: không hành động.
- ReorderPointPolicy: policy (s, S) — khi inventory position
  (available + in_transit + pipeline) chạm reorder_point thì đặt lên target
  reorder_point + safety_multiplier * safety_stock từ supplier được assign.
- GreedyTransferPolicy: kho thiếu (available < safety_stock) nhận hàng từ
  kho dư (available > reorder_point).
- OraclePolicy: look-ahead — dùng demand schedule tương lai (full state)
  để đặt hàng đủ cho horizon lead_time + 2 ngày. Không phải optimal solver,
  nhưng là reference mạnh cho NTC.

`run_policy` chạy một episode theo policy; các policy deterministic.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from mosaic.simulator.actions import (
    ActionExecutor,
    BusinessAction,
    NoAction,
    PlaceOrder,
    TransferStock,
)
from mosaic.simulator.core import Simulator
from mosaic.simulator.episodes import Episode
from mosaic.simulator.models import OrderStatus
from mosaic.simulator.supplier import SupplierBehavior


class Policy(Protocol):
    """Quyết định actions cho ngày hiện tại (gọi sau `Simulator.advance_day`)."""

    def decide(self, simulator: Simulator) -> Sequence[BusinessAction]: ...


def run_policy(episode: Episode, policy: Policy) -> Simulator:
    """Chạy toàn bộ episode theo policy; deterministic."""
    behavior = SupplierBehavior(seed=episode.config.seed, stochastic=episode.supplier_stochastic)
    simulator = Simulator(episode.config, supplier_behavior=behavior)
    executor = ActionExecutor(simulator)
    for _ in range(episode.config.num_days):
        simulator.advance_day()
        for action in policy.decide(simulator):
            executor.apply(action)
    return simulator


class DoNothingPolicy:
    """Không thực hiện action nào."""

    def decide(self, simulator: Simulator) -> Sequence[BusinessAction]:
        return [NoAction()]


class ReorderPointPolicy:
    """Policy (s, S): đặt hàng khi inventory position chạm reorder_point."""

    def __init__(self, safety_multiplier: float = 1.0) -> None:
        if safety_multiplier < 0.0:
            raise ValueError("safety_multiplier phải lớn hơn hoặc bằng 0")
        self._safety_multiplier = safety_multiplier

    def decide(self, simulator: Simulator) -> Sequence[BusinessAction]:
        state = simulator.state
        actions: list[BusinessAction] = []
        for (sku_id, warehouse_id), sku_state in state.sku_states.items():
            position = (
                sku_state.available
                + sku_state.in_transit
                + _open_po_quantity(simulator, sku_id, warehouse_id)
            )
            if position > sku_state.reorder_point:
                continue
            supplier_id = state.supplier_assignments.get(sku_id)
            if supplier_id is None:
                continue
            supplier = state.suppliers.get(supplier_id)
            if supplier is None:
                continue
            target = sku_state.reorder_point + self._safety_multiplier * sku_state.safety_stock
            quantity = min(supplier.capacity, max(1, round(target - position)))
            actions.append(
                PlaceOrder(
                    supplier_id=supplier_id,
                    warehouse_id=warehouse_id,
                    sku_id=sku_id,
                    quantity=quantity,
                )
            )
        return actions


class GreedyTransferPolicy:
    """Chuyển hàng từ kho dư sang kho thiếu theo từng SKU."""

    def decide(self, simulator: Simulator) -> Sequence[BusinessAction]:
        state = simulator.state
        route_pairs = {
            (route.source_warehouse, route.destination_warehouse) for route in state.routes.values()
        }
        sku_ids = {sku_id for sku_id, _ in state.sku_states}
        actions: list[BusinessAction] = []
        for sku_id in sorted(sku_ids):
            warehouses = [wh for (s, wh) in state.sku_states if s == sku_id]
            for deficit_warehouse in warehouses:
                deficit = state.sku_states[(sku_id, deficit_warehouse)]
                shortfall = deficit.safety_stock - deficit.available
                if shortfall <= 0:
                    continue
                for surplus_warehouse in warehouses:
                    if surplus_warehouse == deficit_warehouse:
                        continue
                    if (surplus_warehouse, deficit_warehouse) not in route_pairs:
                        continue
                    surplus = state.sku_states[(sku_id, surplus_warehouse)]
                    excess = surplus.available - surplus.reorder_point
                    if excess <= 0:
                        continue
                    quantity = min(shortfall, excess)
                    actions.append(
                        TransferStock(
                            source_warehouse=surplus_warehouse,
                            destination_warehouse=deficit_warehouse,
                            sku_id=sku_id,
                            quantity=quantity,
                        )
                    )
                    shortfall -= quantity
                    if shortfall <= 0:
                        break
        return actions


class OraclePolicy:
    """Look-ahead policy: đặt hàng đủ demand tương lai trong horizon."""

    def __init__(self, buffer_days: int = 2) -> None:
        if buffer_days < 0:
            raise ValueError("buffer_days phải lớn hơn hoặc bằng 0")
        self._buffer_days = buffer_days

    def decide(self, simulator: Simulator) -> Sequence[BusinessAction]:
        state = simulator.state
        schedule = simulator.config.demand_schedule
        region_by_warehouse = {
            warehouse.warehouse_id: warehouse.region_id for warehouse in simulator.config.warehouses
        }
        actions: list[BusinessAction] = []
        for (sku_id, warehouse_id), sku_state in state.sku_states.items():
            supplier_id = state.supplier_assignments.get(sku_id)
            if supplier_id is None:
                continue
            supplier = state.suppliers.get(supplier_id)
            if supplier is None:
                continue
            horizon = supplier.current_lead_time + self._buffer_days
            demand_ahead = 0
            for day in range(state.day + 1, min(state.day + 1 + horizon, len(schedule.days) + 1)):
                demand_ahead += (
                    schedule.get(day)
                    .quantities.get(region_by_warehouse[warehouse_id], {})
                    .get(sku_id, 0)
                )
            coverage = (
                sku_state.available
                + sku_state.in_transit
                + _open_po_quantity(simulator, sku_id, warehouse_id)
            )
            shortfall = demand_ahead - coverage
            if shortfall > 0:
                quantity = min(supplier.capacity, shortfall)
                actions.append(
                    PlaceOrder(
                        supplier_id=supplier_id,
                        warehouse_id=warehouse_id,
                        sku_id=sku_id,
                        quantity=quantity,
                    )
                )
        return actions


def _open_po_quantity(simulator: Simulator, sku_id: str, warehouse_id: str) -> int:
    return sum(
        order.quantity - order.received_quantity
        for order in simulator.state.purchase_orders
        if order.sku_id == sku_id
        and order.warehouse_id == warehouse_id
        and order.status in (OrderStatus.OPEN, OrderStatus.PARTIAL)
    )
