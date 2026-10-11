"""Simulator core: vòng lặp hằng ngày và chuyển trạng thái (Sprint 1, task 4).

Thứ tự trong một ngày:
1. `advance_day` tăng ngày, nhận purchase order đến hạn (received_quantity
   vào on_hand, tính late penalty), shipment đến hạn cập nhật on_hand tại
   kho đích, fulfill demand (thiếu → backorder + stockout cost), charge
   holding cost theo on_hand cuối ngày, ghi DayCosts vào ledger.
2. Sau `advance_day`, agent/executor đặt action cho ngày hiện tại
   (placed_at = ngày hiện tại). Chi phí action đặt vào `_pending_costs`
   và được gộp vào DayCosts của lần `advance_day` kế tiếp; cumulative_costs
   luôn đúng khi gộp.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from mosaic.simulator.demand import DemandSchedule
from mosaic.simulator.models import (
    DisruptionEvent,
    DisruptionType,
    OrderStatus,
    PurchaseOrder,
    Shipment,
    ShipmentStatus,
    SKUState,
    Supplier,
    TransportRoute,
    Warehouse,
)
from mosaic.simulator.supplier import SupplierBehavior


class CostLedger(BaseModel):
    """Chi phí tích lũy; mỗi DayCosts lưu delta của một ngày."""

    model_config = ConfigDict(extra="forbid")

    stockout_cost: float = Field(ge=0.0, default=0.0)
    holding_cost: float = Field(ge=0.0, default=0.0)
    ordering_cost: float = Field(ge=0.0, default=0.0)
    transport_cost: float = Field(ge=0.0, default=0.0)
    late_delivery_cost: float = Field(ge=0.0, default=0.0)
    supplier_switch_cost: float = Field(ge=0.0, default=0.0)


class DayCosts(BaseModel):
    """Kết quả một ngày mô phỏng."""

    model_config = ConfigDict(extra="forbid")

    day: int = Field(ge=1)
    total_demand: int = Field(ge=0)
    fulfilled_demand: int = Field(ge=0)
    stockout_units: int = Field(ge=0)
    late_deliveries: int = Field(ge=0)
    costs: CostLedger


def _empty_purchase_orders() -> list[PurchaseOrder]:
    return []


def _empty_shipments() -> list[Shipment]:
    return []


def _empty_day_costs() -> list[DayCosts]:
    return []


def _empty_disruptions() -> list[DisruptionEvent]:
    return []


def _empty_assignments() -> dict[str, str]:
    return {}


class SimulationState(BaseModel):
    """Trạng thái hiện tại của mô phỏng."""

    model_config = ConfigDict(extra="forbid")

    day: int = Field(ge=0, default=0)
    sku_states: dict[tuple[str, str], SKUState]
    suppliers: dict[str, Supplier]
    routes: dict[str, TransportRoute]
    supplier_assignments: dict[str, str] = Field(
        default_factory=_empty_assignments, description="sku_id -> supplier_id đang cung cấp"
    )
    purchase_orders: list[PurchaseOrder] = Field(default_factory=_empty_purchase_orders)
    shipments: list[Shipment] = Field(default_factory=_empty_shipments)
    ledger: list[DayCosts] = Field(default_factory=_empty_day_costs)
    cumulative_costs: CostLedger = Field(default_factory=CostLedger)


class SimulatorConfig(BaseModel):
    """Cấu hình bất biến của một episode."""

    model_config = ConfigDict(extra="forbid")

    num_days: int = Field(ge=1)
    seed: int
    warehouses: list[Warehouse]
    suppliers: list[Supplier]
    routes: list[TransportRoute]
    initial_sku_states: list[SKUState]
    demand_schedule: DemandSchedule
    disruptions: list[DisruptionEvent] = Field(default_factory=_empty_disruptions)
    initial_supplier_assignments: dict[str, str] = Field(
        default_factory=_empty_assignments,
        description="sku_id -> supplier_id; bỏ trống nếu mỗi SKU chỉ có một supplier hỗ trợ",
    )
    late_delivery_penalty_per_day: float = Field(ge=0.0, default=5.0)
    supplier_switch_cost: float = Field(ge=0.0, default=100.0)


class Simulator:
    """Điều phối vòng lặp ngày và cung cấp primitives cho action executor."""

    def __init__(self, config: SimulatorConfig, supplier_behavior: SupplierBehavior) -> None:
        self._config = config
        self._supplier_behavior = supplier_behavior
        self._pending_costs = CostLedger()
        self._validate_config()
        assignments = self._resolve_supplier_assignments()
        self._state = SimulationState(
            sku_states={
                (s.sku_id, s.warehouse_id): s.model_copy(deep=True)
                for s in config.initial_sku_states
            },
            suppliers={s.supplier_id: s.model_copy(deep=True) for s in config.suppliers},
            routes={r.route_id: r.model_copy(deep=True) for r in config.routes},
            supplier_assignments=assignments,
        )

    @property
    def state(self) -> SimulationState:
        return self._state

    @property
    def config(self) -> SimulatorConfig:
        return self._config

    @property
    def supplier_switch_cost(self) -> float:
        return self._config.supplier_switch_cost

    def advance_day(self) -> DayCosts:
        """Tiến một ngày và trả về chi phí của ngày đó."""
        if self._state.day >= self._config.num_days:
            raise ValueError(f"Episode đã kết thúc sau {self._config.num_days} ngày")
        self._state.day += 1
        day = self._state.day
        self._apply_active_disruptions(day)
        costs = self._merge_costs(CostLedger(), self._pending_costs)
        self._pending_costs = CostLedger()
        late_deliveries = self._receive_purchase_orders(day, costs)
        self._arrive_shipments(day)
        total_demand, fulfilled_demand, stockout_units = self._fulfill_demand(day, costs)
        self._charge_holding_cost(costs)
        day_costs = DayCosts(
            day=day,
            total_demand=total_demand,
            fulfilled_demand=fulfilled_demand,
            stockout_units=stockout_units,
            late_deliveries=late_deliveries,
            costs=costs,
        )
        self._state.ledger.append(day_costs)
        self._state.cumulative_costs = self._merge_costs(self._state.cumulative_costs, costs)
        return day_costs

    def place_order(
        self, supplier_id: str, warehouse_id: str, sku_id: str, quantity: int
    ) -> PurchaseOrder:
        """Tạo purchase order, quyết định kết quả giao hàng và charge ordering cost."""
        supplier = self._state.suppliers.get(supplier_id)
        if supplier is None:
            raise ValueError(f"Không có supplier {supplier_id}")
        if sku_id not in supplier.supported_skus:
            raise ValueError(f"Supplier {supplier_id} không cung cấp {sku_id}")
        if quantity <= 0:
            raise ValueError("quantity phải lớn hơn 0")
        if quantity > supplier.capacity:
            raise ValueError(
                f"quantity {quantity} vượt capacity {supplier.capacity} của supplier {supplier_id}"
            )
        self._require_sku_warehouse(sku_id, warehouse_id)
        order = PurchaseOrder(
            order_id=f"PO-{self._config.seed}-{len(self._state.purchase_orders) + 1}",
            supplier_id=supplier_id,
            warehouse_id=warehouse_id,
            sku_id=sku_id,
            quantity=quantity,
            unit_cost=supplier.unit_cost,
            order_cost=supplier.order_cost,
            placed_at=self._state.day,
            expected_arrival=self._state.day + supplier.current_lead_time,
        )
        outcome = self._supplier_behavior.resolve_order_outcome(
            supplier, order, self._config.disruptions
        )
        order.expected_arrival = outcome.expected_arrival
        order.actual_arrival = outcome.actual_arrival
        if outcome.cancelled:
            order.status = OrderStatus.CANCELLED
        else:
            order.received_quantity = order.quantity
        self._state.purchase_orders.append(order)
        self._pending_costs.ordering_cost += supplier.order_cost
        return order

    def calculate_transfer_cost(self, route: TransportRoute, quantity: int) -> float:
        """Return the rounded current-state route charge."""
        if quantity <= 0:
            raise ValueError("quantity phải lớn hơn 0")
        return round(
            quantity * route.cost_per_unit * self._active_transport_cost_multiplier(route), 2
        )

    def dispatch_shipment(
        self, source_warehouse: str, destination_warehouse: str, sku_id: str, quantity: int
    ) -> Shipment:
        """Chuyển hàng giữa hai kho qua tuyến đã khai báo; charge transport cost."""
        route = self._find_route(source_warehouse, destination_warehouse)
        if route is None:
            raise ValueError(
                f"Không có tuyến vận chuyển {source_warehouse} -> {destination_warehouse}"
            )
        if quantity <= 0:
            raise ValueError("quantity phải lớn hơn 0")
        source = self._require_sku_warehouse(sku_id, source_warehouse)
        self._require_sku_warehouse(sku_id, destination_warehouse)
        if quantity > source.available:
            raise ValueError(
                f"Không đủ hàng tại {source_warehouse}: cần {quantity}, chỉ còn {source.available}"
            )
        source.on_hand -= quantity
        source.in_transit += quantity
        transport_cost = self.calculate_transfer_cost(route, quantity)
        shipment = Shipment(
            shipment_id=f"SHP-{self._config.seed}-{len(self._state.shipments) + 1}",
            source=source_warehouse,
            destination=destination_warehouse,
            sku_id=sku_id,
            quantity=quantity,
            dispatch_time=self._state.day,
            expected_arrival=self._state.day + route.current_travel_time,
            actual_arrival=self._state.day + route.current_travel_time,
            transport_cost=transport_cost,
        )
        self._state.shipments.append(shipment)
        self._pending_costs.transport_cost += transport_cost
        return shipment

    def add_pending_costs(self, costs: CostLedger) -> None:
        """Gộp chi phí action (task 5) vào ngày hiện tại."""
        self._pending_costs = self._merge_costs(self._pending_costs, costs)

    def _validate_config(self) -> None:
        if self._config.num_days > len(self._config.demand_schedule.days):
            raise ValueError("num_days vượt quá số ngày trong demand schedule")
        warehouse_ids = {w.warehouse_id for w in self._config.warehouses}
        self._warehouse_by_region = {w.region_id: w.warehouse_id for w in self._config.warehouses}
        scheduled_regions = {
            region for day in self._config.demand_schedule.days for region in day.quantities
        }
        missing_regions = scheduled_regions - set(self._warehouse_by_region)
        if missing_regions:
            raise ValueError(f"Không có warehouse phục vụ các region: {sorted(missing_regions)}")
        for state in self._config.initial_sku_states:
            if state.warehouse_id not in warehouse_ids:
                raise ValueError(
                    f"SKUState {state.sku_id} thuộc kho không tồn tại {state.warehouse_id}"
                )
        for route in self._config.routes:
            if route.source_warehouse not in warehouse_ids:
                raise ValueError(
                    f"Route {route.route_id} bắt đầu từ kho không tồn tại {route.source_warehouse}"
                )
            if route.destination_warehouse not in warehouse_ids:
                raise ValueError(
                    f"Route {route.route_id} kết thúc tại kho không tồn tại "
                    f"{route.destination_warehouse}"
                )

    def _resolve_supplier_assignments(self) -> dict[str, str]:
        sku_ids = {state.sku_id for state in self._config.initial_sku_states}
        for day in self._config.demand_schedule.days:
            for products in day.quantities.values():
                sku_ids.update(products)
        assignments: dict[str, str] = {}
        for sku_id in sorted(sku_ids):
            if sku_id in self._config.initial_supplier_assignments:
                supplier_id = self._config.initial_supplier_assignments[sku_id]
            else:
                supporting = [
                    s.supplier_id for s in self._config.suppliers if sku_id in s.supported_skus
                ]
                if len(supporting) != 1:
                    raise ValueError(
                        f"SKU {sku_id} cần khai báo initial_supplier_assignment: "
                        f"có {len(supporting)} supplier hỗ trợ ({sorted(supporting)})"
                    )
                supplier_id = supporting[0]
            supplier = next(
                (s for s in self._config.suppliers if s.supplier_id == supplier_id), None
            )
            if supplier is None or sku_id not in supplier.supported_skus:
                raise ValueError(
                    f"Supplier {supplier_id} không hỗ trợ SKU {sku_id} trong assignment"
                )
            assignments[sku_id] = supplier_id
        return assignments

    def _apply_active_disruptions(self, day: int) -> None:
        """Cập nhật tác động disruption đang hoạt động vào ngày hiện tại.

        SUPPLIER_DELAY / TRANSPORT_DELAY: cộng extra_days vào current lead
        time / travel time (reset về base mỗi ngày rồi áp dụng lại).
        WAREHOUSE_STOCKOUT: trừ units khỏi on_hand các SKU tại kho, một lần
        vào đúng ngày bắt đầu. DEMAND_SPIKE/DROP và SUPPLIER_CANCELLATION
        được xử lý lần lượt ở demand generator và supplier behavior.
        """
        self._refresh_dynamic_parameters()
        for event in self._config.disruptions:
            if not event.start_day <= day:
                continue
            if event.end_day is not None and day > event.end_day:
                continue
            if event.type is DisruptionType.SUPPLIER_DELAY and event.subject_type == "supplier":
                supplier = self._state.suppliers.get(event.subject_id)
                if supplier is not None:
                    extra_days = self._int_param(event, "extra_days", default=0)
                    supplier.current_lead_time = supplier.base_lead_time + extra_days
            elif event.type is DisruptionType.TRANSPORT_DELAY and event.subject_type == "route":
                route = self._state.routes.get(event.subject_id)
                if route is not None:
                    extra_days = self._int_param(event, "extra_days", default=0)
                    route.current_travel_time = route.base_travel_time + extra_days
            elif event.type is DisruptionType.WAREHOUSE_STOCKOUT and day == event.start_day:
                self._apply_warehouse_stockout(event)

    def _refresh_dynamic_parameters(self) -> None:
        for supplier in self._state.suppliers.values():
            supplier.current_lead_time = supplier.base_lead_time
        for route in self._state.routes.values():
            route.current_travel_time = route.base_travel_time

    def _apply_warehouse_stockout(self, event: DisruptionEvent) -> None:
        units = self._int_param(event, "units", default=100)
        affected = [
            (sku_id, warehouse_id)
            for sku_id, warehouse_id in self._state.sku_states
            if warehouse_id == event.subject_id
        ]
        for sku_id, warehouse_id in affected:
            state = self._state.sku_states[(sku_id, warehouse_id)]
            removed = min(units, state.on_hand)
            state.on_hand -= removed

    def _active_transport_cost_multiplier(self, route: TransportRoute) -> float:
        multiplier = 1.0
        for event in self._config.disruptions:
            if event.type is not DisruptionType.TRANSPORT_DELAY:
                continue
            if event.subject_type != "route" or event.subject_id != route.route_id:
                continue
            if not event.start_day <= self._state.day:
                continue
            if event.end_day is not None and self._state.day > event.end_day:
                continue
            multiplier *= self._float_param(event, "cost_multiplier", default=1.0)
        return multiplier

    @staticmethod
    def _int_param(event: DisruptionEvent, name: str, default: int) -> int:
        value = event.params.get(name, default)
        if not isinstance(value, (int, float)):
            raise ValueError(f"Disruption {event.disruption_id} thiếu tham số '{name}' kiểu số")
        return int(value)

    @staticmethod
    def _float_param(event: DisruptionEvent, name: str, default: float) -> float:
        value = event.params.get(name, default)
        if not isinstance(value, (int, float)):
            raise ValueError(f"Disruption {event.disruption_id} thiếu tham số '{name}' kiểu số")
        return float(value)

    def _receive_purchase_orders(self, day: int, costs: CostLedger) -> int:
        late_deliveries = 0
        for order in self._state.purchase_orders:
            if order.status not in (OrderStatus.OPEN, OrderStatus.PARTIAL):
                continue
            if order.actual_arrival != day:
                continue
            state = self._require_sku_warehouse(order.sku_id, order.warehouse_id)
            state.on_hand += order.received_quantity
            order.status = (
                OrderStatus.COMPLETED
                if order.received_quantity == order.quantity
                else OrderStatus.PARTIAL
            )
            if order.actual_arrival is not None and order.actual_arrival > order.expected_arrival:
                delay_days = order.actual_arrival - order.expected_arrival
                costs.late_delivery_cost += delay_days * self._config.late_delivery_penalty_per_day
                late_deliveries += 1
        return late_deliveries

    def _arrive_shipments(self, day: int) -> None:
        for shipment in self._state.shipments:
            if shipment.status is not ShipmentStatus.IN_TRANSIT:
                continue
            if shipment.actual_arrival != day:
                continue
            source = self._require_sku_warehouse(shipment.sku_id, shipment.source)
            destination = self._require_sku_warehouse(shipment.sku_id, shipment.destination)
            source.in_transit -= shipment.quantity
            destination.on_hand += shipment.quantity
            shipment.status = ShipmentStatus.ARRIVED

    def _fulfill_demand(self, day: int, costs: CostLedger) -> tuple[int, int, int]:
        demand_day = self._config.demand_schedule.get(day)
        total_demand = 0
        fulfilled_demand = 0
        stockout_units = 0
        for region_id, products in demand_day.quantities.items():
            warehouse_id = self._warehouse_by_region[region_id]
            for sku_id, quantity in products.items():
                state = self._require_sku_warehouse(sku_id, warehouse_id)
                total_demand += quantity
                fulfilled = min(state.available, quantity)
                state.on_hand -= fulfilled
                fulfilled_demand += fulfilled
                shortfall = quantity - fulfilled
                if shortfall > 0:
                    state.backordered += shortfall
                    stockout_units += shortfall
                    costs.stockout_cost += shortfall * state.stockout_cost
        return total_demand, fulfilled_demand, stockout_units

    def _charge_holding_cost(self, costs: CostLedger) -> None:
        for state in self._state.sku_states.values():
            costs.holding_cost += state.on_hand * state.holding_cost

    def _find_route(self, source: str, destination: str) -> TransportRoute | None:
        for route in self._state.routes.values():
            if route.source_warehouse == source and route.destination_warehouse == destination:
                return route
        return None

    def _require_sku_warehouse(self, sku_id: str, warehouse_id: str) -> SKUState:
        state = self._state.sku_states.get((sku_id, warehouse_id))
        if state is None:
            raise ValueError(f"Không có SKUState cho {sku_id} tại kho {warehouse_id}")
        return state

    @staticmethod
    def _merge_costs(target: CostLedger, source: CostLedger) -> CostLedger:
        merged = target.model_copy()
        for field_name in type(target).model_fields:
            current = getattr(target, field_name)
            addition = getattr(source, field_name)
            setattr(merged, field_name, current + addition)
        return merged
