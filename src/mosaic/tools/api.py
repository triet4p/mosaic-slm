"""Tool API cho agent (Sprint 1, task 10).

Read tools: get_inventory_state, get_demand_history, get_supplier_status,
get_open_orders, get_shipments.
Compute tools: forecast_demand (moving average), calculate_reorder_point,
calculate_transfer_cost, calculate_order_cost, simulate_action.

Tool context gắn vào một Simulator + RewardCalculator. ToolRunner validate
arguments bằng pydantic (JSON schema sinh từ input model), bắt lỗi và trả
{"success": bool, "result" | "error"}. Tool memory (query_shared_graph,
write_private_memory, propose_shared_fact) thuộc Phase 3.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mosaic.simulator.actions import ActionExecutor, BusinessAction
from mosaic.simulator.core import Simulator
from mosaic.simulator.models import OrderStatus, ShipmentStatus
from mosaic.simulator.reward import RewardCalculator


class ToolContext(BaseModel):
    """Ngữ cảnh thực thi tool: simulator + reward calculator."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    simulator: Simulator
    reward_calculator: RewardCalculator = Field(default_factory=RewardCalculator)


class ToolSpec(BaseModel):
    """Khai báo tool cho LLM (JSON schema từ input model)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters_schema: dict[str, object]


class GetInventoryStateParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sku_id: str | None = None
    warehouse_id: str | None = None


class GetDemandHistoryParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sku_id: str | None = None
    region_id: str | None = None
    days: int = Field(ge=1, le=30, default=7)


class ForecastDemandParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sku_id: str
    region_id: str
    horizon: int = Field(ge=1, le=14, default=7)
    window: int = Field(ge=1, le=14, default=5)


class GetSupplierStatusParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    supplier_id: str | None = None


class GetOpenOrdersParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sku_id: str | None = None
    warehouse_id: str | None = None
    supplier_id: str | None = None


class GetShipmentsParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sku_id: str | None = None
    status: ShipmentStatus | None = None


class CalculateReorderPointParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sku_id: str
    warehouse_id: str
    daily_demand: float = Field(gt=0.0)
    lead_time_days: int = Field(ge=1)
    safety_factor: float = Field(ge=0.0, default=1.5, description="Số ngày demand dự trữ an toàn")


class CalculateTransferCostParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_warehouse: str
    destination_warehouse: str
    sku_id: str
    quantity: int = Field(gt=0)


class CalculateOrderCostParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    supplier_id: str
    sku_id: str
    quantity: int = Field(gt=0)


class SimulateActionParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: BusinessAction
    horizon: int = Field(ge=1, le=30, default=1)


ToolHandler = Callable[
    [ToolContext, Any], dict[str, object]
]  # Any: params là BaseModel con cụ thể theo từng tool


def _get_inventory_state(
    context: ToolContext, params: GetInventoryStateParams
) -> dict[str, object]:
    states: list[dict[str, object]] = []
    for (sku_id, warehouse_id), state in context.simulator.state.sku_states.items():
        if params.sku_id is not None and sku_id != params.sku_id:
            continue
        if params.warehouse_id is not None and warehouse_id != params.warehouse_id:
            continue
        states.append(state.model_dump(mode="json"))
    return {"states": states}


def _get_demand_history(context: ToolContext, params: GetDemandHistoryParams) -> dict[str, object]:
    schedule = context.simulator.config.demand_schedule
    current_day = context.simulator.state.day
    history: list[dict[str, object]] = []
    first_day = max(1, current_day - params.days + 1)
    for day in range(first_day, current_day + 1):
        demand_day = schedule.get(day)
        for region_id, products in demand_day.quantities.items():
            for sku_id, quantity in products.items():
                if params.sku_id is not None and sku_id != params.sku_id:
                    continue
                if params.region_id is not None and region_id != params.region_id:
                    continue
                history.append(
                    {"day": day, "region_id": region_id, "sku_id": sku_id, "quantity": quantity}
                )
    return {"history": history}


def _forecast_demand(context: ToolContext, params: ForecastDemandParams) -> dict[str, object]:
    schedule = context.simulator.config.demand_schedule
    current_day = context.simulator.state.day
    history: list[int] = []
    first_day = max(1, current_day - params.window + 1)
    for day in range(first_day, current_day + 1):
        demand_day = schedule.get(day)
        quantity = demand_day.quantities.get(params.region_id, {}).get(params.sku_id)
        if quantity is not None:
            history.append(quantity)
    if not history:
        raise ValueError(
            f"Chưa có lịch sử demand cho {params.sku_id} tại {params.region_id} "
            f"(cần ít nhất 1 ngày)"
        )
    forecast_value = round(sum(history) / len(history))
    forecast = [
        {"day": current_day + offset, "quantity": forecast_value}
        for offset in range(1, params.horizon + 1)
    ]
    return {"forecast": forecast}


def _get_supplier_status(
    context: ToolContext, params: GetSupplierStatusParams
) -> dict[str, object]:
    suppliers: list[dict[str, object]] = []
    for supplier in context.simulator.state.suppliers.values():
        if params.supplier_id is not None and supplier.supplier_id != params.supplier_id:
            continue
        suppliers.append(supplier.model_dump(mode="json"))
    return {"suppliers": suppliers}


def _get_open_orders(context: ToolContext, params: GetOpenOrdersParams) -> dict[str, object]:
    orders: list[dict[str, object]] = []
    for order in context.simulator.state.purchase_orders:
        if order.status in (OrderStatus.COMPLETED, OrderStatus.CANCELLED):
            continue
        if params.sku_id is not None and order.sku_id != params.sku_id:
            continue
        if params.warehouse_id is not None and order.warehouse_id != params.warehouse_id:
            continue
        if params.supplier_id is not None and order.supplier_id != params.supplier_id:
            continue
        orders.append(order.model_dump(mode="json"))
    return {"orders": orders}


def _get_shipments(context: ToolContext, params: GetShipmentsParams) -> dict[str, object]:
    shipments: list[dict[str, object]] = []
    for shipment in context.simulator.state.shipments:
        if params.sku_id is not None and shipment.sku_id != params.sku_id:
            continue
        if params.status is not None and shipment.status is not params.status:
            continue
        shipments.append(shipment.model_dump(mode="json"))
    return {"shipments": shipments}


def _calculate_reorder_point(
    context: ToolContext, params: CalculateReorderPointParams
) -> dict[str, object]:
    if (params.sku_id, params.warehouse_id) not in context.simulator.state.sku_states:
        raise ValueError(f"Không có SKUState cho {params.sku_id} tại {params.warehouse_id}")
    reorder_point = (
        params.daily_demand * params.lead_time_days + params.safety_factor * params.daily_demand
    )
    return {"reorder_point": round(reorder_point, 2)}


def _calculate_transfer_cost(
    context: ToolContext, params: CalculateTransferCostParams
) -> dict[str, object]:
    route = next(
        (
            route
            for route in context.simulator.state.routes.values()
            if route.source_warehouse == params.source_warehouse
            and route.destination_warehouse == params.destination_warehouse
        ),
        None,
    )
    if route is None:
        raise ValueError(
            f"Không có tuyến vận chuyển {params.source_warehouse} -> {params.destination_warehouse}"
        )
    cost = params.quantity * route.cost_per_unit
    return {"transfer_cost": round(cost, 2), "travel_time_days": route.current_travel_time}


def _calculate_order_cost(
    context: ToolContext, params: CalculateOrderCostParams
) -> dict[str, object]:
    supplier = context.simulator.state.suppliers.get(params.supplier_id)
    if supplier is None:
        raise ValueError(f"Không có supplier {params.supplier_id}")
    unit_total = params.quantity * supplier.unit_cost
    return {
        "fixed_order_cost": supplier.order_cost,
        "unit_total_cost": round(unit_total, 2),
        "total_cost": round(supplier.order_cost + unit_total, 2),
    }


def _simulate_action(context: ToolContext, params: SimulateActionParams) -> dict[str, object]:
    simulator = copy.deepcopy(context.simulator)
    ActionExecutor(simulator).apply(params.action)
    for _ in range(params.horizon):
        simulator.advance_day()
    breakdown = context.reward_calculator.business_reward(simulator.state)
    return {
        "final_day": simulator.state.day,
        "total_cost": round(breakdown.total_cost, 2),
        "fill_rate": breakdown.fill_rate,
        "reward": round(breakdown.reward, 2),
    }


class ToolRunner:
    """Registry + executor các tool; lỗi trả về trong kết quả, không raise."""

    def __init__(self, context: ToolContext) -> None:
        self._context = context
        self._handlers: dict[str, tuple[str, type[BaseModel], ToolHandler]] = {}
        self._register_all()

    @property
    def context(self) -> ToolContext:
        """Ngữ cảnh (simulator + reward calculator) cho backbone đọc state."""
        return self._context

    @property
    def specs(self) -> list[ToolSpec]:
        """Khai báo tool (JSON schema) cho LLM."""
        return [
            ToolSpec(
                name=name,
                description=description,
                parameters_schema=input_model.model_json_schema(),
            )
            for name, (description, input_model, _) in self._handlers.items()
        ]

    def run(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
        """Thực thi tool; lỗi → {"success": false, "error": ...}."""
        entry = self._handlers.get(name)
        if entry is None:
            return {"success": False, "error": f"Tool không tồn tại: {name}"}
        _, input_model, handler = entry
        try:
            params = input_model.model_validate(arguments)
            result = handler(self._context, params)
        except (ValidationError, ValueError, IndexError, KeyError) as exc:
            return {"success": False, "error": str(exc)}
        return {"success": True, "result": result}

    def _register_all(self) -> None:
        self._register(
            "get_inventory_state",
            "Lấy trạng thái tồn kho (on_hand, available, in_transit, backordered, "
            "safety_stock) theo sku/warehouse.",
            GetInventoryStateParams,
            _get_inventory_state,
        )
        self._register(
            "get_demand_history",
            "Lấy lịch sử demand theo kế hoạch (schedule) của các ngày đã qua.",
            GetDemandHistoryParams,
            _get_demand_history,
        )
        self._register(
            "forecast_demand",
            "Dự báo demand ngắn hạn bằng moving average của lịch sử schedule.",
            ForecastDemandParams,
            _forecast_demand,
        )
        self._register(
            "get_supplier_status",
            "Lấy trạng thái supplier: lead time, reliability, cost, capacity, disruptions.",
            GetSupplierStatusParams,
            _get_supplier_status,
        )
        self._register(
            "get_open_orders",
            "Lấy các purchase order đang mở (chưa hoàn thành/hủy).",
            GetOpenOrdersParams,
            _get_open_orders,
        )
        self._register(
            "get_shipments",
            "Lấy danh sách shipment theo sku/status.",
            GetShipmentsParams,
            _get_shipments,
        )
        self._register(
            "calculate_reorder_point",
            "Tính reorder point = daily_demand * lead_time + safety_factor * daily_demand.",
            CalculateReorderPointParams,
            _calculate_reorder_point,
        )
        self._register(
            "calculate_transfer_cost",
            "Tính chi phí chuyển hàng giữa hai kho qua tuyến hiện có.",
            CalculateTransferCostParams,
            _calculate_transfer_cost,
        )
        self._register(
            "calculate_order_cost",
            "Tính chi phí đặt hàng: fixed order cost + quantity * unit_cost.",
            CalculateOrderCostParams,
            _calculate_order_cost,
        )
        self._register(
            "simulate_action",
            "Mô phỏng what-if: áp dụng action trên bản sao và chạy thêm horizon "
            "ngày, trả về reward.",
            SimulateActionParams,
            _simulate_action,
        )

    def _register(
        self,
        name: str,
        description: str,
        input_model: type[BaseModel],
        handler: ToolHandler,
    ) -> None:
        self._handlers[name] = (description, input_model, handler)
