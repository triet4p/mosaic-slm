"""Action space và executor (Sprint 1, task 5).

Business actions khớp mục 4.1 INITIALIZATION.md:
NO_ACTION, PLACE_ORDER, TRANSFER_STOCK, SWITCH_SUPPLIER,
ADJUST_SAFETY_STOCK, ACCEPT_BACKORDER.

Thứ tự gọi: action được áp dụng SAU `Simulator.advance_day()`
(đánh giá cuối ngày); chi phí action gộp vào DayCosts lần advance kế tiếp.

Semantics ACCEPT_BACKORDER: xóa quantity đơn vị backorder đang chờ
(ghi nhận bán mất), stockout cost đã tính tại thời điểm thiếu hàng.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from mosaic.simulator.core import CostLedger, Simulator


class NoAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["NO_ACTION"] = "NO_ACTION"


class PlaceOrder(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["PLACE_ORDER"] = "PLACE_ORDER"
    supplier_id: str
    warehouse_id: str
    sku_id: str
    quantity: int = Field(gt=0)


class TransferStock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["TRANSFER_STOCK"] = "TRANSFER_STOCK"
    source_warehouse: str
    destination_warehouse: str
    sku_id: str
    quantity: int = Field(gt=0)


class SwitchSupplier(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["SWITCH_SUPPLIER"] = "SWITCH_SUPPLIER"
    old_supplier: str
    new_supplier: str
    sku_id: str


class AdjustSafetyStock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["ADJUST_SAFETY_STOCK"] = "ADJUST_SAFETY_STOCK"
    warehouse_id: str
    sku_id: str
    new_level: int = Field(ge=0)


class AcceptBackorder(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["ACCEPT_BACKORDER"] = "ACCEPT_BACKORDER"
    warehouse_id: str
    sku_id: str
    quantity: int = Field(gt=0)


type BusinessAction = Annotated[
    NoAction | PlaceOrder | TransferStock | SwitchSupplier | AdjustSafetyStock | AcceptBackorder,
    Field(discriminator="type"),
]


class ActionExecutor:
    """Validate và áp dụng business action lên simulator."""

    def __init__(self, simulator: Simulator) -> None:
        self._simulator = simulator

    def apply(self, action: BusinessAction) -> None:
        """Áp dụng action; action không hợp lệ → ValueError, state không đổi."""
        match action:
            case NoAction():
                return
            case PlaceOrder():
                self._place_order(action)
            case TransferStock():
                self._transfer_stock(action)
            case SwitchSupplier():
                self._switch_supplier(action)
            case AdjustSafetyStock():
                self._adjust_safety_stock(action)
            case AcceptBackorder():
                self._accept_backorder(action)
            case _:
                raise ValueError(f"Action không hỗ trợ: {action!r}")

    def _place_order(self, action: PlaceOrder) -> None:
        self._simulator.place_order(
            action.supplier_id, action.warehouse_id, action.sku_id, action.quantity
        )

    def _transfer_stock(self, action: TransferStock) -> None:
        self._simulator.dispatch_shipment(
            action.source_warehouse, action.destination_warehouse, action.sku_id, action.quantity
        )

    def _switch_supplier(self, action: SwitchSupplier) -> None:
        assignments = self._simulator.state.supplier_assignments
        current = assignments.get(action.sku_id)
        if current is None:
            raise ValueError(f"Chưa có supplier assignment cho SKU {action.sku_id}")
        if action.old_supplier != current:
            raise ValueError(
                f"old_supplier {action.old_supplier} không khớp assignment hiện tại {current}"
            )
        if action.old_supplier == action.new_supplier:
            raise ValueError("old_supplier và new_supplier phải khác nhau")
        suppliers = self._simulator.state.suppliers
        new_supplier = suppliers.get(action.new_supplier)
        if new_supplier is None:
            raise ValueError(f"Không có supplier {action.new_supplier}")
        if action.sku_id not in new_supplier.supported_skus:
            raise ValueError(f"Supplier {action.new_supplier} không cung cấp SKU {action.sku_id}")
        assignments[action.sku_id] = action.new_supplier
        self._simulator.add_pending_costs(
            CostLedger(supplier_switch_cost=self._simulator.supplier_switch_cost)
        )

    def _adjust_safety_stock(self, action: AdjustSafetyStock) -> None:
        state = self._require_sku_state(action.sku_id, action.warehouse_id)
        state.safety_stock = action.new_level

    def _accept_backorder(self, action: AcceptBackorder) -> None:
        state = self._require_sku_state(action.sku_id, action.warehouse_id)
        if state.backordered < action.quantity:
            raise ValueError(
                f"Backorder hiện tại {state.backordered} nhỏ hơn quantity {action.quantity}"
            )
        state.backordered -= action.quantity

    def _require_sku_state(self, sku_id: str, warehouse_id: str):
        state = self._simulator.state.sku_states.get((sku_id, warehouse_id))
        if state is None:
            raise ValueError(f"Không có SKUState cho {sku_id} tại kho {warehouse_id}")
        return state
