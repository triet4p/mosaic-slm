"""Domain models của supply-chain simulator (Phase 1).

Khớp với schema tại mục 3.1–3.3 và danh sách entity tại mục 6.1
của docs/INITIALIZATION.md.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, computed_field

JsonScalar = str | int | float | bool


class DisruptionType(StrEnum):
    """Sáu loại disruption của MVP (mục 1.2 INITIALIZATION.md)."""

    DEMAND_SPIKE = "DEMAND_SPIKE"
    DEMAND_DROP = "DEMAND_DROP"
    SUPPLIER_DELAY = "SUPPLIER_DELAY"
    SUPPLIER_CANCELLATION = "SUPPLIER_CANCELLATION"
    WAREHOUSE_STOCKOUT = "WAREHOUSE_STOCKOUT"
    TRANSPORT_DELAY = "TRANSPORT_DELAY"


class ShipmentStatus(StrEnum):
    IN_TRANSIT = "IN_TRANSIT"
    DELAYED = "DELAYED"
    ARRIVED = "ARRIVED"


class OrderStatus(StrEnum):
    OPEN = "OPEN"
    PARTIAL = "PARTIAL"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


class CustomerOrderStatus(StrEnum):
    OPEN = "OPEN"
    FULFILLED = "FULFILLED"
    BACKORDERED = "BACKORDERED"
    CANCELLED = "CANCELLED"


class Product(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_id: str
    name: str | None = None
    unit_value: float = Field(
        gt=0.0, description="Giá trị 1 đơn vị, dùng làm cơ sở tính holding cost"
    )


class CustomerRegion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region_id: str
    name: str | None = None


class Warehouse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    warehouse_id: str
    region_id: str
    name: str | None = None
    capacity: int = Field(ge=0, description="Sức chứa tối đa tính theo đơn vị hàng")


class SKUState(BaseModel):
    """Trạng thái một SKU tại một kho (mục 3.1 INITIALIZATION.md)."""

    model_config = ConfigDict(extra="forbid")

    sku_id: str
    warehouse_id: str
    on_hand: int = Field(ge=0)
    reserved: int = Field(ge=0)
    in_transit: int = Field(ge=0, default=0)
    backordered: int = Field(ge=0, default=0)
    safety_stock: int = Field(ge=0)
    reorder_point: int = Field(ge=0)
    holding_cost: float = Field(ge=0.0, description="Chi phí lưu kho mỗi đơn vị mỗi ngày")
    stockout_cost: float = Field(ge=0.0, description="Chi phí thiếu hàng mỗi đơn vị mỗi ngày")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def available(self) -> int:
        """Số hàng có thể phục vụ, sau khi trừ reserved."""
        return self.on_hand - self.reserved


class Supplier(BaseModel):
    """Dữ liệu supplier (mục 3.2 INITIALIZATION.md)."""

    model_config = ConfigDict(extra="forbid")

    supplier_id: str
    name: str | None = None
    supported_skus: list[str] = Field(default_factory=list)
    base_lead_time: int = Field(ge=0, description="Lead time cơ bản theo ngày")
    current_lead_time: int = Field(
        ge=0, description="Lead time hiện tại (có thể tăng khi bị disruption)"
    )
    unit_cost: float = Field(ge=0.0, description="Chi phí mỗi đơn vị")
    order_cost: float = Field(ge=0.0, description="Chi phí cố định mỗi lần đặt hàng")
    reliability_score: float = Field(ge=0.0, le=1.0, description="Độ tin cậy 0–1")
    capacity: int = Field(ge=0, description="Số đơn vị tối đa cho một đơn hàng")
    active_disruptions: list[str] = Field(
        default_factory=list, description="Danh sách disruption_id đang ảnh hưởng"
    )


class TransportRoute(BaseModel):
    """Tuyến vận chuyển giữa hai kho."""

    model_config = ConfigDict(extra="forbid")

    route_id: str
    source_warehouse: str
    destination_warehouse: str
    base_travel_time: int = Field(ge=1, description="Thời gian vận chuyển cơ bản theo ngày")
    current_travel_time: int = Field(
        ge=1, description="Thời gian hiện tại, tăng khi tuyến bị disruption"
    )
    cost_per_unit: float = Field(ge=0.0, description="Chi phí vận chuyển mỗi đơn vị")
    active_disruptions: list[str] = Field(default_factory=list)


class PurchaseOrder(BaseModel):
    """Đơn đặt hàng nhà cung cấp."""

    model_config = ConfigDict(extra="forbid")

    order_id: str
    supplier_id: str
    warehouse_id: str
    sku_id: str
    quantity: int = Field(gt=0)
    unit_cost: float = Field(ge=0.0)
    order_cost: float = Field(ge=0.0)
    placed_at: int = Field(ge=0, description="Ngày mô phỏng đặt hàng")
    expected_arrival: int = Field(ge=0, description="Ngày dự kiến giao hàng")
    actual_arrival: int | None = Field(default=None, description="Ngày giao hàng thực tế")
    received_quantity: int = Field(ge=0, default=0)
    status: OrderStatus = OrderStatus.OPEN


class Shipment(BaseModel):
    """Lô hàng vận chuyển (mục 3.3 INITIALIZATION.md)."""

    model_config = ConfigDict(extra="forbid")

    shipment_id: str
    source: str
    destination: str
    sku_id: str
    quantity: int = Field(gt=0)
    dispatch_time: int = Field(ge=0)
    expected_arrival: int = Field(ge=0)
    actual_arrival: int | None = Field(default=None)
    transport_cost: float = Field(ge=0.0)
    status: ShipmentStatus = ShipmentStatus.IN_TRANSIT


class CustomerOrder(BaseModel):
    """Đơn hàng khách hàng theo vùng, dùng để đo fill rate."""

    model_config = ConfigDict(extra="forbid")

    order_id: str
    region_id: str
    sku_id: str
    quantity: int = Field(gt=0)
    placed_at: int = Field(ge=0)
    fulfilled_quantity: int = Field(ge=0, default=0)
    status: CustomerOrderStatus = CustomerOrderStatus.OPEN


class DisruptionEvent(BaseModel):
    """Sự kiện disruption được lập lịch trong episode."""

    model_config = ConfigDict(extra="forbid")

    disruption_id: str
    type: DisruptionType
    start_day: int = Field(ge=0)
    end_day: int | None = Field(default=None, description="None nếu kéo dài đến hết episode")
    subject_type: str = Field(
        description="Loại subject: supplier, warehouse, route, product hoặc region"
    )
    subject_id: str
    params: dict[str, JsonScalar] = Field(
        default_factory=dict,
        description="Tham số theo loại disruption, ví dụ extra_days, multiplier",
    )
