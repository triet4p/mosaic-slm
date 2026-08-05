import pytest
from pydantic import ValidationError

from mosaic.simulator.models import (
    CustomerOrder,
    CustomerOrderStatus,
    DisruptionEvent,
    DisruptionType,
    OrderStatus,
    Product,
    PurchaseOrder,
    Shipment,
    ShipmentStatus,
    SKUState,
    Supplier,
    TransportRoute,
    Warehouse,
)


def test_sku_state_available_is_on_hand_minus_reserved():
    state = SKUState(
        sku_id="SKU001",
        warehouse_id="WH1",
        on_hand=10,
        reserved=3,
        safety_stock=2,
        reorder_point=5,
        holding_cost=0.1,
        stockout_cost=10.0,
    )
    assert state.available == 7


def test_sku_state_available_serializes_in_json_dump():
    state = SKUState(
        sku_id="SKU001",
        warehouse_id="WH1",
        on_hand=10,
        reserved=3,
        safety_stock=2,
        reorder_point=5,
        holding_cost=0.1,
        stockout_cost=10.0,
    )
    assert state.model_dump(mode="json")["available"] == 7


def test_sku_state_rejects_negative_on_hand():
    with pytest.raises(ValidationError):
        SKUState(
            sku_id="SKU001",
            warehouse_id="WH1",
            on_hand=-1,
            reserved=0,
            safety_stock=0,
            reorder_point=0,
            holding_cost=0.0,
            stockout_cost=0.0,
        )


def test_sku_state_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        SKUState.model_validate(
            {
                "sku_id": "SKU001",
                "warehouse_id": "WH1",
                "on_hand": 1,
                "reserved": 0,
                "safety_stock": 0,
                "reorder_point": 0,
                "holding_cost": 0.0,
                "stockout_cost": 0.0,
                "typo_field": 1,
            }
        )


def test_supplier_rejects_reliability_out_of_range():
    with pytest.raises(ValidationError):
        Supplier(
            supplier_id="SUP_A",
            base_lead_time=5,
            current_lead_time=5,
            unit_cost=1.0,
            order_cost=10.0,
            reliability_score=1.5,
            capacity=1000,
        )


def test_supplier_supported_skus_defaults_to_empty():
    supplier = Supplier(
        supplier_id="SUP_A",
        base_lead_time=5,
        current_lead_time=5,
        unit_cost=1.0,
        order_cost=10.0,
        reliability_score=0.9,
        capacity=1000,
    )
    assert supplier.supported_skus == []
    assert supplier.active_disruptions == []


def test_shipment_defaults_to_in_transit_without_actual_arrival():
    shipment = Shipment(
        shipment_id="S1",
        source="WH1",
        destination="WH2",
        sku_id="SKU001",
        quantity=50,
        dispatch_time=1,
        expected_arrival=3,
        transport_cost=25.0,
    )
    assert shipment.status == ShipmentStatus.IN_TRANSIT
    assert shipment.actual_arrival is None


def test_purchase_order_partial_status_serializes_as_value():
    order = PurchaseOrder(
        order_id="PO1",
        supplier_id="SUP_A",
        warehouse_id="WH1",
        sku_id="SKU001",
        quantity=100,
        unit_cost=2.0,
        order_cost=10.0,
        placed_at=1,
        expected_arrival=6,
        received_quantity=40,
        status=OrderStatus.PARTIAL,
    )
    assert order.model_dump(mode="json")["status"] == "PARTIAL"


def test_disruption_event_round_trips_json():
    event = DisruptionEvent(
        disruption_id="D1",
        type=DisruptionType.SUPPLIER_DELAY,
        start_day=5,
        subject_type="supplier",
        subject_id="SUP_A",
        params={"extra_days": 3},
    )
    restored = DisruptionEvent.model_validate_json(event.model_dump_json())
    assert restored == event
    assert restored.type is DisruptionType.SUPPLIER_DELAY


def test_customer_order_defaults_to_open():
    order = CustomerOrder(
        order_id="CO1",
        region_id="R1",
        sku_id="SKU001",
        quantity=20,
        placed_at=1,
    )
    assert order.status is CustomerOrderStatus.OPEN
    assert order.fulfilled_quantity == 0


def test_transport_route_rejects_zero_travel_time():
    with pytest.raises(ValidationError):
        TransportRoute(
            route_id="R1",
            source_warehouse="WH1",
            destination_warehouse="WH2",
            base_travel_time=0,
            current_travel_time=1,
            cost_per_unit=1.0,
        )


def test_product_and_warehouse_require_positive_value():
    with pytest.raises(ValidationError):
        Product(product_id="SKU001", unit_value=0.0)
    with pytest.raises(ValidationError):
        Warehouse(warehouse_id="WH1", region_id="R1", capacity=-1)
