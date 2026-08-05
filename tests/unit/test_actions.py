import pytest
from pydantic import TypeAdapter, ValidationError

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
from mosaic.simulator.core import Simulator, SimulatorConfig
from mosaic.simulator.demand import DemandGenerator, DemandProfile, DemandSchedule
from mosaic.simulator.models import SKUState, Supplier, TransportRoute, Warehouse
from mosaic.simulator.supplier import SupplierBehavior

SKU = "SKU001"
ADAPTER: TypeAdapter[BusinessAction] = TypeAdapter(BusinessAction)


def make_supplier(supplier_id: str, lead_time: int = 3) -> Supplier:
    return Supplier(
        supplier_id=supplier_id,
        base_lead_time=lead_time,
        current_lead_time=lead_time,
        unit_cost=2.0,
        order_cost=10.0,
        reliability_score=0.9,
        capacity=500,
        supported_skus=[SKU],
    )


def make_warehouse(warehouse_id: str, region_id: str) -> Warehouse:
    return Warehouse(warehouse_id=warehouse_id, region_id=region_id, capacity=10000)


def make_sku_state(warehouse_id: str, on_hand: int = 100) -> SKUState:
    return SKUState(
        sku_id=SKU,
        warehouse_id=warehouse_id,
        on_hand=on_hand,
        reserved=0,
        safety_stock=10,
        reorder_point=20,
        holding_cost=0.1,
        stockout_cost=5.0,
    )


def make_schedule(num_days: int = 30, demand: int = 0) -> DemandSchedule:
    generator = DemandGenerator(
        seed=0,
        profiles=[DemandProfile(region_id="R1", product_id=SKU, base_demand=demand)],
    )
    return generator.generate_schedule(num_days=num_days)


def make_simulator(
    suppliers: list[Supplier] | None = None,
    initial_assignments: dict[str, str] | None = None,
    switch_cost: float = 100.0,
    demand: int = 0,
    on_hand: int = 100,
) -> Simulator:
    warehouses = [make_warehouse("WH1", "R1"), make_warehouse("WH2", "R2")]
    route = TransportRoute(
        route_id="R1",
        source_warehouse="WH1",
        destination_warehouse="WH2",
        base_travel_time=2,
        current_travel_time=2,
        cost_per_unit=1.0,
    )
    config = SimulatorConfig(
        num_days=30,
        seed=1,
        warehouses=warehouses,
        suppliers=suppliers or [make_supplier("SUP_A")],
        routes=[route],
        initial_sku_states=[
            make_sku_state("WH1", on_hand=on_hand),
            make_sku_state("WH2", on_hand=0),
        ],
        demand_schedule=make_schedule(demand=demand),
        initial_supplier_assignments=initial_assignments or {},
        supplier_switch_cost=switch_cost,
    )
    return Simulator(config=config, supplier_behavior=SupplierBehavior(seed=config.seed))


def test_no_action_does_not_change_state():
    simulator = make_simulator()
    executor = ActionExecutor(simulator)
    before = simulator.state.model_dump()
    executor.apply(NoAction())
    assert simulator.state.model_dump() == before


def test_place_order_creates_order_and_charges_ordering_cost():
    simulator = make_simulator()
    executor = ActionExecutor(simulator)
    simulator.advance_day()
    executor.apply(PlaceOrder(supplier_id="SUP_A", warehouse_id="WH1", sku_id=SKU, quantity=50))
    assert len(simulator.state.purchase_orders) == 1
    simulator.advance_day()
    assert simulator.state.cumulative_costs.ordering_cost == pytest.approx(10.0)


def test_transfer_stock_dispatches_shipment():
    simulator = make_simulator()
    executor = ActionExecutor(simulator)
    simulator.advance_day()
    executor.apply(
        TransferStock(source_warehouse="WH1", destination_warehouse="WH2", sku_id=SKU, quantity=20)
    )
    assert len(simulator.state.shipments) == 1
    simulator.advance_day()
    assert simulator.state.cumulative_costs.transport_cost == pytest.approx(20.0)


def test_switch_supplier_updates_assignment_and_charges_switch_cost():
    simulator = make_simulator(
        suppliers=[make_supplier("SUP_A"), make_supplier("SUP_B")],
        initial_assignments={SKU: "SUP_A"},
    )
    executor = ActionExecutor(simulator)
    simulator.advance_day()
    executor.apply(SwitchSupplier(old_supplier="SUP_A", new_supplier="SUP_B", sku_id=SKU))
    assert simulator.state.supplier_assignments[SKU] == "SUP_B"
    simulator.advance_day()
    assert simulator.state.cumulative_costs.supplier_switch_cost == pytest.approx(100.0)


def test_switch_supplier_rejects_mismatched_old_supplier():
    simulator = make_simulator(
        suppliers=[make_supplier("SUP_A"), make_supplier("SUP_B")],
        initial_assignments={SKU: "SUP_A"},
    )
    executor = ActionExecutor(simulator)
    with pytest.raises(ValueError):
        executor.apply(SwitchSupplier(old_supplier="SUP_B", new_supplier="SUP_A", sku_id=SKU))


def test_switch_supplier_rejects_same_supplier():
    simulator = make_simulator(initial_assignments={SKU: "SUP_A"})
    executor = ActionExecutor(simulator)
    with pytest.raises(ValueError):
        executor.apply(SwitchSupplier(old_supplier="SUP_A", new_supplier="SUP_A", sku_id=SKU))


def test_switch_supplier_rejects_supplier_without_sku():
    unsupported = make_supplier("SUP_B")
    unsupported.supported_skus = []
    simulator = make_simulator(
        suppliers=[make_supplier("SUP_A"), unsupported],
        initial_assignments={SKU: "SUP_A"},
    )
    executor = ActionExecutor(simulator)
    with pytest.raises(ValueError):
        executor.apply(SwitchSupplier(old_supplier="SUP_A", new_supplier="SUP_B", sku_id=SKU))


def test_adjust_safety_stock_updates_level():
    simulator = make_simulator()
    executor = ActionExecutor(simulator)
    executor.apply(AdjustSafetyStock(warehouse_id="WH1", sku_id=SKU, new_level=25))
    assert simulator.state.sku_states[(SKU, "WH1")].safety_stock == 25


def test_adjust_safety_stock_rejects_unknown_sku_warehouse():
    simulator = make_simulator()
    executor = ActionExecutor(simulator)
    with pytest.raises(ValueError):
        executor.apply(AdjustSafetyStock(warehouse_id="WH9", sku_id=SKU, new_level=25))


def test_accept_backorder_writes_off_pending_backorder():
    simulator = make_simulator(demand=10, on_hand=0)
    executor = ActionExecutor(simulator)
    state = simulator.state.sku_states[(SKU, "WH1")]
    simulator.advance_day()
    assert state.backordered == 10
    executor.apply(AcceptBackorder(warehouse_id="WH1", sku_id=SKU, quantity=5))
    assert state.backordered == 5
    executor.apply(AcceptBackorder(warehouse_id="WH1", sku_id=SKU, quantity=5))
    assert state.backordered == 0


def test_accept_backorder_rejects_exceeding_pending():
    simulator = make_simulator(demand=10, on_hand=0)
    executor = ActionExecutor(simulator)
    simulator.advance_day()
    with pytest.raises(ValueError):
        executor.apply(AcceptBackorder(warehouse_id="WH1", sku_id=SKU, quantity=11))


def test_business_action_round_trips_json():
    action: BusinessAction = PlaceOrder(
        supplier_id="SUP_A", warehouse_id="WH1", sku_id=SKU, quantity=30
    )
    restored = ADAPTER.validate_json(ADAPTER.dump_json(action))
    assert restored == action
    assert isinstance(restored, PlaceOrder)


def test_business_action_rejects_unknown_type():
    with pytest.raises(ValidationError):
        ADAPTER.validate_python({"type": "DO_SOMETHING"})


def test_action_executor_rejects_invalid_transfer():
    simulator = make_simulator()
    executor = ActionExecutor(simulator)
    with pytest.raises(ValueError):
        executor.apply(
            TransferStock(
                source_warehouse="WH1", destination_warehouse="WH3", sku_id=SKU, quantity=10
            )
        )


def test_config_requires_assignment_for_ambiguous_sku():
    with pytest.raises(ValueError):
        make_simulator(suppliers=[make_supplier("SUP_A"), make_supplier("SUP_B")])
