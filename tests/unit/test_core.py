import pytest

from mosaic.simulator.core import CostLedger, Simulator, SimulatorConfig
from mosaic.simulator.demand import DemandGenerator, DemandProfile, DemandSchedule
from mosaic.simulator.models import (
    DisruptionEvent,
    DisruptionType,
    OrderStatus,
    ShipmentStatus,
    SKUState,
    Supplier,
    TransportRoute,
    Warehouse,
)
from mosaic.simulator.supplier import SupplierBehavior

SKU = "SKU001"


def make_supplier(
    supplier_id: str = "SUP_A",
    lead_time: int = 3,
    reliability: float = 1.0,
    capacity: int = 500,
) -> Supplier:
    return Supplier(
        supplier_id=supplier_id,
        base_lead_time=lead_time,
        current_lead_time=lead_time,
        unit_cost=2.0,
        order_cost=10.0,
        reliability_score=reliability,
        capacity=capacity,
        supported_skus=[SKU],
    )


def make_warehouse(warehouse_id: str, region_id: str) -> Warehouse:
    return Warehouse(warehouse_id=warehouse_id, region_id=region_id, capacity=10000)


def make_sku_state(
    warehouse_id: str,
    on_hand: int = 100,
    holding_cost: float = 0.1,
    stockout_cost: float = 5.0,
) -> SKUState:
    return SKUState(
        sku_id=SKU,
        warehouse_id=warehouse_id,
        on_hand=on_hand,
        reserved=0,
        safety_stock=0,
        reorder_point=0,
        holding_cost=holding_cost,
        stockout_cost=stockout_cost,
    )


def make_schedule(region_id: str = "R1", quantity: int = 10, num_days: int = 30) -> DemandSchedule:
    generator = DemandGenerator(
        seed=0,
        profiles=[DemandProfile(region_id=region_id, product_id=SKU, base_demand=quantity)],
    )
    return generator.generate_schedule(num_days=num_days)


def make_config(
    num_days: int = 30,
    seed: int = 1,
    on_hand: int = 100,
    demand: int = 10,
    suppliers: list[Supplier] | None = None,
    routes: list[TransportRoute] | None = None,
    disruptions: list[DisruptionEvent] | None = None,
    warehouses: list[Warehouse] | None = None,
) -> SimulatorConfig:
    warehouses = warehouses or [make_warehouse("WH1", "R1")]
    return SimulatorConfig(
        num_days=num_days,
        seed=seed,
        warehouses=warehouses,
        suppliers=suppliers or [make_supplier()],
        routes=routes or [],
        initial_sku_states=[make_sku_state("WH1", on_hand=on_hand)],
        demand_schedule=make_schedule(quantity=demand, num_days=num_days),
        disruptions=disruptions or [],
    )


def make_simulator(config: SimulatorConfig, stochastic: bool = False) -> Simulator:
    behavior = SupplierBehavior(seed=config.seed, stochastic=stochastic)
    return Simulator(config=config, supplier_behavior=behavior)


def test_advance_day_charges_holding_cost_on_end_of_day_stock():
    simulator = make_simulator(make_config(on_hand=100, demand=0))
    day_costs = simulator.advance_day()
    assert day_costs.costs.holding_cost == pytest.approx(10.0)
    assert simulator.state.cumulative_costs.holding_cost == pytest.approx(10.0)


def test_demand_is_fulfilled_from_available_stock():
    simulator = make_simulator(make_config(on_hand=100, demand=10))
    day_costs = simulator.advance_day()
    assert day_costs.total_demand == 10
    assert day_costs.fulfilled_demand == 10
    assert day_costs.stockout_units == 0
    assert simulator.state.sku_states[(SKU, "WH1")].on_hand == 90


def test_stockout_cost_charged_when_demand_exceeds_stock():
    simulator = make_simulator(make_config(on_hand=5, demand=10))
    day_costs = simulator.advance_day()
    assert day_costs.stockout_units == 5
    assert day_costs.costs.stockout_cost == pytest.approx(25.0)
    state = simulator.state.sku_states[(SKU, "WH1")]
    assert state.on_hand == 0
    assert state.backordered == 5


def test_backorder_accumulates_across_days():
    simulator = make_simulator(make_config(on_hand=5, demand=10))
    first = simulator.advance_day()
    second = simulator.advance_day()
    assert first.stockout_units == 5
    assert second.stockout_units == 10
    assert simulator.state.sku_states[(SKU, "WH1")].backordered == 15


def test_purchase_order_arrives_and_increases_stock():
    simulator = make_simulator(make_config(on_hand=0, demand=0))
    simulator.advance_day()
    order = simulator.place_order("SUP_A", "WH1", SKU, 50)
    assert order.placed_at == 1
    assert order.expected_arrival == 4
    for _ in range(3):
        simulator.advance_day()
    state = simulator.state.sku_states[(SKU, "WH1")]
    assert state.on_hand == 50
    assert order.status is OrderStatus.COMPLETED
    assert simulator.state.cumulative_costs.ordering_cost == pytest.approx(10.0)


def test_cancelled_order_never_arrives():
    disruption = DisruptionEvent(
        disruption_id="D1",
        type=DisruptionType.SUPPLIER_CANCELLATION,
        start_day=1,
        end_day=30,
        subject_type="supplier",
        subject_id="SUP_A",
        params={},
    )
    simulator = make_simulator(make_config(on_hand=0, demand=0, disruptions=[disruption]))
    simulator.advance_day()
    order = simulator.place_order("SUP_A", "WH1", SKU, 50)
    assert order.status is OrderStatus.CANCELLED
    assert order.actual_arrival is None
    for _ in range(10):
        simulator.advance_day()
    assert simulator.state.sku_states[(SKU, "WH1")].on_hand == 0


def test_late_order_charges_late_delivery_penalty():
    simulator = make_simulator(make_config(on_hand=0, demand=0, seed=5), stochastic=True)
    supplier = simulator.state.suppliers["SUP_A"]
    supplier.reliability_score = 0.0
    simulator.advance_day()
    order = simulator.place_order("SUP_A", "WH1", SKU, 50)
    assert order.actual_arrival is not None
    assert order.actual_arrival > order.expected_arrival
    delay_days = order.actual_arrival - order.expected_arrival
    for _ in range(order.actual_arrival - 1):
        simulator.advance_day()
    expected_penalty = delay_days * 5.0
    assert simulator.state.cumulative_costs.late_delivery_cost == pytest.approx(expected_penalty)


def test_shipment_moves_inventory_between_warehouses():
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
        num_days=10,
        seed=1,
        warehouses=warehouses,
        suppliers=[make_supplier()],
        routes=[route],
        initial_sku_states=[make_sku_state("WH1", on_hand=50), make_sku_state("WH2", on_hand=0)],
        demand_schedule=make_schedule(region_id="R1", quantity=0, num_days=10),
    )
    simulator = make_simulator(config)
    simulator.advance_day()
    shipment = simulator.dispatch_shipment("WH1", "WH2", SKU, 20)
    source = simulator.state.sku_states[(SKU, "WH1")]
    assert source.on_hand == 30
    assert source.in_transit == 20
    for _ in range(3):
        simulator.advance_day()
    assert simulator.state.cumulative_costs.transport_cost == pytest.approx(20.0)
    assert simulator.state.sku_states[(SKU, "WH1")].in_transit == 0
    assert simulator.state.sku_states[(SKU, "WH2")].on_hand == 20
    assert shipment.status is ShipmentStatus.ARRIVED


def test_place_order_rejects_unsupported_sku():
    simulator = make_simulator(make_config())
    with pytest.raises(ValueError):
        simulator.place_order("SUP_A", "WH1", "SKU999", 10)


def test_place_order_rejects_quantity_over_capacity():
    simulator = make_simulator(make_config())
    with pytest.raises(ValueError):
        simulator.place_order("SUP_A", "WH1", SKU, 1000)


def test_place_order_rejects_unknown_supplier():
    simulator = make_simulator(make_config())
    with pytest.raises(ValueError):
        simulator.place_order("SUP_X", "WH1", SKU, 10)


def test_dispatch_rejects_missing_route():
    simulator = make_simulator(make_config())
    with pytest.raises(ValueError):
        simulator.dispatch_shipment("WH1", "WH2", SKU, 10)


def test_dispatch_rejects_insufficient_stock():
    warehouses = [make_warehouse("WH1", "R1"), make_warehouse("WH2", "R2")]
    route = TransportRoute(
        route_id="R1",
        source_warehouse="WH1",
        destination_warehouse="WH2",
        base_travel_time=1,
        current_travel_time=1,
        cost_per_unit=1.0,
    )
    config = SimulatorConfig(
        num_days=10,
        seed=1,
        warehouses=warehouses,
        suppliers=[make_supplier()],
        routes=[route],
        initial_sku_states=[make_sku_state("WH1", on_hand=5), make_sku_state("WH2", on_hand=0)],
        demand_schedule=make_schedule(quantity=0, num_days=10),
    )
    simulator = make_simulator(config)
    with pytest.raises(ValueError):
        simulator.dispatch_shipment("WH1", "WH2", SKU, 10)


def test_config_rejects_region_without_serving_warehouse():
    config = SimulatorConfig(
        num_days=5,
        seed=1,
        warehouses=[make_warehouse("WH1", "R1")],
        suppliers=[make_supplier()],
        routes=[],
        initial_sku_states=[make_sku_state("WH1")],
        demand_schedule=make_schedule(region_id="R9", quantity=10, num_days=5),
    )
    with pytest.raises(ValueError):
        make_simulator(config)


def test_advance_day_beyond_episode_raises():
    simulator = make_simulator(make_config(num_days=1, demand=0))
    simulator.advance_day()
    with pytest.raises(ValueError):
        simulator.advance_day()


def test_simulation_is_deterministic_across_two_runs():
    def run_episode() -> Simulator:
        simulator = make_simulator(make_config(on_hand=40, demand=15, seed=42), stochastic=True)
        simulator.place_order("SUP_A", "WH1", SKU, 60)
        for _ in range(30):
            simulator.advance_day()
        return simulator

    first = run_episode()
    second = run_episode()
    assert first.state == second.state


def test_cumulative_costs_match_sum_of_daily_costs():
    simulator = make_simulator(make_config(on_hand=40, demand=15))
    for _ in range(5):
        simulator.advance_day()
    expected = CostLedger()
    for day_costs in simulator.state.ledger:
        for field_name in CostLedger.model_fields:
            value = getattr(expected, field_name) + getattr(day_costs.costs, field_name)
            setattr(expected, field_name, value)
    assert simulator.state.cumulative_costs == expected
