import pytest

from mosaic.simulator.core import Simulator, SimulatorConfig
from mosaic.simulator.demand import DemandGenerator, DemandProfile, DemandSchedule
from mosaic.simulator.disruptions import DisruptionEngine, DisruptionEngineConfig
from mosaic.simulator.models import (
    DisruptionEvent,
    DisruptionType,
    SKUState,
    Supplier,
    TransportRoute,
    Warehouse,
)
from mosaic.simulator.supplier import SupplierBehavior

SKU = "SKU001"


def make_engine(
    seed: int = 7,
    difficulty: int = 2,
    num_disruptions: int = 2,
    num_days: int = 30,
) -> DisruptionEngine:
    return DisruptionEngine(
        DisruptionEngineConfig(
            seed=seed,
            difficulty=difficulty,
            num_disruptions=num_disruptions,
            num_days=num_days,
            regions=["R1", "R2"],
            products=["P1", "P2"],
            suppliers=["SUP_A", "SUP_B"],
            warehouses=["WH1", "WH2"],
            routes=["RTE_AB"],
        )
    )


def test_engine_is_deterministic_for_same_seed():
    first = make_engine(seed=42).generate()
    second = make_engine(seed=42).generate()
    assert first == second


def test_engine_difficulty_1_generates_exactly_one_event():
    events = make_engine(difficulty=1).generate()
    assert len(events) == 1
    assert events[0].type in DisruptionType
    assert events[0].end_day is not None
    assert events[0].end_day >= events[0].start_day


def test_engine_difficulty_2_honors_num_disruptions():
    events = make_engine(difficulty=2, num_disruptions=3).generate()
    assert len(events) == 3


def test_engine_difficulty_3_can_repeat_events():
    events = make_engine(difficulty=3, num_disruptions=2).generate()
    assert len(events) >= 2
    recurring = [e for e in events if e.disruption_id.endswith("R")]
    assert len(recurring) >= 1


def test_engine_events_have_valid_subjects():
    events = make_engine(difficulty=2, num_disruptions=5).generate()
    suppliers = {"SUP_A", "SUP_B"}
    warehouses = {"WH1", "WH2"}
    routes = {"RTE_AB"}
    for event in events:
        assert event.end_day is not None
        assert 1 <= event.start_day <= event.end_day <= 30
        if event.type in (DisruptionType.DEMAND_SPIKE, DisruptionType.DEMAND_DROP):
            assert event.subject_type in {"region", "product"}
        elif event.type in (DisruptionType.SUPPLIER_DELAY, DisruptionType.SUPPLIER_CANCELLATION):
            assert event.subject_type == "supplier"
            assert event.subject_id in suppliers
        elif event.type is DisruptionType.WAREHOUSE_STOCKOUT:
            assert event.subject_type == "warehouse"
            assert event.subject_id in warehouses
        elif event.type is DisruptionType.TRANSPORT_DELAY:
            assert event.subject_type == "route"
            assert event.subject_id in routes


def test_engine_events_have_required_params():
    events = make_engine(difficulty=2, num_disruptions=5).generate()
    for event in events:
        if event.type is DisruptionType.DEMAND_SPIKE:
            assert float(event.params["multiplier"]) > 1.0
        elif event.type is DisruptionType.DEMAND_DROP:
            assert float(event.params["multiplier"]) < 1.0
        elif event.type is DisruptionType.SUPPLIER_DELAY:
            assert float(event.params["extra_days"]) >= 1
        elif event.type is DisruptionType.WAREHOUSE_STOCKOUT:
            assert float(event.params["units"]) >= 1
        elif event.type is DisruptionType.TRANSPORT_DELAY:
            assert float(event.params["extra_days"]) >= 1
            assert float(event.params["cost_multiplier"]) >= 1.0


def test_engine_different_seeds_generate_different_events():
    first = make_engine(seed=1, difficulty=2, num_disruptions=3).generate()
    second = make_engine(seed=2, difficulty=2, num_disruptions=3).generate()
    assert first != second


def test_engine_rejects_invalid_difficulty():
    with pytest.raises(ValueError):
        make_engine(difficulty=4)


def test_engine_rejects_empty_subject_pools():
    with pytest.raises(ValueError):
        DisruptionEngine(
            DisruptionEngineConfig(seed=1, difficulty=2, num_disruptions=1, num_days=30)
        )


def make_supplier(supplier_id: str, lead_time: int = 3) -> Supplier:
    return Supplier(
        supplier_id=supplier_id,
        base_lead_time=lead_time,
        current_lead_time=lead_time,
        unit_cost=2.0,
        order_cost=10.0,
        reliability_score=1.0,
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
        safety_stock=0,
        reorder_point=0,
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
    disruptions: list[DisruptionEvent],
    schedule: DemandSchedule | None = None,
) -> Simulator:
    route = TransportRoute(
        route_id="RTE_AB",
        source_warehouse="WH1",
        destination_warehouse="WH2",
        base_travel_time=2,
        current_travel_time=2,
        cost_per_unit=1.0,
    )
    config = SimulatorConfig(
        num_days=30,
        seed=1,
        warehouses=[make_warehouse("WH1", "R1"), make_warehouse("WH2", "R2")],
        suppliers=[make_supplier("SUP_A")],
        routes=[route],
        initial_sku_states=[make_sku_state("WH1"), make_sku_state("WH2", on_hand=0)],
        demand_schedule=schedule or make_schedule(),
        disruptions=disruptions,
    )
    return Simulator(config=config, supplier_behavior=SupplierBehavior(seed=config.seed))


def delay_event(
    supplier_id: str = "SUP_A", start: int = 5, end: int = 8, extra: int = 3
) -> DisruptionEvent:
    return DisruptionEvent(
        disruption_id="DELAY1",
        type=DisruptionType.SUPPLIER_DELAY,
        start_day=start,
        end_day=end,
        subject_type="supplier",
        subject_id=supplier_id,
        params={"extra_days": extra},
    )


def test_supplier_delay_increases_lead_time_during_window_only():
    simulator = make_simulator([delay_event()])
    for _ in range(5):
        simulator.advance_day()
    assert simulator.state.suppliers["SUP_A"].current_lead_time == 6
    order = simulator.place_order("SUP_A", "WH1", SKU, 50)
    assert order.expected_arrival == 5 + 6
    for _ in range(4):
        simulator.advance_day()
    assert simulator.state.suppliers["SUP_A"].current_lead_time == 3
    order_after = simulator.place_order("SUP_A", "WH1", SKU, 50)
    assert order_after.expected_arrival == 9 + 3


def test_transport_delay_increases_travel_time_and_cost():
    event = DisruptionEvent(
        disruption_id="TD1",
        type=DisruptionType.TRANSPORT_DELAY,
        start_day=3,
        end_day=5,
        subject_type="route",
        subject_id="RTE_AB",
        params={"extra_days": 2, "cost_multiplier": 2.0},
    )
    simulator = make_simulator([event])
    for _ in range(3):
        simulator.advance_day()
    shipment = simulator.dispatch_shipment("WH1", "WH2", SKU, 10)
    assert shipment.expected_arrival == 3 + 4
    assert shipment.transport_cost == pytest.approx(20.0)
    for _ in range(3):
        simulator.advance_day()
    assert simulator.state.suppliers["SUP_A"].current_lead_time == 3
    shipment_after = simulator.dispatch_shipment("WH1", "WH2", SKU, 10)
    assert shipment_after.transport_cost == pytest.approx(10.0)


def test_warehouse_stockout_reduces_on_hand_once():
    event = DisruptionEvent(
        disruption_id="WS1",
        type=DisruptionType.WAREHOUSE_STOCKOUT,
        start_day=2,
        end_day=5,
        subject_type="warehouse",
        subject_id="WH1",
        params={"units": 60},
    )
    simulator = make_simulator([event])
    for _ in range(2):
        simulator.advance_day()
    assert simulator.state.sku_states[(SKU, "WH1")].on_hand == 40
    for _ in range(4):
        simulator.advance_day()
    assert simulator.state.sku_states[(SKU, "WH1")].on_hand == 40


def test_episode_with_engine_disruptions_is_deterministic():
    events = make_engine(seed=9, difficulty=2, num_disruptions=3).generate()

    def run() -> Simulator:
        simulator = make_simulator(disruptions=events)
        for _ in range(30):
            simulator.advance_day()
        return simulator

    assert run().state == run().state


def test_demand_disruption_flows_into_schedule_and_simulator():
    spike = DisruptionEvent(
        disruption_id="SP1",
        type=DisruptionType.DEMAND_SPIKE,
        start_day=5,
        end_day=7,
        subject_type="region",
        subject_id="R1",
        params={"multiplier": 2.0},
    )
    generator = DemandGenerator(
        seed=0,
        profiles=[DemandProfile(region_id="R1", product_id=SKU, base_demand=10)],
    )
    schedule = generator.generate_schedule(num_days=30, disruptions=[spike])
    simulator = make_simulator(disruptions=[spike], schedule=schedule)
    for _ in range(5):
        simulator.advance_day()
    assert simulator.state.sku_states[(SKU, "WH1")].on_hand == 100 - (4 * 10 + 20)
