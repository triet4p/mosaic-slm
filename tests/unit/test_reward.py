import pytest

from mosaic.simulator.core import CostLedger, DayCosts, Simulator, SimulatorConfig
from mosaic.simulator.demand import DemandGenerator, DemandProfile, DemandSchedule
from mosaic.simulator.models import SKUState, Supplier, Warehouse
from mosaic.simulator.reward import RewardCalculator
from mosaic.simulator.supplier import SupplierBehavior

SKU = "SKU001"


def make_supplier() -> Supplier:
    return Supplier(
        supplier_id="SUP_A",
        base_lead_time=3,
        current_lead_time=3,
        unit_cost=2.0,
        order_cost=10.0,
        reliability_score=1.0,
        capacity=500,
        supported_skus=[SKU],
    )


def make_schedule(demand: int, num_days: int = 30) -> DemandSchedule:
    generator = DemandGenerator(
        seed=0,
        profiles=[DemandProfile(region_id="R1", product_id=SKU, base_demand=demand)],
    )
    return generator.generate_schedule(num_days=num_days)


def make_simulator(on_hand: int, demand: int) -> Simulator:
    warehouse = Warehouse(warehouse_id="WH1", region_id="R1", capacity=10000)
    config = SimulatorConfig(
        num_days=30,
        seed=1,
        warehouses=[warehouse],
        suppliers=[make_supplier()],
        routes=[],
        initial_sku_states=[
            SKUState(
                sku_id=SKU,
                warehouse_id="WH1",
                on_hand=on_hand,
                reserved=0,
                safety_stock=0,
                reorder_point=0,
                holding_cost=0.1,
                stockout_cost=5.0,
            )
        ],
        demand_schedule=make_schedule(demand),
    )
    return Simulator(config=config, supplier_behavior=SupplierBehavior(seed=config.seed))


def run_days(simulator: Simulator, days: int) -> None:
    for _ in range(days):
        simulator.advance_day()


def test_business_reward_matches_manual_calculation():
    simulator = make_simulator(on_hand=100, demand=10)
    run_days(simulator, 5)
    breakdown = RewardCalculator().business_reward(simulator.state)
    costs = simulator.state.cumulative_costs
    manual_cost = (
        costs.stockout_cost
        + costs.holding_cost
        + costs.ordering_cost
        + costs.transport_cost
        + costs.late_delivery_cost
        + costs.supplier_switch_cost
    )
    assert breakdown.total_cost == pytest.approx(manual_cost)
    assert breakdown.fill_rate == pytest.approx(1.0)
    assert breakdown.service_bonus == pytest.approx(1.0)
    assert breakdown.reward == pytest.approx(-manual_cost + 1.0)


def test_fill_rate_is_fulfilled_over_total_demand():
    simulator = make_simulator(on_hand=100, demand=10)
    run_days(simulator, 3)
    breakdown = RewardCalculator().business_reward(simulator.state)
    assert breakdown.fill_rate == pytest.approx(1.0)


def test_reward_with_no_demand_has_zero_fill_rate():
    simulator = make_simulator(on_hand=100, demand=0)
    run_days(simulator, 3)
    breakdown = RewardCalculator().business_reward(simulator.state)
    assert breakdown.fill_rate == 0.0
    assert breakdown.service_bonus == 0.0


def test_reward_is_negative_when_stockout_occurs():
    simulator = make_simulator(on_hand=0, demand=10)
    run_days(simulator, 3)
    breakdown = RewardCalculator().business_reward(simulator.state)
    assert breakdown.fill_rate == 0.0
    assert breakdown.reward < 0.0
    assert breakdown.total_cost == pytest.approx(simulator.state.cumulative_costs.stockout_cost)


def test_reward_is_deterministic_for_same_state():
    first = make_simulator(on_hand=40, demand=15)
    second = make_simulator(on_hand=40, demand=15)
    run_days(first, 10)
    run_days(second, 10)
    assert first.state == second.state
    assert RewardCalculator().business_reward(first.state) == RewardCalculator().business_reward(
        second.state
    )


def test_day_reward_matches_manual_math():
    day_costs = DayCosts(
        day=1,
        total_demand=10,
        fulfilled_demand=7,
        stockout_units=3,
        late_deliveries=0,
        costs=CostLedger(stockout_cost=15.0, holding_cost=2.0),
    )
    breakdown = RewardCalculator(service_level_bonus_eta=1.0).day_reward(day_costs)
    assert breakdown.total_cost == pytest.approx(17.0)
    assert breakdown.fill_rate == pytest.approx(0.7)
    assert breakdown.reward == pytest.approx(-17.0 + 0.7)


def test_eta_scales_service_bonus():
    simulator = make_simulator(on_hand=100, demand=10)
    run_days(simulator, 2)
    base = RewardCalculator(service_level_bonus_eta=1.0).business_reward(simulator.state)
    scaled = RewardCalculator(service_level_bonus_eta=2.0).business_reward(simulator.state)
    assert scaled.service_bonus == pytest.approx(2.0 * base.service_bonus)


def test_eta_rejects_negative():
    with pytest.raises(ValueError):
        RewardCalculator(service_level_bonus_eta=-1.0)
