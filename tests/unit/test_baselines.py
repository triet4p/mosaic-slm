from pathlib import Path

import pytest

from mosaic.simulator.baselines import (
    DoNothingPolicy,
    GreedyTransferPolicy,
    OraclePolicy,
    ReorderPointPolicy,
    run_policy,
)
from mosaic.simulator.core import Simulator, SimulatorConfig
from mosaic.simulator.demand import DemandGenerator, DemandProfile
from mosaic.simulator.episodes import EpisodeGenerator, EpisodeGeneratorConfig
from mosaic.simulator.models import SKUState, Supplier, TransportRoute, Warehouse
from mosaic.simulator.supplier import SupplierBehavior

SPEC = EpisodeGeneratorConfig(difficulty=1)


def test_do_nothing_places_no_orders():
    episode = EpisodeGenerator(SPEC).generate(seed=1)
    simulator = run_policy(episode, DoNothingPolicy())
    assert simulator.state.purchase_orders == []
    assert simulator.state.shipments == []
    assert simulator.state.day == 30


def test_reorder_point_places_order_when_position_hits_reorder_point():
    simulator = Simulator(
        SimulatorConfig(
            num_days=10,
            seed=1,
            warehouses=[Warehouse(warehouse_id="WH1", region_id="R1", capacity=10000)],
            suppliers=[
                Supplier(
                    supplier_id="SUP_A",
                    supported_skus=["SKU001"],
                    base_lead_time=2,
                    current_lead_time=2,
                    unit_cost=2.0,
                    order_cost=10.0,
                    reliability_score=1.0,
                    capacity=500,
                )
            ],
            routes=[],
            initial_sku_states=[
                SKUState(
                    sku_id="SKU001",
                    warehouse_id="WH1",
                    on_hand=10,
                    reserved=0,
                    safety_stock=5,
                    reorder_point=20,
                    holding_cost=0.1,
                    stockout_cost=5.0,
                )
            ],
            demand_schedule=DemandGenerator(
                seed=0,
                profiles=[DemandProfile(region_id="R1", product_id="SKU001", base_demand=1)],
            ).generate_schedule(num_days=10),
        ),
        supplier_behavior=SupplierBehavior(seed=1),
    )
    simulator.advance_day()
    actions = ReorderPointPolicy().decide(simulator)
    assert len(actions) == 1
    assert actions[0].quantity == 20 + 5 - 9  # type: ignore[attr-defined]


def test_reorder_point_policy_runs_full_episode():
    episode = EpisodeGenerator(SPEC).generate(seed=1)
    simulator = run_policy(episode, ReorderPointPolicy())
    assert simulator.state.day == 30
    assert len(simulator.state.purchase_orders) > 0


def test_greedy_transfer_runs_full_episode_without_errors():
    episode = EpisodeGenerator(SPEC).generate(seed=1)
    simulator = run_policy(episode, GreedyTransferPolicy())
    assert simulator.state.day == 30


def test_greedy_transfer_proposes_transfer_when_surplus_and_deficit_exist():
    simulator = Simulator(
        SimulatorConfig(
            num_days=10,
            seed=1,
            warehouses=[
                Warehouse(warehouse_id="WH1", region_id="R1", capacity=10000),
                Warehouse(warehouse_id="WH2", region_id="R2", capacity=10000),
            ],
            suppliers=[
                Supplier(
                    supplier_id="SUP_A",
                    supported_skus=["SKU001"],
                    base_lead_time=2,
                    current_lead_time=2,
                    unit_cost=2.0,
                    order_cost=10.0,
                    reliability_score=1.0,
                    capacity=500,
                )
            ],
            routes=[
                TransportRoute(
                    route_id="RTE_21",
                    source_warehouse="WH2",
                    destination_warehouse="WH1",
                    base_travel_time=1,
                    current_travel_time=1,
                    cost_per_unit=0.5,
                )
            ],
            initial_sku_states=[
                SKUState(
                    sku_id="SKU001",
                    warehouse_id="WH1",
                    on_hand=5,
                    reserved=0,
                    safety_stock=10,
                    reorder_point=15,
                    holding_cost=0.1,
                    stockout_cost=5.0,
                ),
                SKUState(
                    sku_id="SKU001",
                    warehouse_id="WH2",
                    on_hand=100,
                    reserved=0,
                    safety_stock=10,
                    reorder_point=15,
                    holding_cost=0.1,
                    stockout_cost=5.0,
                ),
            ],
            demand_schedule=DemandGenerator(
                seed=0,
                profiles=[
                    DemandProfile(region_id="R1", product_id="SKU001", base_demand=0),
                    DemandProfile(region_id="R2", product_id="SKU001", base_demand=0),
                ],
            ).generate_schedule(num_days=10),
        ),
        supplier_behavior=SupplierBehavior(seed=1),
    )
    actions = GreedyTransferPolicy().decide(simulator)
    assert len(actions) == 1
    assert actions[0].source_warehouse == "WH2"  # type: ignore[attr-defined]
    assert actions[0].destination_warehouse == "WH1"  # type: ignore[attr-defined]
    assert actions[0].quantity == 5  # type: ignore[attr-defined]


def test_oracle_places_orders_to_cover_future_demand():
    episode = EpisodeGenerator(SPEC).generate(seed=1)
    simulator = run_policy(episode, OraclePolicy())
    assert simulator.state.day == 30
    assert len(simulator.state.purchase_orders) > 0


def test_all_policies_are_deterministic():
    episode = EpisodeGenerator(SPEC).generate(seed=2)
    for policy in (DoNothingPolicy(), ReorderPointPolicy(), GreedyTransferPolicy(), OraclePolicy()):
        first = run_policy(episode, policy)
        second = run_policy(episode, policy)
        assert first.state == second.state


def test_oracle_beats_do_nothing_on_fill_rate_and_stockout():
    episode = EpisodeGenerator(SPEC).generate(seed=1)
    do_nothing = run_policy(episode, DoNothingPolicy())
    oracle = run_policy(episode, OraclePolicy())

    def fill_rate(simulator: Simulator) -> float:
        total = sum(day.total_demand for day in simulator.state.ledger)
        fulfilled = sum(day.fulfilled_demand for day in simulator.state.ledger)
        return fulfilled / total if total > 0 else 0.0

    assert fill_rate(oracle) > fill_rate(do_nothing)
    assert (
        oracle.state.cumulative_costs.stockout_cost
        < do_nothing.state.cumulative_costs.stockout_cost
    )


def test_policies_are_replayable_from_file(tmp_path: Path):
    from mosaic.simulator.replay import load_episodes, save_episodes

    episode = EpisodeGenerator(SPEC).generate(seed=3)
    path = tmp_path / "ep.jsonl"
    save_episodes([episode], path)
    loaded = load_episodes(path)[0]
    direct = run_policy(episode, ReorderPointPolicy())
    replayed = run_policy(loaded, ReorderPointPolicy())
    assert replayed.state == direct.state


def test_reorder_point_rejects_negative_safety_multiplier():
    with pytest.raises(ValueError):
        ReorderPointPolicy(safety_multiplier=-1.0)


def test_oracle_rejects_negative_buffer_days():
    with pytest.raises(ValueError):
        OraclePolicy(buffer_days=-1)
