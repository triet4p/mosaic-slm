from typing import Any, cast

import pytest

from mosaic.simulator.actions import ActionExecutor, PlaceOrder, TransferStock
from mosaic.simulator.core import Simulator
from mosaic.simulator.episodes import EpisodeGenerator, EpisodeGeneratorConfig
from mosaic.simulator.supplier import SupplierBehavior
from mosaic.tools.api import ToolContext, ToolRunner

SPEC = EpisodeGeneratorConfig(difficulty=1)


def make_runner(days: int = 3) -> tuple[ToolRunner, ToolContext]:
    episode = EpisodeGenerator(SPEC).generate(seed=1)
    simulator = Simulator(
        episode.config,
        supplier_behavior=SupplierBehavior(
            seed=episode.config.seed, stochastic=episode.supplier_stochastic
        ),
    )
    for _ in range(days):
        simulator.advance_day()
    context = ToolContext(simulator=simulator)
    return ToolRunner(context), context


def ok_payload(result: dict[str, object]) -> dict[str, Any]:
    """Result của tool là ranh giới JSON với LLM nên payload dùng Any."""
    assert result["success"] is True
    return cast(dict[str, Any], result["result"])


def test_specs_contain_all_ten_tools_with_schemas():
    runner, _ = make_runner()
    names = [spec.name for spec in runner.specs]
    assert names == [
        "get_inventory_state",
        "get_demand_history",
        "forecast_demand",
        "get_supplier_status",
        "get_open_orders",
        "get_shipments",
        "calculate_reorder_point",
        "calculate_transfer_cost",
        "calculate_order_cost",
        "simulate_action",
    ]
    for spec in runner.specs:
        assert spec.description
        assert "properties" in spec.parameters_schema


def test_get_inventory_state_returns_all_states_with_available():
    runner, _ = make_runner()
    payload = ok_payload(runner.run("get_inventory_state", {}))
    assert len(payload["states"]) == 60
    assert "available" in payload["states"][0]


def test_get_inventory_state_filters_by_sku_and_warehouse():
    runner, _ = make_runner()
    payload = ok_payload(
        runner.run("get_inventory_state", {"sku_id": "SKU001", "warehouse_id": "WH1"})
    )
    assert len(payload["states"]) == 1
    assert payload["states"][0]["sku_id"] == "SKU001"
    assert payload["states"][0]["warehouse_id"] == "WH1"


def test_get_demand_history_returns_past_days_only():
    runner, _ = make_runner(days=3)
    payload = ok_payload(runner.run("get_demand_history", {"days": 3}))
    assert len(payload["history"]) == 3 * 3 * 20
    days = {entry["day"] for entry in payload["history"]}
    assert days == {1, 2, 3}


def test_forecast_demand_uses_moving_average_of_schedule():
    runner, _ = make_runner(days=3)
    payload = ok_payload(
        runner.run(
            "forecast_demand", {"sku_id": "SKU001", "region_id": "R1", "horizon": 3, "window": 5}
        )
    )
    assert len(payload["forecast"]) == 3
    assert payload["forecast"][0]["day"] == 4
    assert (
        payload["forecast"][0]["quantity"]
        == payload["forecast"][1]["quantity"]
        == payload["forecast"][2]["quantity"]
    )


def test_get_supplier_status_returns_suppliers():
    runner, _ = make_runner()
    payload = ok_payload(runner.run("get_supplier_status", {}))
    assert {s["supplier_id"] for s in payload["suppliers"]} == {"SUP_A", "SUP_B"}


def test_get_open_orders_returns_only_open_orders():
    runner, context = make_runner()
    ActionExecutor(context.simulator).apply(
        PlaceOrder(supplier_id="SUP_A", warehouse_id="WH1", sku_id="SKU001", quantity=100)
    )
    payload = ok_payload(runner.run("get_open_orders", {}))
    assert len(payload["orders"]) == 1
    assert payload["orders"][0]["status"] == "OPEN"


def test_get_shipments_filters_by_status():
    runner, context = make_runner()
    ActionExecutor(context.simulator).apply(
        TransferStock(
            source_warehouse="WH1", destination_warehouse="WH2", sku_id="SKU001", quantity=10
        )
    )
    payload = ok_payload(runner.run("get_shipments", {"status": "IN_TRANSIT"}))
    assert len(payload["shipments"]) == 1


def test_calculate_reorder_point_matches_formula():
    runner, _ = make_runner()
    payload = ok_payload(
        runner.run(
            "calculate_reorder_point",
            {"sku_id": "SKU001", "warehouse_id": "WH1", "daily_demand": 10, "lead_time_days": 3},
        )
    )
    assert payload["reorder_point"] == pytest.approx(10 * 3 + 1.5 * 10)


def test_calculate_reorder_point_rejects_unknown_sku():
    runner, _ = make_runner()
    result = runner.run(
        "calculate_reorder_point",
        {"sku_id": "SKU999", "warehouse_id": "WH1", "daily_demand": 10, "lead_time_days": 3},
    )
    assert result["success"] is False
    assert "SKU999" in result["error"]  # type: ignore[operator]


def test_calculate_transfer_cost_uses_route():
    runner, _ = make_runner()
    payload = ok_payload(
        runner.run(
            "calculate_transfer_cost",
            {
                "source_warehouse": "WH1",
                "destination_warehouse": "WH2",
                "sku_id": "SKU001",
                "quantity": 10,
            },
        )
    )
    assert payload["transfer_cost"] > 0
    assert payload["travel_time_days"] == 2


def test_calculate_transfer_cost_rejects_missing_route():
    runner, _ = make_runner()
    result = runner.run(
        "calculate_transfer_cost",
        {
            "source_warehouse": "WH1",
            "destination_warehouse": "WH3",
            "sku_id": "SKU001",
            "quantity": 10,
        },
    )
    assert result["success"] is False


def test_calculate_order_cost_sums_fixed_and_unit():
    runner, _ = make_runner()
    payload = ok_payload(
        runner.run(
            "calculate_order_cost", {"supplier_id": "SUP_A", "sku_id": "SKU001", "quantity": 100}
        )
    )
    assert payload["fixed_order_cost"] == 10.0
    assert payload["unit_total_cost"] == pytest.approx(100 * 2.0)
    assert payload["total_cost"] == pytest.approx(10.0 + 200.0)


def test_simulate_action_does_not_mutate_original_state():
    runner, context = make_runner(days=3)
    simulator = context.simulator
    before = simulator.state.model_dump()
    payload = ok_payload(
        runner.run(
            "simulate_action",
            {
                "action": {
                    "type": "PLACE_ORDER",
                    "supplier_id": "SUP_A",
                    "warehouse_id": "WH1",
                    "sku_id": "SKU001",
                    "quantity": 100,
                },
                "horizon": 3,
            },
        )
    )
    assert payload["final_day"] == 6
    assert simulator.state.model_dump() == before


def test_simulate_action_reports_cost_and_fill_rate():
    runner, _ = make_runner(days=3)
    payload = ok_payload(
        runner.run(
            "simulate_action",
            {
                "action": {
                    "type": "TRANSFER_STOCK",
                    "source_warehouse": "WH1",
                    "destination_warehouse": "WH2",
                    "sku_id": "SKU001",
                    "quantity": 10,
                },
                "horizon": 1,
            },
        )
    )
    assert "reward" in payload
    assert "fill_rate" in payload


def test_run_rejects_unknown_tool():
    runner, _ = make_runner()
    result = runner.run("do_something", {})
    assert result["success"] is False
    assert "do_something" in result["error"]  # type: ignore[operator]


def test_run_rejects_invalid_arguments():
    runner, _ = make_runner()
    result = runner.run("forecast_demand", {"sku_id": "SKU001", "region_id": "R1", "horizon": 0})
    assert result["success"] is False
