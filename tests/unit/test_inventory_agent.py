from __future__ import annotations

from copy import deepcopy
from typing import cast

import pytest
from pydantic import ValidationError

from mosaic.agents.inventory import InventoryAgent
from mosaic.agents.messages import (
    AgentRole,
    AnalystRecommendation,
    AnalystRequest,
    ToolEvidence,
    allowed_tools,
    ensure_tool_allowed,
    executable_actions,
)
from mosaic.simulator.actions import ActionExecutor, NoAction, PlaceOrder, TransferStock
from mosaic.simulator.core import Simulator
from mosaic.simulator.episodes import Episode, EpisodeGenerator, EpisodeGeneratorConfig
from mosaic.simulator.models import DisruptionEvent, DisruptionType, TransportRoute
from mosaic.simulator.supplier import SupplierBehavior
from mosaic.tools.api import ToolContext, ToolRunner

_SEED = 53
_CURRENT_DAY = 6
_SKU = "SKU001"


def make_simulator(
    *,
    current_day: int = _CURRENT_DAY,
    routes: list[TransportRoute] | None = None,
    disruptions: list[DisruptionEvent] | None = None,
) -> tuple[Episode, Simulator, ToolRunner]:
    generated = EpisodeGenerator(EpisodeGeneratorConfig(difficulty=1)).generate(seed=_SEED)
    config_updates: dict[str, object] = {
        "disruptions": disruptions if disruptions is not None else []
    }
    if routes is not None:
        config_updates["routes"] = routes
    config = generated.config.model_copy(update=config_updates)
    episode = generated.model_copy(update={"config": config})
    simulator = Simulator(
        config,
        supplier_behavior=SupplierBehavior(seed=episode.seed, stochastic=False),
    )
    for _ in range(current_day):
        simulator.advance_day()
    for state in simulator.state.sku_states.values():
        state.on_hand = 5_000
        state.reserved = 0
    return episode, simulator, ToolRunner(ToolContext(simulator=simulator))


def make_request(episode: Episode, day: int, message_id: str = "req-inventory") -> AnalystRequest:
    return AnalystRequest(
        message_id=message_id,
        episode_seed=episode.seed,
        day=day,
        to_role=AgentRole.INVENTORY,
        instruction="Assess inventory position and propose justified stock transfers.",
    )


def set_available(simulator: Simulator, warehouse_id: str, quantity: int) -> None:
    state = simulator.state.sku_states[(_SKU, warehouse_id)]
    state.on_hand = quantity
    state.reserved = 0


def evidence_rows(evidence: ToolEvidence, key: str) -> list[dict[str, object]]:
    assert evidence.success
    assert evidence.result is not None
    rows = evidence.result.get(key)
    assert isinstance(rows, list)
    return cast(list[dict[str, object]], rows)


def assert_proposals_are_valid_without_mutating_live_simulator(
    simulator: Simulator, recommendation: AnalystRecommendation
) -> None:
    validation_simulator = deepcopy(simulator)
    for action in recommendation.proposed_actions:
        assert isinstance(action, (NoAction, TransferStock))
        ActionExecutor(validation_simulator).apply(action)
    with pytest.raises(TypeError):
        executable_actions(recommendation)


def test_normal_inventory_returns_deterministic_correlated_read_only_recommendation() -> None:
    episode, simulator, tools = make_simulator()
    request = make_request(episode, _CURRENT_DAY, "req-inventory-normal")
    before_state = simulator.state.model_dump()
    before_config = simulator.config.model_dump()

    recommendation = InventoryAgent().recommend(request, tools)

    assert recommendation.role is AgentRole.INVENTORY
    assert recommendation.request_id == request.message_id
    assert recommendation.episode_seed == episode.seed
    assert recommendation.day == simulator.state.day == request.day
    assert recommendation.proposed_actions == [NoAction()]
    assert "no transfer is recommended" in recommendation.summary
    assert [item.tool for item in recommendation.evidence[:2]] == [
        "get_inventory_state",
        "get_demand_history",
    ]
    assert {item.tool for item in recommendation.evidence} <= allowed_tools(AgentRole.INVENTORY)
    assert any(item.tool == "forecast_demand" for item in recommendation.evidence)
    assert any(item.tool == "calculate_reorder_point" for item in recommendation.evidence)
    assert_proposals_are_valid_without_mutating_live_simulator(simulator, recommendation)
    assert simulator.state.model_dump() == before_state
    assert simulator.config.model_dump() == before_config
    assert InventoryAgent().recommend(request, tools) == recommendation


def test_shortage_with_surplus_and_valid_route_proposes_tool_quoted_transfer() -> None:
    episode, simulator, tools = make_simulator()
    set_available(simulator, "WH1", 5_000)
    set_available(simulator, "WH2", 0)
    request = make_request(episode, _CURRENT_DAY, "req-inventory-transfer")
    before_state = simulator.state.model_dump()
    before_config = simulator.config.model_dump()

    recommendation = InventoryAgent().recommend(request, tools)

    assert (
        AnalystRecommendation.model_validate_json(recommendation.model_dump_json())
        == recommendation
    )

    assert recommendation.request_id == request.message_id
    assert recommendation.episode_seed == episode.seed
    assert recommendation.day == _CURRENT_DAY
    assert len(recommendation.proposed_actions) == 1
    action = recommendation.proposed_actions[0]
    assert isinstance(action, TransferStock)
    assert action.sku_id == _SKU
    assert action.source_warehouse == "WH1"
    assert action.destination_warehouse == "WH2"
    assert 0 < action.quantity <= simulator.state.sku_states[(_SKU, "WH1")].available
    assert action.quantity <= simulator.state.sku_states[(_SKU, "WH2")].safety_stock
    assert "tool-calculated reorder point" in recommendation.summary
    assert "transfer cost" in recommendation.summary
    assert {item.tool for item in recommendation.evidence} <= allowed_tools(AgentRole.INVENTORY)
    exact_quote = next(
        item
        for item in recommendation.evidence
        if item.tool == "calculate_transfer_cost"
        and item.arguments.get("sku_id") == _SKU
        and item.arguments.get("source_warehouse") == "WH1"
        and item.arguments.get("destination_warehouse") == "WH2"
        and item.arguments.get("quantity") == action.quantity
    )
    assert exact_quote.success
    assert exact_quote.result is not None
    assert isinstance(exact_quote.result.get("transfer_cost"), (int, float))
    forecast = next(
        item
        for item in recommendation.evidence
        if item.tool == "forecast_demand" and item.arguments.get("sku_id") == _SKU
    )
    assert evidence_rows(forecast, "forecast")[0]["day"] == _CURRENT_DAY + 1
    assert_proposals_are_valid_without_mutating_live_simulator(simulator, recommendation)
    assert simulator.state.model_dump() == before_state
    assert simulator.config.model_dump() == before_config
    assert InventoryAgent().recommend(request, tools) == recommendation


def test_missing_route_produces_truthful_no_transfer_assessment() -> None:
    generated = EpisodeGenerator(EpisodeGeneratorConfig(difficulty=1)).generate(seed=_SEED)
    remaining_routes = [
        route
        for route in generated.config.routes
        if (route.source_warehouse, route.destination_warehouse) in {("WH1", "WH2"), ("WH3", "WH1")}
    ]
    episode, simulator, tools = make_simulator(routes=remaining_routes)
    set_available(simulator, "WH1", 5_000)
    set_available(simulator, "WH2", 5_000)
    set_available(simulator, "WH3", 0)
    request = make_request(episode, _CURRENT_DAY, "req-inventory-no-route")

    recommendation = InventoryAgent().recommend(request, tools)

    assert recommendation.proposed_actions == [NoAction()]
    assert "candidate route(s) are unavailable" in recommendation.summary
    failed_route = next(
        item
        for item in recommendation.evidence
        if item.tool == "calculate_transfer_cost"
        and item.arguments.get("source_warehouse") == "WH1"
        and item.arguments.get("destination_warehouse") == "WH3"
        and item.arguments.get("quantity") == 1
    )
    assert failed_route.success is False
    assert failed_route.result is not None
    assert "Không có tuyến vận chuyển" in str(failed_route.result.get("error"))
    assert_proposals_are_valid_without_mutating_live_simulator(simulator, recommendation)


def test_transfer_cost_above_one_day_stockout_exposure_returns_no_action() -> None:
    generated = EpisodeGenerator(EpisodeGeneratorConfig(difficulty=1)).generate(seed=_SEED)
    expensive_routes = [
        route.model_copy(update={"cost_per_unit": 1_000.0})
        if (route.source_warehouse, route.destination_warehouse) == ("WH1", "WH2")
        else route
        for route in generated.config.routes
    ]
    episode, simulator, tools = make_simulator(routes=expensive_routes)
    set_available(simulator, "WH1", 5_000)
    set_available(simulator, "WH2", 0)
    request = make_request(episode, _CURRENT_DAY, "req-inventory-high-cost")

    recommendation = InventoryAgent().recommend(request, tools)

    assert recommendation.proposed_actions == [NoAction()]
    assert "met or exceeded the estimated one-day avoidable stockout cost" in recommendation.summary
    assert "quoted cost" in recommendation.summary
    assert "one-day stockout exposure" in recommendation.summary
    quotes = [
        item
        for item in recommendation.evidence
        if item.tool == "calculate_transfer_cost"
        and item.arguments.get("source_warehouse") == "WH1"
        and item.arguments.get("destination_warehouse") == "WH2"
    ]
    final_quote = next(item for item in quotes if item.arguments.get("quantity", 0) > 1)
    assert final_quote.success
    assert final_quote.result is not None
    reorder = next(
        item
        for item in recommendation.evidence
        if item.tool == "calculate_reorder_point"
        and item.arguments.get("sku_id") == _SKU
        and item.arguments.get("warehouse_id") == "WH2"
    )
    assert reorder.success
    assert_proposals_are_valid_without_mutating_live_simulator(simulator, recommendation)


def test_disrupted_transport_quote_rejects_unjustified_transfer() -> None:
    generated = EpisodeGenerator(EpisodeGeneratorConfig(difficulty=1)).generate(seed=_SEED)
    route = next(
        route
        for route in generated.config.routes
        if (route.source_warehouse, route.destination_warehouse) == ("WH1", "WH2")
    )
    disruption = DisruptionEvent(
        disruption_id="D-TRANSPORT-COST",
        type=DisruptionType.TRANSPORT_DELAY,
        start_day=1,
        end_day=_CURRENT_DAY,
        subject_type="route",
        subject_id=route.route_id,
        params={"extra_days": 1, "cost_multiplier": 10.0},
    )
    episode, simulator, tools = make_simulator(routes=[route], disruptions=[disruption])
    set_available(simulator, "WH1", 5_000)
    set_available(simulator, "WH2", 0)
    request = make_request(episode, _CURRENT_DAY, "req-inventory-disrupted-cost")
    before_state = simulator.state.model_dump()
    before_config = simulator.config.model_dump()

    recommendation = InventoryAgent().recommend(request, tools)

    assert recommendation.proposed_actions == [NoAction()]
    quote = next(
        item
        for item in recommendation.evidence
        if item.tool == "calculate_transfer_cost"
        and item.arguments.get("source_warehouse") == "WH1"
        and item.arguments.get("destination_warehouse") == "WH2"
        and item.arguments.get("quantity", 0) > 1
    )
    assert quote.success
    assert quote.result is not None
    transfer_cost = quote.result.get("transfer_cost")
    quantity = quote.arguments.get("quantity")
    assert isinstance(transfer_cost, (int, float))
    assert isinstance(quantity, int)
    exposure = round(quantity * simulator.state.sku_states[(_SKU, "WH2")].stockout_cost, 2)
    assert transfer_cost >= exposure
    rejection_detail = (
        f"quoted cost {transfer_cost:.2f} >= one-day stockout exposure {exposure:.2f}"
    )
    assert rejection_detail in recommendation.summary
    assert_proposals_are_valid_without_mutating_live_simulator(simulator, recommendation)
    assert simulator.state.model_dump() == before_state
    assert simulator.config.model_dump() == before_config


@pytest.mark.parametrize("inbound_kind", ["open_order", "shipment"])
def test_due_inbound_order_or_shipment_prevents_duplicate_transfer(inbound_kind: str) -> None:
    episode, simulator, tools = make_simulator()
    set_available(simulator, "WH1", 5_000)
    set_available(simulator, "WH2", 0)
    if inbound_kind == "open_order":
        simulator.state.suppliers["SUP_A"].current_lead_time = 1
        ActionExecutor(simulator).apply(
            PlaceOrder(supplier_id="SUP_A", warehouse_id="WH2", sku_id=_SKU, quantity=500)
        )
        evidence_tool = "get_open_orders"
        evidence_key = "orders"
    else:
        ActionExecutor(simulator).apply(
            TransferStock(
                source_warehouse="WH1",
                destination_warehouse="WH2",
                sku_id=_SKU,
                quantity=500,
            )
        )
        evidence_tool = "get_shipments"
        evidence_key = "shipments"
    request = make_request(episode, _CURRENT_DAY, f"req-inventory-{inbound_kind}")
    before_state = simulator.state.model_dump()
    before_config = simulator.config.model_dump()

    recommendation = InventoryAgent().recommend(request, tools)

    assert recommendation.proposed_actions == [NoAction()]
    inbound_evidence = next(item for item in recommendation.evidence if item.tool == evidence_tool)
    inbound_rows = evidence_rows(inbound_evidence, evidence_key)
    assert any(
        row.get("sku_id") == _SKU and row.get("warehouse_id", row.get("destination")) == "WH2"
        for row in inbound_rows
    )
    assert "after inbound orders and shipments due by route arrival" in recommendation.summary
    assert_proposals_are_valid_without_mutating_live_simulator(simulator, recommendation)
    assert simulator.state.model_dump() == before_state
    assert simulator.config.model_dump() == before_config


def test_inventory_permissions_reject_action_simulation() -> None:
    assert allowed_tools(AgentRole.INVENTORY) == frozenset(
        {
            "calculate_reorder_point",
            "calculate_transfer_cost",
            "forecast_demand",
            "get_demand_history",
            "get_inventory_state",
            "get_open_orders",
            "get_shipments",
        }
    )
    with pytest.raises(ValueError, match="khong duoc dung tool"):
        ensure_tool_allowed(AgentRole.INVENTORY, "simulate_action")
    with pytest.raises(ValidationError, match="khong duoc dung tool"):
        AnalystRecommendation(
            message_id="rec-inventory-forbidden",
            request_id="req-inventory-forbidden",
            episode_seed=_SEED,
            day=_CURRENT_DAY,
            role=AgentRole.INVENTORY,
            summary="Action simulation is not allowed for this analyst.",
            proposed_actions=[NoAction()],
            evidence=[ToolEvidence(tool="simulate_action")],
        )


def test_inventory_agent_rejects_stale_request_correlation() -> None:
    episode, _, tools = make_simulator()
    agent = InventoryAgent()
    request = make_request(episode, _CURRENT_DAY)

    with pytest.raises(ValueError, match="episode_seed"):
        agent.recommend(request.model_copy(update={"episode_seed": 999}), tools)
    with pytest.raises(ValueError, match="simulator day"):
        agent.recommend(make_request(episode, _CURRENT_DAY - 1), tools)
    with pytest.raises(ValueError, match="addressed to the inventory role"):
        agent.recommend(request.model_copy(update={"to_role": AgentRole.DEMAND}), tools)


def test_day_zero_without_observed_demand_returns_no_action() -> None:
    episode, simulator, tools = make_simulator(current_day=0)
    request = make_request(episode, day=1, message_id="req-inventory-day-zero")
    before_state = simulator.state.model_dump()
    before_config = simulator.config.model_dump()

    recommendation = InventoryAgent().recommend(request, tools)

    assert recommendation.proposed_actions == [NoAction()]
    assert "No demand has been observed by simulator day 0" in recommendation.summary
    assert [item.tool for item in recommendation.evidence] == [
        "get_inventory_state",
        "get_demand_history",
    ]
    assert simulator.state.model_dump() == before_state
    assert simulator.config.model_dump() == before_config
