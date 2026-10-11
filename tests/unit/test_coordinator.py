from __future__ import annotations

from copy import deepcopy

import pytest

from mosaic.agents.coordinator import CoordinationResult, CoordinatorAgent
from mosaic.agents.messages import (
    AgentRole,
    AnalystRecommendation,
    AnalystRequest,
    ToolEvidence,
    executable_actions,
)
from mosaic.simulator.actions import (
    ActionExecutor,
    NoAction,
    PlaceOrder,
    TransferStock,
)
from mosaic.simulator.core import Simulator
from mosaic.simulator.episodes import Episode, EpisodeGenerator, EpisodeGeneratorConfig
from mosaic.simulator.supplier import SupplierBehavior
from mosaic.tools.api import ToolContext, ToolRunner

_SEED = 811
_DAY = 6
_PREFIX = "req-coord"


def make_environment(
    *,
    seed: int = _SEED,
    day: int = _DAY,
    shortfalls: dict[tuple[str, str], int] | None = None,
) -> tuple[Episode, Simulator, ToolRunner]:
    generated = EpisodeGenerator(EpisodeGeneratorConfig(difficulty=1)).generate(seed=seed)
    config = generated.config.model_copy(update={"disruptions": []})
    episode = generated.model_copy(update={"config": config})
    simulator = Simulator(
        config, supplier_behavior=SupplierBehavior(seed=config.seed, stochastic=False)
    )
    for _ in range(day):
        simulator.advance_day()
    tools = ToolRunner(ToolContext(simulator=simulator))
    if shortfalls is None:
        for state in simulator.state.sku_states.values():
            state.on_hand = 5_000
            state.reserved = 0
    else:
        for state in simulator.state.sku_states.values():
            state.on_hand = 5_000
            state.reserved = 0
        for (sku_id, warehouse_id), quantity in shortfalls.items():
            simulator.state.sku_states[(sku_id, warehouse_id)].on_hand = quantity
            simulator.state.sku_states[(sku_id, warehouse_id)].reserved = 0
    tools = ToolRunner(ToolContext(simulator=simulator))
    return episode, simulator, tools


def assert_only_decision_executes(result: CoordinationResult) -> None:
    for recommendation in result.recommendations:
        with pytest.raises(TypeError):
            executable_actions(recommendation)
    assert isinstance(executable_actions(result.decision), list)


def test_sequential_coordination_is_deterministic_correlated_and_read_only() -> None:
    episode, simulator, tools = make_environment()
    before_state = simulator.state.model_dump()
    before_config = simulator.config.model_dump()

    result = CoordinatorAgent().coordinate(
        episode_seed=episode.seed, day=_DAY, tools=tools, request_prefix=_PREFIX
    )

    assert result.outcome == "decided"
    assert result.episode_seed == episode.seed
    assert result.day == _DAY
    assert [request.to_role for request in result.requests[:2]] == [
        AgentRole.DEMAND,
        AgentRole.INVENTORY,
    ]
    assert {rec.role for rec in result.recommendations} >= {
        AgentRole.DEMAND,
        AgentRole.INVENTORY,
    }
    for request, recommendation in zip(result.requests, result.recommendations, strict=False):
        if recommendation.request_id == request.message_id:
            assert recommendation.episode_seed == episode.seed
            assert recommendation.day == _DAY
    assert set(result.decision.recommendation_ids) <= {
        rec.message_id for rec in result.recommendations
    }
    assert len(result.decision.recommendation_ids) >= 2
    assert result.decision.episode_seed == episode.seed
    assert result.decision.day == _DAY
    assert "non-LLM" in result.decision.rationale
    assert any(rec.message_id in result.decision.rationale for rec in result.recommendations)
    assert_only_decision_executes(result)
    assert simulator.state.model_dump() == before_state
    assert simulator.config.model_dump() == before_config
    repeat = CoordinatorAgent().coordinate(
        episode_seed=episode.seed, day=_DAY, tools=tools, request_prefix=_PREFIX
    )
    assert repeat == result


def test_conflicting_transfer_and_order_resolve_to_single_traced_action() -> None:
    episode, simulator, tools = make_environment(
        seed=702, shortfalls={("SKU001", "WH2"): 0, ("SKU001", "WH3"): 0}
    )
    result = CoordinatorAgent().coordinate(
        episode_seed=episode.seed, day=_DAY, tools=tools, request_prefix="req-conflict"
    )

    assert result.outcome == "decided"
    assert result.decision.final_actions == [
        TransferStock(
            source_warehouse="WH1",
            destination_warehouse="WH2",
            sku_id="SKU001",
            quantity=33,
        )
    ]
    assert "avoid double-supplying" in " ".join(result.trace)
    assert "smoke-conflict" not in result.decision.rationale
    assert "req-conflict-inventory:inventory" in result.decision.rationale
    assert "req-conflict-supplier-0:supplier-risk" in result.decision.rationale
    validation = deepcopy(simulator)
    for action in executable_actions(result.decision):
        ActionExecutor(validation).apply(action)
    assert_only_decision_executes(result)


def test_information_request_skips_supplier_without_observable_shortfall() -> None:
    episode, simulator, tools = make_environment(seed=701)
    for state in simulator.state.sku_states.values():
        state.on_hand = 5_000
        state.reserved = 0
        state.safety_stock = 0

    result = CoordinatorAgent().coordinate(
        episode_seed=episode.seed, day=_DAY, tools=tools, request_prefix="req-no-need"
    )

    assert result.outcome == "decided"
    assert all(request.to_role is not AgentRole.SUPPLIER_RISK for request in result.requests)
    assert "no observable safety-stock shortfall" in " ".join(result.trace)
    # Inventory may still propose an economically justified transfer on its own
    # screen; either way no fabricated supplier request exists.
    for recommendation in result.recommendations:
        assert recommendation.role in {AgentRole.DEMAND, AgentRole.INVENTORY}
    assert_only_decision_executes(result)


def test_no_action_supplier_response_tries_next_candidate_within_revision_budget() -> None:
    generated = EpisodeGenerator(EpisodeGeneratorConfig(difficulty=1)).generate(seed=407)
    suppliers = [
        supplier.model_copy(update={"capacity": 1}) for supplier in generated.config.suppliers
    ]
    config = generated.config.model_copy(update={"suppliers": suppliers, "disruptions": []})
    episode = generated.model_copy(update={"config": config})
    simulator = Simulator(
        config, supplier_behavior=SupplierBehavior(seed=config.seed, stochastic=False)
    )
    simulator.advance_day()
    for state in simulator.state.sku_states.values():
        state.on_hand = 5_000
        state.reserved = 0
    simulator.state.sku_states[("SKU001", "WH2")].on_hand = 0
    simulator.state.sku_states[("SKU002", "WH3")].on_hand = 0
    tools = ToolRunner(ToolContext(simulator=simulator))

    result = CoordinatorAgent().coordinate(
        episode_seed=episode.seed, day=1, tools=tools, request_prefix="req-revise"
    )

    assert result.outcome == "decided"
    assert result.analyst_calls == 4
    assert result.revision_count == 1
    supplier_requests = [
        request for request in result.requests if request.to_role is AgentRole.SUPPLIER_RISK
    ]
    assert len(supplier_requests) == 2
    assert "valid NO_ACTION" in " ".join(result.trace)
    assert_only_decision_executes(result)


def test_exhausted_budget_yields_bounded_stop_without_invented_success() -> None:
    episode, _, tools = make_environment(seed=704)

    result = CoordinatorAgent(max_analyst_calls=1, max_revisions=0).coordinate(
        episode_seed=episode.seed, day=_DAY, tools=tools, request_prefix="req-stop"
    )

    assert result.outcome == "bounded_stop"
    assert result.analyst_calls == 1
    assert result.decision.final_actions == [NoAction()]
    assert "BOUNDED STOP" in result.decision.rationale
    assert "No business action is claimed" in result.decision.rationale
    assert_only_decision_executes(result)


def test_stale_recommendation_fails_correlation_before_authority() -> None:
    request = AnalystRequest(
        message_id="req-stale",
        episode_seed=_SEED,
        day=_DAY,
        to_role=AgentRole.INVENTORY,
        instruction="Assess inventory position.",
    )
    stale = AnalystRecommendation(
        message_id="stale:inventory",
        request_id="req-other",
        episode_seed=_SEED + 999,
        day=_DAY,
        role=AgentRole.INVENTORY,
        summary="stale",
        proposed_actions=[
            TransferStock(
                source_warehouse="WH1",
                destination_warehouse="WH2",
                sku_id="SKU001",
                quantity=5,
            )
        ],
        evidence=[ToolEvidence(tool="get_inventory_state", success=True, result={"states": []})],
    )

    assert (
        CoordinatorAgent.validate_recommendation(
            request, stale, expected_seed=_SEED, expected_day=_DAY
        )
        is not None
    )


def test_coordinator_rejects_invalid_budgets() -> None:
    with pytest.raises(ValueError, match="max_analyst_calls"):
        CoordinatorAgent(max_analyst_calls=0)
    with pytest.raises(ValueError, match="max_revisions"):
        CoordinatorAgent(max_revisions=-1)


def test_coordinator_rejects_mismatched_episode_and_day() -> None:
    episode, _, tools = make_environment()
    with pytest.raises(ValueError, match="episode_seed"):
        CoordinatorAgent().coordinate(episode_seed=episode.seed + 1, day=_DAY, tools=tools)
    with pytest.raises(ValueError, match="Request day"):
        CoordinatorAgent().coordinate(episode_seed=episode.seed, day=_DAY + 1, tools=tools)


def test_final_actions_apply_on_live_copy_without_mutating_policy_input() -> None:
    episode, simulator, tools = make_environment(
        seed=702, shortfalls={("SKU001", "WH2"): 0, ("SKU001", "WH3"): 0}
    )
    result = CoordinatorAgent().coordinate(
        episode_seed=episode.seed, day=_DAY, tools=tools, request_prefix="req-apply"
    )
    before = simulator.state.model_dump()

    live_copy = deepcopy(simulator)
    applied = 0
    for action in executable_actions(result.decision):
        if isinstance(action, NoAction):
            continue
        ActionExecutor(live_copy).apply(action)
        applied += 1

    assert simulator.state.model_dump() == before
    assert applied == len(
        [action for action in result.decision.final_actions if not isinstance(action, NoAction)]
    )
    assert applied >= 1


def test_place_order_only_path_is_preserved_when_no_transfer_route() -> None:
    generated = EpisodeGenerator(EpisodeGeneratorConfig(difficulty=1)).generate(seed=53)
    config = generated.config.model_copy(update={"disruptions": [], "routes": []})
    episode = generated.model_copy(update={"config": config})
    simulator = Simulator(
        config, supplier_behavior=SupplierBehavior(seed=config.seed, stochastic=False)
    )
    for _ in range(_DAY):
        simulator.advance_day()
    tools = ToolRunner(ToolContext(simulator=simulator))

    result = CoordinatorAgent().coordinate(
        episode_seed=episode.seed, day=_DAY, tools=tools, request_prefix="req-order"
    )

    assert result.outcome == "decided"
    assert len(result.decision.final_actions) == 1
    action = result.decision.final_actions[0]
    assert isinstance(action, PlaceOrder)
    validation = deepcopy(simulator)
    ActionExecutor(validation).apply(action)
    assert_only_decision_executes(result)
