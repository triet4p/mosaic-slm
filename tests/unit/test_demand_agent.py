from __future__ import annotations

from typing import cast

import pytest
from pydantic import ValidationError

from mosaic.agents.demand import DemandAgent
from mosaic.agents.messages import (
    AgentRole,
    AnalystRecommendation,
    AnalystRequest,
    ToolEvidence,
    allowed_tools,
    ensure_tool_allowed,
    executable_actions,
)
from mosaic.simulator.actions import NoAction
from mosaic.simulator.core import Simulator
from mosaic.simulator.demand import DemandGenerator, DemandProfile
from mosaic.simulator.episodes import Episode, EpisodeGenerator, EpisodeGeneratorConfig
from mosaic.simulator.models import DisruptionEvent, DisruptionType
from mosaic.simulator.supplier import SupplierBehavior
from mosaic.tools.api import ToolContext, ToolRunner

_SEED = 41
_CURRENT_DAY = 6


def evidence_rows(evidence: ToolEvidence, key: str) -> list[dict[str, object]]:
    rows = evidence_result(evidence).get(key)
    assert isinstance(rows, list)
    return cast(list[dict[str, object]], rows)


def make_simulator(
    scenario: str, *, seed: int = _SEED, current_day: int = _CURRENT_DAY
) -> tuple[Episode, Simulator, ToolRunner]:
    generated = EpisodeGenerator(EpisodeGeneratorConfig(difficulty=1)).generate(seed=seed)
    disruptions: list[DisruptionEvent] = []
    if scenario in {"spike", "drop"}:
        disruptions.append(
            DisruptionEvent(
                disruption_id=f"{scenario}-001",
                type=(
                    DisruptionType.DEMAND_SPIKE
                    if scenario == "spike"
                    else DisruptionType.DEMAND_DROP
                ),
                start_day=_CURRENT_DAY,
                end_day=_CURRENT_DAY,
                subject_type="product_region",
                subject_id="R1:SKU001",
                params={"multiplier": 2.0 if scenario == "spike" else 0.5},
            )
        )
    schedule = DemandGenerator(
        seed=seed,
        profiles=[
            DemandProfile(
                region_id="R1",
                product_id="SKU001",
                base_demand=20,
                seasonality_amplitude=0.0,
                noise_std=0.0,
            )
        ],
    ).generate_schedule(num_days=generated.config.num_days, disruptions=disruptions)
    config = generated.config.model_copy(
        update={"demand_schedule": schedule, "disruptions": disruptions}
    )
    episode = generated.model_copy(update={"config": config})
    simulator = Simulator(
        episode.config,
        supplier_behavior=SupplierBehavior(
            seed=episode.config.seed, stochastic=episode.supplier_stochastic
        ),
    )
    for _ in range(current_day):
        simulator.advance_day()
    return episode, simulator, ToolRunner(ToolContext(simulator=simulator))


def make_request(episode: Episode, day: int, message_id: str = "req-demand") -> AnalystRequest:
    return AnalystRequest(
        message_id=message_id,
        episode_seed=episode.seed,
        day=day,
        to_role=AgentRole.DEMAND,
        instruction="Assess the latest demand observations and short-horizon risk.",
    )


def evidence_result(evidence: ToolEvidence) -> dict[str, object]:
    assert evidence.success
    assert evidence.result is not None
    return evidence.result


@pytest.mark.parametrize(
    ("scenario", "expected_risk", "expected_forecast"),
    [
        ("normal", "No material demand spike/drop", 20),
        ("spike", "DEMAND_SPIKE", 24),
        ("drop", "DEMAND_DROP", 18),
    ],
)
def test_demand_scenarios_produce_correlated_tool_backed_recommendations(
    scenario: str, expected_risk: str, expected_forecast: int
) -> None:
    episode, simulator, tools = make_simulator(scenario)
    request = make_request(episode, _CURRENT_DAY, f"req-{scenario}")
    before_state = simulator.state.model_dump()
    before_config = simulator.config.model_dump()

    recommendation = DemandAgent().recommend(request, tools)

    assert recommendation.role is AgentRole.DEMAND
    assert recommendation.request_id == request.message_id
    assert recommendation.episode_seed == episode.seed
    assert recommendation.day == simulator.state.day == request.day
    assert expected_risk in recommendation.summary
    assert f"day {_CURRENT_DAY + 1}={expected_forecast}" in recommendation.summary
    assert recommendation.proposed_actions == [NoAction()]
    assert [item.tool for item in recommendation.evidence][0] == "get_demand_history"
    assert {item.tool for item in recommendation.evidence} <= allowed_tools(AgentRole.DEMAND)
    assert any(item.tool == "forecast_demand" for item in recommendation.evidence)

    history = evidence_rows(recommendation.evidence[0], "history")
    latest = [
        row
        for row in history
        if row["sku_id"] == "SKU001"
        and row["region_id"] == "R1"
        and row["day"] == _CURRENT_DAY
    ]
    assert len(latest) == 1
    assert latest[0]["quantity"] == {"normal": 20, "spike": 40, "drop": 10}[scenario]

    forecast_evidence = next(
        item for item in recommendation.evidence if item.tool == "forecast_demand"
    )
    forecast = evidence_rows(forecast_evidence, "forecast")
    assert forecast[0]["quantity"] == expected_forecast
    assert forecast[0]["day"] == _CURRENT_DAY + 1
    with pytest.raises(TypeError):
        executable_actions(recommendation)

    assert simulator.state.model_dump() == before_state
    assert simulator.config.model_dump() == before_config
    assert DemandAgent().recommend(request, tools) == recommendation


def test_empty_history_returns_no_forecast_or_unobserved_demand() -> None:
    episode, simulator, tools = make_simulator("normal", current_day=0)
    request = make_request(episode, day=1)

    recommendation = DemandAgent().recommend(request, tools)

    assert "No demand history" in recommendation.summary
    assert "unavailable" in recommendation.summary
    assert len(recommendation.evidence) == 1
    assert recommendation.evidence[0].tool == "get_demand_history"
    assert evidence_result(recommendation.evidence[0])["history"] == []
    assert recommendation.day == 1
    assert simulator.state.day == 0


def test_first_day_forecast_does_not_claim_a_spike_without_a_baseline() -> None:
    episode, _, tools = make_simulator("normal", current_day=1)

    recommendation = DemandAgent().recommend(make_request(episode, day=1), tools)

    assert "assessment is unavailable" in recommendation.summary
    assert "day 2=20" in recommendation.summary
    assert any(item.tool == "forecast_demand" for item in recommendation.evidence)


def test_final_day_history_stops_at_observed_day_and_forecast_is_projected() -> None:
    episode_day = 30
    episode, _, tools = make_simulator("normal", current_day=episode_day)
    recommendation = DemandAgent().recommend(make_request(episode, day=episode_day), tools)

    history = evidence_rows(recommendation.evidence[0], "history")
    assert {row["day"] for row in history} == set(range(episode_day - 6, episode_day + 1))
    forecast_evidence = next(
        item for item in recommendation.evidence if item.tool == "forecast_demand"
    )
    forecast = evidence_rows(forecast_evidence, "forecast")
    assert [point["day"] for point in forecast] == [31, 32, 33]
    assert "day 31=20" in recommendation.summary


def test_demand_permissions_reject_simulation_tools_and_corrupt_evidence() -> None:
    assert allowed_tools(AgentRole.DEMAND) == frozenset(
        {"get_demand_history", "forecast_demand"}
    )
    with pytest.raises(ValueError, match="khong duoc dung tool"):
        ensure_tool_allowed(AgentRole.DEMAND, "simulate_action")
    with pytest.raises(ValidationError, match="khong duoc dung tool"):
        AnalystRecommendation(
            message_id="rec-forbidden",
            request_id="req-forbidden",
            episode_seed=_SEED,
            day=_CURRENT_DAY,
            role=AgentRole.DEMAND,
            summary="Forbidden simulation evidence must not validate.",
            proposed_actions=[NoAction()],
            evidence=[ToolEvidence(tool="simulate_action")],
        )


def test_demand_agent_rejects_stale_correlation_context() -> None:
    episode, _, tools = make_simulator("normal")
    agent = DemandAgent()

    wrong_seed_request = make_request(episode, _CURRENT_DAY).model_copy(
        update={"episode_seed": 99}
    )
    with pytest.raises(ValueError, match="episode_seed"):
        agent.recommend(wrong_seed_request, tools)
    with pytest.raises(ValueError, match="simulator day"):
        agent.recommend(make_request(episode, _CURRENT_DAY - 1), tools)
    with pytest.raises(ValueError, match="addressed to the demand role"):
        agent.recommend(
            make_request(episode, _CURRENT_DAY).model_copy(
                update={"to_role": AgentRole.INVENTORY}
            ),
            tools,
        )
