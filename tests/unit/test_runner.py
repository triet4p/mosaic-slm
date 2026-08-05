from collections.abc import Sequence

from mosaic.agents.runner import (
    AgentDecision,
    OpenAICompatibleBackbone,
    RuleBasedBackbone,
    SingleAgentRunner,
    ToolCallRecord,
    build_observation,
)
from mosaic.simulator.actions import NoAction, PlaceOrder
from mosaic.simulator.episodes import EpisodeGenerator, EpisodeGeneratorConfig
from mosaic.simulator.reward import RewardCalculator
from mosaic.tools.api import ToolRunner

SPEC = EpisodeGeneratorConfig(difficulty=1)


class FixedDecisionBackbone:
    def __init__(self, decision: AgentDecision) -> None:
        self._decision = decision

    def decide(self, observation: str, tools: ToolRunner) -> AgentDecision:
        return self._decision.model_copy(deep=True)


class FakeChatClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.messages: list[dict[str, str]] = []

    def chat_completion(self, messages: Sequence[dict[str, str]]) -> str:
        self.messages = list(messages)
        return self.text


def make_episode() -> EpisodeGenerator:
    return EpisodeGenerator(SPEC)


def test_rule_based_backbone_runs_episode_and_records_trajectory():
    episode = make_episode().generate(seed=1)
    result = SingleAgentRunner(episode, RuleBasedBackbone()).run()
    assert len(result.trajectory) == 30
    assert result.final_state.day == 30
    assert any(step.actions for step in result.trajectory)
    for step in result.trajectory:
        assert step.day_reward.reward == step.day_reward.reward


def test_runner_is_deterministic():
    episode = make_episode().generate(seed=1)
    first = SingleAgentRunner(episode, RuleBasedBackbone()).run()
    second = SingleAgentRunner(episode, RuleBasedBackbone()).run()
    assert first.final_state == second.final_state
    assert first.trajectory == second.trajectory
    assert first.business_reward == second.business_reward


def test_runner_reward_matches_reward_calculator():
    episode = make_episode().generate(seed=2)
    result = SingleAgentRunner(episode, RuleBasedBackbone()).run()
    expected = RewardCalculator().business_reward(result.final_state).reward
    assert result.business_reward == expected


def test_runner_executes_tool_calls_from_backbone():
    episode = make_episode().generate(seed=3)
    decision = AgentDecision(
        tool_calls=[
            ToolCallRecord(
                tool="get_inventory_state",
                arguments={"sku_id": "SKU001", "warehouse_id": "WH1"},
            )
        ],
        actions=[NoAction()],
        note="test",
    )
    result = SingleAgentRunner(episode, FixedDecisionBackbone(decision)).run()
    first_step = result.trajectory[0]
    assert len(first_step.tool_calls) == 1
    assert first_step.tool_calls[0].success is True
    payload = first_step.tool_calls[0].result
    assert payload is not None
    assert "states" in payload["result"]  # type: ignore[operator]


def test_runner_records_failed_tool_call():
    episode = make_episode().generate(seed=3)
    decision = AgentDecision(
        tool_calls=[ToolCallRecord(tool="not_a_tool", arguments={})],
        actions=[],
    )
    result = SingleAgentRunner(episode, FixedDecisionBackbone(decision)).run()
    call = result.trajectory[0].tool_calls[0]
    assert call.success is False
    assert "not_a_tool" in call.result["error"]  # type: ignore[index]


def test_openai_backbone_parses_structured_output_and_applies_action():
    episode = make_episode().generate(seed=4)
    episode_text = (
        '{"tool_calls": [], '
        '"actions": [{"type": "PLACE_ORDER", "supplier_id": "SUP_A", '
        '"warehouse_id": "WH1", "sku_id": "SKU001", "quantity": 50}], '
        '"note": "test"}'
    )
    backbone = OpenAICompatibleBackbone(FakeChatClient(episode_text), model="fake")
    runner = SingleAgentRunner(episode, backbone)
    result = runner.run()
    assert any(
        isinstance(action, PlaceOrder) for step in result.trajectory for action in step.actions
    )


def test_openai_backbone_rejects_invalid_json():
    import pytest

    episode = make_episode().generate(seed=5)
    backbone = OpenAICompatibleBackbone(FakeChatClient("không phải json"), model="fake")
    runner = SingleAgentRunner(episode, backbone)
    with pytest.raises(Exception):
        runner.run()


def test_build_observation_contains_day_and_skus():
    episode = make_episode().generate(seed=1)
    runner = SingleAgentRunner(episode, FixedDecisionBackbone(AgentDecision(actions=[])))
    runner.simulator.advance_day()
    observation = build_observation(runner.simulator)
    assert "Day 1/30" in observation
    assert "SKU001@WH1" in observation
    assert "supplier SUP_A" in observation


def test_openai_backbone_prompt_contains_tool_specs():
    episode = make_episode().generate(seed=6)
    client = FakeChatClient('{"tool_calls": [], "actions": [{"type": "NO_ACTION"}], "note": ""}')
    backbone = OpenAICompatibleBackbone(client, model="fake")
    runner = SingleAgentRunner(episode, backbone)
    runner.simulator.advance_day()
    observation = build_observation(runner.simulator)
    backbone.decide(observation, runner._tools)  # type: ignore[attr-defined]
    user_content = client.messages[1]["content"]
    assert "get_inventory_state" in user_content
    assert "Day 1/30" in user_content
