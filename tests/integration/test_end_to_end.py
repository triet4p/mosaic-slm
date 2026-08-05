"""End-to-end pipeline integration (Sprint 1, task 15).

Một episode hoàn chỉnh: episode generator → single-agent runner (rule-based)
→ trajectory store → evaluation metrics. Các test replay từ file như môi
trường sản xuất (không dùng generator trực tiếp cho phần replay).
"""

from pathlib import Path

import pytest

from mosaic.agents.runner import RuleBasedBackbone, SingleAgentRunner
from mosaic.agents.trajectory import load_trajectory, save_trajectory
from mosaic.evaluation.metrics import evaluate_episode
from mosaic.simulator.baselines import DoNothingPolicy, run_policy
from mosaic.simulator.episodes import EpisodeGenerator, EpisodeGeneratorConfig
from mosaic.simulator.replay import load_episodes, run_episode, save_episodes

SPEC = EpisodeGeneratorConfig(difficulty=1)


def test_full_pipeline_end_to_end(tmp_path: Path):
    episode = EpisodeGenerator(SPEC).generate(seed=1)
    result = SingleAgentRunner(episode, RuleBasedBackbone()).run()
    assert result.episode_seed == 1
    assert len(result.trajectory) == 30

    path = tmp_path / "trajectory_ep1.jsonl"
    save_trajectory(result, path)
    record = load_trajectory(path)
    assert len(record.steps) == 30
    assert record.header.episode_seed == 1
    assert record.header.final_state == result.final_state
    assert record.header.business_reward == result.business_reward
    assert record.steps[0].state_snapshot.day == 1
    assert record.steps[-1].state_snapshot.day == 30

    metrics = evaluate_episode(episode, RuleBasedBackbone())
    assert metrics.episode_seed == 1
    assert metrics.agent_cost == result.total_cost
    denominator = metrics.reorder_cost - metrics.oracle_cost
    assert metrics.ntc == pytest.approx((metrics.agent_cost - metrics.oracle_cost) / denominator)


def test_trajectory_state_snapshots_are_frozen():
    episode = EpisodeGenerator(SPEC).generate(seed=2)
    result = SingleAgentRunner(episode, RuleBasedBackbone()).run()
    first_snapshot = result.trajectory[0].state_snapshot
    assert first_snapshot.day == 1
    assert result.final_state.day == 30
    assert first_snapshot != result.final_state
    assert result.trajectory[0].state_snapshot == first_snapshot


def test_dev_set_replay_and_agent_run_are_deterministic(tmp_path: Path):
    episodes = EpisodeGenerator(SPEC).generate_dev_set(count=3, base_seed=0)
    path = tmp_path / "dev.jsonl"
    save_episodes(episodes, path)
    loaded = load_episodes(path)

    first = SingleAgentRunner(loaded[1], RuleBasedBackbone()).run()
    second = SingleAgentRunner(load_episodes(path)[1], RuleBasedBackbone()).run()
    assert first.final_state == second.final_state
    assert first.trajectory == second.trajectory

    direct = run_episode(loaded[2])
    replayed = run_episode(load_episodes(path)[2])
    assert direct.state == replayed.state


def test_no_action_baseline_runs_on_generated_episode():
    episode = EpisodeGenerator(SPEC).generate(seed=4)
    simulator = run_policy(episode, DoNothingPolicy())
    assert simulator.state.day == 30
