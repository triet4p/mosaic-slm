"""Evaluation metrics (Sprint 1, task 14).

NTC (Normalized Total Operating Cost, mục 14.1 INITIALIZATION.md):
    NTC = (C_agent - C_oracle) / (C_reorder - C_oracle)
- NTC = 0: đạt mức oracle; NTC = 1: ngang heuristic baseline; > 1: kém hơn.
- C_baseline = ReorderPointPolicy (heuristic baseline bắt buộc).
- C_oracle = best-reference: min cost của họ policy look-ahead
  (OraclePolicy buffer 0..2) + reorder-point — đảm bảo oracle <= baseline
  nên NTC luôn có nghĩa (mẫu số dương).
- Agent = kết quả của SingleAgentRunner (backbone truyền vào).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from pydantic import BaseModel, ConfigDict

from mosaic.agents.runner import Backbone, SingleAgentRunner
from mosaic.simulator.baselines import (
    DoNothingPolicy,
    OraclePolicy,
    ReorderPointPolicy,
    run_policy,
)
from mosaic.simulator.core import SimulationState, Simulator
from mosaic.simulator.episodes import Episode

_ORACLE_BUFFER_DAYS = (0, 1, 2)


class EpisodeMetrics(BaseModel):
    """Metric của một episode."""

    model_config = ConfigDict(extra="forbid")

    episode_seed: int
    agent_cost: float
    agent_fill_rate: float
    agent_stockout_rate: float
    do_nothing_cost: float
    reorder_cost: float
    oracle_cost: float
    ntc: float


class EvaluationReport(BaseModel):
    """Tổng hợp metric trên một tập episode."""

    model_config = ConfigDict(extra="forbid")

    n_episodes: int
    mean_ntc: float
    mean_agent_cost: float
    mean_fill_rate: float
    mean_stockout_rate: float
    mean_do_nothing_cost: float
    mean_reorder_cost: float
    mean_oracle_cost: float
    per_episode: list[EpisodeMetrics]


def evaluate_episode(episode: Episode, backbone: Backbone) -> EpisodeMetrics:
    """Đánh giá một episode: agent + 3 baseline, tính NTC."""
    agent = SingleAgentRunner(episode, backbone).run()
    do_nothing_cost = _total_cost(run_policy(episode, DoNothingPolicy()))
    reorder_cost = _total_cost(run_policy(episode, ReorderPointPolicy()))
    oracle_cost = min(
        [reorder_cost]
        + [
            _total_cost(run_policy(episode, OraclePolicy(buffer_days=buffer)))
            for buffer in _ORACLE_BUFFER_DAYS
        ]
    )
    denominator = reorder_cost - oracle_cost
    ntc = (agent.total_cost - oracle_cost) / denominator if abs(denominator) > 1e-9 else 1.0
    return EpisodeMetrics(
        episode_seed=episode.seed,
        agent_cost=agent.total_cost,
        agent_fill_rate=agent.fill_rate,
        agent_stockout_rate=_stockout_rate(agent.final_state),
        do_nothing_cost=do_nothing_cost,
        reorder_cost=reorder_cost,
        oracle_cost=oracle_cost,
        ntc=ntc,
    )


def evaluate_episodes(
    episodes: Sequence[Episode],
    backbone: Backbone,
    max_episodes: int | None = None,
) -> EvaluationReport:
    """Đánh giá nhiều episode và tổng hợp report."""
    selected = list(episodes) if max_episodes is None else list(episodes[:max_episodes])
    per_episode = [evaluate_episode(episode, backbone) for episode in selected]
    return EvaluationReport(
        n_episodes=len(per_episode),
        mean_ntc=_mean(m.ntc for m in per_episode),
        mean_agent_cost=_mean(m.agent_cost for m in per_episode),
        mean_fill_rate=_mean(m.agent_fill_rate for m in per_episode),
        mean_stockout_rate=_mean(m.agent_stockout_rate for m in per_episode),
        mean_do_nothing_cost=_mean(m.do_nothing_cost for m in per_episode),
        mean_reorder_cost=_mean(m.reorder_cost for m in per_episode),
        mean_oracle_cost=_mean(m.oracle_cost for m in per_episode),
        per_episode=per_episode,
    )


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _total_cost(simulator: Simulator) -> float:
    costs = simulator.state.cumulative_costs
    return (
        costs.stockout_cost
        + costs.holding_cost
        + costs.ordering_cost
        + costs.transport_cost
        + costs.late_delivery_cost
        + costs.supplier_switch_cost
    )


def _stockout_rate(state: SimulationState) -> float:
    total = sum(day.total_demand for day in state.ledger)
    stockout = sum(day.stockout_units for day in state.ledger)
    return stockout / total if total > 0 else 0.0
