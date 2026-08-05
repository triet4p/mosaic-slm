"""Single-agent runner (Sprint 1, task 12).

Vòng lặp mỗi ngày: `advance_day` → build observation → backbone quyết định
(tool_calls + actions) → thực thi tool qua ToolRunner → áp dụng actions qua
ActionExecutor → ghi TrajectoryStep.

Backbone pluggable:
- `RuleBasedBackbone`: mặc định (reorder-point + greedy transfer), không dùng tool.
- `OpenAICompatibleBackbone`: client OpenAI-compatible (vLLM ở Phase 4),
  trả về JSON khớp AgentDecision.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from mosaic.simulator.actions import ActionExecutor, BusinessAction
from mosaic.simulator.baselines import GreedyTransferPolicy, ReorderPointPolicy
from mosaic.simulator.core import SimulationState, Simulator
from mosaic.simulator.episodes import Episode
from mosaic.simulator.reward import RewardBreakdown, RewardCalculator
from mosaic.simulator.supplier import SupplierBehavior
from mosaic.tools.api import ToolContext, ToolRunner


def _empty_tool_calls() -> list[ToolCallRecord]:
    return []


def _empty_actions() -> list[BusinessAction]:
    return []


class ToolCallRecord(BaseModel):
    """Tool call của agent trong một bước; `result` được runner điền sau khi chạy."""

    model_config = ConfigDict(extra="forbid")

    tool: str
    arguments: dict[str, object] = Field(default_factory=dict)
    success: bool = True
    result: dict[str, object] | None = None


class AgentDecision(BaseModel):
    """Quyết định của backbone trong một bước."""

    model_config = ConfigDict(extra="forbid")

    tool_calls: list[ToolCallRecord] = Field(default_factory=_empty_tool_calls)
    actions: list[BusinessAction] = Field(default_factory=_empty_actions)
    note: str = Field(default="")


class TrajectoryStep(BaseModel):
    """Bản ghi một bước của trajectory (state snapshot tại thời điểm quan sát)."""

    model_config = ConfigDict(extra="forbid")

    day: int = Field(ge=1)
    observation: str
    state_snapshot: SimulationState
    tool_calls: list[ToolCallRecord]
    actions: list[BusinessAction]
    day_reward: RewardBreakdown


class RunnerResult(BaseModel):
    """Kết quả chạy một episode qua runner."""

    model_config = ConfigDict(extra="forbid")

    episode_seed: int
    final_state: SimulationState
    trajectory: list[TrajectoryStep]
    business_reward: float
    total_cost: float
    fill_rate: float


class Backbone(Protocol):
    """Quyết định tool_calls + actions từ observation và tool runner."""

    def decide(self, observation: str, tools: ToolRunner) -> AgentDecision: ...


class RuleBasedBackbone:
    """Backbone mặc định: reorder-point + greedy transfer, không dùng tool."""

    def __init__(self, reorder_safety_multiplier: float = 1.0) -> None:
        self._reorder = ReorderPointPolicy(safety_multiplier=reorder_safety_multiplier)
        self._transfer = GreedyTransferPolicy()

    def decide(self, observation: str, tools: ToolRunner) -> AgentDecision:
        simulator = tools.context.simulator
        actions = list(self._reorder.decide(simulator)) + list(self._transfer.decide(simulator))
        return AgentDecision(actions=actions, note="rule-based: reorder-point + greedy transfer")


class ChatClient(Protocol):
    """Client chat tối thiểu (adapter cho OpenAI SDK / vLLM ở Phase 4)."""

    def chat_completion(self, messages: Sequence[dict[str, str]]) -> str: ...


BACKBONE_SYSTEM_PROMPT = (
    "Bạn là coordinator của hệ thống quản lý tồn kho. "
    "Trả về DUY NHẤT một JSON khớp schema: "
    '{"tool_calls": [{"tool": "...", "arguments": {...}}], '
    '"actions": [{"type": "NO_ACTION" | "PLACE_ORDER" | "TRANSFER_STOCK" | '
    '"SWITCH_SUPPLIER" | "ADJUST_SAFETY_STOCK" | "ACCEPT_BACKORDER", ...}], '
    '"note": "lý do ngắn"}. '
    "Không tự tính toán nếu có tool phù hợp. Không dùng markdown."
)


class OpenAICompatibleBackbone:
    """Backbone gọi client OpenAI-compatible; yêu cầu output JSON khớp AgentDecision."""

    def __init__(self, client: ChatClient, model: str) -> None:
        self._client = client
        self._model = model

    def decide(self, observation: str, tools: ToolRunner) -> AgentDecision:
        tool_specs = [spec.model_dump() for spec in tools.specs]
        user_content = (
            f"{observation}\n\n"
            f"Available tools (JSON schemas):\n{json.dumps(tool_specs, ensure_ascii=False)}"
        )
        messages = [
            {"role": "system", "content": BACKBONE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        text = self._client.chat_completion(messages)
        return AgentDecision.model_validate_json(text)


class SingleAgentRunner:
    """Chạy một episode với một backbone; ghi trajectory mỗi bước."""

    def __init__(
        self,
        episode: Episode,
        backbone: Backbone,
        reward_calculator: RewardCalculator | None = None,
    ) -> None:
        self._episode = episode
        self._backbone = backbone
        self._reward_calculator = reward_calculator or RewardCalculator()
        behavior = SupplierBehavior(
            seed=episode.config.seed, stochastic=episode.supplier_stochastic
        )
        self._simulator = Simulator(episode.config, supplier_behavior=behavior)
        self._tools = ToolRunner(ToolContext(simulator=self._simulator))
        self._executor = ActionExecutor(self._simulator)

    @property
    def simulator(self) -> Simulator:
        return self._simulator

    def run(self) -> RunnerResult:
        """Chạy toàn bộ episode; deterministic nếu backbone deterministic."""
        trajectory: list[TrajectoryStep] = []
        for _ in range(self._episode.config.num_days):
            day_costs = self._simulator.advance_day()
            state_snapshot = self._simulator.state.model_copy(deep=True)
            observation = build_observation(self._simulator)
            decision = self._backbone.decide(observation, self._tools)
            for call in decision.tool_calls:
                result = self._tools.run(call.tool, call.arguments)
                call.result = result
                call.success = bool(result.get("success"))
            for action in decision.actions:
                self._executor.apply(action)
            trajectory.append(
                TrajectoryStep(
                    day=self._simulator.state.day,
                    observation=observation,
                    state_snapshot=state_snapshot,
                    tool_calls=decision.tool_calls,
                    actions=decision.actions,
                    day_reward=self._reward_calculator.day_reward(day_costs),
                )
            )
        breakdown = self._reward_calculator.business_reward(self._simulator.state)
        return RunnerResult(
            episode_seed=self._episode.seed,
            final_state=self._simulator.state,
            trajectory=trajectory,
            business_reward=breakdown.reward,
            total_cost=breakdown.total_cost,
            fill_rate=breakdown.fill_rate,
        )


def build_observation(simulator: Simulator) -> str:
    """Tóm tắt state dạng text cho backbone."""
    state = simulator.state
    lines = [f"Day {state.day}/{simulator.config.num_days}"]
    for (sku_id, warehouse_id), sku in state.sku_states.items():
        risk = " RISK" if sku.available <= sku.reorder_point else ""
        lines.append(
            f"{sku_id}@{warehouse_id}: on_hand={sku.on_hand} available={sku.available} "
            f"in_transit={sku.in_transit} backordered={sku.backordered} "
            f"reorder={sku.reorder_point} safety={sku.safety_stock}{risk}"
        )
    for supplier in state.suppliers.values():
        lines.append(
            f"supplier {supplier.supplier_id}: lead={supplier.current_lead_time} "
            f"reliability={supplier.reliability_score} unit_cost={supplier.unit_cost}"
        )
    lines.append(f"open_orders={len(state.purchase_orders)} shipments={len(state.shipments)}")
    costs = state.cumulative_costs
    lines.append(
        f"costs: stockout={costs.stockout_cost:.1f} holding={costs.holding_cost:.1f} "
        f"ordering={costs.ordering_cost:.1f} transport={costs.transport_cost:.1f}"
    )
    return "\n".join(lines)
