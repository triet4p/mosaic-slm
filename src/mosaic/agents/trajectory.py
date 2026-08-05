"""Trajectory store (Sprint 1, task 13).

File JSONL theo episode trong `data/trajectories/`:
- Dòng 1: header — episode_seed, num_days, reward, cost, fill_rate, final_state.
- Các dòng sau: từng TrajectoryStep (có state_snapshot).

Lưu ý: `SimulationState.sku_states` dùng dict với tuple keys — pydantic
`model_dump_json` serialize tuple key thành chuỗi `"SKU001,WH1"` và không
validate ngược lại được. Giải pháp: dump qua `model_dump(mode="python")`
+ `json.dumps` với key tuple được mã hóa thành chuỗi `SKU001|WH1`, và load
bằng reconstruction thủ công. Trajectory ghi state snapshot mỗi bước để
phục vụ SFT/RL (mục 3.4 INITIALIZATION.md).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, TypeAdapter

from mosaic.agents.runner import RunnerResult, ToolCallRecord, TrajectoryStep
from mosaic.simulator.actions import BusinessAction
from mosaic.simulator.core import SimulationState
from mosaic.simulator.reward import RewardBreakdown

_STATE_KEY_SEPARATOR = "|"
_ACTION_ADAPTER: TypeAdapter[list[BusinessAction]] = TypeAdapter(list[BusinessAction])


class TrajectoryHeader(BaseModel):
    """Tóm tắt episode + trạng thái cuối (đủ để tái dựng RunnerResult)."""

    model_config = ConfigDict(extra="forbid")

    episode_seed: int
    num_days: int
    business_reward: float
    total_cost: float
    fill_rate: float
    final_state: SimulationState


class TrajectoryRecord(BaseModel):
    """Toàn bộ trajectory của một episode (header + các step)."""

    model_config = ConfigDict(extra="forbid")

    header: TrajectoryHeader
    steps: list[TrajectoryStep]


def save_trajectory(result: RunnerResult, path: Path) -> None:
    """Ghi trajectory JSONL: header + mỗi step một dòng."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = TrajectoryHeader(
        episode_seed=result.episode_seed,
        num_days=len(result.trajectory),
        business_reward=result.business_reward,
        total_cost=result.total_cost,
        fill_rate=result.fill_rate,
        final_state=result.final_state,
    )
    lines = [_dump_json(header)]
    lines.extend(_dump_json(step) for step in result.trajectory)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_trajectory(path: Path) -> TrajectoryRecord:
    """Đọc JSONL → TrajectoryRecord (header + steps)."""
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Trajectory rỗng: {path}")
    header = _load_header(_load_json(lines[0]))
    steps = [_load_step(_load_json(line)) for line in lines[1:]]
    return TrajectoryRecord(header=header, steps=steps)


def _dump_json(model: BaseModel) -> str:
    payload = _json_ready(model.model_dump(mode="python", exclude_computed_fields=True))
    return json.dumps(payload, ensure_ascii=False)


def _load_json(line: str) -> dict[str, Any]:
    value = json.loads(line)
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def _json_ready(value: Any) -> Any:
    """Chuyển tuple keys thành chuỗi `a|b` để json.dumps chấp nhận.

    Any: payload là dump python của pydantic (cấu trúc hỗn hợp), ranh giới serialization.
    """
    if isinstance(value, dict):
        return {
            (
                _STATE_KEY_SEPARATOR.join(cast(tuple[str, ...], key))
                if isinstance(key, tuple)
                else key
            ): _json_ready(item)
            for key, item in cast(dict[Any, Any], value).items()
        }
    if isinstance(value, list):
        return [_json_ready(item) for item in cast(list[Any], value)]
    return value


def _load_state(payload: dict[str, Any]) -> SimulationState:
    payload = dict(payload)
    raw = payload.pop("sku_states")
    assert isinstance(raw, dict)
    payload["sku_states"] = {
        tuple(key.split(_STATE_KEY_SEPARATOR, 1)): value
        for key, value in cast(dict[Any, Any], raw).items()
    }
    return SimulationState.model_validate(payload)


def _load_header(payload: dict[str, Any]) -> TrajectoryHeader:
    return TrajectoryHeader(
        episode_seed=payload["episode_seed"],
        num_days=payload["num_days"],
        business_reward=payload["business_reward"],
        total_cost=payload["total_cost"],
        fill_rate=payload["fill_rate"],
        final_state=_load_state(payload["final_state"]),
    )


def _load_step(payload: dict[str, Any]) -> TrajectoryStep:
    return TrajectoryStep(
        day=payload["day"],
        observation=payload["observation"],
        state_snapshot=_load_state(payload["state_snapshot"]),
        tool_calls=[ToolCallRecord.model_validate(call) for call in payload["tool_calls"]],
        actions=_ACTION_ADAPTER.validate_python(payload["actions"]),
        day_reward=RewardBreakdown.model_validate(payload["day_reward"]),
    )
