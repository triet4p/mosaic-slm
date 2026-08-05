"""Reward function (Sprint 1, task 6).

Khớp mục 5.1–5.2 INITIALIZATION.md:

    R_business = -(C_stockout + C_holding + C_ordering + C_transport
                   + C_late + C_switch) + eta * FillRate

    FillRate = Demand fulfilled immediately / Total demand

Deterministic: không dùng RNG, chỉ phụ thuộc ledger/state. Reward
compute-aware (lambda tokens/latency/GPU/communication) thuộc Phase 4+.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from mosaic.simulator.core import CostLedger, DayCosts, SimulationState


class RewardBreakdown(BaseModel):
    """Chi tiết reward để phục vụ evaluation và explainability."""

    model_config = ConfigDict(extra="forbid")

    total_cost: float
    fill_rate: float = Field(ge=0.0, le=1.0)
    service_bonus: float = Field(ge=0.0)
    reward: float


class RewardCalculator:
    """Tính business reward từ ledger/state."""

    def __init__(self, service_level_bonus_eta: float = 1.0) -> None:
        if service_level_bonus_eta < 0.0:
            raise ValueError("service_level_bonus_eta phải lớn hơn hoặc bằng 0")
        self._eta = service_level_bonus_eta

    @property
    def service_level_bonus_eta(self) -> float:
        return self._eta

    def business_reward(self, state: SimulationState) -> RewardBreakdown:
        """Reward cả episode từ cumulative costs và fill rate toàn episode."""
        total_demand = sum(day.total_demand for day in state.ledger)
        fulfilled_demand = sum(day.fulfilled_demand for day in state.ledger)
        return self._compute(
            state.cumulative_costs, fulfilled_demand=fulfilled_demand, total_demand=total_demand
        )

    def day_reward(self, day_costs: DayCosts) -> RewardBreakdown:
        """Reward của một ngày (phục vụ trajectory và RL step-level)."""
        return self._compute(
            day_costs.costs,
            fulfilled_demand=day_costs.fulfilled_demand,
            total_demand=day_costs.total_demand,
        )

    def _compute(
        self, costs: CostLedger, fulfilled_demand: int, total_demand: int
    ) -> RewardBreakdown:
        total_cost = (
            costs.stockout_cost
            + costs.holding_cost
            + costs.ordering_cost
            + costs.transport_cost
            + costs.late_delivery_cost
            + costs.supplier_switch_cost
        )
        fill_rate = fulfilled_demand / total_demand if total_demand > 0 else 0.0
        service_bonus = self._eta * fill_rate
        return RewardBreakdown(
            total_cost=total_cost,
            fill_rate=fill_rate,
            service_bonus=service_bonus,
            reward=-total_cost + service_bonus,
        )
