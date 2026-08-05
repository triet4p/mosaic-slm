"""Disruption engine (Sprint 1, task 7).

Sinh danh sách DisruptionEvent deterministic theo seed, theo difficulty
(mục 13.2 INITIALIZATION.md):

- Level 1 (Deterministic): đúng 1 sự kiện, window cố định, params cố định.
- Level 2 (Stochastic): num_disruptions sự kiện, window và params ngẫu nhiên.
- Level 3 (Partial information): như Level 2, thêm sự kiện lặp lại
  (cùng type + subject với window sau).

Engine chỉ SINH sự kiện; việc áp dụng lên simulator (lead time, travel time,
stockout) nằm trong `Simulator.advance_day` (xem core.py).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from mosaic.simulator.models import DisruptionEvent, DisruptionType, JsonScalar

DIFFICULTY_LEVELS = frozenset({1, 2, 3})

_L1_PARAMS: dict[DisruptionType, dict[str, JsonScalar]] = {
    DisruptionType.DEMAND_SPIKE: {"multiplier": 2.0},
    DisruptionType.DEMAND_DROP: {"multiplier": 0.5},
    DisruptionType.SUPPLIER_DELAY: {"extra_days": 3},
    DisruptionType.SUPPLIER_CANCELLATION: {},
    DisruptionType.WAREHOUSE_STOCKOUT: {"units": 100},
    DisruptionType.TRANSPORT_DELAY: {"extra_days": 2, "cost_multiplier": 1.5},
}


class DisruptionEngineConfig(BaseModel):
    """Cấu hình sinh disruption của một episode."""

    model_config = ConfigDict(extra="forbid")

    seed: int
    difficulty: int = Field(ge=1, le=3)
    num_disruptions: int = Field(ge=1, le=5)
    num_days: int = Field(ge=10)
    regions: list[str] = Field(default_factory=list)
    products: list[str] = Field(default_factory=list)
    suppliers: list[str] = Field(default_factory=list)
    warehouses: list[str] = Field(default_factory=list)
    routes: list[str] = Field(default_factory=list)


class DisruptionEngine:
    """Sinh danh sách DisruptionEvent deterministic theo seed."""

    def __init__(self, config: DisruptionEngineConfig) -> None:
        self._config = config
        if config.difficulty not in DIFFICULTY_LEVELS:
            raise ValueError(f"difficulty phải thuộc {sorted(DIFFICULTY_LEVELS)}")
        self._available_types = self._resolve_available_types()

    def generate(self) -> list[DisruptionEvent]:
        """Sinh danh sách sự kiện; cùng seed → cùng kết quả."""
        rng = np.random.default_rng(self._config.seed)
        count = 1 if self._config.difficulty == 1 else self._config.num_disruptions
        events: list[DisruptionEvent] = []
        for i in range(count):
            event_type = self._pick_type(rng)
            event = self._make_event(rng, event_type, f"DIS-{self._config.seed}-{i + 1}")
            events.append(event)
            if self._config.difficulty == 3 and rng.random() < 0.5:
                recurring = self._make_event(
                    rng, event_type, f"DIS-{self._config.seed}-{i + 1}R", later=True
                )
                events.append(recurring)
        return events

    def _resolve_available_types(self) -> tuple[DisruptionType, ...]:
        available = [
            event_type for event_type in DisruptionType if self._has_subject_pool(event_type)
        ]
        if not available:
            raise ValueError(
                "Không có subject pool nào (regions/products/suppliers/warehouses/routes) "
                "để sinh disruption"
            )
        return tuple(available)

    def _has_subject_pool(self, event_type: DisruptionType) -> bool:
        if event_type in (DisruptionType.DEMAND_SPIKE, DisruptionType.DEMAND_DROP):
            return bool(self._config.regions or self._config.products)
        if event_type in (
            DisruptionType.SUPPLIER_DELAY,
            DisruptionType.SUPPLIER_CANCELLATION,
        ):
            return bool(self._config.suppliers)
        if event_type is DisruptionType.WAREHOUSE_STOCKOUT:
            return bool(self._config.warehouses)
        if event_type is DisruptionType.TRANSPORT_DELAY:
            return bool(self._config.routes)
        return False

    def _pick_type(self, rng: np.random.Generator) -> DisruptionType:
        index = int(rng.integers(0, len(self._available_types)))
        return self._available_types[index]

    def _make_event(
        self,
        rng: np.random.Generator,
        event_type: DisruptionType,
        event_id: str,
        *,
        later: bool = False,
    ) -> DisruptionEvent:
        subject_type, subject_id = self._sample_subject(rng, event_type)
        start_day, end_day = self._sample_window(rng, later=later)
        params = self._sample_params(rng, event_type)
        return DisruptionEvent(
            disruption_id=event_id,
            type=event_type,
            start_day=start_day,
            end_day=end_day,
            subject_type=subject_type,
            subject_id=subject_id,
            params=params,
        )

    def _sample_subject(
        self, rng: np.random.Generator, event_type: DisruptionType
    ) -> tuple[str, str]:
        if event_type in (DisruptionType.DEMAND_SPIKE, DisruptionType.DEMAND_DROP):
            use_region = bool(self._config.regions) and (
                not self._config.products or rng.random() < 0.5
            )
            if use_region:
                return "region", self._pick(self._config.regions, rng)
            return "product", self._pick(self._config.products, rng)
        if event_type in (
            DisruptionType.SUPPLIER_DELAY,
            DisruptionType.SUPPLIER_CANCELLATION,
        ):
            return "supplier", self._pick(self._config.suppliers, rng)
        if event_type is DisruptionType.WAREHOUSE_STOCKOUT:
            return "warehouse", self._pick(self._config.warehouses, rng)
        if event_type is DisruptionType.TRANSPORT_DELAY:
            return "route", self._pick(self._config.routes, rng)
        raise ValueError(f"Chưa hỗ trợ subject cho {event_type}")

    def _sample_window(self, rng: np.random.Generator, *, later: bool) -> tuple[int, int]:
        num_days = self._config.num_days
        if self._config.difficulty == 1:
            start_day = num_days // 3
            duration = max(1, num_days // 5)
        else:
            duration = int(rng.integers(3, 8))
            if later:
                low = max(2, num_days * 2 // 3)
                high = max(low + 1, num_days - duration)
                start_day = int(rng.integers(low, high))
            else:
                start_day = int(rng.integers(2, max(3, num_days - duration)))
        end_day = min(start_day + duration - 1, num_days)
        return start_day, end_day

    def _sample_params(
        self, rng: np.random.Generator, event_type: DisruptionType
    ) -> dict[str, JsonScalar]:
        if self._config.difficulty == 1:
            return dict(_L1_PARAMS[event_type])
        if event_type is DisruptionType.DEMAND_SPIKE:
            return {"multiplier": round(float(rng.uniform(1.3, 3.0)), 2)}
        if event_type is DisruptionType.DEMAND_DROP:
            return {"multiplier": round(float(rng.uniform(0.0, 0.5)), 2)}
        if event_type is DisruptionType.SUPPLIER_DELAY:
            return {"extra_days": int(rng.integers(2, 6))}
        if event_type is DisruptionType.SUPPLIER_CANCELLATION:
            return {}
        if event_type is DisruptionType.WAREHOUSE_STOCKOUT:
            return {"units": int(rng.integers(50, 201))}
        if event_type is DisruptionType.TRANSPORT_DELAY:
            return {
                "extra_days": int(rng.integers(1, 4)),
                "cost_multiplier": round(float(rng.uniform(1.2, 2.0)), 2),
            }
        raise ValueError(f"Chưa hỗ trợ params cho {event_type}")

    @staticmethod
    def _pick(values: Sequence[str], rng: np.random.Generator) -> str:
        index = int(rng.integers(0, len(values)))
        return values[index]
