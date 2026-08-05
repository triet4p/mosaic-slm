"""Bộ sinh demand hằng ngày theo region/product (Sprint 1, task 2).

Demand = base * (1 + seasonality) * noise, deterministic theo seed.
Sự kiện DEMAND_SPIKE / DEMAND_DROP nhân hệ số multiplier vào các ngày
bị ảnh hưởng (subject: region, product hoặc product_region "region:product").
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from mosaic.simulator.models import DisruptionEvent, DisruptionType

DEMAND_DISRUPTION_TYPES = frozenset({DisruptionType.DEMAND_SPIKE, DisruptionType.DEMAND_DROP})
DEMAND_SUBJECT_TYPES = frozenset({"region", "product", "product_region"})


class DemandProfile(BaseModel):
    """Cấu hình demand của một cặp (region, product)."""

    model_config = ConfigDict(extra="forbid")

    region_id: str
    product_id: str
    base_demand: float = Field(
        ge=0.0, description="Nhu cầu trung bình mỗi ngày; 0 = không có demand"
    )
    seasonality_amplitude: float = Field(
        ge=0.0, le=1.0, default=0.0, description="Biên độ mùa vụ, tính theo phần của base_demand"
    )
    seasonality_period: int = Field(ge=1, default=7, description="Chu kỳ mùa vụ theo ngày")
    noise_std: float = Field(
        ge=0.0,
        default=0.0,
        description="Độ lệch chuẩn của noise (phần của base_demand); 0 = deterministic",
    )


class DailyDemand(BaseModel):
    """Demand của một ngày: region_id -> product_id -> số đơn vị."""

    model_config = ConfigDict(extra="forbid")

    day: int = Field(ge=1)
    quantities: dict[str, dict[str, int]]


class DemandSchedule(BaseModel):
    """Toàn bộ lịch trình demand của một episode."""

    model_config = ConfigDict(extra="forbid")

    days: list[DailyDemand]

    def get(self, day: int) -> DailyDemand:
        """Demand của ngày (1-based)."""
        if not 1 <= day <= len(self.days):
            raise IndexError(f"day {day} nằm ngoài episode (1..{len(self.days)})")
        return self.days[day - 1]


class DemandGenerator:
    """Sinh lịch trình demand deterministic, hỗ trợ demand spike/drop."""

    def __init__(self, seed: int, profiles: Sequence[DemandProfile]) -> None:
        self._seed = seed
        self._profiles = tuple(profiles)
        if not self._profiles:
            raise ValueError("Cần ít nhất một DemandProfile")

    def generate_schedule(
        self, num_days: int, disruptions: Sequence[DisruptionEvent] = ()
    ) -> DemandSchedule:
        """Sinh demand cho num_days ngày, áp dụng các sự kiện spike/drop."""
        if num_days <= 0:
            raise ValueError("num_days phải lớn hơn 0")
        rng = np.random.default_rng(self._seed)
        events = [e for e in disruptions if e.type in DEMAND_DISRUPTION_TYPES]
        self._validate_demand_events(events)
        days: list[DailyDemand] = []
        for day in range(1, num_days + 1):
            quantities: dict[str, dict[str, int]] = {}
            for profile in self._profiles:
                seasonal = 1.0 + profile.seasonality_amplitude * math.sin(
                    2.0 * math.pi * (day - 1) / profile.seasonality_period
                )
                noise = 1.0 + rng.normal(0.0, profile.noise_std)
                value = profile.base_demand * seasonal * noise
                for event in events:
                    if self._affects(event, profile.region_id, profile.product_id, day):
                        value *= self._multiplier(event)
                quantity = max(0, round(value))
                quantities.setdefault(profile.region_id, {})[profile.product_id] = quantity
            days.append(DailyDemand(day=day, quantities=quantities))
        return DemandSchedule(days=days)

    @classmethod
    def _validate_demand_events(cls, events: Sequence[DisruptionEvent]) -> None:
        for event in events:
            if event.subject_type not in DEMAND_SUBJECT_TYPES:
                raise ValueError(
                    f"Disruption {event.disruption_id} có subject_type không hợp lệ cho demand: "
                    f"{event.subject_type!r}"
                )
            cls._multiplier(event)

    @classmethod
    def _multiplier(cls, event: DisruptionEvent) -> float:
        multiplier = event.params.get("multiplier")
        if not isinstance(multiplier, (int, float)):
            raise ValueError(f"Disruption {event.disruption_id} thiếu tham số 'multiplier' kiểu số")
        return float(multiplier)

    @staticmethod
    def _affects(event: DisruptionEvent, region_id: str, product_id: str, day: int) -> bool:
        if day < event.start_day:
            return False
        if event.end_day is not None and day > event.end_day:
            return False
        if event.subject_type == "region":
            return event.subject_id == region_id
        if event.subject_type == "product":
            return event.subject_id == product_id
        if event.subject_type == "product_region":
            return event.subject_id == f"{region_id}:{product_id}"
        return False
