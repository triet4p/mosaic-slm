import math

import pytest

from mosaic.simulator.demand import DemandGenerator, DemandProfile
from mosaic.simulator.models import DisruptionEvent, DisruptionType

BASE_PROFILES = (
    DemandProfile(region_id="R1", product_id="P1", base_demand=10.0),
    DemandProfile(region_id="R2", product_id="P1", base_demand=20.0),
    DemandProfile(region_id="R1", product_id="P2", base_demand=30.0),
)

NOISY_PROFILES = (
    DemandProfile(region_id="R1", product_id="P1", base_demand=10.0, noise_std=0.3),
    DemandProfile(region_id="R2", product_id="P1", base_demand=20.0, noise_std=0.3),
)


def make_demand_event(
    disruption_id: str,
    event_type: DisruptionType,
    start_day: int,
    end_day: int,
    subject_type: str,
    subject_id: str,
    multiplier: float,
) -> DisruptionEvent:
    return DisruptionEvent(
        disruption_id=disruption_id,
        type=event_type,
        start_day=start_day,
        end_day=end_day,
        subject_type=subject_type,
        subject_id=subject_id,
        params={"multiplier": multiplier},
    )


def test_demand_schedule_is_deterministic_for_same_seed():
    generator = DemandGenerator(seed=42, profiles=NOISY_PROFILES)
    first = generator.generate_schedule(num_days=10)
    second = generator.generate_schedule(num_days=10)
    assert first == second


def test_demand_schedule_differs_for_different_seeds():
    first = DemandGenerator(seed=1, profiles=NOISY_PROFILES).generate_schedule(num_days=10)
    second = DemandGenerator(seed=2, profiles=NOISY_PROFILES).generate_schedule(num_days=10)
    assert first != second


def test_demand_matches_base_when_no_seasonality_and_no_noise():
    schedule = DemandGenerator(seed=0, profiles=BASE_PROFILES).generate_schedule(num_days=5)
    for day in schedule.days:
        assert day.quantities["R1"]["P1"] == 10
        assert day.quantities["R2"]["P1"] == 20
        assert day.quantities["R1"]["P2"] == 30


def test_demand_follows_seasonality_deterministically():
    profiles = (
        DemandProfile(
            region_id="R1",
            product_id="P1",
            base_demand=10.0,
            seasonality_amplitude=0.5,
            seasonality_period=7,
        ),
    )
    schedule = DemandGenerator(seed=0, profiles=profiles).generate_schedule(num_days=7)
    for day in schedule.days:
        seasonal = 1.0 + 0.5 * math.sin(2.0 * math.pi * (day.day - 1) / 7)
        assert day.quantities["R1"]["P1"] == round(10.0 * seasonal)


def test_demand_spike_multiplies_affected_days_only():
    spike = make_demand_event("D1", DisruptionType.DEMAND_SPIKE, 5, 7, "region", "R1", 2.0)
    with_event = DemandGenerator(seed=0, profiles=BASE_PROFILES).generate_schedule(
        num_days=10, disruptions=[spike]
    )
    without_event = DemandGenerator(seed=0, profiles=BASE_PROFILES).generate_schedule(num_days=10)
    for day in with_event.days:
        if 5 <= day.day <= 7:
            assert (
                day.quantities["R1"]["P1"] == 2 * without_event.get(day.day).quantities["R1"]["P1"]
            )
            assert (
                day.quantities["R1"]["P2"] == 2 * without_event.get(day.day).quantities["R1"]["P2"]
            )
            assert day.quantities["R2"]["P1"] == without_event.get(day.day).quantities["R2"]["P1"]
        else:
            assert day == without_event.get(day.day)


def test_demand_drop_multiplies_affected_days_only():
    drop = make_demand_event("D2", DisruptionType.DEMAND_DROP, 3, 4, "product", "P1", 0.5)
    schedule = DemandGenerator(seed=0, profiles=BASE_PROFILES).generate_schedule(
        num_days=10, disruptions=[drop]
    )
    baseline = DemandGenerator(seed=0, profiles=BASE_PROFILES).generate_schedule(num_days=10)
    for day in schedule.days:
        if 3 <= day.day <= 4:
            assert day.quantities["R1"]["P1"] == 5
            assert day.quantities["R2"]["P1"] == 10
            assert day.quantities["R1"]["P2"] == 30
        else:
            assert day == baseline.get(day.day)


def test_demand_spike_targets_single_product_in_region():
    spike = make_demand_event(
        "D3", DisruptionType.DEMAND_SPIKE, 2, 2, "product_region", "R1:P1", 3.0
    )
    schedule = DemandGenerator(seed=0, profiles=BASE_PROFILES).generate_schedule(
        num_days=5, disruptions=[spike]
    )
    assert schedule.get(2).quantities["R1"]["P1"] == 30
    assert schedule.get(2).quantities["R1"]["P2"] == 30
    assert schedule.get(2).quantities["R2"]["P1"] == 20


def test_demand_is_never_negative_with_extreme_drop():
    drop = make_demand_event("D4", DisruptionType.DEMAND_DROP, 1, 30, "region", "R1", 0.0)
    schedule = DemandGenerator(seed=0, profiles=BASE_PROFILES).generate_schedule(
        num_days=30, disruptions=[drop]
    )
    for day in schedule.days:
        assert day.quantities["R1"]["P1"] == 0
        assert day.quantities["R1"]["P2"] == 0


def test_demand_schedule_covers_all_days_and_profiles():
    schedule = DemandGenerator(seed=7, profiles=BASE_PROFILES).generate_schedule(num_days=30)
    assert len(schedule.days) == 30
    for day in schedule.days:
        assert set(day.quantities) == {"R1", "R2"}
        assert set(day.quantities["R1"]) == {"P1", "P2"}
        assert set(day.quantities["R2"]) == {"P1"}


def test_demand_schedule_get_returns_requested_day():
    schedule = DemandGenerator(seed=0, profiles=BASE_PROFILES).generate_schedule(num_days=3)
    assert schedule.get(1).day == 1
    assert schedule.get(3).day == 3
    with pytest.raises(IndexError):
        schedule.get(4)


def test_demand_spike_requires_multiplier():
    event = DisruptionEvent(
        disruption_id="D5",
        type=DisruptionType.DEMAND_SPIKE,
        start_day=1,
        subject_type="region",
        subject_id="R1",
        params={},
    )
    with pytest.raises(ValueError):
        DemandGenerator(seed=0, profiles=BASE_PROFILES).generate_schedule(
            num_days=5, disruptions=[event]
        )


def test_demand_event_rejects_invalid_subject_type():
    event = make_demand_event("D6", DisruptionType.DEMAND_SPIKE, 1, 2, "warehouse", "WH1", 2.0)
    with pytest.raises(ValueError):
        DemandGenerator(seed=0, profiles=BASE_PROFILES).generate_schedule(
            num_days=5, disruptions=[event]
        )


def test_demand_generator_rejects_empty_profiles():
    with pytest.raises(ValueError):
        DemandGenerator(seed=0, profiles=[])
