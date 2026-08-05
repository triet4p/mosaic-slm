from pathlib import Path

import pytest

from mosaic.simulator.core import Simulator
from mosaic.simulator.episodes import EpisodeGenerator, EpisodeGeneratorConfig
from mosaic.simulator.supplier import SupplierBehavior

SPEC = EpisodeGeneratorConfig(difficulty=2)
CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "environment" / "episode_default.yaml"
)


def test_generate_is_deterministic_for_same_seed():
    generator = EpisodeGenerator(SPEC)
    first = generator.generate(seed=7)
    second = generator.generate(seed=7)
    assert first == second


def test_generate_differs_for_different_seeds():
    generator = EpisodeGenerator(SPEC)
    assert generator.generate(seed=1) != generator.generate(seed=2)


def test_episode_matches_locked_dimensions():
    episode = EpisodeGenerator(SPEC).generate(seed=1)
    config = episode.config
    assert config.num_days == 30
    assert len(config.warehouses) == 3
    assert len(config.suppliers) == 2
    assert len(config.initial_sku_states) == 3 * 20
    assert 1 <= len(config.disruptions) <= 5
    assert len(config.demand_schedule.days) == 30


def test_episode_difficulty_1_is_deterministic_and_single_disruption():
    generator = EpisodeGenerator(EpisodeGeneratorConfig(difficulty=1))
    episode = generator.generate(seed=3)
    assert episode.difficulty == 1
    assert len(episode.config.disruptions) == 1
    assert episode.supplier_stochastic is False
    assert (
        episode.config.suppliers[0].current_lead_time == episode.config.suppliers[0].base_lead_time
    )


def test_episode_difficulty_2_is_stochastic():
    episode = EpisodeGenerator(EpisodeGeneratorConfig(difficulty=2)).generate(seed=5)
    assert episode.supplier_stochastic is True


def test_episode_difficulty_3_has_noise_and_recurrences():
    episode = EpisodeGenerator(EpisodeGeneratorConfig(difficulty=3)).generate(seed=9)
    assert episode.supplier_stochastic is True
    recurring = [e for e in episode.config.disruptions if e.disruption_id.endswith("R")]
    assert len(recurring) >= 1


def test_generate_dev_set_respects_count_and_seeds():
    generator = EpisodeGenerator(SPEC)
    episodes = generator.generate_dev_set(count=5, base_seed=10)
    assert [e.seed for e in episodes] == [10, 11, 12, 13, 14]
    assert len({e.model_dump_json() for e in episodes}) == 5


def test_generate_dev_set_rejects_zero_count():
    with pytest.raises(ValueError):
        EpisodeGenerator(SPEC).generate_dev_set(count=0)


def test_spec_rejects_invalid_ranges():
    with pytest.raises(ValueError):
        EpisodeGeneratorConfig(min_disruptions=3, max_disruptions=1)
    with pytest.raises(ValueError):
        EpisodeGeneratorConfig(min_demand=10, max_demand=5)


def test_spec_loads_from_yaml():
    spec = EpisodeGeneratorConfig.from_yaml(CONFIG_PATH)
    assert spec.num_days == 30
    assert spec.num_warehouses == 3
    assert spec.num_suppliers == 2
    assert spec.num_products == 20
    assert spec.difficulty == 2


def test_episode_round_trips_json():
    episode = EpisodeGenerator(SPEC).generate(seed=2)
    dumped = episode.model_dump_json(exclude_computed_fields=True)
    restored = type(episode).model_validate_json(dumped)
    assert restored == episode


def test_episode_constructs_simulator_and_replays_deterministically():
    def run(seed: int) -> Simulator:
        episode = EpisodeGenerator(SPEC).generate(seed=seed)
        behavior = SupplierBehavior(
            seed=episode.config.seed, stochastic=episode.supplier_stochastic
        )
        simulator = Simulator(episode.config, supplier_behavior=behavior)
        for _ in range(episode.config.num_days):
            simulator.advance_day()
        return simulator

    assert run(4).state == run(4).state


def test_each_sku_has_exactly_one_supplier_assignment():
    episode = EpisodeGenerator(SPEC).generate(seed=6)
    simulator = Simulator(
        episode.config, supplier_behavior=SupplierBehavior(seed=episode.config.seed)
    )
    assert len(simulator.state.supplier_assignments) == 20
