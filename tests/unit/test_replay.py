from pathlib import Path

import pytest

from mosaic.simulator.episodes import Episode, EpisodeGenerator, EpisodeGeneratorConfig
from mosaic.simulator.replay import load_episodes, run_episode, save_episodes

SPEC = EpisodeGeneratorConfig(difficulty=2)


def make_episodes(count: int, base_seed: int = 1) -> list[Episode]:
    return EpisodeGenerator(SPEC).generate_dev_set(count=count, base_seed=base_seed)


def test_save_load_round_trips_episodes(tmp_path: Path):
    episodes = make_episodes(3, base_seed=1)
    path = tmp_path / "episodes.jsonl"
    save_episodes(episodes, path)
    loaded = load_episodes(path)
    assert loaded == episodes


def test_run_episode_is_deterministic():
    episode = EpisodeGenerator(SPEC).generate(seed=5)
    first = run_episode(episode)
    second = run_episode(episode)
    assert first.state == second.state
    assert first.state.ledger == second.state.ledger


def test_replay_from_file_matches_fresh_generation(tmp_path: Path):
    episode = EpisodeGenerator(SPEC).generate(seed=3)
    direct = run_episode(episode)
    path = tmp_path / "single.jsonl"
    save_episodes([episode], path)
    loaded = load_episodes(path)[0]
    replayed = run_episode(loaded)
    assert replayed.state == direct.state


def test_replay_from_dev_set_file(tmp_path: Path):
    episodes = make_episodes(5, base_seed=0)
    path = tmp_path / "dev.jsonl"
    save_episodes(episodes, path)
    loaded = load_episodes(path)
    assert [e.seed for e in loaded] == [0, 1, 2, 3, 4]
    simulator = run_episode(loaded[0])
    assert simulator.state.day == 30
    assert len(simulator.state.ledger) == 30


def test_run_episode_does_not_mutate_episode_config():
    episode = EpisodeGenerator(SPEC).generate(seed=8)
    config_before = episode.config.model_dump(exclude_computed_fields=True)
    run_episode(episode)
    assert episode.config.model_dump(exclude_computed_fields=True) == config_before


def test_run_episode_requires_no_external_services():
    episode = EpisodeGenerator(SPEC).generate(seed=1)
    simulator = run_episode(episode)
    assert simulator.state.day == 30
    assert simulator.state.cumulative_costs is not None


def test_load_episodes_skips_empty_lines(tmp_path: Path):
    episodes = make_episodes(2)
    path = tmp_path / "with_blank.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        fh.write(episodes[0].model_dump_json(exclude_computed_fields=True) + "\n\n")
        fh.write(episodes[1].model_dump_json(exclude_computed_fields=True) + "\n")
    assert load_episodes(path) == episodes


def test_run_episode_rejects_invalid_episode_file(tmp_path: Path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"seed": 1}\n', encoding="utf-8")
    with pytest.raises(Exception):
        load_episodes(path)
