"""Sinh dev set episodes ra data/generated/ (Sprint 1, task 8)."""

import argparse
from pathlib import Path

from mosaic.simulator.episodes import EpisodeGenerator, EpisodeGeneratorConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Sinh dev set episodes")
    parser.add_argument(
        "--config", default="configs/environment/episode_default.yaml", help="Path tới spec YAML"
    )
    parser.add_argument("--count", type=int, default=100, help="Số episode")
    parser.add_argument("--base-seed", type=int, default=0, help="Seed bắt đầu")
    parser.add_argument("--output", default="data/generated/dev_episodes.jsonl", help="File đầu ra")
    args = parser.parse_args()

    spec = EpisodeGeneratorConfig.from_yaml(Path(args.config))
    episodes = EpisodeGenerator(spec).generate_dev_set(count=args.count, base_seed=args.base_seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for episode in episodes:
            fh.write(episode.model_dump_json(exclude_computed_fields=True) + "\n")
    print(f"Wrote {len(episodes)} episodes to {output}")


if __name__ == "__main__":
    main()
