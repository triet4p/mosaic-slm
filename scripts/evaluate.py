"""Chạy evaluation trên dev set và ghi report (Sprint 1, task 14)."""

import argparse
from pathlib import Path

from mosaic.agents.runner import RuleBasedBackbone
from mosaic.evaluation.metrics import evaluate_episodes
from mosaic.simulator.replay import load_episodes


class Args(argparse.Namespace):
    episode_file: str
    output: str
    max_episodes: int | None


def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá agent trên dev set")
    parser.add_argument("--episode-file", default="data/generated/dev_episodes.jsonl")
    parser.add_argument("--output", default="data/results/evaluation_report.json")
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args(namespace=Args())

    episodes = load_episodes(Path(args.episode_file))
    report = evaluate_episodes(episodes, RuleBasedBackbone(), max_episodes=args.max_episodes)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        f"Episodes {report.n_episodes} | NTC {report.mean_ntc:.3f} | "
        f"agent_cost {report.mean_agent_cost:.0f} | reorder {report.mean_reorder_cost:.0f} | "
        f"oracle {report.mean_oracle_cost:.0f} | fill_rate {report.mean_fill_rate:.3f} | "
        f"stockout_rate {report.mean_stockout_rate:.3f}"
    )


if __name__ == "__main__":
    main()
