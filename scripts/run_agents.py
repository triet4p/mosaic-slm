"""Chạy single-agent trên một episode và in tóm tắt (Sprint 1, task 12)."""

import argparse
from pathlib import Path

from mosaic.agents.runner import RuleBasedBackbone, SingleAgentRunner
from mosaic.simulator.replay import load_episodes


class Args(argparse.Namespace):
    episode_file: str
    index: int


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy single-agent trên một episode")
    parser.add_argument("--episode-file", default="data/generated/dev_episodes.jsonl")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args(namespace=Args())

    episodes = load_episodes(Path(args.episode_file))
    if not 0 <= args.index < len(episodes):
        raise SystemExit(f"index {args.index} ngoài khoảng 0..{len(episodes) - 1}")
    episode = episodes[args.index]
    result = SingleAgentRunner(episode, RuleBasedBackbone()).run()
    n_tool_calls = sum(len(step.tool_calls) for step in result.trajectory)
    n_actions = sum(len(step.actions) for step in result.trajectory)
    print(
        f"Episode seed {episode.seed} | days {len(result.trajectory)} | "
        f"reward {result.business_reward:.2f} | cost {result.total_cost:.2f} | "
        f"fill_rate {result.fill_rate:.3f} | actions {n_actions} | "
        f"tool_calls {n_tool_calls}"
    )


if __name__ == "__main__":
    main()
