"""Chạy replay một episode từ file và in tóm tắt (Sprint 1, task 9)."""

import argparse
from pathlib import Path

from mosaic.simulator.replay import load_episodes, run_episode
from mosaic.simulator.reward import RewardCalculator


class Args(argparse.Namespace):
    episode_file: str
    index: int


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy replay một episode")
    parser.add_argument("--episode-file", default="data/generated/dev_episodes.jsonl")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args(namespace=Args())

    episodes = load_episodes(Path(args.episode_file))
    if not 0 <= args.index < len(episodes):
        raise SystemExit(f"index {args.index} ngoài khoảng 0..{len(episodes) - 1}")
    episode = episodes[args.index]
    simulator = run_episode(episode)
    state = simulator.state
    breakdown = RewardCalculator().business_reward(state)
    print(
        f"Episode seed {episode.seed} | days {state.day} | cost "
        f"{breakdown.total_cost:.2f} | fill_rate {breakdown.fill_rate:.3f} | "
        f"reward {breakdown.reward:.2f}"
    )


if __name__ == "__main__":
    main()
