"""Replay episode (Sprint 1, task 9).

Episode chứa toàn bộ thông tin environment (demand schedule, disruptions,
suppliers, initial stock, seed...) nên replay không cần generator hay dịch
vụ bên ngoài. Chạy với cùng seed và cùng SupplierBehavior cho kết quả
bit-identical (không cần RNG khác).

Lưu ý: mọi dump episode phải dùng `exclude_computed_fields=True` vì
`SKUState.available` là computed field, không validate lại được.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from mosaic.simulator.core import Simulator
from mosaic.simulator.episodes import Episode
from mosaic.simulator.supplier import SupplierBehavior


def save_episodes(episodes: Sequence[Episode], path: Path) -> None:
    """Ghi danh sách episode thành JSONL (bỏ computed fields)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for episode in episodes:
            fh.write(episode.model_dump_json(exclude_computed_fields=True) + "\n")


def load_episodes(path: Path) -> list[Episode]:
    """Đọc JSONL → list[Episode]."""
    episodes: list[Episode] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                episodes.append(Episode.model_validate_json(line))
    return episodes


def run_episode(episode: Episode) -> Simulator:
    """Chạy toàn bộ episode (không action); deterministic theo seed."""
    behavior = SupplierBehavior(seed=episode.config.seed, stochastic=episode.supplier_stochastic)
    simulator = Simulator(episode.config, supplier_behavior=behavior)
    for _ in range(episode.config.num_days):
        simulator.advance_day()
    return simulator
