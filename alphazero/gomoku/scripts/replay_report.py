"""Report self-play training health per iteration from replay shards.

    python -m gomoku.scripts.replay_report runs/<run_id>/replay --last 5
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
import glob
import os
import re

import numpy as np
import pyarrow.parquet as pq

EARLY_STONES = 4
COLUMNS = ["board", "policy_probs", "value", "next_player"]


@dataclass
class IterationStats:
    """Accumulated targets of one self-play iteration."""

    rows: int = 0
    games: int = 0
    first_player_wins: int = 0
    early_rows: int = 0
    early_one_hot: int = 0
    entropy: dict[int, list[float]] = field(default_factory=lambda: {1: [], 2: []})

    def add(self, row: dict) -> None:
        board = np.frombuffer(row["board"], dtype=np.int8)
        target = np.frombuffer(row["policy_probs"], dtype=np.float16).astype(np.float64)
        stones = int((board != 0).sum())
        support = target[target > 0]
        self.rows += 1
        self.entropy[int(row["next_player"])].append(float(-(support * np.log(support)).sum()))
        if stones == 0:
            self.games += 1
            self.first_player_wins += int(row["value"] > 0)
        if stones < EARLY_STONES:
            self.early_rows += 1
            self.early_one_hot += int(target.max() > 0.999)

    def summary(self) -> str:
        games = max(self.games, 1)
        return (
            f"games={self.games} mean_plies={self.rows / games:.1f} "
            f"first_player_win={self.first_player_wins / games:.2f} "
            f"early_one_hot={self.early_one_hot / max(self.early_rows, 1):.3f} "
            f"entropy_p1={np.mean(self.entropy[1] or [0]):.2f} "
            f"entropy_p2={np.mean(self.entropy[2] or [0]):.2f}"
        )


def collect(replay_dir: str) -> dict[int, IterationStats]:
    """Read every replay shard, grouped by the iteration in its file name."""
    stats: dict[int, IterationStats] = defaultdict(IterationStats)
    for path in sorted(glob.glob(os.path.join(replay_dir, "*.parquet"))):
        match = re.search(r"shard-iter(\d+)-", os.path.basename(path))
        iteration = int(match.group(1)) if match else 0
        for row in pq.read_table(path, columns=COLUMNS).to_pylist():
            stats[iteration].add(row)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay_dir")
    parser.add_argument("--last", type=int, default=0, help="Only show the last N iterations.")
    args = parser.parse_args()
    stats = collect(args.replay_dir)
    for iteration in sorted(stats)[-args.last :]:
        print(f"iter {iteration:4d} {stats[iteration].summary()}")


if __name__ == "__main__":
    main()
