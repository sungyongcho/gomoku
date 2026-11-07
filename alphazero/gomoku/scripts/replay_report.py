"""Report self-play training health per iteration from replay shards.

Works on a local folder or a GCS path. With a checkpoint, it also checks how many
recent positions the network already scores as nearly won or lost.

    python -m gomoku.scripts.replay_report gs://<bucket>/v5-small-6x96/replay --last 5 \\
        --checkpoint gs://<bucket>/v5-small-6x96/ckpt/iteration_0010.pt \\
        --config configs/v5-small-6x96.yaml
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
import io
import random
import re

import fsspec
import numpy as np
import pyarrow.parquet as pq

EARLY_STONES = 4
SATURATED = 0.95
COLUMNS = ["board", "policy_probs", "value", "next_player"]


@dataclass
class IterationStats:
    """Accumulated targets of one self-play iteration."""

    rows: int = 0
    games: int = 0
    first_player_wins: int = 0
    draws: int = 0
    early_rows: int = 0
    early_one_hot: int = 0
    entropy: dict[int, list[float]] = field(default_factory=lambda: {1: [], 2: []})
    width: dict[int, list[int]] = field(default_factory=lambda: {1: [], 2: []})

    def add(self, row: dict) -> None:
        board = np.frombuffer(row["board"], dtype=np.int8)
        target = np.frombuffer(row["policy_probs"], dtype=np.float16).astype(np.float64)
        stones = int((board != 0).sum())
        support = target[target > 0]
        player = int(row["next_player"])
        self.rows += 1
        self.entropy[player].append(float(-(support * np.log(support)).sum()))
        self.width[player].append(int(support.size))
        if stones == 0:
            self.games += 1
            self.first_player_wins += int(row["value"] > 0)
            self.draws += int(row["value"] == 0)
        if stones < EARLY_STONES:
            self.early_rows += 1
            self.early_one_hot += int(target.max() > 0.999)

    def summary(self) -> str:
        games = max(self.games, 1)
        return (
            f"games={self.games} mean_plies={self.rows / games:.1f} "
            f"first_player_win={self.first_player_wins / games:.2f} "
            f"draws={self.draws / games:.2f} "
            f"early_one_hot={self.early_one_hot / max(self.early_rows, 1):.3f} "
            f"entropy_p1={np.mean(self.entropy[1] or [0]):.2f} "
            f"entropy_p2={np.mean(self.entropy[2] or [0]):.2f} "
            f"width_p1={np.mean(self.width[1] or [0]):.0f} "
            f"width_p2={np.mean(self.width[2] or [0]):.0f}"
        )


def _shards(replay_dir: str) -> tuple[fsspec.AbstractFileSystem, list[tuple[int, str]]]:
    fs, root = fsspec.core.url_to_fs(replay_dir)
    shards = []
    for path in sorted(fs.glob(f"{root.rstrip('/')}/*.parquet")):
        match = re.search(r"shard-iter(\d+)-", path.rsplit("/", 1)[-1])
        shards.append((int(match.group(1)) if match else 0, path))
    return fs, shards


def _rows(fs: fsspec.AbstractFileSystem, path: str, columns: list[str] | None = COLUMNS) -> list[dict]:
    with fs.open(path, "rb") as handle:
        return pq.read_table(handle, columns=columns).to_pylist()


def collect(replay_dir: str, last: int = 0) -> dict[int, IterationStats]:
    """Read replay shards grouped by the iteration in their file names."""
    fs, shards = _shards(replay_dir)
    keep = set(sorted({iteration for iteration, _ in shards})[-last:]) if last else None
    stats: dict[int, IterationStats] = defaultdict(IterationStats)
    for iteration, path in shards:
        if keep is None or iteration in keep:
            for row in _rows(fs, path):
                stats[iteration].add(row)
    return stats


def value_saturation(
    replay_dir: str, checkpoint: str, config: str, samples: int = 512, seed: int = 0
) -> tuple[float, float]:
    """Share of latest-iteration positions the network scores beyond +-0.95, and its empty-board value."""
    import torch

    from gomoku.alphazero.learning.dataset import reconstruct_state
    from gomoku.core.gomoku import Gomoku
    from gomoku.model.policy_value_net import PolicyValueNet
    from gomoku.utils.config.loader import load_and_parse_config
    from gomoku.utils.state_dict_utils import align_state_dict_to_model

    cfg = load_and_parse_config(config)
    game = Gomoku(cfg.board)
    model = PolicyValueNet(game, cfg.model, torch.device("cpu"))
    with fsspec.open(checkpoint, "rb") as handle:
        raw = torch.load(io.BytesIO(handle.read()), map_location="cpu", weights_only=False)
    state = raw.get("model_state_dict", raw.get("state_dict", raw)) if isinstance(raw, dict) else raw
    align_state_dict_to_model(state, model.state_dict())
    model.load_state_dict(state)
    model.eval()

    fs, shards = _shards(replay_dir)
    latest = max(iteration for iteration, _ in shards)
    rows = [row for iteration, path in shards if iteration == latest for row in _rows(fs, path, None)]
    picked = random.Random(seed).sample(rows, min(samples, len(rows)))
    states = [reconstruct_state(row, game) for row in picked] + [game.get_initial_state()]
    encoded = torch.as_tensor(np.asarray(game.get_encoded_state(states), dtype=np.float32))
    with torch.no_grad():
        _, values = model(encoded)
    values = values.reshape(-1).numpy()
    return float(np.mean(np.abs(values[:-1]) > SATURATED)), float(values[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay_dir", help="Local folder or gs:// path of replay shards.")
    parser.add_argument("--last", type=int, default=0, help="Only show the last N iterations.")
    parser.add_argument("--checkpoint", help="Also measure value saturation with this checkpoint.")
    parser.add_argument("--config", help="Config matching the checkpoint's model.")
    args = parser.parse_args()
    stats = collect(args.replay_dir, args.last)
    for iteration in sorted(stats):
        print(f"iter {iteration:4d} {stats[iteration].summary()}")
    if args.checkpoint:
        if not args.config:
            parser.error("--checkpoint needs --config")
        saturated, empty_value = value_saturation(args.replay_dir, args.checkpoint, args.config)
        print(f"value_saturated={saturated:.3f} (|v|>{SATURATED}) empty_board_value={empty_value:+.2f}")


if __name__ == "__main__":
    main()
