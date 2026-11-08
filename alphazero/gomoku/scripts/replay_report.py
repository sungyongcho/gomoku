"""Report self-play training health per iteration from replay shards.

Works on a local folder or a GCS path. Each iteration also shows how games ended. With a
checkpoint, it checks the network on positions from the latest iteration: how many it
scores as nearly won or lost, and whether its policy alone finds an immediate win, blocks
the opponent's immediate win and avoids double-three points.

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
COLUMNS = ["board", "policy_probs", "value", "next_player", "p1_pts", "p2_pts"]
CAPTURE_GOAL = 5  # v5 board rules; a mover one capture short most likely won by capture


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
    five_wins: int = 0
    capture_wins: int = 0
    ended_draws: int = 0
    _last: dict | None = None

    def end_game(self) -> None:
        """Classify the finished game from its last stored position (the final mover to play)."""
        row, self._last = self._last, None
        if row is None:
            return
        if row["value"] == 0:
            self.ended_draws += 1
            return
        pts = row["p1_pts"] if int(row["next_player"]) == 1 else row["p2_pts"]
        if pts >= CAPTURE_GOAL - 1:
            self.capture_wins += 1
        else:
            self.five_wins += 1

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
            self.end_game()
            self.games += 1
            self.first_player_wins += int(row["value"] > 0)
            self.draws += int(row["value"] == 0)
        if stones < EARLY_STONES:
            self.early_rows += 1
            self.early_one_hot += int(target.max() > 0.999)
        self._last = row

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
            f"width_p2={np.mean(self.width[2] or [0]):.0f} "
            f"five_win={self.five_wins / games:.2f} "
            f"capture_win={self.capture_wins / games:.2f}"
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
            stats[iteration].end_game()  # shards hold whole games
    return stats


def _load_model(checkpoint: str, config: str):
    """Return (game, model) for a checkpoint and the config describing its network."""
    import torch

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
    return game, model


def _latest_rows(replay_dir: str) -> list[dict]:
    fs, shards = _shards(replay_dir)
    latest = max(iteration for iteration, _ in shards)
    return [row for iteration, path in shards if iteration == latest for row in _rows(fs, path, None)]


def _evaluate(game, model, states) -> tuple[np.ndarray, np.ndarray]:
    """Policy probabilities over all points and values for a list of states."""
    import torch

    encoded = torch.as_tensor(np.asarray(game.get_encoded_state(states), dtype=np.float32))
    with torch.no_grad():
        logits, values = model(encoded)
    probs = torch.softmax(logits.reshape(len(states), -1), dim=-1).numpy()
    return probs, values.reshape(-1).numpy()


def value_saturation(
    replay_dir: str, checkpoint: str, config: str, samples: int = 512, seed: int = 0
) -> tuple[float, float]:
    """Share of latest-iteration positions the network scores beyond +-0.95, and its empty-board value."""
    from gomoku.alphazero.learning.dataset import reconstruct_state

    game, model = _load_model(checkpoint, config)
    rows = _latest_rows(replay_dir)
    picked = random.Random(seed).sample(rows, min(samples, len(rows)))
    states = [reconstruct_state(row, game) for row in picked] + [game.get_initial_state()]
    _, values = _evaluate(game, model, states)
    return float(np.mean(np.abs(values[:-1]) > SATURATED)), float(values[-1])


def _winning_points(game, state, player: int, candidates: list[int]) -> list[int]:
    wins = []
    for idx in candidates:
        x, y = idx % game.col_count, idx // game.col_count
        if game.check_win(game.get_next_state(state, (x, y), player), (x, y)):
            wins.append(idx)
    return wins


def tactics(
    replay_dir: str, checkpoint: str, config: str, samples: int = 300, seed: int = 0
) -> dict[str, float]:
    """Check the raw policy (no search) on latest-iteration positions.

    win: the side to move can win at once; hit = top policy move wins, mass = probability on wins.
    block: no win, but the opponent threatens to win at once; hit = top move takes that point.
    forbidden: probability on empty points that are illegal for the mover (double-three),
    next to what a uniform policy would put there.
    """
    from gomoku.alphazero.learning.dataset import reconstruct_state

    game, model = _load_model(checkpoint, config)
    rows = _latest_rows(replay_dir)
    random.Random(seed).shuffle(rows)
    found = {"win": [], "block": [], "forbidden": []}
    for row in rows:
        if all(len(v) >= samples for v in found.values()):
            break
        state = reconstruct_state(row, game)
        board = state.board.reshape(-1)
        if int((board != 0).sum()) < 6:
            continue
        game.get_legal_moves(state)
        legal = {int(i) for i in state.legal_indices_cache}
        empty = {int(i) for i in np.flatnonzero(board == 0)}
        if empty - legal and len(found["forbidden"]) < samples:
            found["forbidden"].append((state, sorted(empty - legal), len(empty)))
        # A move that wins at once, by five or by capture, touches an existing stone.
        padded = np.pad((state.board != 0).astype(np.int8), 1)
        h, w = state.board.shape
        near = np.zeros((h, w), dtype=np.int8)
        for dy in range(3):
            for dx in range(3):
                near |= padded[dy : dy + h, dx : dx + w]
        candidates = sorted(legal & {int(i) for i in np.flatnonzero(near.reshape(-1))})
        mover = int(state.next_player)
        mine = _winning_points(game, state, mover, candidates)
        if mine:
            if len(found["win"]) < samples:
                found["win"].append((state, mine))
        elif len(found["block"]) < samples:
            theirs = _winning_points(game, state, 3 - mover, candidates)
            if theirs:
                found["block"].append((state, theirs))
    out: dict[str, float] = {}
    for kind in ("win", "block"):
        items = found[kind]
        out[f"{kind}_positions"] = len(items)
        if items:
            probs, _ = _evaluate(game, model, [s for s, _ in items])
            out[f"{kind}_top1"] = float(np.mean([int(np.argmax(probs[i])) in set(t) for i, (_, t) in enumerate(items)]))
            out[f"{kind}_mass"] = float(np.mean([probs[i][t].sum() for i, (_, t) in enumerate(items)]))
    items = found["forbidden"]
    out["forbidden_positions"] = len(items)
    if items:
        probs, _ = _evaluate(game, model, [s for s, _, _ in items])
        out["forbidden_mass"] = float(np.mean([probs[i][f].sum() for i, (_, f, _) in enumerate(items)]))
        out["forbidden_uniform"] = float(np.mean([len(f) / e for _, f, e in items]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay_dir", help="Local folder or gs:// path of replay shards.")
    parser.add_argument("--last", type=int, default=0, help="Only show the last N iterations.")
    parser.add_argument("--checkpoint", help="Also measure value saturation with this checkpoint.")
    parser.add_argument("--config", help="Config matching the checkpoint's model.")
    parser.add_argument(
        "--tactic-samples", type=int, default=300, help="Positions per tactic check (win, block, forbidden)."
    )
    args = parser.parse_args()
    stats = collect(args.replay_dir, args.last)
    for iteration in sorted(stats):
        print(f"iter {iteration:4d} {stats[iteration].summary()}")
    if args.checkpoint:
        if not args.config:
            parser.error("--checkpoint needs --config")
        saturated, empty_value = value_saturation(args.replay_dir, args.checkpoint, args.config)
        print(f"value_saturated={saturated:.3f} (|v|>{SATURATED}) empty_board_value={empty_value:+.2f}")
        t = tactics(args.replay_dir, args.checkpoint, args.config, samples=args.tactic_samples)
        print(
            "policy without search: "
            f"win top1={t.get('win_top1', float('nan')):.2f} mass={t.get('win_mass', float('nan')):.2f} "
            f"(n={t['win_positions']}) | block top1={t.get('block_top1', float('nan')):.2f} "
            f"mass={t.get('block_mass', float('nan')):.2f} (n={t['block_positions']}) | "
            f"forbidden mass={t.get('forbidden_mass', float('nan')):.3f} "
            f"(uniform {t.get('forbidden_uniform', float('nan')):.3f}, n={t['forbidden_positions']})"
        )


if __name__ == "__main__":
    main()
