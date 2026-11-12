"""Train a fresh network on the game records of an earlier run, to seed a larger model.

A larger network cannot take over a smaller one's weights, but it can learn from the games
the smaller one played: the search's visit shares are the policy target and the game result
is the value target, exactly as in the run that produced them.

    python gomoku/scripts/pretrain.py --config configs/v6-10x128.yaml \\
        --replay gs://bucket/v5-small-6x96/replay --first 96 --last 143 --holdout 144 \\
        --lr 0.001,0.001,0.0003,0.0001 \\
        --baseline gs://bucket/v5-small-6x96/ckpt/iteration_0143.pt@configs/v5.yaml \\
        --out gs://bucket/v6-10x128/ckpt/seed_v5games.pt

One pass over the records per ``--lr`` entry. After each pass the loss on the held-out
iteration is printed and the weights are saved to ``--out``, so a stopped run keeps its last
pass. ``--baseline`` prints the same loss for an existing model: pick one that has not
trained on the held-out iteration (iteration N's games are played by model N-1 and first
trained on by model N). ``--init`` starts from weights saved by an earlier call, to add
passes or widen the range of iterations. ``--out`` named ``seed_*.pt`` in a new run's checkpoint directory is
picked up by train.py as that run's starting weights.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import shutil
import time

import fsspec
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gomoku.alphazero.learning.dataset import (
    GameSample,
    ReplayDataset,
    decode_policy,
    reconstruct_state,
)
from gomoku.alphazero.learning.optimizer import build_optimizer
from gomoku.alphazero.learning.trainer import AlphaZeroTrainer
from gomoku.core.gomoku import Gomoku
from gomoku.model.policy_value_net import PolicyValueNet
from gomoku.utils.config.loader import load_and_parse_config
from gomoku.utils.io_helpers import list_replay_shards, read_parquet_shard
from gomoku.utils.paths import load_state_dict_from_fs, save_state
from gomoku.utils.state_dict_utils import align_state_dict_to_model


def _fs_for(path: str) -> fsspec.AbstractFileSystem:
    return fsspec.filesystem("gcs" if path.startswith("gs://") else "file")


def shards_by_iteration(replay_dir: str) -> dict[int, list[str]]:
    """Replay shard paths grouped by the iteration in their file name."""
    grouped: dict[int, list[str]] = {}
    fs = _fs_for(replay_dir)
    for path in list_replay_shards(fs, replay_dir):
        match = re.search(r"shard-iter(\d+)-", os.path.basename(path))
        if match:
            # Listings drop the protocol, and without it a bucket path reads as a local one.
            grouped.setdefault(int(match.group(1)), []).append(fs.unstrip_protocol(path))
    return grouped


def load_samples(paths: list[str], game: Gomoku, cache_dir: str | None) -> list[GameSample]:
    """Read shards into training samples, through a local cache when one is given."""
    local = fsspec.filesystem("file")
    samples: list[GameSample] = []
    for path in paths:
        fs, read_path = _fs_for(path), path
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cached = os.path.join(cache_dir, os.path.basename(path))
            if not os.path.exists(cached):
                with fs.open(path, "rb") as src, open(cached + ".part", "wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
                os.replace(cached + ".part", cached)
            fs, read_path = local, cached
        for row in read_parquet_shard(fs, read_path):
            if "last_move_idx" not in row and "last_move" in row:
                row = {**row, "last_move_idx": row["last_move"]}
            samples.append(
                GameSample(
                    reconstruct_state(row, game),
                    decode_policy(row["policy_probs"]),
                    float(row.get("value", 0.0)),
                    1.0,
                )
            )
    return samples


def holdout_losses(
    model: torch.nn.Module,
    samples: list[GameSample],
    game: Gomoku,
    *,
    channels_last: bool = False,
    num_workers: int = 0,
    seed: int = 0,
) -> tuple[float, float]:
    """Mean policy cross entropy and value squared error on ``samples``, without training."""
    # The dataset draws a random board symmetry per sample: fix it so models share positions.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    dataset = ReplayDataset(samples=samples, game=game, include_priority=False)
    loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=num_workers)
    was_training = model.training
    model.eval()
    policy_total = value_total = 0.0
    with torch.no_grad():
        for states, policies, values in loader:
            states = states.to(device)
            if channels_last:
                states = states.contiguous(memory_format=torch.channels_last)
            logits, predicted = model(states)
            log_probs = F.log_softmax(logits.float(), dim=1)
            policy_total += float(-(policies.to(device) * log_probs).sum())
            value_total += float(
                (predicted.float().reshape(-1) - values.to(device).reshape(-1)).pow(2).sum()
            )
    model.train(was_training)
    return policy_total / len(samples), value_total / len(samples)


def _load_baseline(spec: str, device: torch.device) -> tuple[torch.nn.Module, Gomoku]:
    checkpoint, _, config_path = spec.partition("@")
    if not config_path:
        raise SystemExit("--baseline must be checkpoint@config")
    cfg = load_and_parse_config(config_path)
    game = Gomoku(cfg.board, use_native=False)
    model = PolicyValueNet(game, cfg.model, device=str(device))
    state = load_state_dict_from_fs(_fs_for(checkpoint), checkpoint, device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    align_state_dict_to_model(state, model.state_dict())
    model.load_state_dict(state, strict=False)
    return model, game


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, help="YAML of the network to train")
    parser.add_argument("--replay", required=True, help="directory of the earlier run's shards")
    parser.add_argument("--first", type=int, required=True, help="first iteration to train on")
    parser.add_argument("--last", type=int, required=True, help="last iteration to train on")
    parser.add_argument("--holdout", type=int, required=True, help="iteration kept for testing")
    parser.add_argument("--lr", required=True, help="learning rate of each pass, comma separated")
    parser.add_argument("--baseline", default=None, help="checkpoint@config to compare with")
    parser.add_argument("--init", default=None, help="weights of this network to start from")
    parser.add_argument("--workers", type=int, default=None, help="data loader workers")
    parser.add_argument("--cache", default=None, help="local directory to keep shards in")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not args.first <= args.last or args.first <= args.holdout <= args.last:
        raise SystemExit("--holdout must lie outside --first..--last")
    rates = [float(value) for value in args.lr.split(",")]

    cfg = load_and_parse_config(args.config)
    training = cfg.training.model_copy(update={"num_epochs": 1})
    if args.workers is not None:
        training = training.model_copy(update={"dataloader_num_workers": args.workers})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    game = Gomoku(cfg.board, use_native=False)
    channels_last = bool(getattr(training, "use_channels_last", False))
    model = PolicyValueNet(game, cfg.model, device=str(device))
    if args.init:
        model.load_state_dict(load_state_dict_from_fs(_fs_for(args.init), args.init, device))
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    optimizer = build_optimizer(model, training, rates[0])

    grouped = shards_by_iteration(args.replay)
    train_paths = [p for i in range(args.first, args.last + 1) for p in grouped.get(i, [])]
    held_paths = grouped.get(args.holdout, [])
    if not train_paths or not held_paths:
        raise SystemExit(f"no shards for training or for iteration {args.holdout}")
    start = time.monotonic()
    held = load_samples(held_paths, game, args.cache)
    samples = load_samples(train_paths, game, args.cache)
    print(
        f"[Pretrain] {len(samples)} samples from iterations {args.first}-{args.last}, "
        f"{len(held)} held out from {args.holdout}, loaded in {time.monotonic() - start:.0f}s",
        flush=True,
    )
    workers = int(training.dataloader_num_workers)
    if args.baseline:
        baseline, baseline_game = _load_baseline(args.baseline, device)
        policy, value = holdout_losses(baseline, held, baseline_game, num_workers=workers)
        print(f"[Pretrain] baseline holdout policy={policy:.4f} value={value:.4f}", flush=True)
        del baseline

    trainer = AlphaZeroTrainer(training, model, optimizer, game)
    out_fs = _fs_for(args.out)
    for index, rate in enumerate(rates, start=1):
        for group in optimizer.param_groups:
            group["lr"] = rate
        start = time.monotonic()
        metrics = trainer.train_one_iteration(samples)
        policy, value = holdout_losses(
            model, held, game, channels_last=channels_last, num_workers=workers
        )
        save_state(out_fs, model.state_dict(), args.out)
        print(
            f"[Pretrain] pass {index}/{len(rates)} lr={rate:g} "
            f"train policy={metrics['policy_loss']:.4f} value={metrics['value_loss']:.4f} | "
            f"holdout policy={policy:.4f} value={value:.4f} | "
            f"{time.monotonic() - start:.0f}s, saved {args.out}",
            flush=True,
        )


if __name__ == "__main__":
    main()
