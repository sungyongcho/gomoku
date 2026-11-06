"""Train a small student network on noise-free teacher search labels.

Replay data is deliberately unused: its policy targets carry the Dirichlet noise and
temperature of the self-play run. Random opening plies only widen position coverage,
they never reach a label. Labels from a flat teacher stay flat, so a student still has
to learn opening shape from its own self-play.

    python -m gomoku.scripts.distill label --teacher models/champion.pt \\
        --positions 2000 --sims 800 --out runs/distill/labels/shard_0.parquet --seed 0
    python -m gomoku.scripts.distill train --config configs/small-6x96.yaml \\
        --labels runs/distill/labels --out runs/distill/student.pt
"""

import argparse
import glob
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

from gomoku.alphazero.learning.dataset import (
    GameSample,
    ShardDataset,
    save_records_to_parquet_shard,
)
from gomoku.alphazero.learning.trainer import AlphaZeroTrainer
from gomoku.alphazero.types import GameRecord
from gomoku.core.gomoku import GameState, Gomoku
from gomoku.model.policy_value_net import PolicyValueNet
from gomoku.utils.config.loader import load_and_parse_config
from server.engine import AlphaZeroEngine


def _outcome(winner: int, player: int) -> float:
    if winner == 0:
        return 0.0
    return 1.0 if winner == player else -1.0


def label_game(
    teacher: AlphaZeroEngine, sims: int, opening_plies: int, rng: random.Random
) -> list[GameSample]:
    """Play one teacher game and label each searched position.

    Policies are noise-free visit shares. Values are the game result because the
    teacher's own value estimates saturate and would copy that bias into the student.
    """
    game = teacher.game
    state = teacher._ensure_native_state(game.get_initial_state())
    labelled: list[tuple[GameState, np.ndarray]] = []
    winner = 0
    for ply in range(game.action_size):
        if ply < opening_plies:
            root, _ = teacher.search(state, 2)
            xy = rng.choice(sorted(root.children))
            action = xy[0] + xy[1] * game.col_count
        else:
            root, policy = teacher.search(state, sims)
            labelled.append((state, np.asarray(policy, dtype=np.float32).reshape(-1)))
            action = teacher._most_visited_action(root, policy)
            xy = (action % game.col_count, action // game.col_count)
        mover = int(state.next_player)
        state, _ = teacher.apply_move(state, action)
        value, terminal = game.get_value_and_terminated(state, xy)
        if terminal:
            winner = 0 if value == 0 else (mover if value > 0 else 3 - mover)
            break
    return [
        GameSample(
            state=position,
            policy_probs=policy,
            value=_outcome(winner, int(position.next_player)),
        )
        for position, policy in labelled
    ]


def collect_labels(
    teacher: AlphaZeroEngine,
    positions: int,
    sims: int,
    random_plies: int,
    rng: random.Random,
) -> list[GameSample]:
    """Label positions from complete teacher games that start after random plies."""
    samples: list[GameSample] = []
    while len(samples) < positions:
        samples.extend(label_game(teacher, sims, rng.randint(0, random_plies), rng))
    return samples[:positions]


def _to_record(samples: list[GameSample]) -> GameRecord:
    return GameRecord(
        states_raw=[sample.state for sample in samples],
        policies=np.stack([sample.policy_probs for sample in samples]),
        moves=np.array(
            [int(np.argmax(sample.policy_probs)) for sample in samples], dtype=np.int32
        ),
        players=np.array([sample.state.next_player for sample in samples], dtype=np.int8),
        outcomes=np.array([sample.value for sample in samples], dtype=np.float32),
        config_snapshot={"source": "distill"},
    )


def run_label(args: argparse.Namespace) -> None:
    with tempfile.TemporaryDirectory() as cache:
        os.environ.setdefault("ALPHAZERO_INFER_BACKEND", "onnx")
        os.environ["ONNX_CACHE_DIR"] = cache
        teacher = AlphaZeroEngine(args.teacher_config, args.teacher, "cpu")
        teacher.mcts_config = teacher.mcts_config.model_copy(update={"time_limit_s": None})
        samples = collect_labels(
            teacher, args.positions, args.sims, args.random_plies, random.Random(args.seed)
        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_records_to_parquet_shard([_to_record(samples)], args.out)
    print(f"labelled {len(samples)} positions at {args.sims} sims: {args.out}")


def holdout_losses(
    model: torch.nn.Module, dataset: ShardDataset, indices: list[int], device: torch.device
) -> tuple[float, float]:
    """Return mean policy cross entropy and value MSE on held-out labels."""
    model.eval()
    policy_loss, value_loss = 0.0, 0.0
    with torch.no_grad():
        for index in indices:
            state, policy, value = dataset[index][:3]
            logits, predicted = model(state.unsqueeze(0).to(device))
            log_probs = F.log_softmax(logits.reshape(1, -1), dim=1)
            policy_loss += float(-(policy.to(device) * log_probs[0]).sum())
            value_loss += float((predicted.reshape(-1) - value.to(device)).pow(2).mean())
    return policy_loss / len(indices), value_loss / len(indices)


def run_train(args: argparse.Namespace) -> None:
    config = load_and_parse_config(args.config)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game = Gomoku(config.board, use_native=False)
    model = PolicyValueNet(game, config.model, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.training.learning_rate),
        weight_decay=config.training.weight_decay,
    )
    shards = sorted(glob.glob(os.path.join(args.labels, "*.parquet")))
    if not shards:
        raise FileNotFoundError(f"no label shards in {args.labels}")
    dataset = ShardDataset(shards, game, include_priority=False)
    order = list(range(len(dataset)))
    random.Random(args.seed).shuffle(order)
    held = order[: max(1, len(order) // 10)]
    train_part = torch.utils.data.Subset(dataset, order[len(held) :])

    trainer = AlphaZeroTrainer(config.training, model, optimizer, game)
    for round_index in range(args.rounds):
        metrics = trainer.train_one_iteration(train_part)
        policy, value = holdout_losses(model, dataset, held, device)
        print(
            f"round {round_index + 1}/{args.rounds} train_policy={metrics['policy_loss']:.3f} "
            f"train_value={metrics['value_loss']:.3f} "
            f"holdout_policy={policy:.3f} holdout_value={value:.3f}",
            flush=True,
        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, args.out)
    print(f"saved {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    label = commands.add_parser("label")
    label.add_argument("--teacher", default="models/champion.pt")
    label.add_argument("--teacher-config", default="configs/deploy.yaml")
    label.add_argument("--positions", type=int, default=2000)
    label.add_argument("--sims", type=int, default=800)
    label.add_argument("--random-plies", type=int, default=6)
    label.add_argument("--out", required=True)
    label.add_argument("--seed", type=int, default=0)
    train = commands.add_parser("train")
    train.add_argument("--config", default="configs/small-6x96.yaml")
    train.add_argument("--labels", required=True)
    train.add_argument("--rounds", type=int, default=5)
    train.add_argument("--out", required=True)
    train.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.command == "label":
        run_label(args)
    else:
        run_train(args)


if __name__ == "__main__":
    main()
