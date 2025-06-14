from __future__ import annotations

import argparse
import multiprocessing as mp
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import List, Tuple

import torch
from ai.ai_config import CAPACITY, TrainerConfig
from ai.dataset import ReplayBuffer
from ai.policy_value_net import PolicyValueNet
from ai.pv_mcts import PVMCTS
from ai.self_play import SelfPlayConfig, SelfPlayWorker, play_one_game
from core.game_config import PLAYER_1, PLAYER_2
from core.gomoku import Gomoku

# ─────────────────────────────── configs ────────────────────────────────


@dataclass
class EvalConfig:
    games: int = 20  # matches per eval
    sims: int = 200  # MCTS simulations per move during eval
    interval: int = 1_000  # learner steps between evaluations
    gating_threshold: float = 0.55  # promote if win‑rate ≥ 55 %
    K: int = 32  # Elo K‑factor


# ─────────────────────────── helper utils ───────────────────────────────

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
BEST_PATH = CHECKPOINT_DIR / "best.pkl"


def save_checkpoint(
    model: torch.nn.Module, step: int, cycle: int, buffer: ReplayBuffer
) -> None:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = CHECKPOINT_DIR / f"{ts}-cycle{cycle}.pkl"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "step": step,
            "cycle": cycle,
            "buffer": buffer,
        },
        fname,
    )
    print(f"[ckpt] saved → {fname} (buffer {len(buffer)})")


def load_checkpoint(
    path: Path, model: torch.nn.Module, device: str
) -> Tuple[int, int, ReplayBuffer]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[ckpt] loaded {path.name}")
    return (
        ckpt.get("step", 0),
        ckpt.get("cycle", 0),
        ckpt.get("buffer", ReplayBuffer(CAPACITY)),
    )


# ───────────────────────── loss function ────────────────────────────────


def alpha_zero_loss(
    p_pred: torch.Tensor, pi: torch.Tensor, v_pred: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    policy_loss = -(pi * torch.log(p_pred + 1e-8)).sum(dim=(1, 2)).mean()
    value_loss = torch.nn.functional.mse_loss(v_pred.squeeze(), z.squeeze())
    return policy_loss + value_loss


# ─────────────────────────── evaluation logic ───────────────────────────


def play_match(
    net_a: torch.nn.Module, net_b: torch.nn.Module, sims: int, device: str
) -> int:
    """Return +1 if A wins, -1 if B wins, 0 for draw. Color alternates."""
    game = Gomoku()
    mcts_black = PVMCTS(net_a, sims=sims, device=device)
    mcts_white = PVMCTS(net_b, sims=sims, device=device)
    while game.winner is None and len(game.history) < 400:
        root = (
            mcts_black.search(game.board)
            if game.current_player == PLAYER_1
            else mcts_white.search(game.board)
        )
        move, _ = (
            mcts_black.get_move_and_pi(root)
            if game.current_player == PLAYER_1
            else mcts_white.get_move_and_pi(root)
        )
        game.play_move(*move)
    if game.winner == PLAYER_1:
        return +1  # black(net_a) win
    if game.winner == PLAYER_2:
        return -1  # white(net_b) win
    return 0  # draw


def evaluate(
    candidate: torch.nn.Module, best: torch.nn.Module, cfg: EvalConfig, device: str
) -> Tuple[float, float]:
    """Play cfg.games matches and return (win_rate, elo_delta)."""
    wins = draws = 0
    # alternate first‑move advantage
    for g in range(cfg.games):
        if g % 2 == 0:
            result = play_match(candidate, best, cfg.sims, device)
        else:
            result = -play_match(best, candidate, cfg.sims, device)  # switch colors
        if result == 1:
            wins += 1
        elif result == 0:
            draws += 1
    losses = cfg.games - wins - draws

    win_rate = wins / max(1, wins + losses)
    # simple Elo delta estimation against equal‑rating opponent
    expected = 0.5
    elo_delta = cfg.K * ((wins + 0.5 * draws) / cfg.games - expected)
    return win_rate, elo_delta


# ───────────────────────── argument parsing ─────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AlphaZero trainer + evaluator")
    p.add_argument(
        "--workers", type=int, default=0, help="self‑play processes (0:inline)"
    )
    p.add_argument("--gpu-workers", type=int, default=0, help="workers allowed on GPU")
    p.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--train-steps", type=int, default=None, help="SGD steps per cycle override"
    )
    p.add_argument("--load", type=str, help="checkpoint to resume")
    p.add_argument("--eval-games", type=int, default=20)
    p.add_argument("--eval-sims", type=int, default=200)
    p.add_argument(
        "--gating", type=float, default=0.55, help="promote if win‑rate ≥ x (0~1)"
    )
    return p.parse_args()


# ─────────────────────────── main control ───────────────────────────────


def main() -> None:
    mp.set_start_method("spawn", force=True)
    args = parse_args()

    # configs
    train_cfg = TrainerConfig()
    if args.train_steps is not None:
        train_cfg.train_steps_per_cycle = args.train_steps
    eval_cfg = EvalConfig(
        games=args.eval_games, sims=args.eval_sims, gating_threshold=args.gating
    )

    device = args.device
    model = PolicyValueNet().to(device)
    model.share_memory()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=1e-4
    )

    step, cycle, buffer = 0, 0, ReplayBuffer(train_cfg.buffer_capacity)
    if args.load:
        step, cycle, buffer = load_checkpoint(Path(args.load), model, device)

    # best model snapshot
    best_model = deepcopy(model).to(device)
    if BEST_PATH.exists():
        best_sd = torch.load(BEST_PATH, map_location=device)
        best_model.load_state_dict(best_sd["model_state_dict"])
        print("[best] loaded existing best.pkl")

    # queues & workers
    data_q: mp.Queue | None = None
    param_q: mp.Queue | None = None
    workers: List[mp.Process] = []
    if args.workers > 0:
        data_q, param_q = mp.Queue(10_000), mp.Queue(64)
        for i in range(args.workers):
            w_dev = (
                "cuda"
                if (i < args.gpu_workers and torch.cuda.is_available())
                else "cpu"
            )
            w = SelfPlayWorker(SelfPlayConfig(device=w_dev), data_q, param_q, model)
            w.start()
            workers.append(w)
        print(f"[spawn] {len(workers)} workers")

    # training loop
    WARMUP = 5_000
    next_eval_at = eval_cfg.interval
    while True:
        cycle += 1
        print(f"\n=== Cycle {cycle} ===")

        # 1. collect games
        collected = 0
        if data_q is None:
            for _ in range(train_cfg.games_per_cycle):
                buffer.extend(play_one_game(model, SelfPlayConfig(device=device)))
                collected += 1
        else:
            while collected < train_cfg.games_per_cycle:
                buffer.extend(data_q.get())
                collected += 1
        print(f"[buffer] {len(buffer)} samples")

        if len(buffer) < WARMUP:
            print(f"< warm-up {WARMUP} – keep collecting…")
            continue
        if len(buffer) < train_cfg.batch_size:
            print("< buffer < batch_size – collecting…")
            continue
        # 2. SGD steps
        model.train()
        for _ in range(train_cfg.train_steps_per_cycle):
            s, pi, z = buffer.sample(train_cfg.batch_size)
            s, pi, z = s.to(device), pi.to(device), z.to(device)
            p_pred, v_pred = model(s)
            loss = alpha_zero_loss(p_pred, pi, v_pred, z)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if step % 100 == 0:
                print(f"step {step:>7} | loss {loss.item():.4f}")

        model.eval()

        # 3. param broadcast
        if param_q is not None:
            while not param_q.empty():
                try:
                    param_q.get_nowait()
                except Exception:
                    break
            param_q.put(model.state_dict())

        # 4. checkpoint
        if step % 5_000 == 0:
            save_checkpoint(model, step, cycle, buffer)

        # 5. evaluation / gating
        if step >= next_eval_at:
            print(f"[eval] start {eval_cfg.games} games…")
            win_rate, elo_delta = evaluate(model, best_model, eval_cfg, device)
            print(f"[eval] win‑rate {win_rate * 100:.1f}%  | ΔElo ≈ {elo_delta:+.1f}")
            if win_rate >= eval_cfg.gating_threshold:
                best_model.load_state_dict(model.state_dict())
                torch.save({"model_state_dict": model.state_dict()}, BEST_PATH)
                print(f"[gating] promoted new best → win‑rate {win_rate * 100:.1f}%")
            next_eval_at += eval_cfg.interval

        sleep(0.2)


if __name__ == "__main__":
    main()
