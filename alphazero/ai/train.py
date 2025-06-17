from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import List, Tuple

import torch
from ai.ai_config import (
    BEST_PATH,
    CAPACITY,
    CHECKPOINT_DIR,
    WARMUP,
    EvalConfig,
    TrainerConfig,
)
from ai.dataset import ReplayBuffer
from ai.policy_value_net import PolicyValueNet
from ai.pv_mcts import PVMCTS
from ai.self_play import SelfPlayConfig, SelfPlayWorker, play_one_game
from core.game_config import PLAYER_1, PLAYER_2
from core.gomoku import Gomoku

# ───────────────────────── configs ─────────────────────────


# ───────────────────────── utils ───────────────────────────


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


# ───────────────────── loss & eval helpers ─────────────────


def alpha_zero_loss(
    p_pred: torch.Tensor, pi: torch.Tensor, v_pred: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    """
    * p_pred : (B,N,N) **or** (B,N²) – Softmax 확률 **또는** LogSoftmax(log π)
    * pi     : (B,N,N) **or** (B,N²) – 타깃 확률 π*
    → 두 텐서를 (B,N²) 로 맞춘 뒤 정책/가치 손실을 계산한다.
    """
    # --- 모양 맞추기 ---------------------------------------------------------
    if p_pred.dim() == 3:
        p_pred = p_pred.flatten(start_dim=1)  # (B,N²)
    if pi.dim() == 3:
        pi = pi.flatten(start_dim=1)  # (B,N²)

    # --- p_pred 가 확률인지(log 확률인지) 자동 판별 ----------------------------
    if p_pred.min() >= 0.0:  # Softmax 확률일 때
        log_p = torch.log(torch.clamp(p_pred, min=1e-8))
    else:  # 이미 LogSoftmax 일 때
        log_p = p_pred

    policy_loss = -(pi * log_p).sum(dim=1).mean()  # (B,)
    value_loss = torch.nn.functional.mse_loss(v_pred.squeeze(), z.squeeze())
    return policy_loss + value_loss


def play_match(
    net_a: torch.nn.Module, net_b: torch.nn.Module, sims: int, device: str
) -> int:
    """Return +1 if A wins, −1 if B wins, 0 for draw (colors alternate)."""
    game = Gomoku()
    mcts_black = PVMCTS(net_a, sims=sims, device=device)
    mcts_white = PVMCTS(net_b, sims=sims, device=device)
    while game.winner is None and len(game.history) < 400:
        if game.current_player == PLAYER_1:
            root = mcts_black.search(game.board)
            move, _ = mcts_black.get_move_and_pi(root)
        else:
            root = mcts_white.search(game.board)
            move, _ = mcts_white.get_move_and_pi(root)
        game.play_move(*move)
    if game.winner == PLAYER_1:
        return +1
    if game.winner == PLAYER_2:
        return -1
    return 0


def evaluate(
    candidate: torch.nn.Module, best: torch.nn.Module, cfg: EvalConfig, device: str
) -> Tuple[float, float]:
    wins = draws = 0
    for g in range(cfg.games):
        if g % 2 == 0:
            res = play_match(candidate, best, cfg.sims, device)
        else:
            res = -play_match(best, candidate, cfg.sims, device)  # switch colors
        if res == 1:
            wins += 1
        elif res == 0:
            draws += 1
    losses = cfg.games - wins - draws
    win_rate = wins / max(1, wins + losses)
    elo_delta = cfg.K * ((wins + 0.5 * draws) / cfg.games - 0.5)
    return win_rate, elo_delta


# ───────────────────── argument parsing ───────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AlphaZero trainer (full)")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--gpu-workers", type=int, default=0)
    p.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--train-steps", type=int, help="SGD steps per cycle override")
    p.add_argument("--load", type=str)
    p.add_argument("--eval-games", type=int, default=20)
    p.add_argument("--eval-sims", type=int, default=200)
    p.add_argument("--gating", type=float, default=0.55)
    # auto‑stop
    p.add_argument("--max-steps", type=int)
    p.add_argument("--max-cycles", type=int)
    p.add_argument("--max-games", type=int)
    return p.parse_args()


# ─────────────────────── main loop ────────────────────────


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

    # best network
    best_model = deepcopy(model)
    if BEST_PATH.exists():
        best_model.load_state_dict(
            torch.load(BEST_PATH, map_location=device)["model_state_dict"]
        )
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

    # loop state
    next_eval_at = eval_cfg.interval
    total_games = 0

    try:
        while True:
            cycle += 1
            print(f"\n=== Cycle {cycle} ===")

            # 1) self‑play collection
            collected = 0
            if data_q is None:
                for _ in range(train_cfg.games_per_cycle):
                    buffer.extend(play_one_game(model, SelfPlayConfig(device=device)))
                    collected += 1
            else:
                start = time.time()
                while collected < train_cfg.games_per_cycle:
                    buffer.extend(data_q.get())
                    collected += 1
                    if collected % 10 == 0:
                        print(
                            f"[debug] collected {collected} games "
                            f"in {time.time() - start:.1f}s"
                        )
            total_games += collected
            print(f"[buffer] {len(buffer)} samples | games {total_games}")

            if len(buffer) < WARMUP:
                print(f"< warm‑up {WARMUP} – keep collecting…")
                continue
            if len(buffer) < train_cfg.batch_size:
                print("< buffer < batch_size – collecting…")
                continue

            # 2) SGD updates
            model.train()
            for _ in range(train_cfg.train_steps_per_cycle):
                s, pi, z = buffer.sample(train_cfg.batch_size)
                s, pi, z = s.to(device), pi.to(device), z.to(device)
                p_pred, v_pred = model(s)
                loss = alpha_zero_loss(p_pred, pi, v_pred, z)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                step += 1
                if step % 100 == 0:
                    print(f"step {step:>7} | loss {loss.item():.4f}")
                    with torch.no_grad():
                        s_dbg, pi_tgt, z_tgt = buffer.sample(256)
                        p_log, v_pred = model(s_dbg.to(device))
                        corr = torch.corrcoef(torch.stack([
                                    v_pred.cpu().flatten(),
                                    z_tgt.flatten()]))[0, 1]
                        print(f"[debug] value-z corr {corr:.2f} | pi_max {pi_tgt.max():.2f}")

            model.eval()

            # 3) broadcast params
            if param_q is not None and step % 500 == 0:
                while not param_q.empty():
                    try:
                        param_q.get_nowait()
                    except Exception:
                        break
                param_q.put(model.state_dict())

            # 4) periodic checkpoint
            if step % 5_000 == 0:
                save_checkpoint(model, step, cycle, buffer)

            # 5) evaluation & gating
            if step >= next_eval_at:
                print(f"[eval] {eval_cfg.games} games…")
                win_rate, elo_delta = evaluate(model, best_model, eval_cfg, device)
                print(
                    f"[eval] win-rate {win_rate * 100:.1f}% | ΔElo ≈ {elo_delta:+.1f}"
                )
                if win_rate >= eval_cfg.gating_threshold:
                    best_model.load_state_dict(model.state_dict())
                    torch.save({"model_state_dict": model.state_dict()}, BEST_PATH)
                    print("[gating] promoted new best ✔")
                next_eval_at += eval_cfg.interval

            # 6) auto-stop checks
            stop = False
            if args.max_steps is not None and step >= args.max_steps:
                stop = True
            if args.max_cycles is not None and cycle >= args.max_cycles:
                stop = True
            if args.max_games is not None and total_games >= args.max_games:
                stop = True
            if stop:
                print(
                    "[done] reached stop criterion — saving checkpoint & shutting down…"
                )
                save_checkpoint(model, step, cycle, buffer)
                break

            sleep(0.2)
    finally:
        # ─ 1) 브로드캐스트 큐 정리 (GPU·CPU 공통) ─
        if param_q is not None:
            param_q.close()  # 생산자 파이프 닫기
            param_q.join_thread()  # 백그라운드 스레드 종료

        # ─ 2) 워커 프로세스 정상 종료 대기 ─
        for w in workers:
            w.terminate()
            w.join()

        # ─ 3) CUDA IPC 공유 메모리 회수 (CPU 실행 시 no-op) ─
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()


if __name__ == "__main__":
    main()
