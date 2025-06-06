from __future__ import annotations

import argparse
import multiprocessing as mp
import pickle
from datetime import datetime
from pathlib import Path
from time import sleep

import torch
import torch.nn.functional as F
from ai.dataset import ReplayBuffer
from ai.policy_value_net import PolicyValueNet
from ai.self_play import SelfPlayConfig, SelfPlayWorker, play_one_game
from core.game_config import CAPACITY


def alpha_zero_loss(
    p_pred: torch.Tensor,
    pi_target: torch.Tensor,
    v_pred: torch.Tensor,
    z_target: torch.Tensor,
    l2_coef: float = 0.0,
    model: torch.nn.Module | None = None,
) -> torch.Tensor:
    """AlphaZero 총 손실 = policy + value + L2(reg)"""
    # policy loss: −Σ π* log p  (batch 평균)
    policy_loss = -(pi_target * torch.log(p_pred + 1e-8)).sum(dim=(1, 2)).mean()
    # value loss: MSE
    value_loss = F.mse_loss(v_pred.squeeze(), z_target.squeeze())
    l2_loss = (
        l2_coef * sum(p.pow(2).sum() for p in model.parameters())
        if (l2_coef and model)
        else 0.0
    )
    return policy_loss + value_loss + l2_loss


class TrainerConfig:
    buffer_capacity: int = CAPACITY
    batch_size: int = 192
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    games_per_cycle: int = 30
    train_steps_per_cycle: int = 5
    fp16: bool = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load", type=str, default=None, help="checkpoint .pkl to load"
    )
    parser.add_argument(
        "--save", type=str, default=None, help="output checkpoint path (.pkl)"
    )
    parser.add_argument(
        "--train-steps", type=int, default=None, help="train steps per cycle"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="device for learner (main process)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="self-play worker count; 0 = single process",
    )
    parser.add_argument(
        "--worker-device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="device used inside worker processes",
    )
    parser.add_argument(
        "--gpu-workers",
        type=int,
        default=0,
        help="how many workers may use GPU (others forced to CPU)",
    )
    parser.add_argument(
        "--gpu-fraction",
        type=float,
        default=None,
        help="per‑process GPU memory fraction cap (0~1)",
    )
    parser.add_argument(
        "--fp16", action="store_true", help="run model in half precision (FP16)"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="real‑time loss curve (matplotlib)"
    )
    return parser.parse_args()

# ───────── util 함수 ─────────
def load_checkpoint(path: Path, model: torch.nn.Module, device: str
                    ) -> tuple[int, int, ReplayBuffer]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    step   = ckpt.get("step", 0)
    cycle  = ckpt.get("cycle", 0)
    buffer = ckpt.get("buffer", ReplayBuffer(CAPACITY))
    print(f"Loaded checkpoint {path} (step={step}, cycle={cycle}, "
          f"buffer size={len(buffer)})")
    return step, cycle, buffer


def save_checkpoint(model: torch.nn.Module, step: int,
                    buffer: ReplayBuffer, cycle: int):
    fname = f"{datetime.now():%Y%m%d-%H%M%S}-cycle{cycle}.pkl"
    path  = Path("checkpoints")/fname
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "step"  : step,
        "cycle" : cycle,
        "buffer": buffer,
    }, path)
    print(f"Checkpoint saved → {path} (buffer {len(buffer)})")



def main():
    args = parse_args()
    cfg = TrainerConfig()
    if args.train_steps is not None:
        cfg.train_steps_per_cycle = args.train_steps

    if args.fp16:
        cfg.fp16 = True
    # learner device 확정
    learner_device = (
        "cuda"
        if (args.device == "auto" and torch.cuda.is_available())
        else args.device
        if args.device != "auto"
        else "cpu"
    )

    # GPU memory fraction 제한 (PyTorch 2.3+)
    if args.gpu_fraction and learner_device == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction, 0)

    # 시각화 세팅
    if args.visualize:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        losses_plot: list[tuple[int, float]] = []

        def save_plot():
            if not losses_plot:
                return
            steps, ls = zip(*losses_plot)
            plt.figure(figsize=(6, 4))
            plt.plot(steps, ls)
            plt.title("Training Loss")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.tight_layout()
            plt.savefig("loss_curve.png")
            plt.close()
    else:
        losses_plot = None

        def save_plot():
            pass

    save_path = None
    # 모델 준비
    model = PolicyValueNet().to(learner_device)
    if args.fp16:
        model = model.half()
    step = 0
    model.share_memory()
    model.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    step, cycle_loaded, buffer = 0, 0, ReplayBuffer(cfg.buffer_capacity)
    if args.load:
        step, cycle_loaded, buffer = load_checkpoint(Path(args.load), model, learner_device)


    sp_cfg_template = SelfPlayConfig(device=args.worker_device)

    # ────────── Worker spawn ──────────
    workers: list[mp.Process] = []
    queue: mp.Queue | None = None
    if args.workers > 0:
        queue = mp.Queue(maxsize=10000)
        for w_id in range(args.workers):
            # GPU 허용 개수까지만 GPU, 나머지는 CPU
            w_device = (
                "cuda"
                if (w_id < args.gpu_workers and torch.cuda.is_available())
                else args.worker_device
            )
            w_cfg = SelfPlayConfig(
                sims=sp_cfg_template.sims,
                c_puct=sp_cfg_template.c_puct,
                temperature_turns=sp_cfg_template.temperature_turns,
                temperature=sp_cfg_template.temperature,
                dirichlet_alpha=sp_cfg_template.dirichlet_alpha,
                dirichlet_epsilon=sp_cfg_template.dirichlet_epsilon,
                resign_threshold=sp_cfg_template.resign_threshold,
                max_turns=sp_cfg_template.max_turns,
                device=w_device,
            )
            if args.fp16:
                w_cfg.sims = int(w_cfg.sims * 0.75)  # FP16 + sims 조정 예시
            w = SelfPlayWorker(w_cfg, queue, model)
            w.start()
            workers.append(w)
        print(
            f"Spawned {len(workers)} self-play workers (GPU workers: {args.gpu_workers})"
        )

    try:
        run_training_loop(
            cfg,
            model,
            optimizer,
            cycle_loaded,
            buffer,
            sp_cfg_template,
            step,
            save_path,
            learner_device,
            queue,
            losses_plot,
            save_plot,
        )
    finally:
        for w in workers:
            w.terminate()
            w.join()
        save_checkpoint(save_path, model, step, buffer)
        if args.visualize:
            save_plot()


# ────────────────────────────────────────────────────────────────────────


def run_training_loop(
    cfg,
    model,
    optimizer,
    cycle_start:int,
    buffer,
    sp_cfg,
    step,
    save_path,
    device,
    queue,
    losses_plot,
    save_plot,
):
    cycle = cycle_start
    while True:
        cycle += 1
        print(f"=== Cycle {cycle} ===")

        # ─ Self‑play 샘플 수집 ─
        if queue is None:  # 싱글 프로세스
            for _ in range(cfg.games_per_cycle):
                buffer.extend(play_one_game(model, sp_cfg))
        else:  # 멀티 프로세스
            collected = 0
            while collected < cfg.games_per_cycle:
                game_samples = queue.get()  # 한 게임 분량 리스트
                buffer.extend(game_samples)
                collected += 1
        print(f"Buffer size → {len(buffer)}")
        WARMUP = 5000
        if len(buffer) < WARMUP:
            print(f"Buffer < {WARMUP}  – skip training, keep collecting…")
            continue

        if len(buffer) < cfg.batch_size:
            print("Buffer too small, continue collecting…")
            continue

        # ─ 학습 ─
        model.train()
        for _ in range(cfg.train_steps_per_cycle):
            states, pis, zs = buffer.sample(cfg.batch_size)
            states, pis, zs = states.to(device), pis.to(device), zs.to(device)
            if cfg.fp16:
                states, pis = states.half(), pis.half()
            p_pred, v_pred = model(states)
            loss = alpha_zero_loss(p_pred, pis, v_pred, zs, l2_coef=1e-4, model=model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if losses_plot is not None:
                losses_plot.append((step, float(loss)))
            if step % 50 == 0:
                print(f"Step {step} | Loss {loss.item():.4f}")
                if losses_plot is not None:
                    save_plot()

        model.eval()
        if (step % 5000 == 0):
            save_checkpoint(model, step, buffer, cycle)
        sleep(0.2)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
