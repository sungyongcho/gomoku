from __future__ import annotations

import argparse
import pickle
from datetime import datetime
from pathlib import Path
from time import sleep

import torch
import torch.nn.functional as F
from ai.dataset import ReplayBuffer
from ai.policy_value_net import PolicyValueNet
from ai.self_play import SelfPlayConfig, play_one_game


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
    buffer_capacity: int = 1_000_000
    batch_size: int = 256
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    games_per_cycle: int = 10
    train_steps_per_cycle: int = 5


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
    return parser.parse_args()


def load_checkpoint(path: Path, model: torch.nn.Module, device: str):
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    model.load_state_dict(ckpt["model_state_dict"])
    step = ckpt.get("step", 0)
    print(f"Loaded checkpoint {path} (step={step})")
    return step


def save_checkpoint(path: Path, model: torch.nn.Module, step: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"model_state_dict": model.state_dict(), "step": step}, f)
    print(f"Checkpoint saved → {path}")


# ────────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    cfg = TrainerConfig()
    if args.train_steps is not None:
        cfg.train_steps_per_cycle = args.train_steps

    # 0) 경로 설정
    save_path = (
        Path(args.save)
        if args.save
        else Path(
            f"checkpoints/{datetime.now().strftime('%Y%m%d-%H%M%S')}-{cfg.train_steps_per_cycle}steps.pkl"
        )
    )

    # 1) 모델 초기화/로드
    model = PolicyValueNet().to(cfg.device)
    step = 0
    if args.load:
        step = load_checkpoint(Path(args.load), model, cfg.device)
    model.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    buffer = ReplayBuffer(cfg.buffer_capacity)
    sp_cfg = SelfPlayConfig(device=cfg.device)

    cycle = 0
    try:
        while True:
            cycle += 1
            print(f"=== Cycle {cycle} ===")

            # 2) Self-play
            for _ in range(cfg.games_per_cycle):
                buffer.extend(play_one_game(model, sp_cfg))
            print(f"Buffer size → {len(buffer)}")
            if len(buffer) < cfg.batch_size:
                print("Buffer too small, continue collecting…")
                continue

            # 3) Training
            model.train()
            for _ in range(cfg.train_steps_per_cycle):
                states, pis, zs = buffer.sample(cfg.batch_size)
                states, pis, zs = (
                    states.to(cfg.device),
                    pis.to(cfg.device),
                    zs.to(cfg.device),
                )
                p_pred, v_pred = model(states)
                loss = alpha_zero_loss(
                    p_pred, pis, v_pred, zs, l2_coef=1e-4, model=model
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1
                if step % 50 == 0:
                    print(f"Step {step} | Loss {loss.item():.4f}")

            model.eval()
            save_checkpoint(save_path, model, step)
            sleep(0.1)
    except KeyboardInterrupt:
        print("Interrupted – final save…")
        save_checkpoint(save_path, model, step)


if __name__ == "__main__":
    main()
