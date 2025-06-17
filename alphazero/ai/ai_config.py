from dataclasses import dataclass
from pathlib import Path

import torch

# MCTS
WIN = 1
LOSE = -1
DRAW = 0

# Replay Buffer
CAPACITY = 10000


WARMUP = 4_000


@dataclass
class SelfPlayConfig:
    sims: int = 300
    c_puct: float = 1.4
    temperature_turns: int = 15
    temperature: float = 1.0
    dirichlet_alpha: float = 0.2
    dirichlet_epsilon: float = 0.40
    resign_threshold: float = -1.1  # value 예측 Q
    max_turns: int = 400  # 안전 캡
    device: str = "cpu"


class TrainerConfig:
    buffer_capacity: int = CAPACITY
    batch_size: int = 128
    lr: float = 3e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    games_per_cycle: int = 40
    train_steps_per_cycle: int = 80
    fp16: bool = False


@dataclass
class EvalConfig:
    games: int = 40  # matches per eval
    sims: int = 300  # MCTS sims per move during eval
    interval: int = 300  # learner steps between evaluations
    gating_threshold: float = 0.55  # promote if win‑rate ≥ 55 %
    K: int = 32  # Elo K‑factor


CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
BEST_PATH = CHECKPOINT_DIR / "best.pkl"


# Checkpoint loading path for websocket
CHECKPOINT_PATH = Path("checkpoints/20250606-233201-cycle1040.pkl")  # 원하는 모델 경로
