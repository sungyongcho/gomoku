from dataclasses import dataclass
from pathlib import Path

import torch

# MCTS
WIN = 1
LOSE = -1
DRAW = 0

# Replay Buffer
CAPACITY = 5000


@dataclass
class SelfPlayConfig:
    sims: int = 600
    c_puct: float = 1.4
    temperature_turns: int = 20
    temperature: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    resign_threshold: float = -0.8  # value 예측 Q
    max_turns: int = 400  # 안전 캡
    device: str = "cpu"


class TrainerConfig:
    buffer_capacity: int = CAPACITY
    batch_size: int = 192
    lr: float = 3e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    games_per_cycle: int = 50
    train_steps_per_cycle: int = 120
    fp16: bool = False


# Checkpoint loading path for websocket
CHECKPOINT_PATH = Path("checkpoints/20250606-233201-cycle1040.pkl")  # 원하는 모델 경로
