from __future__ import annotations

from collections import deque
from typing import Deque, List

import numpy as np
import torch

# Sample 타입 alias 가져오기
from ai.self_play import Sample  # <-- 추가 (타입 체크용)
from core.game_config import CAPACITY
from torch.utils.data import Dataset


class ReplayBuffer:
    """간단한 ring buffer (thread-safe 필요 X)."""

    def __init__(self, capacity: int = CAPACITY):
        self.capacity = capacity
        self.data: Deque[Sample] = deque(maxlen=self.capacity)

    def add(self, sample: Sample):
        self.data.append(sample)

    def extend(self, samples: List[Sample]):
        self.data.extend(samples)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, len(self.data), size=batch_size)
        states, pis, zs = zip(*(self.data[i] for i in idxs))
        states = torch.from_numpy(np.stack(states)).float()
        pis = torch.from_numpy(np.stack(pis)).float()
        zs = torch.tensor(zs).float().unsqueeze(1)
        return states, pis, zs

    def __len__(self):
        return len(self.data)


class GomokuDataset(Dataset):
    """Dataset wrapper (주로 DataLoader shuffle/num_workers 용)"""

    def __init__(self, buffer: ReplayBuffer):
        self.buffer = buffer

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        state, pi, z = self.buffer.data[idx]
        return (
            torch.from_numpy(state).float(),
            torch.from_numpy(pi).float(),
            torch.tensor([z], dtype=torch.float32),
        )
