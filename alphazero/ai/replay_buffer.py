# ai/replay_buffer.py
# ------------------------------------------------------------
# 간단한 FIFO Ring-buffer 형태의 Replay Buffer
# (AlphaZero self-play 경험 저장용)
# ------------------------------------------------------------
import random
from collections import deque
from typing import Deque, Tuple

import numpy as np
import torch


class ReplayBuffer:
    """
    (state_planes, policy π, value z) 튜플을 저장/샘플링하는 버퍼
      • state_planes : np.ndarray (C, N, N)  float32
      • π            : np.ndarray (N, N)     float32  (합 = 1)
      • z            : float32  (승=+1, 무=0, 패=-1)
    """

    def __init__(self, max_len: int = 100_000):
        self._buf: Deque[Tuple[np.ndarray, np.ndarray, np.float32]] = deque(
            maxlen=max_len
        )

    # --------------------------------------------------------
    # 버퍼에 경험 추가
    # --------------------------------------------------------
    def add(self, state: np.ndarray, pi: np.ndarray, z: float):
        """
        Parameters
        ----------
        state : (C, N, N)   NumPy float32
        pi    : (N, N)      NumPy float32  (softmax 확률)
        z     : float       -1 / 0 / +1
        """
        self._buf.append(
            (state.astype(np.float32), pi.astype(np.float32), np.float32(z))
        )

    # --------------------------------------------------------
    # 미니배치 샘플링
    # --------------------------------------------------------
    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        states : (B, C, N, N)  torch.float32
        pis    : (B, N, N)     torch.float32
        zs     : (B, 1)        torch.float32
        """
        batch = random.sample(self._buf, batch_size)
        s, p, z = map(np.stack, zip(*batch))
        return (
            torch.from_numpy(s),
            torch.from_numpy(p),
            torch.from_numpy(z).unsqueeze(1),
        )

    # --------------------------------------------------------
    # 버퍼 크기
    # --------------------------------------------------------
    def __len__(self) -> int:
        return len(self._buf)
