from typing import Tuple

import torch
import torch.nn as nn
from core.game_config import NUM_LINES


class PolicyValueNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        filters: int = 128,
        blocks: int = 10,
    ):
        super().__init__()
        self.N2 = NUM_LINES * NUM_LINES

        # ─ shared stem ─
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
        )
        self.resblocks = nn.Sequential(
            *[self._residual_block(filters) for _ in range(blocks)]
        )

        # ─ heads (함수형 헬퍼로 생성) ─
        self.policy_head = self._make_policy_head(filters)
        self.value_head = self._make_value_head(filters)

    # ─────────── forward ───────────
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.resblocks(x)

        p = self.policy_head(x)  # (B, N, N) TODO: log-prob
        v = self.value_head(x)  # (B, 1)  TODO:tanh

        return p, v

    # ─────────── helper blocks ───────────
    @staticmethod
    def _residual_block(channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    # def _make_policy_head(self, in_channels: int) -> nn.Sequential:
    #     """1x1 Conv → BN → ReLU → FC → log_softmax"""
    #     return nn.Sequential(
    #         nn.Conv2d(in_channels, 2, kernel_size=1, bias=False),
    #         nn.BatchNorm2d(2),
    #         nn.ReLU(inplace=True),
    #         nn.Flatten(),  # (B, 2*N²)
    #         nn.Linear(2 * self.N2, self.N2),
    #         nn.LogSoftmax(dim=1),  # 학습엔 log-prob 권장
    #     )

    # def _make_value_head(self, in_channels: int) -> nn.Sequential:
    #     """1x1 Conv → BN → ReLU → FC(256) → ReLU → FC(1) → tanh"""
    #     return nn.Sequential(
    #         nn.Conv2d(in_channels, 1, kernel_size=1, bias=False),
    #         nn.BatchNorm2d(1),
    #         nn.ReLU(inplace=True),
    #         nn.Flatten(),  # (B, N²)
    #         nn.Linear(self.N2, 256),
    #         nn.ReLU(inplace=True),
    #         nn.Linear(256, 1),
    #         nn.Tanh(),
    #     )
    def _make_policy_head(self, in_channels: int) -> nn.Sequential:
        """1x1 Conv → BN → ReLU → FC → softmax(평면 19x19)"""
        return nn.Sequential(
            nn.Conv2d(in_channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),  # (B, 2*N²)
            nn.Linear(2 * self.N2, self.N2),  # (B, N²)
            nn.Softmax(dim=1),  # 확률값
            nn.Unflatten(1, (NUM_LINES, NUM_LINES)),  # (B, N, N)
        )

    def _make_value_head(self, in_channels: int) -> nn.Sequential:
        """1x1 Conv → BN → ReLU → FC(256) → ReLU → FC(1) → sigmoid(0~1)"""
        return nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),  # (B, N²)
            nn.Linear(self.N2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),  # 0~1
        )
