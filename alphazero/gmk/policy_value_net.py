import math

import torch.nn as nn
import torch.nn.functional as F

from game_config import N_PLANES
from gomoku import Gomoku


def calc_conv2d_output(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """takes a tuple of (h,w) and returns a tuple of (h,w)"""

    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, kernel_size)
    h = math.floor(
        ((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1
    )
    w = math.floor(
        ((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1
    )
    return h, w


class PolicyValueNet(nn.Module):
    def __init__(self, game: Gomoku, num_ResBlocks, num_hidden, device):
        super().__init__()
        self.device = device

        conv_out_hw = calc_conv2d_output(
            (game.row_count, game.col_count), kernel_size=3, stride=1, pad=3
        )
        conv_out = conv_out_hw[0] * conv_out_hw[1]

        self.startBlock = nn.Sequential(
            nn.Conv2d(
                N_PLANES, num_hidden, kernel_size=3, padding=3, bias=False
            ),  # selected kernel size of 3 for gomoku only edge problem
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
        )

        self.res_blocks = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_ResBlocks)]
        )

        POLICY_OUT_CHANNELS = 2
        VALUE_OUT_CHANNELS = 1

        self.policy_head = nn.Sequential(
            nn.Conv2d(num_hidden, POLICY_OUT_CHANNELS, kernel_size=1, bias=False),
            nn.BatchNorm2d(POLICY_OUT_CHANNELS),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(POLICY_OUT_CHANNELS * conv_out, game.action_size),
        )

        num_fc_units = 256

        self.value_head = nn.Sequential(
            nn.Conv2d(num_hidden, VALUE_OUT_CHANNELS, kernel_size=1, bias=False),
            nn.BatchNorm2d(VALUE_OUT_CHANNELS),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(VALUE_OUT_CHANNELS * conv_out, num_fc_units),
            nn.ReLU(),
            nn.Linear(num_fc_units, 1),
            nn.Tanh(),
        )

        self.to(device)

    def forward(self, x):
        x = self.startBlock(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value


class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x
