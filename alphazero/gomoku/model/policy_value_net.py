import math

import torch
import torch.nn as nn
import torch.nn.functional as fn

from gomoku.core.gomoku import Gomoku
from gomoku.utils.config.loader import ModelConfig

# CNN configuration constants to avoid scattering magic numbers through the network.
START_BLOCK_KERNEL_SIZE = 3
START_BLOCK_PADDING = 1
RESBLOCK_KERNEL_SIZE = 3
RESBLOCK_PADDING = 1
VALUE_HEAD_FC_UNITS = 256


def calc_conv2d_output(
    h_w: tuple[int, int],
    kernel_size: int | tuple[int, int] = 1,
    stride: int = 1,
    pad: int = 0,
    dilation: int = 1,
) -> tuple[int, int]:
    """Return 2D convolution output dimensions.

    Note: With kernel_size=3, pad=1, stride=1, dilation=1 the output size equals
    the input size. This helper keeps the computation explicit if the
    architecture changes later.

    Parameters
    ----------
    h_w :
        Tuple of height and width for the input feature map.
    kernel_size :
        Kernel size for the convolution (int or tuple). If int, the same value is
        used for both dimensions.
    stride :
        Stride of the convolution.
    pad :
        Zero padding applied to both sides.
    dilation :
        Spacing between kernel elements.

    Returns
    -------
    tuple[int, int]
        Height and width after the convolution operation.
    """
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
    """Combined policy and value network tailored for Gomoku boards.

    The policy head returns logits (not softmaxed); apply softmax externally
    where needed (e.g., inference) to obtain probabilities.
    """

    def __init__(
        self,
        game: Gomoku,
        config: ModelConfig,
        device: str | torch.device,
    ):
        """Create a policy/value network for Gomoku.

        Parameters
        ----------
        game :
            Gomoku game instance providing board dimensions and action size.
        config :
            Model configuration providing channel sizes and block counts.
        device :
            Torch device string to place the model on. Movement is handled by the
            caller; the device string is stored for reference.
        """
        super().__init__()

        num_planes = config.num_planes
        num_hidden = config.num_hidden
        num_resblocks = config.num_resblocks
        policy_channels = config.policy_channels
        value_channels = config.value_channels
        target_device = torch.device(device)
        self.device = target_device

        # Keep spatial size to preserve board coordinate alignment.
        conv_out_hw = calc_conv2d_output(
            (game.row_count, game.col_count),
            kernel_size=START_BLOCK_KERNEL_SIZE,
            stride=1,
            pad=START_BLOCK_PADDING,
        )
        conv_out = conv_out_hw[0] * conv_out_hw[1]

        self.start_block = nn.Sequential(
            nn.Conv2d(
                num_planes,
                num_hidden,
                kernel_size=START_BLOCK_KERNEL_SIZE,
                padding=START_BLOCK_PADDING,
                bias=False,
            ),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
        )

        self.res_blocks = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resblocks)]
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(num_hidden, policy_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(policy_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(policy_channels * conv_out, game.action_size),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(num_hidden, value_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(value_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(value_channels * conv_out, VALUE_HEAD_FC_UNITS),
            nn.ReLU(),
            nn.Linear(VALUE_HEAD_FC_UNITS, 1),
            nn.Tanh(),
        )
        self.to(target_device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass returning policy logits and scalar value."""
        x = self.start_block(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value


class ResBlock(nn.Module):
    """Residual block with two 3x3 convolutions and skip connection."""

    def __init__(self, num_hidden: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            num_hidden,
            num_hidden,
            kernel_size=RESBLOCK_KERNEL_SIZE,
            padding=RESBLOCK_PADDING,
        )
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(
            num_hidden,
            num_hidden,
            kernel_size=RESBLOCK_KERNEL_SIZE,
            padding=RESBLOCK_PADDING,
        )
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual block transformation."""
        residual = x
        x = fn.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = fn.relu(x)
        return x
