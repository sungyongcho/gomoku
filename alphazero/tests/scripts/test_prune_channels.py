"""Removing unused residual-block channels leaves the network's outputs unchanged."""

from __future__ import annotations

import pytest
import torch

from gomoku.core.gomoku import Gomoku
from gomoku.model.policy_value_net import PolicyValueNet
from gomoku.scripts.prune_channels import largest_difference, prune_state_dict
from gomoku.utils.config.loader import BoardConfig, ModelConfig

BOARD = BoardConfig(
    num_lines=9,
    enable_doublethree=False,
    enable_capture=False,
    capture_goal=5,
    gomoku_goal=5,
    history_length=1,
)


def _trained_like_state(game: Gomoku, cfg: ModelConfig, dead: dict[int, list[int]]) -> dict:
    """Random weights with the given channels zeroed the way weight decay leaves them."""
    torch.manual_seed(0)
    model = PolicyValueNet(game, cfg, "cpu")
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.running_mean.normal_()
            module.running_var.uniform_(0.5, 1.5)
            module.weight.data.uniform_(0.5, 1.5)
            module.bias.data.normal_()
    state = model.state_dict()
    for block, channels in dead.items():
        prefix = f"res_blocks.{block}."
        for channel in channels:
            state[prefix + "conv1.weight"][channel] = 0.0
            state[prefix + "bn1.weight"][channel] = 0.0
            state[prefix + "conv2.weight"][:, channel] = 0.0
    return state


def test_unused_channels_are_removed_and_outputs_stay_the_same() -> None:
    game = Gomoku(BOARD)
    planes = int(game.get_encoded_state(game.get_initial_state()).shape[1])
    cfg = ModelConfig(num_planes=planes, num_hidden=8, num_resblocks=3)
    state = _trained_like_state(game, cfg, {0: [1, 5], 2: [0, 3, 7]})

    pruned, widths = prune_state_dict(state)

    assert widths == [6, 8, 5]
    assert pruned["res_blocks.0.conv1.weight"].shape == (6, 8, 3, 3)
    assert pruned["res_blocks.0.bn1.running_var"].shape == (6,)
    assert pruned["res_blocks.2.conv2.weight"].shape == (8, 5, 3, 3)
    assert pruned["res_blocks.1.conv1.weight"].shape == state["res_blocks.1.conv1.weight"].shape
    policy_diff, value_diff = largest_difference(game, cfg, state, pruned, widths, count=20)
    assert policy_diff < 1e-4 and value_diff < 1e-4


def test_a_channel_the_second_convolution_still_reads_is_kept() -> None:
    game = Gomoku(BOARD)
    planes = int(game.get_encoded_state(game.get_initial_state()).shape[1])
    cfg = ModelConfig(num_planes=planes, num_hidden=8, num_resblocks=1)
    state = _trained_like_state(game, cfg, {})
    state["res_blocks.0.conv1.weight"][2] = 0.0  # a dead filter whose constant output is still used

    _, widths = prune_state_dict(state)

    assert widths == [8]


def test_config_needs_one_width_per_block() -> None:
    assert ModelConfig(num_hidden=8, num_resblocks=2, resblock_mid_channels=[5, 8])
    with pytest.raises(ValueError):
        ModelConfig(num_hidden=8, num_resblocks=2, resblock_mid_channels=[5])
    with pytest.raises(ValueError):
        ModelConfig(num_hidden=8, num_resblocks=2, resblock_mid_channels=[5, 0])
