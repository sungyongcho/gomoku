"""Architecture checks for PolicyValueNet."""

import torch
from torch import nn

from gomoku.model.policy_value_net import (
    RESBLOCK_KERNEL_SIZE,
    START_BLOCK_KERNEL_SIZE,
    VALUE_HEAD_FC_UNITS,
)
from tests.helpers import make_model

# TODO: need to review


def _expected_param_count(
    num_planes: int,
    num_hidden: int,
    num_resblocks: int,
    policy_channels: int,
    value_channels: int,
    board_hw: int,
    action_size: int,
) -> int:
    """Return manual parameter count for PolicyValueNet given config."""
    params = 0
    # start block
    params += num_hidden * num_planes * (START_BLOCK_KERNEL_SIZE**2)  # conv (no bias)
    params += 2 * num_hidden  # bn
    # resblocks
    conv_block = num_hidden * num_hidden * (RESBLOCK_KERNEL_SIZE**2)
    per_block = conv_block + num_hidden  # conv1 + bias
    per_block += conv_block + num_hidden  # conv2 + bias
    per_block += 2 * num_hidden + 2 * num_hidden  # bn1 + bn2
    params += num_resblocks * per_block
    # policy head
    params += policy_channels * num_hidden  # conv1x1 (no bias)
    params += 2 * policy_channels  # bn
    params += action_size * policy_channels * board_hw + action_size  # linear
    # value head
    params += value_channels * num_hidden  # conv1x1 (no bias)
    params += 2 * value_channels  # bn
    params += VALUE_HEAD_FC_UNITS * value_channels * board_hw + VALUE_HEAD_FC_UNITS
    params += VALUE_HEAD_FC_UNITS + 1  # final linear
    return params


def test_forward_output_shapes():
    """Policy/Value outputs should have expected shapes."""
    model, game, config = make_model()
    x = torch.randn(2, config.num_planes, game.row_count, game.col_count)
    policy, value = model(x)
    print("Policy shape:", tuple(policy.shape))
    print("Value shape:", tuple(value.shape))
    assert policy.shape == (2, game.action_size)
    assert value.shape == (2, 1)


def test_value_head_range():
    """Value head output should be bounded to [-1, 1]."""
    model, game, config = make_model()
    x = torch.randn(4, config.num_planes, game.row_count, game.col_count)
    _, value = model(x)
    print("Value head min/max:", float(value.min()), float(value.max()))
    assert torch.all(value <= 1.0)
    assert torch.all(value >= -1.0)


def test_backward_propagates_gradients():
    """Backward pass should produce gradients for all parameters."""
    model, game, config = make_model()
    x = torch.randn(3, config.num_planes, game.row_count, game.col_count)
    target_policy = torch.zeros(3, dtype=torch.long)
    target_value = torch.ones(3, 1)
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()

    policy_logits, value = model(x)
    loss = criterion_policy(policy_logits, target_policy) + criterion_value(
        value, target_value
    )
    loss.backward()
    print("Loss value:", float(loss))

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient missing for {name}"
        assert torch.all(torch.isfinite(param.grad)), f"Non-finite grad at {name}"
        grad_min = float(param.grad.min())
        grad_max = float(param.grad.max())
        print(f"Grad stats for {name}: min={grad_min}, max={grad_max}")


def test_parameter_count_matches_config():
    """Parameter count should match manual calculation from config."""
    model, game, config = make_model()
    conv_out = game.row_count * game.col_count
    expected = _expected_param_count(
        num_planes=config.num_planes,
        num_hidden=config.num_hidden,
        num_resblocks=config.num_resblocks,
        policy_channels=config.policy_channels,
        value_channels=config.value_channels,
        board_hw=conv_out,
        action_size=game.action_size,
    )
    actual = sum(p.numel() for p in model.parameters())
    print("Expected params:", expected)
    print("Actual params:", actual)
    assert actual == expected
