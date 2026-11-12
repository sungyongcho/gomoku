"""Optimizer choice: how weight decay reaches a channel the loss does not use."""

from __future__ import annotations

import torch

from gomoku.alphazero.learning.optimizer import build_optimizer
from gomoku.utils.config.loader import TrainingConfig


def _training(optimizer: str, weight_decay: float) -> TrainingConfig:
    return TrainingConfig(
        num_iterations=1,
        num_selfplay_iterations=1,
        num_epochs=1,
        batch_size=1,
        learning_rate=0.01,
        weight_decay=weight_decay,
        optimizer=optimizer,
        temperature=1.0,
        replay_buffer_size=1,
        min_samples_to_train=1,
    )


def _net() -> torch.nn.Sequential:
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Conv2d(1, 2, 3), torch.nn.BatchNorm2d(2))


def test_adam_is_the_default_and_decays_every_parameter() -> None:
    training = _training("adam", 0.1)
    assert TrainingConfig.model_fields["optimizer"].default == "adam"

    optimizer = build_optimizer(_net(), training, 0.01)

    assert isinstance(optimizer, torch.optim.Adam) and not isinstance(optimizer, torch.optim.AdamW)
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.1]


def test_adamw_decays_weights_but_not_normalization_or_biases() -> None:
    net = _net()

    optimizer = build_optimizer(net, _training("adamw", 0.1), 0.01)

    assert isinstance(optimizer, torch.optim.AdamW)
    decayed, plain = optimizer.param_groups
    assert decayed["weight_decay"] == 0.1 and plain["weight_decay"] == 0.0
    assert [tuple(p.shape) for p in decayed["params"]] == [(2, 1, 3, 3)]
    assert len(plain["params"]) == 3  # conv bias, norm scale, norm bias


def test_a_channel_without_gradient_survives_adamw_and_vanishes_under_adam() -> None:
    """With no signal from the loss, only the decay moves the parameters."""
    left = {}
    for name in ("adam", "adamw"):
        net = _net()
        optimizer = build_optimizer(net, _training(name, 0.1), 0.01)
        for _ in range(300):
            optimizer.zero_grad()
            (net[0].weight.sum() * 0.0 + net[1].weight.sum() * 0.0).backward()
            optimizer.step()
        left[name] = (
            float(net[0].weight.detach().abs().max()),
            float(net[1].weight.detach().abs().min()),
        )

    assert left["adam"][0] < 1e-3 and left["adam"][1] < 1e-3
    assert left["adamw"][0] > 0.1 and left["adamw"][1] == 1.0
