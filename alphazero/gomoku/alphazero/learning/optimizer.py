"""Optimizer construction shared by training and pretraining."""

from __future__ import annotations

import torch

from gomoku.utils.config.loader import TrainingConfig


def build_optimizer(
    model: torch.nn.Module, training: TrainingConfig, learning_rate: float
) -> torch.optim.Optimizer:
    """Build the optimizer ``training.optimizer`` names.

    ``adam`` adds the weight decay to the gradient, as every run before v6 did. Adam then
    divides it by the gradient's own scale, so a channel the loss barely depends on shrinks
    at the full learning rate until its filter, its normalization scale and the next layer's
    weights for it are all zero, and it never comes back: v5 lost 14% of its filters this way
    and a 10x128 network pretrained with it lost half of each residual block's first layer.

    ``adamw`` shrinks weights in proportion to their size instead, and only convolution and
    linear weights: normalization parameters and biases are left alone.
    """
    if training.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=training.weight_decay
        )
    decayed = [p for p in model.parameters() if p.requires_grad and p.ndim > 1]
    plain = [p for p in model.parameters() if p.requires_grad and p.ndim <= 1]
    return torch.optim.AdamW(
        [
            {"params": decayed, "weight_decay": training.weight_decay},
            {"params": plain, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )
