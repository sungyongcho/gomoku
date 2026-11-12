"""Cut the channels a trained network no longer uses out of its residual blocks.

Training with weight decay added to the gradient can drive whole channels to zero: the
first convolution's filter, its normalization and the second convolution's weights for that
channel all vanish. Such a channel still costs inference time. Removing it changes nothing
the network computes, because the second convolution multiplies it by zero.

    python gomoku/scripts/prune_channels.py --config configs/deploy-v5.yaml \\
        --checkpoint iteration_0144.pt --out iteration_0144_pruned.pt

Prints ``resblock_mid_channels`` for the model section of the config that goes with the
output, and the largest difference between the two networks' outputs on sample positions.
Only channels between the two convolutions of a block are removed: the channels the blocks
pass on to each other are shared by the skip connections.
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from gomoku.core.gomoku import Gomoku
from gomoku.model.policy_value_net import PolicyValueNet
from gomoku.utils.config.loader import ModelConfig, load_and_parse_config


def prune_state_dict(
    state: dict[str, torch.Tensor], threshold: float = 1e-12
) -> tuple[dict[str, torch.Tensor], list[int]]:
    """Drop every block channel whose weights into the second convolution are all zero.

    Returns the narrower state dict and the channels kept in each block.
    """
    pruned = dict(state)
    widths: list[int] = []
    block = 0
    while f"res_blocks.{block}.conv2.weight" in state:
        prefix = f"res_blocks.{block}."
        used = state[prefix + "conv2.weight"].abs().amax(dim=(0, 2, 3)) >= threshold
        if not bool(used.any()):
            used[0] = True  # a block needs at least one channel
        pruned[prefix + "conv1.weight"] = state[prefix + "conv1.weight"][used].clone()
        pruned[prefix + "conv1.bias"] = state[prefix + "conv1.bias"][used].clone()
        for name in ("weight", "bias", "running_mean", "running_var"):
            pruned[prefix + "bn1." + name] = state[prefix + "bn1." + name][used].clone()
        pruned[prefix + "conv2.weight"] = state[prefix + "conv2.weight"][:, used].clone()
        widths.append(int(used.sum()))
        block += 1
    return pruned, widths


def sample_positions(game: Gomoku, count: int, seed: int) -> torch.Tensor:
    """Encoded positions from random games of varied length."""
    rng = random.Random(seed)
    encoded = []
    while len(encoded) < count:
        state = game.get_initial_state()
        for _ in range(rng.randint(0, 80)):
            game.get_legal_moves(state)
            action = int(rng.choice(sorted(int(i) for i in state.legal_indices_cache)))
            xy = (action % game.col_count, action // game.col_count)
            state = game.get_next_state(state, xy, state.next_player)
            if game.get_value_and_terminated(state, xy)[1]:
                break
        else:
            planes = np.asarray(game.get_encoded_state(state), dtype=np.float32)
            encoded.append(planes[0] if planes.ndim == 4 else planes)
    return torch.from_numpy(np.stack(encoded))


def largest_difference(
    game: Gomoku, model_cfg: ModelConfig, state: dict, pruned: dict, widths: list[int], count: int
) -> tuple[float, float]:
    """Largest change in any policy logit and in the value over ``count`` positions."""
    original = PolicyValueNet(game, model_cfg, "cpu").eval()
    original.load_state_dict(state)
    narrow_cfg = model_cfg.model_copy(update={"resblock_mid_channels": widths})
    narrow = PolicyValueNet(game, narrow_cfg, "cpu").eval()
    narrow.load_state_dict(pruned)
    positions = sample_positions(game, count, seed=0)
    with torch.no_grad():
        logits_a, value_a = original(positions)
        logits_b, value_b = narrow(positions)
    return float((logits_a - logits_b).abs().max()), float((value_a - value_b).abs().max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, help="YAML of the network as trained")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--positions", type=int, default=200, help="positions to compare on")
    args = parser.parse_args()

    cfg = load_and_parse_config(args.config)
    raw = torch.load(args.checkpoint, map_location="cpu")
    state = raw["model_state_dict"] if "model_state_dict" in raw else raw
    pruned, widths = prune_state_dict(state)
    game = Gomoku(cfg.board, use_native=False)
    policy_diff, value_diff = largest_difference(
        game, cfg.model, state, pruned, widths, args.positions
    )
    torch.save(pruned, args.out)

    hidden = int(cfg.model.num_hidden)
    kept = sum(widths) / (hidden * len(widths))
    print(f"resblock_mid_channels: {widths}")
    print(f"kept {sum(widths)} of {hidden * len(widths)} block channels ({kept:.1%})")
    print(f"largest output difference over {args.positions} positions: "
          f"policy logit {policy_diff:.2e}, value {value_diff:.2e}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
