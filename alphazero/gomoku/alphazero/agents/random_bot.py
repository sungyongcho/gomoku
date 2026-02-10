"""Uniform random opponent for self-play diversification."""

from collections.abc import Sequence

import numpy as np

from gomoku.core.gomoku import GameState, Gomoku


class RandomBot:
    """Bot that samples uniformly from legal moves.

    Notes
    -----
    - Provides a minimal ``get_action_probs`` API compatible with self-play runners.
    - Roots are stateless; ``reset``/``update_root`` are no-ops.
    """

    def __init__(self, game: Gomoku):
        self.game = game
        self.is_uniform_random = True

    def reset(self) -> None:
        """Reset internal state (no-op)."""
        return None

    def reset_game(self, slot_idx: int) -> None:
        """Reset a specific slot (no-op to match agent interface)."""
        return None

    def update_root(self, action: int) -> None:
        """Advance internal root (no-op)."""
        return None

    def update_root_batch(self, actions) -> None:  # noqa: ANN001
        """Advance multiple roots (no-op)."""
        return None

    def _uniform_policy(self, state_raw: GameState) -> np.ndarray:
        """Return uniform policy over legal moves."""
        cache = getattr(state_raw, "legal_indices_cache", None)
        if cache is None or cache.size == 0:
            self.game.get_legal_moves(state_raw)
            cache = getattr(state_raw, "legal_indices_cache", None)

        policy = np.zeros(self.game.action_size, dtype=np.float32)
        if cache is None or cache.size == 0:
            return policy

        legal = cache.astype(int)
        prob = 1.0 / len(legal)
        policy[legal] = prob
        return policy

    def get_action_probs(
        self, state_raw: GameState, temperature: float = 1.0, add_noise: bool = False
    ) -> np.ndarray:
        """Return uniform distribution over legal moves."""
        return self._uniform_policy(state_raw)

    def get_action_probs_batch(
        self,
        states_raw: Sequence[GameState],
        temperature: float | Sequence[float] = 1.0,
        add_noise_flags: bool | Sequence[bool] = False,
        **kwargs,  # noqa: ARG002
    ) -> list[np.ndarray]:
        """Return uniform policies for a batch of states."""
        return [self._uniform_policy(state) for state in states_raw]
