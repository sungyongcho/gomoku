from typing import NamedTuple

import numpy as np

from gomoku.alphazero.types import Action, GameRecord
from gomoku.core.gomoku import GameState


class SelfPlaySample(NamedTuple):
    """Temporary container for a single self-play turn."""

    state: GameState
    policy_probs: np.ndarray
    move: Action
    player: int


def sample_action(pi: np.ndarray, turn: int, exploration_turns: int) -> Action:
    """Pick an action by sampling early and using argmax after exploration turns.

    Parameters
    ----------
    pi : np.ndarray
        Policy distribution for the current turn.
    turn : int
        Zero-based turn index.
    exploration_turns : int
        Number of initial turns to sample stochastically.

    Returns
    -------
    Action
        Chosen flat action index.
    """
    if turn >= exploration_turns:
        return int(np.argmax(pi))

    probs = np.nan_to_num(pi, nan=0.0, posinf=0.0, neginf=0.0)
    probs = np.maximum(probs, 0.0)
    total = probs.sum()
    if not np.isfinite(total) or total <= 1e-9:
        return int(np.argmax(pi))

    probs /= total
    return int(np.random.choice(len(probs), p=probs))


def random_legal_action(game, state) -> Action | None:
    """Return a uniformly random legal action index for the given state."""
    cache = getattr(state, "legal_indices_cache", None)
    if cache is None or cache.size == 0:
        game.get_legal_moves(state)
        cache = getattr(state, "legal_indices_cache", None)
    if cache is None or cache.size == 0:
        return None
    return int(np.random.choice(cache.astype(int)))


def build_game_record(
    memory: list[SelfPlaySample], final_value: float, last_player: int
) -> GameRecord:
    """Convert buffered turn data into a ``GameRecord``.

    Parameters
    ----------
    memory : list[SelfPlaySample]
        Buffered per-turn samples collected during a game.
    final_value : float
        Terminal value from the perspective of the player who just moved.
    last_player : int
        Player id of the last mover (aligns outcomes with perspective).

    Returns
    -------
    GameRecord
        Training-ready record with states, policies, moves, players, outcomes.
    """
    states_raw = []
    policies = []
    moves = []
    players = []
    outcomes = []

    for sample in memory:
        # outcome is stored from each turn player's perspective
        outcome = final_value if sample.player == last_player else -final_value
        states_raw.append(sample.state)
        policies.append(sample.policy_probs)
        moves.append(int(sample.move))
        players.append(int(sample.player))
        outcomes.append(int(outcome))

    return GameRecord(
        states_raw=states_raw,
        policies=np.stack(policies),
        moves=np.asarray(moves, dtype=np.int32),
        players=np.asarray(players, dtype=np.int8),
        outcomes=np.asarray(outcomes, dtype=np.int8),
        config_snapshot=None,
    )
