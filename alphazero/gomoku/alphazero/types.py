from dataclasses import dataclass
from typing import Any

import numpy as np

from gomoku.core.game_config import index_to_xy, xy_to_index

type Action = int
type RawState = Any
type PolicyVector = np.ndarray
type PolicyBatch = list[PolicyVector]
type ConfigSnapshot = dict[str, Any] | None

# Action conversion is consistently defined on the flat index space.
action_to_xy = index_to_xy
xy_to_action = xy_to_index


@dataclass(slots=True)
class GameRecord:
    """Container for a single self-play game's training targets.

    Parameters
    ----------
    states_raw :
        Raw game states per turn.
    policies :
        Visit-count-derived policy vectors ``(T, action_size)``.
    moves :
        Flat action indices actually played ``(T,)``.
    players :
        Player identifier per turn ``(T,)`` (e.g., PLAYER_1=1, PLAYER_2=2).
    outcomes :
        Value targets from each turn player's perspective ``(T,)``.
    config_snapshot :
        Optional snapshot of configuration at data creation time.
    """

    states_raw: list[RawState]
    policies: np.ndarray
    moves: np.ndarray
    players: np.ndarray
    outcomes: np.ndarray
    config_snapshot: ConfigSnapshot = None
