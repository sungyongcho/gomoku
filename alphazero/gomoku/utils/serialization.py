# gmk/utils/serialization.py

import numpy as np

from gomoku.core.gomoku import GameState


def to_savable_sample(
    state: GameState, pi: np.ndarray, z: float | int
) -> dict[str, bytes | float | list[int]]:
    """
    (state, policy, value) 튜플을 저장 가능한 딕셔너리로 변환합니다.
    """
    if not isinstance(state, GameState):
        raise TypeError("`state` must be a GameState object.")

    history_list: list[int] = list(state.history) if state.history else []

    return {
        "board": np.asarray(state.board, dtype=np.int8).tobytes(),
        "p1_pts": int(state.p1_pts),
        "p2_pts": int(state.p2_pts),
        "next_player": int(state.next_player),
        "last_move": int(state.last_move_idx),
        "last_move_idx": int(state.last_move_idx),
        "history": history_list,
        "policy_probs": np.asarray(pi, dtype=np.float16).tobytes(),
        "value": float(z),
        "priority": 1.0,
    }
