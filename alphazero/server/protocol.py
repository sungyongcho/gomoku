from __future__ import annotations

from typing import Any

import numpy as np

from gomoku.core.game_config import (
    EMPTY_SPACE,
    PLAYER_1,
    PLAYER_2,
    index_to_xy,
    xy_to_index,
)
from gomoku.core.gomoku import GameState

_STONE_TO_PLAYER = {
    ".": EMPTY_SPACE,
    "X": PLAYER_1,
    "O": PLAYER_2,
}

_PLAYER_TO_STONE = {
    EMPTY_SPACE: ".",
    PLAYER_1: "X",
    PLAYER_2: "O",
}


def _stone_to_player(stone: str) -> int:
    if stone not in _STONE_TO_PLAYER:
        raise ValueError(f"Invalid stone value: {stone}")
    return _STONE_TO_PLAYER[stone]


def _player_to_stone(player: int) -> str:
    if player not in _PLAYER_TO_STONE:
        raise ValueError(f"Invalid player value: {player}")
    return _PLAYER_TO_STONE[player]


def _board_to_numpy(board: list[list[str]]) -> np.ndarray:
    if not board or not board[0]:
        raise ValueError("Board cannot be empty.")

    size = len(board)
    if any(len(row) != size for row in board):
        raise ValueError("Board must be square.")

    out = np.empty((size, size), dtype=np.int8)
    for y, row in enumerate(board):
        for x, stone in enumerate(row):
            out[y, x] = _stone_to_player(stone)
    return out


def _board_to_frontend(board: np.ndarray) -> list[list[str]]:
    height, width = board.shape
    out = []
    for y in range(height):
        row = []
        for x in range(width):
            row.append(_player_to_stone(int(board[y, x])))
        out.append(row)
    return out


def _extract_scores(scores: list[dict[str, Any]] | None) -> tuple[int, int]:
    p1_pts = 0
    p2_pts = 0
    if not scores:
        return p1_pts, p2_pts

    for item in scores:
        player = item.get("player")
        score = int(item.get("score", 0))
        if player == "X":
            p1_pts = score
        elif player == "O":
            p2_pts = score
    return p1_pts, p2_pts


def _opponent_stone(stone: str) -> str:
    if stone == "X":
        return "O"
    if stone == "O":
        return "X"
    return "."


def build_error_response(message: str) -> dict[str, Any]:
    return {"type": "error", "error": message}


def build_evaluate_response(
    x_eval: float,
    o_eval: float,
    x_percentage: float,
    o_percentage: float,
) -> dict[str, Any]:
    return {
        "type": "evaluate",
        "evalScores": [
            {"player": "O", "evalScores": float(o_eval), "percentage": float(o_percentage)},
            {"player": "X", "evalScores": float(x_eval), "percentage": float(x_percentage)},
        ],
    }


def frontend_to_gamestate(data: dict[str, Any]) -> GameState:
    """
    Frontend JSON -> GameState.

    Mappings:
      board[y][x] "."/"X"/"O"  -> np.int8 0/1/2
      scores[player="X"].score  -> p1_pts (same units, no conversion)
      scores[player="O"].score  -> p2_pts
      nextPlayer "X"/"O"       -> next_player 1/2
      lastPlay.coordinate      -> last_move_idx = x + y * board_size
    """
    board_payload = data.get("board")
    if not isinstance(board_payload, list):
        raise ValueError("Missing or invalid 'board' field.")
    board = _board_to_numpy(board_payload)

    p1_pts, p2_pts = _extract_scores(data.get("scores"))

    next_player_raw = data.get("nextPlayer")
    if next_player_raw not in ("X", "O"):
        raise ValueError("Missing or invalid 'nextPlayer' field.")
    next_player = _stone_to_player(next_player_raw)

    board_size = board.shape[0]
    last_move_idx = -1
    last_play = data.get("lastPlay")
    if isinstance(last_play, dict):
        coord = last_play.get("coordinate")
        if isinstance(coord, dict):
            x = int(coord.get("x", -1))
            y = int(coord.get("y", -1))
            if 0 <= x < board_size and 0 <= y < board_size:
                last_move_idx = xy_to_index(x, y, board_size)

    history: tuple[int, ...] = (int(last_move_idx),) if last_move_idx >= 0 else tuple()
    empty_count = int(np.count_nonzero(board == EMPTY_SPACE))

    return GameState(
        board=board,
        p1_pts=np.int16(p1_pts),
        p2_pts=np.int16(p2_pts),
        next_player=np.int8(next_player),
        last_move_idx=np.int16(last_move_idx),
        empty_count=np.int16(empty_count),
        history=history,
    )


def build_move_response(
    action: int,
    stone: str,
    new_state: GameState,
    captured_indices: list[int],
    execution_time_ns: int,
) -> dict[str, Any]:
    """
    Build SocketMoveResponse compatible with frontend/minimax payload shape.
    """
    board_size = int(new_state.board.shape[0])
    x, y = index_to_xy(action, board_size)
    captured_stone = _opponent_stone(stone)

    captured = []
    for idx in captured_indices:
        cx, cy = index_to_xy(int(idx), board_size)
        captured.append({"x": int(cx), "y": int(cy), "stone": captured_stone})

    elapsed_s = execution_time_ns / 1_000_000_000.0
    elapsed_ms = execution_time_ns / 1_000_000.0

    return {
        "type": "move",
        "status": "success",
        "lastPlay": {"coordinate": {"x": int(x), "y": int(y)}, "stone": stone},
        "board": _board_to_frontend(new_state.board),
        "capturedStones": captured,
        "scores": [
            {"player": "X", "score": int(new_state.p1_pts)},
            {"player": "O", "score": int(new_state.p2_pts)},
        ],
        "evalScores": [
            {"player": "O", "evalScores": 0.0, "percentage": 50.0},
            {"player": "X", "evalScores": 0.0, "percentage": 50.0},
        ],
        "executionTime": {"s": elapsed_s, "ms": elapsed_ms, "ns": int(execution_time_ns)},
    }
