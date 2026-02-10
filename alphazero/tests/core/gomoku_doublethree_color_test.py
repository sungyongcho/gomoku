import numpy as np

from gomoku.core.game_config import PLAYER_1, PLAYER_2, set_pos
from gomoku.core.gomoku import Gomoku
from gomoku.core.rules.doublethree import detect_doublethree
from gomoku.utils.config.loader import BoardConfig


def _make_game() -> Gomoku:
    cfg = BoardConfig(
        num_lines=9,
        enable_doublethree=True,
        enable_capture=False,
        capture_goal=0,
        gomoku_goal=5,
        history_length=5,
    )
    return Gomoku(cfg)


def test_detect_doublethree_white_pattern() -> None:
    game = _make_game()
    state = game.get_initial_state()
    board = state.board

    for x, y in [(4, 3), (5, 3), (3, 4), (3, 5)]:
        set_pos(board, x, y, PLAYER_2)
    set_pos(board, 0, 0, PLAYER_1)

    assert detect_doublethree(board, 3, 3, PLAYER_2, game.row_count)


def test_white_turn_forbidden_move_removed_from_legals() -> None:
    game = _make_game()
    state = game.get_initial_state()
    board = state.board

    for x, y in [(4, 3), (5, 3), (3, 4), (3, 5)]:
        set_pos(board, x, y, PLAYER_2)
    set_pos(board, 8, 8, PLAYER_1)
    state.next_player = np.int8(PLAYER_2)

    legal_moves = game.get_legal_moves(state)
    assert "D4" not in legal_moves
