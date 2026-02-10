import numpy as np

from gomoku.core.game_config import (
    PLAYER_1,
    PLAYER_2,
    convert_coordinates_to_xy,
    xy_to_index,
)
from gomoku.core.gomoku import Gomoku


def _cpp_board_to_np(cpp_state, size: int) -> np.ndarray:
    return np.array(cpp_state.board, dtype=np.int8).reshape(size, size)


def _py_legal_indices(py_game: Gomoku, state) -> set[int]:
    coords = py_game.get_legal_moves(state)
    indices: set[int] = set()
    for coord in coords:
        xy = convert_coordinates_to_xy(coord, py_game.col_count)
        if xy is None:
            continue
        x, y = xy
        indices.add(xy_to_index(x, y, py_game.col_count))
    return indices


def _build_white_double_three_state(py_game: Gomoku, cpp_core):
    py_state = py_game.get_initial_state()
    cpp_state = cpp_core.initial_state()
    setup_moves = [
        (4, 3, PLAYER_2),
        (0, 0, PLAYER_1),
        (5, 3, PLAYER_2),
        (0, 1, PLAYER_1),
        (3, 4, PLAYER_2),
        (0, 2, PLAYER_1),
        (3, 5, PLAYER_2),
        (8, 8, PLAYER_1),
    ]
    for x, y, player in setup_moves:
        py_state = py_game.get_next_state(py_state, (x, y), player)
        cpp_state = cpp_core.apply_move(cpp_state, x, y, player)
    assert int(py_state.next_player) == PLAYER_2
    assert int(cpp_state.next_player) == PLAYER_2
    return py_state, cpp_state


def test_apply_move_parity(py_game: Gomoku, cpp_core) -> None:
    py_state = py_game.get_initial_state()
    cpp_state = cpp_core.initial_state()
    moves = [(4, 4), (4, 5), (5, 5)]
    player = int(py_state.next_player)

    for x, y in moves:
        py_state = py_game.get_next_state(py_state, (x, y), player)
        cpp_state = cpp_core.apply_move(cpp_state, x, y, player)
        player = 2 if player == 1 else 1

    cpp_board = _cpp_board_to_np(cpp_state, py_game.col_count)
    print("py board:\n", py_state.board)
    print("cpp board:\n", cpp_board)
    assert np.array_equal(cpp_board, py_state.board)
    assert cpp_state.next_player == py_state.next_player
    assert cpp_state.last_move_idx == py_state.last_move_idx
    assert cpp_state.empty_count == py_state.empty_count
    assert list(cpp_state.history) == list(py_state.history)


def test_check_win_parity(py_game: Gomoku, cpp_core) -> None:
    py_state = py_game.get_initial_state()
    cpp_state = cpp_core.initial_state()
    player = 1
    for x in range(5):
        py_state = py_game.get_next_state(py_state, (x, 0), player)
        cpp_state = cpp_core.apply_move(cpp_state, x, 0, player)
        player = 2 if player == 1 else 1

    py_win = py_game.check_win(py_state, (4, 0))
    cpp_win = cpp_core.check_win(cpp_state, 4, 0)
    print("py win:", py_win, "cpp win:", cpp_win)
    assert py_win == cpp_win


def test_get_legal_moves_parity(py_game: Gomoku, cpp_core) -> None:
    py_state = py_game.get_initial_state()
    cpp_state = cpp_core.initial_state()
    setup_moves = [(4, 4), (5, 4), (3, 5), (5, 5)]
    player = 1
    for x, y in setup_moves:
        py_state = py_game.get_next_state(py_state, (x, y), player)
        cpp_state = cpp_core.apply_move(cpp_state, x, y, player)
        player = 2 if player == 1 else 1

    py_legals = _py_legal_indices(py_game, py_state)
    cpp_legals = set(int(idx) for idx in cpp_core.get_legal_moves(cpp_state))
    print("legal sizes -> py:", len(py_legals), "cpp:", len(cpp_legals))
    assert py_legals == cpp_legals


def test_wrapper_use_native_parity(py_game: Gomoku, native_game: Gomoku) -> None:
    py_state = py_game.get_initial_state()
    native_state = native_game.get_initial_state()

    moves = [(4, 4), (4, 5), (5, 5), (6, 5)]
    player = 1
    for x, y in moves:
        py_state = py_game.get_next_state(py_state, (x, y), player)
        native_state = native_game.get_next_state(native_state, (x, y), player)
        player = 2 if player == 1 else 1

    native_board = native_state.board
    print("py board:\n", py_state.board)
    print("native board:\n", native_board)
    assert np.array_equal(native_board, py_state.board)
    assert native_state.next_player == py_state.next_player
    assert native_state.last_move_idx == py_state.last_move_idx

    py_legals = _py_legal_indices(py_game, py_state)
    native_legals = _py_legal_indices(native_game, native_state)
    print(
        "native wrapper legal sizes -> py:",
        len(py_legals),
        "native:",
        len(native_legals),
    )
    assert py_legals == native_legals

    py_feat = py_game.get_encoded_state(py_state)[0]
    native_feat = native_game.get_encoded_state(native_state)[0]
    print("wrapper feature diff max:", float(np.max(np.abs(py_feat - native_feat))))
    np.testing.assert_allclose(py_feat, native_feat, atol=1e-6, rtol=0.0)


def test_write_state_features_parity(py_game: Gomoku, cpp_core) -> None:
    py_state = py_game.get_initial_state()
    py_state = py_game.get_next_state(py_state, (1, 1), 1)
    py_state = py_game.get_next_state(py_state, (2, 2), 2)

    cpp_state = cpp_core.apply_move(cpp_core.initial_state(), 1, 1, 1)
    cpp_state = cpp_core.apply_move(cpp_state, 2, 2, 2)

    py_feat = py_game.get_encoded_state(py_state)[0]
    cpp_feat = np.array(cpp_core.write_state_features(cpp_state), dtype=np.float32)
    cpp_feat = cpp_feat.reshape(py_feat.shape)

    print("write_state_features diff max:", float(np.max(np.abs(py_feat - cpp_feat))))
    np.testing.assert_allclose(py_feat, cpp_feat, atol=1e-6, rtol=0.0)


def test_encode_state_parity(py_game: Gomoku, cpp_core) -> None:
    py_state = py_game.get_initial_state()
    py_state = py_game.get_next_state(py_state, (0, 0), 1)
    cpp_state = cpp_core.apply_move(cpp_core.initial_state(), 0, 0, 1)

    py_feat = py_game.get_encoded_state(py_state)[0]
    cpp_feat = np.array(cpp_core.encode_state(cpp_state), dtype=np.float32)
    cpp_feat = cpp_feat.reshape(py_feat.shape)

    print("feature diff max:", float(np.max(np.abs(py_feat - cpp_feat))))
    np.testing.assert_allclose(py_feat, cpp_feat, atol=1e-6, rtol=0.0)


def test_white_double_three_legal_moves_parity(py_game: Gomoku, cpp_core) -> None:
    py_state, cpp_state = _build_white_double_three_state(py_game, cpp_core)
    forbidden_idx = xy_to_index(3, 3, py_game.col_count)

    py_legals = _py_legal_indices(py_game, py_state)
    cpp_legals = set(int(idx) for idx in cpp_core.get_legal_moves(cpp_state))

    assert py_legals == cpp_legals
    assert forbidden_idx not in py_legals
    assert forbidden_idx not in cpp_legals


def test_white_double_three_forbidden_plane_parity(py_game: Gomoku, cpp_core) -> None:
    py_state, cpp_state = _build_white_double_three_state(py_game, cpp_core)
    py_feat = py_game.get_encoded_state(py_state)[0]
    cpp_feat = np.array(cpp_core.write_state_features(cpp_state), dtype=np.float32)
    cpp_feat = cpp_feat.reshape(py_feat.shape)

    assert py_feat[7, 3, 3] == 1.0
    assert cpp_feat[7, 3, 3] == 1.0
    np.testing.assert_allclose(py_feat, cpp_feat, atol=1e-6, rtol=0.0)
