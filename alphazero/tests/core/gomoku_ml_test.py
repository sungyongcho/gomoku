import numpy as np

from gomoku.core.game_config import EMPTY_SPACE, PLAYER_1, PLAYER_2, set_pos
from gomoku.core.gomoku import Gomoku
from gomoku.utils.config.loader import BoardConfig
from tests.helpers import log_section, log_state, make_game

pytest_plugins = ("tests.helpers",)

"""
Channel Layout:
---------------
0.  **Me (Current Player)**: Binary plane. 1.0 where the current player has stones.
1.  **Opponent**: Binary plane. 1.0 where the opponent has stones.
2.  **Empty**: Binary plane. 1.0 where the intersection is empty.
3.  **Last Move**: One-hot plane. 1.0 at the coordinate of the last move made.
4.  **My Capture Score**: Constant plane. Current player's capture score normalized
    by `capture_goal` (score / goal).
5.  **Opponent Capture Score**: Constant plane. Opponent's capture score normalized.
6.  **Color Plane**: Constant plane indicating the current turn color.
    Filled with 1.0 for Black (Player 1), -1.0 for White (Player 2).
7.  **Forbidden Points**: Binary plane. 1.0 at coordinates that are forbidden
    by Renju rules (Double-Three Only).
8+. **History**: A sequence of `self.history_length` planes representing past moves.
    - Channel 8: The move made at T-1.
    - Channel 9: The move made at T-2.
    - ... and so on up to `self.history_length`.
"""


def test_me_opp_empty_are_mutually_exclusive(game_env):
    """Me/Opp/Empty planes should sum to exactly 1.0 at every coordinate."""
    log_section("ML Features - Me/Opp/Empty Exclusivity")
    game: Gomoku = game_env

    def assert_exclusive(state):
        encoded = game.get_encoded_state(state)  # (1, C, H, W)
        trio = encoded[:, 0:3]  # Me, Opp, Empty
        summed = np.sum(trio, axis=1)
        ones = np.ones(
            (encoded.shape[0], game.row_count, game.col_count), dtype=np.float32
        )
        print("Summed plane stats:", summed.min(), summed.max())
        print("Non-zero Me coords:", np.argwhere(trio[0, 0] == 1))
        print("Non-zero Opp coords:", np.argwhere(trio[0, 1] == 1))
        print("Non-zero Empty coords:", np.argwhere(trio[0, 2] == 1))
        assert np.allclose(
            summed,
            ones,
        ), "Me + Opp + Empty must be exactly 1 at every coordinate"

    # 1. Initial board
    initial_state = game.get_initial_state()
    assert_exclusive(initial_state)

    # 2. After a few moves
    mid_state = game.get_next_state(
        initial_state, (0, 0), int(initial_state.next_player)
    )
    mid_state = game.get_next_state(mid_state, (1, 1), int(mid_state.next_player))
    assert_exclusive(mid_state)


def test_last_move_plane(game_env: Gomoku):
    """Last-move channel should be one-hot at the most recent move."""
    log_section("ML Features - Last Move Plane")
    game: Gomoku = game_env
    state = game.get_initial_state()

    x, y = 3, 4
    state = game.get_next_state(state, (x, y), int(state.next_player))
    encoded = game.get_encoded_state(state)

    last_plane = encoded[0, 3]
    print("Last move non-zero coords:", np.argwhere(last_plane == 1))
    print("Last move plane sum:", np.sum(last_plane))

    assert last_plane[y, x] == 1.0
    assert np.sum(last_plane) == 1.0


def test_capture_score_plane_basic(game_env: Gomoku):
    """Capture score planes should reflect normalized points for each side."""
    log_section("ML Features - Capture Planes")
    game: Gomoku = game_env
    state = game.get_initial_state()

    state.p1_pts = 2
    state.p2_pts = 0
    state.next_player = np.int8(PLAYER_1)

    encoded = game.get_encoded_state(state)
    my_plane = encoded[0, 4]
    opp_plane = encoded[0, 5]

    print("My capture score plane unique:", np.unique(my_plane))
    print("Opp capture score plane unique:", np.unique(opp_plane))

    expected = state.p1_pts / game.capture_goal
    # TODO: didn't quite have a case where i need allclose, or full_like
    assert np.allclose(my_plane, expected)
    assert np.allclose(opp_plane, 0.0)


def test_color_plane(game_env: Gomoku):
    """Color plane should be 1.0 for P1's turn, -1.0 for P2's turn."""
    log_section("ML Features - Color Plane")
    game: Gomoku = game_env
    state = game.get_initial_state()

    encoded = game.get_encoded_state(state)
    color_plane = encoded[0, 6]
    print("Color plane unique (P1 turn):", np.unique(color_plane))
    assert np.all(color_plane == 1.0)

    state = game.get_next_state(state, (0, 0), int(state.next_player))
    encoded = game.get_encoded_state(state)
    color_plane = encoded[0, 6]
    print("Color plane unique (P2 turn):", np.unique(color_plane))
    assert np.all(color_plane == -1.0)


def test_forbidden_plane_coordinates(game_env: Gomoku):
    """Forbidden channel should mark Renju double-three forbiddens for P1."""
    log_section("ML Features - Forbidden Plane Mapping")
    game: Gomoku = game_env
    state = game.get_initial_state()

    pattern_a = [(3, 1), (3, 2), (2, 3), (4, 3)]
    pattern_b = [(1, 9), (4, 9), (2, 10), (4, 10)]
    pattern_c = [(12, 3), (12, 5), (9, 6), (13, 6)]
    pattern_d = [(10, 9), (12, 11), (13, 11), (10, 12)]
    for x, y in pattern_a + pattern_b + pattern_c + pattern_d:
        set_pos(state.board, x, y, PLAYER_1)
    state.next_player = PLAYER_1

    encoded = game.get_encoded_state(state)
    forbidden = encoded[0, 7]
    coords = np.argwhere(forbidden == 1)
    print("Forbidden coords:", coords)
    expected_coords = [(3, 3), (4, 12), (11, 4), (10, 11)]
    for x, y in expected_coords:
        print(f"Expect forbidden at ({y},{x})")
        assert forbidden[y, x] == 1.0


def test_forbidden_plane_toggle(game_env: Gomoku):
    """Forbidden channel should zero out when double-three rule is disabled."""
    log_section("ML Features - Forbidden Plane Toggle")
    pattern_a = [(3, 1), (3, 2), (2, 3), (4, 3)]
    pattern_b = [(1, 9), (4, 9), (2, 10), (4, 10)]
    pattern_c = [(12, 3), (12, 5), (9, 6), (13, 6)]
    pattern_d = [(10, 9), (12, 11), (13, 11), (10, 12)]
    game = game_env

    def apply_pattern(state):
        for x, y in pattern_a + pattern_b + pattern_c + pattern_d:
            set_pos(state.board, x, y, PLAYER_1)
        state.next_player = PLAYER_1
        return state

    state = apply_pattern(game.get_initial_state())
    log_state(game, state, "Forbidden Enabled State")
    encoded = game.get_encoded_state(state)
    print("Forbidden unique (enabled):", np.unique(encoded[0, 7]))

    game2 = Gomoku(
        BoardConfig(
            num_lines=game.row_count,
            gomoku_goal=game.gomoku_goal,
            capture_goal=game.capture_goal,
            enable_doublethree=False,
            enable_capture=game.enable_capture,
        )
    )
    state2 = apply_pattern(game2.get_initial_state())
    log_state(game2, state2, "Forbidden Disabled State")
    encoded2 = game2.get_encoded_state(state2)
    print("Forbidden unique (disabled):", np.unique(encoded2[0, 7]))
    assert np.all(encoded2[0, 7] == 0.0)


def test_history_planes(game_env: Gomoku):
    """History planes should stack past moves in reverse chronological order."""
    log_section("ML Features - History Planes")
    # history length is 5
    game = game_env
    state = game.get_initial_state()
    moves = [(4, 4), (5, 5), (6, 6), (7, 7)]
    for move in moves:
        state = game.get_next_state(state, move, int(state.next_player))
    encoded = game.get_encoded_state(state)
    base = 8
    for k in range(game.history_length):
        plane = encoded[0, base + k]
        coords = np.argwhere(plane == 1)
        print(f"History plane {k} coords:", coords)
        if k < len(moves):
            y, x = moves[len(moves) - 1 - k][1], moves[len(moves) - 1 - k][0]
            assert plane[y, x] == 1.0
        else:
            assert np.sum(plane) == 0


def test_history_temporal_shift(game_env: Gomoku):
    """History planes should shift forward one slot each turn."""
    log_section("ML Features - History Temporal Shift")
    game: Gomoku = game_env
    state = game.get_initial_state()
    state = game.get_next_state(state, (0, 0), int(state.next_player))
    log_state(game, state, "State T")
    encoded_t = game.get_encoded_state(state)
    state = game.get_next_state(state, (1, 1), int(state.next_player))
    log_state(game, state, "State T+1")
    encoded_t1 = game.get_encoded_state(state)
    base = 8
    hist0_t = encoded_t[0, base]
    hist1_t1 = encoded_t1[0, base + 1]
    print("Temporal shift shapes:", hist0_t.shape, hist1_t1.shape)
    assert np.array_equal(hist0_t, hist1_t1)


def test_history_retains_captured_positions(game_env: Gomoku):
    """Captured stones should persist in history planes."""
    log_section("ML Features - History Capture Residual")
    game = game_env
    state = game.get_initial_state()

    moves = [
        ((1, 0), PLAYER_1),
        ((0, 0), PLAYER_2),
        ((2, 0), PLAYER_1),  # capture
        ((3, 0), PLAYER_2),  # extra move
    ]

    for move, player in moves:
        state = game.get_next_state(state, move, player)
        if move == (2, 0):
            log_state(game, state, "State right before capture")

    log_state(game, state, "Applied move")
    encoded = game.get_encoded_state(state)
    base = 8
    hist_planes = [encoded[0, base + i] for i in range(len(moves))]

    print("History plane summaries:")
    for idx, plane in enumerate(hist_planes):
        coords = np.argwhere(plane == 1)
        print(f"  Plane {idx} coords:", coords)

    capture_recent = (2, 0)
    capture_older = (1, 0)
    assert hist_planes[1][capture_recent[1], capture_recent[0]] == 1.0, (
        "Captured (2,0) remains in history"
    )
    assert hist_planes[3][capture_older[1], capture_older[0]] == 1.0, (
        "Captured (1,0) remains in history"
    )


def test_history_queue_after_exceeding_capture_goal(game_env: Gomoku):
    """History queue should retain the latest moves even after captures."""
    log_section("ML Features - History Queue Stability & Capture Clipping")
    game = game_env
    game.capture_goal = 2
    state = game.get_initial_state()
    moves = [
        (4, 4),
        (5, 5),
        (10, 4),
        (6, 6),
        (0, 0),  # dummy
        (9, 5),
        (1, 0),  # dummy
        (8, 6),  # before capture
        (7, 7),  # at capture
    ]

    for move in moves:
        state = game.get_next_state(state, move, int(state.next_player))
        if move == (8, 6):
            log_state(game, state, "State right before capture")

    log_state(game, state, "Applied move")

    encoded = game.get_encoded_state(state)
    planes = [encoded[0, 8 + i] for i in range(game.history_length)]
    print("Recent history planes:")
    for idx, plane in enumerate(planes):
        coords = np.argwhere(plane == 1)
        print(f"  Plane {idx} coords:", coords)

    recent_moves = list(reversed(moves[-game.history_length :]))
    for idx, move in enumerate(recent_moves):
        plane = planes[idx]
        assert plane[move[1], move[0]] == 1.0

    capture_plane = encoded[0, 4]
    opponent_capture_plane = encoded[0, 5]
    print("Capture plane unique values (current):", np.unique(capture_plane))
    print("Capture plane unique values (opponent):", np.unique(opponent_capture_plane))

    current_to_move = state.next_player
    if current_to_move == PLAYER_1:
        assert np.all(capture_plane == 1.0)
    else:
        assert np.all(opponent_capture_plane == 1.0)


def test_history_queue_with_back_to_back_captures(game_env: Gomoku):
    """History queue should stay ordered through consecutive capture turns."""
    log_section("ML Features - History Queue under consecutive captures")
    game = game_env
    state = game.get_initial_state()

    moves = [
        ((0, 0), PLAYER_1),
        ((1, 0), PLAYER_2),
        ((1, 2), PLAYER_1),
        ((0, 2), PLAYER_2),
        ((2, 2), PLAYER_1),
        ((2, 0), PLAYER_2),
        ((3, 0), PLAYER_1),  # capture two whites on row 0
        ((3, 2), PLAYER_2),  # capture two blacks on row 2
    ]

    for move, player in moves:
        state = game.get_next_state(state, move, player)
        if move in {(3, 0), (3, 2)}:
            log_state(game, state, f"After capture move {move}")

    # Captured stones should be removed from the board
    assert state.board[0, 1] == EMPTY_SPACE
    assert state.board[0, 2] == EMPTY_SPACE
    assert state.board[2, 1] == EMPTY_SPACE
    assert state.board[2, 2] == EMPTY_SPACE

    encoded = game.get_encoded_state(state)
    planes = [encoded[0, 8 + i] for i in range(game.history_length)]
    recent_moves = list(reversed([mv for mv, _ in moves][-game.history_length :]))

    for idx, move in enumerate(recent_moves):
        plane = planes[idx]
        assert plane[move[1], move[0]] == 1.0, (
            f"History plane {idx} should mark move {move}"
        )


def test_capture_plane_zero_goal():
    """Capture planes should be zeroed out when capture goal is disabled."""
    log_section("ML Features - Capture Plane Zero Goal")

    zero_cap_game: Gomoku = make_game(capture_goal=0)

    state = zero_cap_game.get_initial_state()
    state.p1_pts = 10
    encoded = zero_cap_game.get_encoded_state(state)
    print(
        "Capture planes with zero goal:",
        np.unique(encoded[:, 4]),
        np.unique(encoded[:, 5]),
    )
    assert np.all(encoded[:, 4] == 0.0)
    assert np.all(encoded[:, 5] == 0.0)


def test_capture_plane_zero_goal_no_nan():
    """capture_goal=0 must not introduce inf/NaN in capture planes."""
    log_section("ML Features - Capture Plane Zero Goal (No NaN)")

    zero_cap_game: Gomoku = make_game(capture_goal=0)

    state = zero_cap_game.get_initial_state()
    state.p1_pts = 7
    state.p2_pts = 3

    encoded = zero_cap_game.get_encoded_state(state)
    my_plane = encoded[:, 4]
    opp_plane = encoded[:, 5]
    print(
        "Capture planes finiteness:",
        np.isfinite(my_plane).all(),
        np.isfinite(opp_plane).all(),
    )

    assert np.isfinite(my_plane).all()
    assert np.isfinite(opp_plane).all()
    assert np.all(my_plane == 0.0)
    assert np.all(opp_plane == 0.0)


def test_encoded_padding_area_is_zero():
    """Encoded tensors should match board size with no stray padding values."""
    log_section("ML Features - Padding Area Zero")
    game: Gomoku = make_game(num_lines=5)
    state = game.get_initial_state()
    state = game.get_next_state(state, (2, 2), PLAYER_1)

    encoded = game.get_encoded_state(state)
    h, w = game.row_count, game.col_count
    assert encoded.shape[2:] == (h, w)

    # Sparse planes (last move and history) should only mark the played move.
    last_plane = encoded[0, 3]
    assert last_plane.sum() == 1.0
    assert last_plane[2, 2] == 1.0

    for k in range(game.history_length):
        plane = encoded[0, 8 + k]
        if k == 0:
            assert plane[2, 2] == 1.0
        else:
            assert plane.sum() == 0.0


def test_binary_planes_are_0_or_1(game_env: Gomoku):
    """Binary feature planes should only contain 0.0 or 1.0 (within tolerance)."""
    log_section("ML Features - Binary Plane Value Range")
    game = game_env
    state = game.get_initial_state()
    moves = [(0, 0), (1, 1), (2, 2)]
    for mv in moves:
        state = game.get_next_state(state, mv, int(state.next_player))

    encoded = game.get_encoded_state(state)
    channels = [0, 1, 2, 3, 7]  # me, opp, empty, last move, forbidden
    channels.extend(range(8, 8 + game.history_length))

    for ch in channels:
        plane = encoded[0, ch]
        unique_vals = np.unique(np.round(plane, decimals=6))
        assert set(unique_vals).issubset({0.0, 1.0}), (
            f"Channel {ch} has non-binary values: {unique_vals}"
        )


def test_get_encoded_state_deterministic():
    """Encoding the same state twice should produce identical tensors."""
    log_section("ML Features - Determinism")
    game: Gomoku = make_game(num_lines=5)
    state = game.get_initial_state()
    scripted_moves = [(0, 0), (1, 1), (2, 2), (3, 3)]
    for mv in scripted_moves:
        state = game.get_next_state(state, mv, int(state.next_player))

    encoded_a = game.get_encoded_state(state)
    encoded_b = game.get_encoded_state(state)
    assert np.array_equal(encoded_a, encoded_b)


def test_batch_encoding_is_isolated():
    """Batch encoding should not bleed values between different states."""
    log_section("ML Features - Batch Isolation")
    game = make_game(num_lines=3)
    empty_state = game.get_initial_state()

    full_state = game.get_initial_state()
    for y in range(game.row_count):
        for x in range(game.col_count):
            set_pos(full_state.board, x, y, PLAYER_1)
    full_state.next_player = PLAYER_2
    full_state.empty_count = 0

    encoded = game.get_encoded_state([empty_state, full_state])
    empty_enc, full_enc = encoded

    assert np.all(empty_enc[0] == 0)  # me plane empty
    assert np.all(empty_enc[1] == 0)  # opp plane empty
    assert np.all(empty_enc[2] == 1)  # all empty cells

    assert np.all(full_enc[0] == 0)  # P2 turn, no P2 stones
    assert np.all(full_enc[1] == 1)  # all stones belong to opponent
    assert np.all(full_enc[2] == 0)  # no empties


def test_encoding_after_terminal_state():
    """Encoding should remain consistent immediately after a winning move."""
    log_section("ML Features - Terminal State Encoding")
    game = make_game(num_lines=5)
    state = game.get_initial_state()

    moves = [
        ((0, 0), PLAYER_1),
        ((0, 1), PLAYER_2),
        ((1, 0), PLAYER_1),
        ((1, 1), PLAYER_2),
        ((2, 0), PLAYER_1),
        ((2, 1), PLAYER_2),
        ((3, 0), PLAYER_1),
        ((3, 1), PLAYER_2),
        ((4, 0), PLAYER_1),  # winning move for P1
    ]
    for move, player in moves:
        state = game.get_next_state(state, move, player)

    log_state(game, state, "Forbidden Enabled State")

    encoded = game.get_encoded_state(state)
    last_plane = encoded[0, 3]
    color_plane = encoded[0, 6]

    assert last_plane.sum() == 1.0
    assert last_plane[0, 4] == 1.0  # last move at (4,0)
    assert np.all(color_plane == -1.0)  # P2 to move next


def test_encoded_state_respects_board_size():
    """Encoded tensor should match the configured board size for different games."""
    log_section("ML Features - Variable Board Size")
    for size in (9, 15):
        game = Gomoku(
            BoardConfig(
                num_lines=size,
                gomoku_goal=5,
                capture_goal=0,
                enable_doublethree=False,
                enable_capture=False,
                history_length=2,
            )
        )
        state = game.get_initial_state()
        encoded = game.get_encoded_state(state)

        expected_channels = 8 + game.history_length
        assert encoded.shape == (1, expected_channels, size, size)
