from gomoku.core.game_config import (
    EMPTY_SPACE,
    PLAYER_1,
    convert_coordinates_to_index,
    get_pos,
    index_to_xy,
    xy_to_index,
)
from gomoku.core.gomoku import GameState, Gomoku
from tests.helpers import log_section, log_state, make_game

# -------------------------------------------------------------------------
# Test Cases
# -------------------------------------------------------------------------


def test_xy_index_conversion_preserves_axes():
    """Ensure xy/index conversions keep x as column and y as row."""
    assert xy_to_index(0, 0) == 0
    assert xy_to_index(1, 0) == 1
    assert xy_to_index(0, 1) == 19

    for x, y in [(0, 0), (2, 3), (4, 4)]:
        idx = xy_to_index(x, y)
        assert index_to_xy(idx) == (x, y)


def test_next_state_places_stone_with_xy_axes(game_env):
    """Place a stone at (2, 1) and verify axes are respected."""
    log_section("Stone Placement Check (Axes)")

    game: Gomoku = game_env
    start_state = game.get_initial_state()

    # Action: (2, 1) -> C2 (x=2, y=1)
    next_state = game.get_next_state(start_state, (2, 1), PLAYER_1)

    log_state(game, next_state, "Placed Stone at C2 (x=2, y=1)")

    # Assertions
    assert get_pos(next_state.board, 2, 1) == PLAYER_1
    assert next_state.board[1, 2] == PLAYER_1
    assert get_pos(next_state.board, 1, 2) == EMPTY_SPACE
    assert next_state.board[2, 1] == EMPTY_SPACE


def test_coordinate_strings_match_xy_axes():
    """Coordinate strings map to expected (x, y) indices."""
    assert convert_coordinates_to_index("A1") == (0, 0)
    assert convert_coordinates_to_index("C2") == (2, 1)
    assert convert_coordinates_to_index("E5") == (4, 4)


def test_horizontal_and_vertical_wins_follow_axes(game_env):
    """Five in a row horizontally or vertically triggers a win."""
    log_section("Win Condition Check (Axes)")

    game: Gomoku = game_env
    # 1. Horizontal Win
    state = game.get_initial_state()
    for x in range(5):
        state = game.get_next_state(state, (x, 3), PLAYER_1)

    log_state(game, state, "Horizontal Line (Row 4)")
    assert game.check_win(state, (4, 3)) is True

    # 2. Vertical Win
    state = game.get_initial_state()
    for y in range(5):
        state = game.get_next_state(state, (2, y), PLAYER_1)

    log_state(game, state, "Vertical Line (Col C)")
    assert game.check_win(state, (2, 4)) is True


def test_format_board_outputs_readable_state(game_env):
    """Board formatting should match the expected text layout."""
    log_section("Board Formatting Check")

    # 1. Setup
    game: Gomoku = make_game(num_lines=3, capture_goal=3)
    state: GameState = game.get_initial_state()
    state = game.get_next_state(state, (1, 0), PLAYER_1)  # B1
    state = game.get_next_state(state, (0, 2), state.next_player)  # A3

    # 2. Expected String
    # fmt: off
    expected = (
        "   A B C\n"
        " 1 0 1 0\n"
        " 2 0 0 0\n"
        " 3 2 0 0\n"
        "Captures  P1:0  P2:0\n\n"
    )
    # fmt: on

    # 3. Visual Comparison
    print("\n>>> Comparing Output:")
    print(f"{' [EXPECTED] ':~^30}")  # ~~~ [EXPECTED] ~~~ 스타일
    print(expected.strip())  # 개행 제거하여 깔끔하게

    print(f"\n{' [ACTUAL] ':~^30}")
    game.print_board(state)
    print("~" * 30)

    assert game.format_board(state) == expected
