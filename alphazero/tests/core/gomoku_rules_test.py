from gomoku.core.game_config import (
    EMPTY_SPACE,
    PLAYER_1,
    PLAYER_2,
    convert_index_to_coordinates,
    set_pos,
)
from gomoku.core.gomoku import Gomoku
from gomoku.core.rules.doublethree import detect_doublethree
from gomoku.utils.config.loader import BoardConfig
from tests.helpers import log_section, log_state, make_game


def test_capture_mechanics(capture_game: Gomoku):
    """Capture removes the sandwiched stones, restores empty count, and scores."""
    log_section("Capture Mechanics (Sandwich)")
    game: Gomoku = capture_game
    state = capture_game.get_initial_state()
    initial_empty = state.empty_count

    # Scenario: Black at A1, White at A2/A3, then Black at A4 captures two whites.
    # Coordinates: (0,0)=B, (0,1)=W, (0,2)=W
    moves = [
        ((0, 0), PLAYER_1),
        ((0, 1), PLAYER_2),
        ((1, 0), PLAYER_1),  # black plays elsewhere
        ((0, 2), PLAYER_2),
    ]
    for action, p in moves:
        state = game.get_next_state(state, action, p)

    log_state(game, state, "Before Capture Move")

    # Decisive move: Black at (0, 3)
    state = game.get_next_state(state, (0, 3), PLAYER_1)

    print(state.board)

    log_state(game, state, "After Capture Move (A2, A3 should be removed)")

    # 1. Captured stones are cleared
    assert state.board[1, 0] == EMPTY_SPACE
    assert state.board[2, 0] == EMPTY_SPACE
    assert state.board[3, 0] == PLAYER_1

    # 2. Score increments (one pair = one point)
    assert state.p1_pts == 1

    # 3. Empty count restored after removals
    # Total moves 5 (-5) + captured 2 (+2) = -3
    assert state.empty_count == initial_empty - 3


def test_win_by_capture_score(capture_game):
    """Winning by reaching the capture goal should terminate the game."""
    log_section("Win by Capture Score")
    game: Gomoku = capture_game
    # Lower goal to 1 for a quick test
    game.capture_goal = 1

    state = game.get_initial_state()

    # Recreate the capture situation
    # B W W . -> B W W B
    moves = [
        ((0, 0), PLAYER_1),
        ((0, 1), PLAYER_2),
        ((5, 5), PLAYER_1),
        ((0, 2), PLAYER_2),
    ]
    for action, p in moves:
        state = game.get_next_state(state, action, p)

    # Decisive move
    action = (0, 3)
    state = game.get_next_state(state, action, PLAYER_1)

    log_state(game, state, "Winning by Capture Score")

    # Check win
    assert game.check_win(state, action) is True
    val, term = game.get_value_and_terminated(state, action)
    assert val == 1
    assert term is True


def test_double_three_logic(strict_game):
    """Double-three forbids the intersecting point when both patterns exist."""
    log_section("Double Three (Forbidden Move)")
    game: Gomoku = strict_game
    state = game.get_initial_state()

    # Gomoku skips double-three if fewer than 4 stones exist, so seed stones.
    # Scenario: (3,3) should be forbidden.
    #   . B .
    # B X B
    #   . B .

    # Direct board manipulation for quick setup
    board = state.board
    setup_stones = [(3, 1), (3, 2), (1, 3), (2, 3)]  # black stones
    for x, y in setup_stones:
        set_pos(board, x, y, PLAYER_1)

    # Dummy white stone to reach count threshold
    set_pos(board, 8, 8, PLAYER_2)

    # Black to move
    state.next_player = PLAYER_1

    log_state(game, state, "Setup for Double Three at D4 (3,3)")

    print("needs to be true", detect_doublethree(board, 3, 3, PLAYER_1, 9))
    # Fetch legal moves
    legal_moves = game.get_legal_moves(state)
    forbidden_coord = "D4"

    print(f"Legal Moves Count: {len(legal_moves)}")
    print("Legal Moves", legal_moves)
    print(f"Is D4 in legal moves? {'Yes' if forbidden_coord in legal_moves else 'No'}")

    # Verify D4 is forbidden
    assert forbidden_coord not in legal_moves


def test_get_legal_moves_cache(strict_game):
    """Legal move caching should reuse the computed indices."""
    log_section("Get Legal Moves - caching / fast path")
    game: Gomoku = strict_game
    state = game.get_initial_state()
    board = state.board

    placements = [(0, 0), (2, 2), (4, 4)]
    for x, y in placements:
        set_pos(board, x, y, PLAYER_1)

    state.next_player = PLAYER_1
    assert state.legal_indices_cache is None

    moves_first = game.get_legal_moves(state)
    assert len(moves_first) == game.action_size - len(placements)
    assert state.legal_indices_cache is not None
    assert len(state.legal_indices_cache) == len(moves_first)

    moves_second = game.get_legal_moves(state)
    assert moves_second == moves_first


def test_double_three_edge_patterns(strict_game):
    """
    Two edge-type open-threes intersecting at D4 should be forbidden.

    Patterns roughly:
      - Horizontal: .$OO. across row 4
      - Vertical:   .$OO. across column D
    """
    log_section("Double Three - edge patterns (.$OO. / .$.OO. / .$O.O.)")
    game: Gomoku = strict_game
    state = game.get_initial_state()
    board = state.board

    forbidden_coord = "D4"  # (3,3)

    set_pos(board, 4, 3, PLAYER_1)
    set_pos(board, 5, 3, PLAYER_1)
    set_pos(board, 3, 4, PLAYER_1)
    set_pos(board, 3, 5, PLAYER_1)
    set_pos(board, 8, 8, PLAYER_2)

    state.next_player = PLAYER_1
    log_state(game, state, "Setup: double edge-type open threes at D4")

    legal_moves = game.get_legal_moves(state)

    print(f"Legal Moves Count: {len(legal_moves)}")
    print("Legal Moves", legal_moves)
    print(f"Is D4 in legal moves? {'Yes' if forbidden_coord in legal_moves else 'No'}")

    # D4 must be forbidden
    assert forbidden_coord not in legal_moves


def test_double_three_middle1_patterns(strict_game):
    """
    Two middle-type open-threes (.O$O.) intersecting at D4 should be forbidden.

    Patterns:
      - Horizontal: .O$O. across row 4
      - Vertical:   .O$O. across column D
    """
    log_section("Double Three - middle1 patterns (.O$O.)")
    game: Gomoku = strict_game
    state = game.get_initial_state()
    board = state.board

    forbidden_coord = "D4"  # (3,3)

    set_pos(board, 2, 3, PLAYER_1)
    set_pos(board, 4, 3, PLAYER_1)
    set_pos(board, 3, 2, PLAYER_1)
    set_pos(board, 3, 4, PLAYER_1)

    set_pos(board, 8, 8, PLAYER_2)

    state.next_player = PLAYER_1
    log_state(game, state, "Setup: double middle1 (.O$O.) open threes at D4")

    legal_moves = game.get_legal_moves(state)

    print(f"Legal Moves Count: {len(legal_moves)}")
    print("Legal Moves", legal_moves)
    print(f"Is D4 in legal moves? {'Yes' if forbidden_coord in legal_moves else 'No'}")

    assert forbidden_coord not in legal_moves


def test_double_three_middle2_patterns(strict_game):
    """
    Two middle-type open-threes (.O$.O.) intersecting at D4 should be forbidden.
    Patterns:
      - Horizontal: .O$.O. across row 4
      - Vertical:   .O$.O. across column D.
    """
    log_section("Double Three - middle2 patterns (.O$.O.)")
    game: Gomoku = strict_game
    state = game.get_initial_state()
    board = state.board

    forbidden_coord = "D4"  # (3,3)

    set_pos(board, 2, 3, PLAYER_1)
    set_pos(board, 5, 3, PLAYER_1)
    set_pos(board, 3, 2, PLAYER_1)
    set_pos(board, 3, 5, PLAYER_1)

    set_pos(board, 8, 8, PLAYER_2)

    state.next_player = PLAYER_1
    log_state(game, state, "Setup: double middle2 (.O$.O.) open threes at D4")

    legal_moves = game.get_legal_moves(state)

    print(f"Legal Moves Count: {len(legal_moves)}")
    print("Legal Moves", legal_moves)
    print(f"Is D4 in legal moves? {'Yes' if forbidden_coord in legal_moves else 'No'}")

    # D4 must be forbidden
    assert forbidden_coord not in legal_moves


def test_double_three_exclusions_edge_not_forbidden_1(strict_game):
    """Edge exclusion pattern X.$OO.X should remain legal at D4."""
    log_section("Double Three - exclusion patterns should be legal")
    game: Gomoku = strict_game
    state = game.get_initial_state()
    board = state.board

    # Target move: D4
    possible_coord = "D4"

    set_pos(board, 3, 1, PLAYER_1)
    set_pos(board, 3, 2, PLAYER_1)

    set_pos(board, 2, 3, PLAYER_2)
    set_pos(board, 4, 3, PLAYER_1)
    set_pos(board, 5, 3, PLAYER_1)
    set_pos(board, 6, 3, PLAYER_2)  # right X

    set_pos(board, 8, 8, PLAYER_2)
    state.next_player = PLAYER_1

    log_state(game, state, "Setup: exclusion shape X.$OO.X at D4 only")

    legal_moves = game.get_legal_moves(state)

    assert possible_coord in legal_moves


# wrong case
def test_double_three_exclusions_edge_not_forbidden_2(strict_game):
    """
    Edge Case 2 exclusion (O.$O.O etc.) must NOT be treated as an open-three.
    D4 should remain legal.

    This isolates the O.$O.O pattern only on the horizontal line through D4
    so no vertical open-threes exist—verifying only the exclusion pattern.
    """
    log_section("Double Three - edge Case 2 exclusion should be legal")
    game: Gomoku = strict_game
    state = game.get_initial_state()
    board = state.board

    forbidden_coord = "D4"  # (3, 3)

    set_pos(board, 1, 3, PLAYER_1)
    set_pos(board, 4, 3, PLAYER_1)
    set_pos(board, 6, 3, PLAYER_1)

    set_pos(board, 3, 1, PLAYER_1)
    set_pos(board, 3, 2, PLAYER_1)

    set_pos(board, 8, 8, PLAYER_2)

    state.next_player = PLAYER_1

    log_state(
        game, state, "Setup: edge Case 2 exclusion pattern around D4 (only horizontal)"
    )

    legal_moves = game.get_legal_moves(state)

    assert forbidden_coord in legal_moves


def test_double_three_exclusions_middle_not_forbidden_1(strict_game):
    """
    Middle Case 1 exclusion: X.O$O.X must NOT be treated as an open-three.
    D4 should remain legal.
    """
    log_section("Double Three - middle Case 1 exclusion X.O$O.X")
    game: Gomoku = strict_game
    state = game.get_initial_state()
    board = state.board

    forbidden_coord = "D4"  # (3,3)

    set_pos(board, 3, 1, PLAYER_1)
    set_pos(board, 3, 2, PLAYER_1)

    set_pos(board, 0, 3, PLAYER_2)
    set_pos(board, 2, 3, PLAYER_1)
    set_pos(board, 4, 3, PLAYER_1)
    set_pos(board, 6, 3, PLAYER_2)

    set_pos(board, 8, 8, PLAYER_2)

    state.next_player = PLAYER_1

    log_state(game, state, "Setup: middle Case 1 exclusion X.O$O.X at D4")

    legal_moves = game.get_legal_moves(state)

    print("needs to be false", detect_doublethree(board, 3, 3, PLAYER_1, 9))

    assert forbidden_coord in legal_moves


def test_double_three_exclusions_middle_not_forbidden_2(strict_game):
    """
    Middle Case 1 second exclusion (O$O.O / .O$O.O type) must NOT be treated
    as an open-three. D4 should remain legal.
    """
    log_section("Double Three - middle Case 1 exclusion O$O.O")
    game: Gomoku = strict_game
    state = game.get_initial_state()
    board = state.board

    forbidden_coord = "D4"  # (3,3)

    set_pos(board, 3, 1, PLAYER_1)
    set_pos(board, 3, 2, PLAYER_1)
    set_pos(board, 2, 3, PLAYER_1)
    set_pos(board, 4, 3, PLAYER_1)
    set_pos(board, 6, 3, PLAYER_1)

    set_pos(board, 8, 8, PLAYER_2)

    state.next_player = PLAYER_1

    log_state(game, state, "Setup: middle Case 1 exclusion O$O.O at D4")

    legal_moves = game.get_legal_moves(state)

    print("needs to be false", detect_doublethree(board, 3, 3, PLAYER_1, 9))

    assert forbidden_coord in legal_moves


def test_draw_condition():
    """Draw when a full board contains no winner."""
    log_section("Draw Condition (Board Full)")
    # Use a 2x2 board for brevity
    cfg = BoardConfig(
        num_lines=2,
        gomoku_goal=5,
        capture_goal=0,
        enable_doublethree=False,
        enable_capture=False,
    )
    game: Gomoku = Gomoku(cfg)
    state = game.get_initial_state()

    # B W
    # W B
    moves = [
        ((0, 0), PLAYER_1),
        ((0, 1), PLAYER_2),
        ((1, 1), PLAYER_1),
        ((1, 0), PLAYER_2),
    ]
    for action, p in moves:
        state = game.get_next_state(state, action, p)

    log_state(game, state, "Board Full (Draw)")

    # Confirm termination with no winner
    val, term = game.get_value_and_terminated(state, (1, 0))
    assert term is True
    assert val == 0  # draw


def test_double_three_chimhaha_1(strict_game):
    """Chimhaha case 1 should keep the target coordinate legal."""
    log_section("Double Three - Chimhaha case 1")
    game: Gomoku = make_game(num_lines=15)
    state = game.get_initial_state()
    board = state.board

    possible_coord = convert_index_to_coordinates(9, 5, 15)

    set_pos(board, 6, 5, PLAYER_1)
    set_pos(board, 7, 5, PLAYER_1)
    set_pos(board, 8, 6, PLAYER_1)
    set_pos(board, 7, 7, PLAYER_1)

    set_pos(board, 12, 2, PLAYER_1)

    set_pos(board, 5, 9, PLAYER_2)

    state.next_player = PLAYER_1
    """
       A B C D E F G H I J K L M N O
     1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     3 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0
     4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     6 0 0 0 0 0 0 1 1 0 A 0 0 0 0 0
     7 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0
     8 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0
     9 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    10 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0
    11 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    12 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    13 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    14 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    15 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    """
    log_state(game, state, "Setup: Chimaha case 1")

    legal_moves = game.get_legal_moves(state)

    print("needs to be false", detect_doublethree(board, 9, 5, PLAYER_1, 15))

    assert possible_coord in legal_moves


def test_double_three_chimhaha_2(strict_game):
    """Chimhaha case 2 mixes allowed and forbidden points across the board."""
    log_section("Double Three - Chimhaha case 1")
    game: Gomoku = make_game(num_lines=15)
    state = game.get_initial_state()
    board = state.board

    coord_e = convert_index_to_coordinates(3, 3, 15)
    coord_f = convert_index_to_coordinates(13, 2, 15)
    coord_g = convert_index_to_coordinates(3, 11, 15)
    coord_h = convert_index_to_coordinates(11, 12, 15)

    set_pos(board, 0, 3, PLAYER_1)
    set_pos(board, 4, 2, PLAYER_1)
    set_pos(board, 4, 3, PLAYER_1)
    set_pos(board, 5, 3, PLAYER_1)
    set_pos(board, 2, 4, PLAYER_1)

    set_pos(board, 14, 1, PLAYER_1)
    set_pos(board, 8, 3, PLAYER_1)
    set_pos(board, 11, 4, PLAYER_1)
    set_pos(board, 10, 5, PLAYER_1)
    set_pos(board, 9, 6, PLAYER_1)
    set_pos(board, 13, 4, PLAYER_1)
    set_pos(board, 13, 5, PLAYER_1)
    set_pos(board, 13, 6, PLAYER_1)

    set_pos(board, 2, 10, PLAYER_1)
    set_pos(board, 2, 11, PLAYER_1)
    set_pos(board, 5, 11, PLAYER_1)
    set_pos(board, 4, 12, PLAYER_1)

    set_pos(board, 11, 10, PLAYER_1)
    set_pos(board, 10, 12, PLAYER_1)
    set_pos(board, 12, 12, PLAYER_1)
    set_pos(board, 11, 13, PLAYER_1)

    set_pos(board, 8, 7, PLAYER_2)
    set_pos(board, 13, 7, PLAYER_2)
    set_pos(board, 6, 14, PLAYER_2)

    set_pos(board, 10, 10, PLAYER_2)
    set_pos(board, 10, 11, PLAYER_2)
    set_pos(board, 9, 13, PLAYER_2)
    set_pos(board, 10, 13, PLAYER_2)
    set_pos(board, 10, 14, PLAYER_2)

    set_pos(board, 12, 10, PLAYER_2)
    set_pos(board, 12, 11, PLAYER_2)
    set_pos(board, 12, 13, PLAYER_2)
    set_pos(board, 13, 13, PLAYER_2)
    set_pos(board, 12, 14, PLAYER_2)

    state.next_player = PLAYER_1
    """
       A B C D E F G H I J K L M N O
     1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
     3 0 0 0 0 1 0 0 0 0 0 0 0 0 F 0
     4 1 0 0 E 1 1 0 0 1 0 0 0 0 0 0
     5 0 0 1 0 0 0 0 0 0 0 0 1 0 1 0
     6 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0
     7 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0
     8 0 0 0 0 0 0 0 0 2 0 0 0 0 2 0
     9 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    10 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    11 0 0 1 0 0 0 0 0 0 0 2 1 2 0 0
    12 0 0 1 G 0 1 0 0 0 0 2 0 2 0 0
    13 0 0 0 0 1 0 0 0 0 0 1 H 1 0 0
    14 0 0 0 0 0 0 0 0 0 2 2 1 2 2 0
    15 0 0 0 0 0 0 2 0 0 0 2 0 2 0 0
    """
    log_state(game, state, "Setup: Chimaha case 1")

    legal_moves = game.get_legal_moves(state)

    assert coord_e in legal_moves
    assert coord_f in legal_moves
    assert coord_g not in legal_moves
    assert coord_h not in legal_moves


def test_double_three_namuwiki_1(strict_game):
    """Namuwiki case 1 marks four distinct forbidden coordinates."""
    log_section("Double Three - namuwiki case 1")
    game: Gomoku = make_game(num_lines=15)
    state = game.get_initial_state()
    board = state.board

    # for A (3, 3)
    forbidden_coord_a = convert_index_to_coordinates(3, 3, 15)
    set_pos(board, 3, 1, PLAYER_1)
    set_pos(board, 3, 2, PLAYER_1)
    set_pos(board, 2, 3, PLAYER_1)
    set_pos(board, 4, 3, PLAYER_1)

    # for B (4, 12)
    forbidden_coord_b = convert_index_to_coordinates(4, 12, 15)
    set_pos(board, 1, 9, PLAYER_1)
    set_pos(board, 4, 9, PLAYER_1)
    set_pos(board, 2, 10, PLAYER_1)
    set_pos(board, 4, 10, PLAYER_1)

    # for C (11, 4)
    forbidden_coord_c = convert_index_to_coordinates(11, 4, 15)
    set_pos(board, 12, 3, PLAYER_1)
    set_pos(board, 12, 5, PLAYER_1)
    set_pos(board, 9, 6, PLAYER_1)
    set_pos(board, 13, 6, PLAYER_1)

    # for D (10, 11)
    forbidden_coord_d = convert_index_to_coordinates(10, 11, 15)
    set_pos(board, 10, 9, PLAYER_1)
    set_pos(board, 12, 11, PLAYER_1)
    set_pos(board, 13, 11, PLAYER_1)
    set_pos(board, 10, 12, PLAYER_1)

    state.next_player = PLAYER_1

    """
       A B C D E F G H I J K L M N O
     1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     2 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0
     3 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0
     4 0 0 1 A 1 0 0 0 0 0 0 0 1 0 0
     5 0 0 0 0 0 0 0 0 0 0 0 C 0 0 0
     6 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0
     7 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0
     8 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     9 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    10 0 1 0 0 1 0 0 0 0 0 1 0 0 0 0
    11 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0
    12 0 0 0 0 0 0 0 0 0 0 D 0 1 1 0
    13 0 0 0 0 B 0 0 0 0 0 1 0 0 0 0
    14 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    15 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    """
    log_state(game, state, "Setup: namuwiki Basic 1")

    print("needs to be true", detect_doublethree(board, 3, 3, PLAYER_1, 15))
    print("needs to be true", detect_doublethree(board, 4, 12, PLAYER_1, 15))
    print("needs to be true", detect_doublethree(board, 11, 4, PLAYER_1, 15))
    print("needs to be true", detect_doublethree(board, 10, 11, PLAYER_1, 15))

    legal_moves = game.get_legal_moves(state)
    assert forbidden_coord_a not in legal_moves
    assert forbidden_coord_b not in legal_moves
    assert forbidden_coord_c not in legal_moves
    assert forbidden_coord_d not in legal_moves


def test_double_three_namuwiki_2(strict_game):
    """Namuwiki case 2 blends forbidden and allowed spots on a 15x15 board."""
    log_section("Double Three - namuwiki case 2")
    game: Gomoku = make_game(num_lines=15)
    state = game.get_initial_state()
    board = state.board

    forbidden_coord_a = convert_index_to_coordinates(1, 2, 15)
    forbidden_coord_b = convert_index_to_coordinates(3, 8, 15)
    forbidden_coord_c = convert_index_to_coordinates(7, 8, 15)
    forbidden_coord_d = convert_index_to_coordinates(7, 12, 15)

    possible_coord_a = convert_index_to_coordinates(7, 5, 15)
    possible_coord_b = convert_index_to_coordinates(11, 1, 15)
    possible_coord_c = convert_index_to_coordinates(9, 11, 15)

    set_pos(board, 1, 1, PLAYER_1)
    set_pos(board, 1, 4, PLAYER_1)
    set_pos(board, 3, 2, PLAYER_1)
    set_pos(board, 4, 2, PLAYER_1)

    set_pos(board, 4, 9, PLAYER_1)
    set_pos(board, 3, 10, PLAYER_1)
    set_pos(board, 3, 11, PLAYER_1)
    set_pos(board, 5, 10, PLAYER_1)

    set_pos(board, 5, 5, PLAYER_1)
    set_pos(board, 6, 5, PLAYER_1)
    set_pos(board, 7, 6, PLAYER_1)
    set_pos(board, 7, 7, PLAYER_1)

    set_pos(board, 9, 10, PLAYER_1)
    set_pos(board, 8, 11, PLAYER_1)
    set_pos(board, 10, 11, PLAYER_1)
    set_pos(board, 9, 12, PLAYER_1)

    set_pos(board, 10, 1, PLAYER_1)
    set_pos(board, 12, 1, PLAYER_1)
    set_pos(board, 11, 3, PLAYER_1)
    set_pos(board, 11, 4, PLAYER_1)
    set_pos(board, 9, 5, PLAYER_1)

    set_pos(board, 8, 0, PLAYER_2)
    set_pos(board, 8, 1, PLAYER_2)
    set_pos(board, 8, 2, PLAYER_2)
    set_pos(board, 8, 3, PLAYER_2)

    set_pos(board, 14, 0, PLAYER_2)
    set_pos(board, 14, 1, PLAYER_2)
    set_pos(board, 14, 2, PLAYER_2)
    set_pos(board, 14, 3, PLAYER_2)

    set_pos(board, 14, 5, PLAYER_2)
    set_pos(board, 14, 6, PLAYER_2)
    set_pos(board, 14, 7, PLAYER_2)
    set_pos(board, 14, 8, PLAYER_2)

    set_pos(board, 14, 10, PLAYER_2)
    set_pos(board, 14, 11, PLAYER_2)
    set_pos(board, 14, 12, PLAYER_2)
    set_pos(board, 14, 13, PLAYER_2)

    set_pos(board, 9, 14, PLAYER_2)

    set_pos(board, 11, 14, PLAYER_2)
    set_pos(board, 12, 14, PLAYER_2)
    set_pos(board, 13, 14, PLAYER_2)

    set_pos(board, 9, 8, PLAYER_2)

    state.next_player = PLAYER_1

    """
      A B C D E F G H I J K L M N O
    1 0 0 0 0 0 0 0 0 2 0 0 0 0 0 2
    2 0 1 0 0 0 0 0 0 2 0 1 b 1 0 2
    3 0 A 0 1 1 0 0 0 2 0 0 0 0 0 2
    4 0 0 0 0 0 0 0 0 2 0 0 1 0 0 2
    5 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0
    6 0 0 0 0 0 1 1 a 0 1 0 0 0 0 2
    7 0 0 0 0 0 0 0 1 0 0 0 0 0 0 2
    8 0 0 0 0 0 0 0 1 0 0 0 0 0 0 2
    9 0 0 0 B 0 0 0 C 0 2 0 0 0 0 2
    0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0
    1 0 0 0 1 0 1 0 0 0 1 0 0 0 0 2
    2 0 0 0 1 0 0 0 0 1 c 1 0 0 0 2
    3 0 0 0 0 0 0 0 D 0 1 0 0 0 0 2
    4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2
    5 0 0 0 0 0 0 0 0 0 2 0 2 2 2 0
    """
    log_state(game, state, "Setup: namuwiki Basic 1")
    print("needs to be true", detect_doublethree(board, 1, 2, PLAYER_1, 15))
    print("needs to be true", detect_doublethree(board, 3, 8, PLAYER_1, 15))
    print("needs to be true", detect_doublethree(board, 7, 8, PLAYER_1, 15))
    print("needs to be true", detect_doublethree(board, 7, 12, PLAYER_1, 15))

    print("needs to be false", detect_doublethree(board, 7, 5, PLAYER_1, 15))
    print("needs to be false", detect_doublethree(board, 11, 1, PLAYER_1, 15))
    print("needs to be false", detect_doublethree(board, 9, 11, PLAYER_1, 15))

    legal_moves = game.get_legal_moves(state)

    forbidden_coord_a = convert_index_to_coordinates(1, 2, 15)
    forbidden_coord_b = convert_index_to_coordinates(3, 8, 15)
    forbidden_coord_c = convert_index_to_coordinates(7, 8, 15)
    forbidden_coord_d = convert_index_to_coordinates(7, 12, 15)

    possible_coord_a = convert_index_to_coordinates(7, 5, 15)
    possible_coord_b = convert_index_to_coordinates(11, 1, 15)
    possible_coord_c = convert_index_to_coordinates(9, 11, 15)

    assert forbidden_coord_a not in legal_moves
    assert forbidden_coord_b not in legal_moves
    assert forbidden_coord_c not in legal_moves
    assert forbidden_coord_d not in legal_moves

    assert possible_coord_a in legal_moves
    assert possible_coord_b in legal_moves
    assert possible_coord_c in legal_moves


def test_double_three_namuwiki_3(strict_game):
    """Namuwiki case 3 leaves the center-right target legal."""
    log_section("Double Three - namuwiki case 3")
    game: Gomoku = make_game(num_lines=11)
    state = game.get_initial_state()
    board = state.board

    possible_coord = convert_index_to_coordinates(6, 4, 15)

    set_pos(board, 1, 4, PLAYER_1)
    set_pos(board, 4, 4, PLAYER_1)
    set_pos(board, 5, 4, PLAYER_1)

    set_pos(board, 6, 2, PLAYER_1)
    set_pos(board, 6, 3, PLAYER_1)

    set_pos(board, 8, 4, PLAYER_2)

    state.next_player = PLAYER_1

    """
      A B C D E F G H I J K
    1 0 0 0 0 0 0 0 0 0 0 0
    2 0 0 0 0 0 0 0 0 0 0 0
    3 0 0 0 0 0 0 1 0 0 0 0
    4 0 0 0 0 0 0 1 0 0 0 0
    5 0 1 0 0 1 1 X 0 2 0 0
    6 0 0 0 0 0 0 0 0 0 0 0
    7 0 0 0 0 0 0 0 0 0 0 0
    8 0 0 0 0 0 0 0 0 0 0 0
    9 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0
    1 0 0 0 0 0 0 0 0 0 0 0
    """
    log_state(game, state, "Setup: namuwiki Basic 1")

    legal_moves = game.get_legal_moves(state)

    assert possible_coord in legal_moves
