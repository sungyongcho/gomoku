"""Feature logic tests for PolicyValueNet input handling."""

import torch

from gomoku.core.game_config import NUM_LINES, PLAYER_1, PLAYER_2, set_pos
from gomoku.model.model_helpers import NUM_PLANES, calc_num_hidden, calc_num_resblocks
from gomoku.utils.config.loader import parse_config
from tests.helpers import (
    log_section,
    log_state,
    log_tensor_summary,
    make_linear_policy_model,
    minimal_raw_config,
)


def test_model_config_uses_safe_defaults():
    """Ensure parser supplies safe default model dimensions."""
    log_section("PolicyValueNet - ModelConfig safe defaults")
    raw = minimal_raw_config()

    cfg = parse_config(raw)

    expected_planes = 8 + cfg.board.history_length
    assert cfg.model.num_planes == expected_planes
    assert cfg.model.num_hidden == calc_num_hidden(num_lines=NUM_LINES)
    assert cfg.model.num_resblocks == calc_num_resblocks(num_lines=NUM_LINES)


def test_last_move_plane_marks_correct_position():
    """Last-move plane should map to the correct policy logit."""
    log_section("PolicyValueNet - Last move plane correctness")
    model, game, config = make_linear_policy_model(focus_channels=(3,))
    move = (1, 2)
    state = game.get_initial_state()
    state = game.get_next_state(state, move, int(state.next_player))
    encoded = torch.tensor(game.get_encoded_state(state))

    with torch.no_grad():
        policy_logits, _ = model(encoded)

    action_idx = move[1] * game.col_count + move[0]
    print("Action index:", action_idx)
    print("Logit at action_idx:", float(policy_logits[0, action_idx]))
    log_tensor_summary(policy_logits[0], "policy_logits", top_k=5)
    assert torch.isclose(policy_logits[0, action_idx], torch.tensor(1.0), atol=1e-3)
    mask = torch.ones_like(policy_logits[0])
    mask[action_idx] = 0.0
    log_tensor_summary(policy_logits[0] * mask, "policy_logits", top_k=5)
    assert torch.allclose(policy_logits[0] * mask, torch.zeros_like(policy_logits[0]))


def test_color_plane_turn_encoding():
    """Color plane should flip between P1 and P2 turns."""
    log_section("PolicyValueNet - Color plane turn encoding")
    model, game, config = make_linear_policy_model(focus_channels=(6,))
    start_state = game.get_initial_state()
    p2_state = game.get_next_state(start_state, (2, 1), PLAYER_1)  # switch turn to P2

    p1_input = torch.tensor(game.get_encoded_state(start_state))
    p2_input = torch.tensor(game.get_encoded_state(p2_state))

    with torch.no_grad():
        p1_logits, _ = model(p1_input)
        p2_logits, _ = model(p2_input)

    log_tensor_summary(p1_logits[0], "color_p1_logits", top_k=5)
    log_tensor_summary(p2_logits[0], "color_p2_logits", top_k=5)
    assert torch.all(p1_logits > 0)
    assert torch.allclose(p2_logits, torch.zeros_like(p2_logits))


def test_capture_score_plane_normalization():
    """Capture score planes must influence logits proportionally."""
    log_section("PolicyValueNet - Capture score normalization")
    model, game, config = make_linear_policy_model(focus_channels=(4,))
    low_state = game.get_initial_state()
    low_state.next_player = PLAYER_1
    low_state.p1_pts = 1
    high_state = game.get_initial_state()
    high_state.next_player = PLAYER_1
    high_state.p1_pts = game.capture_goal
    low = torch.tensor(game.get_encoded_state(low_state))
    high = torch.tensor(game.get_encoded_state(high_state))

    with torch.no_grad():
        low_logits, _ = model(low)
        high_logits, _ = model(high)

    log_tensor_summary(low_logits[0], "capture_low_logits", top_k=5)
    log_tensor_summary(high_logits[0], "capture_high_logits", top_k=5)
    assert torch.all(high_logits > low_logits)


def test_forbidden_plane_marks_double_three():
    """Forbidden plane should map only the marked positions."""
    log_section("PolicyValueNet - Forbidden plane double-three mapping")
    model, game, config = make_linear_policy_model(focus_channels=(7,))
    state = game.get_initial_state()
    pattern_a = [(3, 1), (3, 2), (2, 3), (4, 3)]
    pattern_b = [(1, 9), (4, 9), (2, 10), (4, 10)]
    pattern_c = [(12, 3), (12, 5), (9, 6), (13, 6)]
    pattern_d = [(10, 9), (12, 11), (13, 11), (10, 12)]
    for x_coord, y_coord in pattern_a + pattern_b + pattern_c + pattern_d:
        set_pos(state.board, x_coord, y_coord, PLAYER_1)
    state.next_player = PLAYER_1

    encoded = torch.tensor(game.get_encoded_state(state))
    log_state(game, state, "Forbidden pattern board")
    forbidden_plane = encoded[0, 7]
    forbidden_coords = torch.nonzero(forbidden_plane, as_tuple=False).tolist()
    forbidden_idxs = [
        coord[0] * game.col_count + coord[1] for coord in forbidden_coords
    ]
    print("Forbidden plane nonzero (y,x):", forbidden_coords)
    print("Forbidden plane nonzero flat idx:", forbidden_idxs)

    with torch.no_grad():
        logits, _ = model(encoded)

    expected_coords = [(3, 3), (4, 12), (11, 4), (10, 11)]
    zeros = torch.ones_like(logits[0])
    for x_coord, y_coord in expected_coords:
        idx = y_coord * game.col_count + x_coord
        assert torch.allclose(logits[0, idx], torch.tensor(1.0))
        zeros[idx] = 0.0
    log_tensor_summary(logits[0], "forbidden_logits", top_k=5)
    assert torch.allclose(logits[0] * zeros, torch.zeros_like(logits[0]))


def test_history_planes_stack_and_shift_correctly():
    """History planes must shift to reflect the most recent move."""
    log_section("PolicyValueNet - History plane stacking")
    model, game, config = make_linear_policy_model(focus_channels=(8,))
    state = game.get_initial_state()
    first_move = (0, 1)
    second_move = (2, 2)
    state = game.get_next_state(state, first_move, int(state.next_player))
    encoded_first = torch.tensor(game.get_encoded_state(state))
    log_state(game, state, "After first move")
    state = game.get_next_state(state, second_move, int(state.next_player))
    encoded_second = torch.tensor(game.get_encoded_state(state))
    log_state(game, state, "After second move")

    with torch.no_grad():
        first_logits, _ = model(encoded_first)
        second_logits, _ = model(encoded_second)

    log_tensor_summary(first_logits[0], "history_first_logits", top_k=5)
    log_tensor_summary(second_logits[0], "history_second_logits", top_k=5)
    first_idx = first_move[1] * game.col_count + first_move[0]
    second_idx = second_move[1] * game.col_count + second_move[0]
    assert torch.allclose(first_logits[0, first_idx], torch.tensor(1.0))
    assert torch.allclose(first_logits[0, second_idx], torch.tensor(0.0))
    assert torch.allclose(second_logits[0, second_idx], torch.tensor(1.0))
    assert torch.allclose(second_logits[0, first_idx], torch.tensor(0.0))


def test_feature_consistency_after_terminal_state():
    """Encoding stays consistent immediately after terminal move."""
    log_section("PolicyValueNet - Features after terminal state")
    model, game, config = make_linear_policy_model(focus_channels=(3,))
    moves = [
        ((0, 0), PLAYER_1),
        ((0, 1), PLAYER_2),
        ((1, 0), PLAYER_1),
        ((1, 1), PLAYER_2),
        ((2, 0), PLAYER_1),
        ((2, 1), PLAYER_2),
        ((3, 0), PLAYER_1),
        ((3, 1), PLAYER_2),
        ((4, 0), PLAYER_1),  # winning move (5 in a row for P1)
    ]
    state = game.get_initial_state()
    for mv, pl in moves:
        state = game.get_next_state(state, mv, pl)
    encoded = torch.tensor(game.get_encoded_state(state))

    with torch.no_grad():
        logits, value = model(encoded)

    log_tensor_summary(logits[0], "terminal_logits", top_k=5)
    log_tensor_summary(value[0], "terminal_value", top_k=value[0].numel())
    idx = moves[-1][0][1] * game.col_count + moves[-1][0][0]
    assert torch.allclose(logits[0, idx], torch.tensor(1.0))
    assert torch.all(torch.isfinite(logits))
    assert torch.all(torch.isfinite(value))
