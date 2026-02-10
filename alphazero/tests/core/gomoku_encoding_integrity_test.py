import numpy as np

from gomoku.core.game_config import PLAYER_1, PLAYER_2, set_pos
from tests.helpers import make_game


def test_planes_mutually_exclusive_and_sum_to_one():
    """Me/Opp/Empty planes should be one-hot across players and empty."""
    game = make_game()

    def assert_one_hot(state_label: str):
        encoded = game.get_encoded_state(state)
        trio = encoded[:, 0:3]
        summed = np.sum(trio, axis=1)
        assert np.allclose(
            summed,
            np.ones(
                (encoded.shape[0], game.row_count, game.col_count), dtype=np.float32
            ),
        ), f"{state_label}: Me+Opp+Empty must sum to 1"
        assert np.all((trio == 0.0) | (trio == 1.0)), (
            f"{state_label}: Me/Opp/Empty must be binary"
        )

    state = game.get_initial_state()
    assert_one_hot("initial")

    state = game.get_next_state(state, (0, 0), int(state.next_player))
    state = game.get_next_state(state, (1, 1), int(state.next_player))
    assert_one_hot("midgame")


def test_padding_area_zero_for_history_planes():
    """Unused history planes should stay zero when moves < history_length."""
    game = make_game()
    moves = [(0, 0), (2, 2)]
    state = game.get_initial_state()
    for mv in moves:
        state = game.get_next_state(state, mv, int(state.next_player))

    encoded = game.get_encoded_state(state)
    history_start = 8
    used_planes = len(moves)
    zero_planes = encoded[
        :, history_start + used_planes : history_start + game.history_length
    ]
    assert np.all(zero_planes == 0.0), (
        "History padding planes should remain zero when not filled"
    )


def test_binary_feature_planes_are_0_or_1():
    """Binary planes should contain only 0 or 1 values."""
    game = make_game()
    state = game.get_initial_state()
    # Place a few stones to populate multiple binary planes.
    set_pos(state.board, 0, 0, PLAYER_1)
    set_pos(state.board, 1, 1, PLAYER_2)
    state.next_player = PLAYER_1
    state = game.get_next_state(state, (2, 2), int(state.next_player))

    encoded = game.get_encoded_state(state)
    binary_channels = [0, 1, 2, 3, 7] + list(
        range(8, 8 + game.history_length)
    )  # Me, Opp, Empty, Last, Forbidden, History
    for ch in binary_channels:
        plane = encoded[0, ch]
        assert np.all(np.isin(plane, [0.0, 1.0])), (
            f"Channel {ch} must be binary but found values {np.unique(plane)}"
        )
