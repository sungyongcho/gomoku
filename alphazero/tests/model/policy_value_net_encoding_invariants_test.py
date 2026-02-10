"""Encoding invariants for PolicyValueNet inputs."""

import numpy as np
import torch

from gomoku.core.game_config import PLAYER_1, PLAYER_2
from tests.helpers import log_section, make_model


def test_encoding_invariants_and_constraints():
    """Me/Opp/Empty sum-to-1, binary planes, determinism, batch isolation, finite outputs."""  # noqa: E501
    log_section("PolicyValueNet - Encoding invariants")
    model, game, config = make_model()

    # Build two states: one empty, one with a couple of moves.
    empty_state = game.get_initial_state()
    state = game.get_next_state(empty_state, (0, 0), PLAYER_1)
    state = game.get_next_state(state, (1, 1), PLAYER_2)

    encoded = game.get_encoded_state(state)
    h, w = game.row_count, game.col_count
    expected_channels = 8 + game.history_length
    assert encoded.shape == (1, expected_channels, h, w)
    assert encoded.shape[1] == config.num_planes

    # Me/Opp/Empty invariants.
    trio = encoded[0, 0:3]
    summed = np.sum(trio, axis=0)
    assert np.allclose(summed, np.ones((h, w), dtype=np.float32))

    def _assert_binary_plane(plane: np.ndarray, name: str) -> None:
        unique = np.unique(np.round(plane, decimals=6))
        assert set(unique).issubset({0.0, 1.0}), (
            f"{name} has non-binary values: {unique}"
        )

    # Binary planes (me, opp, empty, last move, forbidden, history).
    binary_channels = [0, 1, 2, 3, 7] + list(range(8, 8 + game.history_length))
    for ch in binary_channels:
        _assert_binary_plane(encoded[0, ch], f"channel {ch}")

    # Last move plane one-hot.
    last_plane = encoded[0, 3]
    assert np.isclose(last_plane.sum(), 1.0)
    assert last_plane[1, 1] == 1.0  # last move at (1,1)
    empty_encoded = game.get_encoded_state(empty_state)
    assert np.all(empty_encoded[0, 3] == 0.0)
    print("Me plane:\n", encoded[0, 0].astype(int))
    print("Opp plane:\n", encoded[0, 1].astype(int))
    print("Empty plane:\n", encoded[0, 2].astype(int))
    print("Last plane:\n", encoded[0, 3].astype(int))
    print("Color plane unique:", np.unique(encoded[0, 6]))
    print(
        "Capture planes (my/opp) unique:",
        np.unique(encoded[0, 4]),
        np.unique(encoded[0, 5]),
    )
    print("Forbidden plane nonzero coords:", np.argwhere(encoded[0, 7] == 1).tolist())
    history_nonzero = {
        k: np.argwhere(encoded[0, 8 + k] == 1).tolist()
        for k in range(game.history_length)
        if np.any(encoded[0, 8 + k])
    }
    print("History nonzero coords per plane:", history_nonzero)

    # Determinism.
    encoded_again = game.get_encoded_state(state)
    assert np.array_equal(encoded, encoded_again)

    # Batch isolation: ensure two different states stay distinct.
    batch = game.get_encoded_state([state, empty_state])
    me_state, me_empty = batch[:, 0]
    assert np.any(me_state != me_empty)
    assert np.all(me_empty == 0.0)

    # Finite values only.
    assert np.isfinite(encoded).all()

    # Model forward with encoded inputs should produce finite outputs.
    x = torch.tensor(encoded)
    with torch.no_grad():
        policy_logits, value = model(x)
    assert policy_logits.shape[0] == 1
    assert value.shape == (1, 1)
    assert torch.isfinite(policy_logits).all()
    assert torch.isfinite(value).all()
