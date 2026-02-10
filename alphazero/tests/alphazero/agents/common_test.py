from __future__ import annotations

import numpy as np

from gomoku.alphazero.runners.common import SelfPlaySample, build_game_record, sample_action
from gomoku.alphazero.types import GameRecord
from gomoku.core.gomoku import Gomoku
from gomoku.utils.config.loader import BoardConfig


def _make_game() -> Gomoku:
    cfg = BoardConfig(
        num_lines=3,
        enable_doublethree=False,
        enable_capture=False,
        capture_goal=5,
        gomoku_goal=3,
        history_length=1,
    )
    return Gomoku(cfg)


def test_sample_action_nan_and_negative_handling() -> None:
    """sample_action should sanitize NaN/negative and fall back to argmax when sum<=0."""
    pi = np.array([np.nan, -1.0, 0.0], dtype=np.float32)
    action = sample_action(pi, turn=0, exploration_turns=5)
    assert action in {0, 1, 2}

    # When sum is zero after sanitization → argmax
    pi_zero = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    action_zero = sample_action(pi_zero, turn=10, exploration_turns=1)
    assert action_zero == int(np.argmax(pi_zero))


def test_sample_action_switches_after_exploration_turns() -> None:
    """Exploration for early turns, argmax after threshold."""
    pi = np.array([0.1, 0.9], dtype=np.float32)
    act_explore = sample_action(pi, turn=0, exploration_turns=1)
    assert act_explore in {0, 1}
    act_exploit = sample_action(pi, turn=1, exploration_turns=1)
    assert act_exploit == 1


def test_build_game_record_outcome_signs() -> None:
    """Outcomes should flip sign based on last player."""
    game = _make_game()
    s0 = game.get_initial_state()
    sample = SelfPlaySample(
        state=s0,
        policy_probs=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        move=0,
        player=1,
    )
    record = build_game_record([sample], final_value=1.0, last_player=1)
    assert isinstance(record, GameRecord)
    assert record.outcomes.tolist() == [1]

    record_lose = build_game_record([sample], final_value=1.0, last_player=2)
    assert record_lose.outcomes.tolist() == [-1]


def test_build_game_record_shapes() -> None:
    """build_game_record should stack policies and convert arrays to expected dtypes."""
    game = _make_game()
    s0 = game.get_initial_state()
    samples = [
        SelfPlaySample(
            state=s0,
            policy_probs=np.eye(3, dtype=np.float32)[i],
            move=i,
            player=1 if i % 2 == 0 else 2,
        )
        for i in range(3)
    ]
    record = build_game_record(samples, final_value=0.5, last_player=2)
    assert record.policies.shape == (3, 3)
    assert record.moves.shape == (3,)
    assert record.players.shape == (3,)
    assert record.outcomes.shape == (3,)
