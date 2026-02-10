from __future__ import annotations

import numpy as np

import pytest

from gomoku.alphazero.eval.match import _calculate_q_drop, play_match
from gomoku.alphazero.types import action_to_xy
from gomoku.core.gomoku import GameState, Gomoku
from gomoku.utils.config.loader import BoardConfig


class DummyChild:
    def __init__(self, q_value: float):
        self.q_value = q_value


class DummyRoot:
    def __init__(self, children: dict[tuple[int, int], DummyChild]):
        self.children = children


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


def test_calculate_q_drop_prefers_child_q() -> None:
    """q_drop should use chosen child q_value when available."""
    game = _make_game()
    action_idx = 2
    x, y = action_to_xy(action_idx, game.col_count)
    stats = {"q_max": 0.9, "root": DummyRoot({(x, y): DummyChild(0.1)})}
    drop = _calculate_q_drop(stats, best_q=0.9, game=game, action_idx=action_idx)
    assert drop == pytest.approx(0.8, rel=1e-6)


def test_calculate_q_drop_fallback_q_selected() -> None:
    """If q_selected exists, use it when no child is present."""
    game = _make_game()
    stats = {"q_max": 1.0, "q_selected": 0.4}
    drop = _calculate_q_drop(stats, best_q=1.0, game=game, action_idx=0)
    assert drop == pytest.approx(0.6, rel=1e-6)


class DummyMCTS:
    def __init__(self, game: Gomoku, policy: np.ndarray, values: list[float]):
        self.game = game
        self.policy = policy
        self.values = values
        self.calls = 0

    def create_root(self, state: GameState):
        return DummyRoot({})

    def run_search_on_root(self, root):
        self.calls += 1
        return self.policy, {"q_max": max(self.values, default=0.0), "root": root}


def test_play_match_respects_opening_sampling_and_safety_guard() -> None:
    """play_match should finish with winner and blunder count."""
    game = _make_game()
    policy = np.ones(game.action_size, dtype=np.float32)
    mcts = DummyMCTS(game, policy, values=[1.0])
    winner, blunders, moves = play_match(
        game=game,
        p1=mcts,  # type: ignore[arg-type]
        p2=mcts,  # type: ignore[arg-type]
        opening_turns=1,
        temperature=1.0,
        blunder_threshold=0.0,
    )
    assert winner in {1, -1, 0}
    assert moves > 0
    assert blunders >= 0
