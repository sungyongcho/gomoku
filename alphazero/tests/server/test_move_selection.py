"""Serving move selection and request parsing."""

import time
from types import SimpleNamespace

import numpy as np
import pytest

from gomoku.cpp_ext import gomoku_cpp
from server.engine import AlphaZeroEngine
from server.protocol import frontend_to_gamestate


def _child(visits: int, prior: float) -> SimpleNamespace:
    return SimpleNamespace(visit_count=visits, prior=prior)


def _select(children: dict) -> int:
    engine = SimpleNamespace(game=SimpleNamespace(col_count=19))
    root = SimpleNamespace(children=children)
    return AlphaZeroEngine._most_visited_action(engine, root, np.zeros(361))


def test_visit_ties_use_the_prior_instead_of_the_lowest_index() -> None:
    action = _select({(3, 0): _child(1, 0.001), (15, 9): _child(1, 0.005)})
    assert action == 15 + 9 * 19


def test_more_visits_beat_a_higher_prior() -> None:
    action = _select({(3, 0): _child(2, 0.001), (15, 9): _child(1, 0.5)})
    assert action == 3


def _payload(**extra) -> dict:
    board = [["."] * 19 for _ in range(19)]
    board[1][0] = "X"
    board[2][0] = "O"
    return {
        "nextPlayer": "X",
        "board": board,
        "lastPlay": {"coordinate": {"x": 0, "y": 2}, "stone": "O"},
        **extra,
    }


def test_recent_moves_become_newest_first_history() -> None:
    state = frontend_to_gamestate(
        _payload(recentMoves=[{"x": 0, "y": 1}, {"x": 0, "y": 2}])
    )
    assert state.history == (2 * 19, 1 * 19)


def test_requests_without_recent_moves_keep_the_last_move() -> None:
    assert frontend_to_gamestate(_payload()).history == (2 * 19,)


def test_recent_moves_must_end_with_the_last_play() -> None:
    with pytest.raises(ValueError, match="lastPlay"):
        frontend_to_gamestate(_payload(recentMoves=[{"x": 0, "y": 1}]))


def _encoded_search(time_limit_s: float, delay_s: float) -> tuple[int, list]:
    core = gomoku_cpp.GomokuCore(9, True, False, 0, 5, 0)
    engine = gomoku_cpp.MctsEngine(core, 2.0)
    size = core.action_size

    def evaluator(features, legal):
        time.sleep(delay_s)
        policy = [0.0] * size
        for index in legal:
            policy[int(index)] = 1.0
        return policy, 0.0

    visits = engine.run_mcts_encoded(
        core.initial_state(), 50, evaluator, time_limit_s=time_limit_s
    )
    return sum(count for _, count in visits), engine.root_priors()


def test_time_limit_stops_the_native_search_early() -> None:
    searched, _ = _encoded_search(time_limit_s=0.05, delay_s=0.02)
    assert 1 <= searched < 49


def test_no_time_limit_runs_every_simulation() -> None:
    searched, priors = _encoded_search(time_limit_s=0.0, delay_s=0.0)
    assert searched >= 48
    assert priors and all(prior > 0 for _, prior in priors)
