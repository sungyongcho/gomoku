from __future__ import annotations

import numpy as np
import pytest

from gomoku.alphazero.agent import AlphaZeroAgent, _apply_temperature
from gomoku.core.gomoku import Gomoku
from gomoku.utils.config.loader import BoardConfig, MctsConfig


class DummyMCTS:
    def __init__(self, policies: list[np.ndarray]):
        self.policies = policies
        self.calls = 0
        self.forward_calls: list[int] = []

    def create_root(self, state):
        return type("Root", (), {"state": state, "children": {}, "parent": None})()

    def run_search(self, roots, add_noise: bool = True):
        idx = self.calls % len(self.policies)
        self.calls += 1
        return [(self.policies[idx], {})]

    def forward(self, action: int) -> None:
        self.forward_calls.append(action)

    def reset(self) -> None:
        self.calls = 0
        self.forward_calls.clear()


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


def _make_mcts_cfg() -> MctsConfig:
    return MctsConfig(
        C=1.5,
        num_searches=1,
        exploration_turns=1,
        dirichlet_epsilon=0.0,
        dirichlet_alpha=0.3,
        batch_infer_size=1,
        max_batch_wait_ms=0,
        min_batch_size=1,
    )


def test_apply_temperature_argmax_when_low_temp() -> None:
    game = _make_game()
    mcts_cfg = _make_mcts_cfg()
    counts = np.array([1, 10, 2], dtype=np.float32)
    pi = _apply_temperature(counts, temperature=1e-6)
    assert pi.argmax() == 1
    assert pi[1] > 0.99


def test_apply_temperature_normalized_at_temp_one() -> None:
    probs = np.array([0.5, 0.5], dtype=np.float32)
    pi = _apply_temperature(probs, temperature=1.0)
    np.testing.assert_allclose(pi, [0.5, 0.5], atol=1e-6)


def test_get_action_probs_masks_illegal_moves(monkeypatch) -> None:
    game = _make_game()
    mcts_cfg = _make_mcts_cfg()
    # occupy center
    state = game.get_initial_state()
    cx, cy = game.col_count // 2, game.row_count // 2
    state = game.get_next_state(state, (cx, cy), state.next_player)

    policy = np.ones(game.action_size, dtype=np.float32)
    center = cy * game.col_count + cx
    policy[center] = 0.0
    policy /= policy.sum()
    dummy_mcts = DummyMCTS([policy])
    agent = AlphaZeroAgent(game, mcts_cfg, inference_client=None, engine_type="sequential")  # type: ignore[arg-type]
    agent.mcts = dummy_mcts  # type: ignore[assignment]

    pi = agent.get_action_probs(state, temperature=1.0, add_noise=False)
    mask_idx = cy * game.col_count + cx
    assert pi[mask_idx] == 0.0
    assert np.isclose(pi.sum(), 1.0, atol=1e-6)


def test_update_root_moves_to_child_or_resets() -> None:
    game = _make_game()
    mcts_cfg = _make_mcts_cfg()
    policy = np.ones(game.action_size, dtype=np.float32) / game.action_size
    dummy_mcts = DummyMCTS([policy])
    agent = AlphaZeroAgent(game, mcts_cfg, inference_client=None, engine_type="sequential")  # type: ignore[arg-type]
    agent.mcts = dummy_mcts  # type: ignore[assignment]

    state = game.get_initial_state()
    pi = agent.get_action_probs(state, temperature=1.0, add_noise=False)
    action = pi.argmax()
    agent.update_root(action)
    # child not present -> root reset to None
    assert agent.roots[0] is None

    # invalid action should reset roots list length to 1
    agent.update_root(game.action_size + 1)
    assert len(agent.roots) == 1
