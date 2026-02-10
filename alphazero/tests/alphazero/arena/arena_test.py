from __future__ import annotations

from collections import deque
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from gomoku.alphazero.eval import arena
from gomoku.alphazero.types import action_to_xy
from gomoku.utils.config.loader import (
    MctsConfig,
    RootConfig,
    SPRTConfig,
    load_and_parse_config,
)


class DummyState:
    def __init__(self, next_player: int = 1, moves: int = 0) -> None:
        self.next_player = next_player
        self.moves = moves


class DummyGame:
    def __init__(self, terminal_after: int = 2, illegal_forfeit: bool = False) -> None:
        self.col_count = 3
        self.row_count = 3
        self.action_size = self.col_count * self.row_count
        self.terminal_after = terminal_after
        self.illegal_forfeit = illegal_forfeit

    def get_initial_state(self) -> DummyState:
        return DummyState(next_player=1, moves=0)

    def get_next_state(
        self, state: DummyState, move: tuple[int, int], player: int
    ) -> DummyState:
        return DummyState(next_player=2 if player == 1 else 1, moves=state.moves + 1)

    def get_value_and_terminated(
        self, state: DummyState, move: tuple[int, int]
    ) -> tuple[float, bool]:
        if self.illegal_forfeit:
            return -1.0, True
        done = state.moves >= self.terminal_after
        return (1.0, done) if done else (0.0, False)


class DummyChild:
    def __init__(self, q_value: float) -> None:
        self.q_value = q_value


class DummyRoot:
    def __init__(self) -> None:
        self.children: dict[tuple[int, int], DummyChild] = {}


class DummyMCTS:
    def __init__(self, action_size: int, col_count: int, q_values: list[float] | None):
        self.action_size = action_size
        self.col_count = col_count
        self._q_values = q_values

    def create_root(self, state: Any) -> DummyRoot:
        return DummyRoot()

    def run_search_on_root(self, root: DummyRoot) -> tuple[np.ndarray, dict]:
        policy = np.zeros(self.action_size, dtype=np.float32)
        policy[0] = 1.0
        q_values = (
            self._q_values
            if self._q_values is not None
            else [0.0 for _ in range(self.action_size)]
        )
        for idx, q in enumerate(q_values):
            x, y = action_to_xy(idx, self.col_count)
            root.children[(x, y)] = DummyChild(q_value=float(q))
        stats = {"q_max": max(q_values) if q_values else 0.0, "root": root}
        return policy, stats


def _make_cfg() -> RootConfig:
    cfg = load_and_parse_config("configs/config_alphazero_test.yaml")
    cfg = cfg.model_copy(
        update={
            "mcts": MctsConfig(
                C=cfg.mcts.C,
                num_searches=1,
                exploration_turns=0,
                dirichlet_epsilon=0.0,
                dirichlet_alpha=cfg.mcts.dirichlet_alpha,
                batch_infer_size=1,
                max_batch_wait_ms=0,
                min_batch_size=1,
            )
        }
    )
    return cfg


def test_arena_promote_flag_by_winrate_and_baseline(monkeypatch: pytest.MonkeyPatch):
    """Verify promotion flag is set by win rate, baseline, and blunder limits."""
    cfg = _make_cfg()
    eval_cfg = cfg.evaluation.model_copy(
        update={
            "promotion_win_rate": 0.55,
            "baseline_wr_min": 0.5,
            "blunder_increase_limit": 0.2,
            "num_baseline_games": 2,
        }
    )

    def _make_metrics(
        win_rate: float, blunder_rate: float, games: int = 10
    ) -> arena.H2HMetrics:
        wins = int(round(win_rate * games))
        losses = games - wins
        metrics = arena.H2HMetrics(wins=wins, losses=losses, draws=0)
        metrics.blunders = int(round(blunder_rate * games))
        metrics.moves = games
        metrics.lengths = [1 for _ in range(games)]
        return metrics

    metrics_seq = deque([_make_metrics(0.6, 0.1), _make_metrics(0.7, 0.05)])
    monkeypatch.setattr(
        arena, "_run_duel", lambda *args, **kwargs: metrics_seq.popleft()
    )

    result = arena.run_arena(
        cfg=cfg,
        inference_new=MagicMock(),
        inference_best=MagicMock(),
        eval_cfg=eval_cfg,
        matches=2,
        baseline_inference=MagicMock(),
    )

    print(f"[promote] summary={result}")
    assert result["promote"] is True
    assert result["pass_promotion"] is True
    assert result["pass_baseline"] is True
    assert result["pass_blunder"] is True


def test_arena_sprt_early_stop_accept_reject(monkeypatch: pytest.MonkeyPatch):
    """Ensure SPRT accept decision stops further games early."""
    cfg = _make_cfg()
    eval_cfg = cfg.evaluation.model_copy(
        update={
            "use_sprt": True,
            "sprt": SPRTConfig(
                p0=0.45,
                p1=0.55,
                alpha=0.05,
                beta=0.05,
                max_games=10,
                ignore_draws=False,
            ),
        }
    )

    game_calls = {"n": 0}

    def _fake_play_game(*args, **kwargs):
        game_calls["n"] += 1
        return 1, 0, 1

    call_counts = {"sprt": 0}

    def _fake_sprt(cfg_obj, wins, losses, draws):
        call_counts["sprt"] += 1
        return "accept_h1"

    monkeypatch.setattr(arena, "play_match", _fake_play_game)
    monkeypatch.setattr(
        arena,
        "_build_eval_mcts",
        lambda *args, **kwargs: DummyMCTS(action_size=9, col_count=3, q_values=None),
    )
    monkeypatch.setattr(arena, "Gomoku", lambda *args, **kwargs: DummyGame())
    monkeypatch.setattr(arena, "check_sprt", _fake_sprt)

    metrics = arena._run_duel(
        cfg=cfg,
        eval_cfg=eval_cfg,
        inference_a=MagicMock(),
        inference_b=MagicMock(),
        games=5,
        opening_turns=0,
        temperature=1.0,
        blunder_th=0.5,
    )

    print(
        f"[sprt] games_played={metrics.total_games}, sprt_calls={call_counts['sprt']}"
    )
    assert metrics.total_games == 1
    assert call_counts["sprt"] == 1


def test_arena_blunder_qdrop_uses_chosen_action(monkeypatch: pytest.MonkeyPatch):
    """Check blunder q_drop uses the Q-value of the chosen action."""
    dummy_game = DummyGame(terminal_after=1)
    q_values = [0.1, 0.9, 0.8]
    mcts = DummyMCTS(action_size=dummy_game.action_size, col_count=3, q_values=q_values)

    winner, blunders, moves = arena.play_match(
        game=dummy_game,
        p1=mcts,
        p2=mcts,
        opening_turns=0,
        temperature=1.0,
        blunder_threshold=0.05,
    )

    print(f"[blunder] winner={winner}, blunders={blunders}, moves={moves}")
    assert winner == 1
    assert blunders == 1
    assert moves == 1


def test_arena_mcts_recreated_per_game(monkeypatch: pytest.MonkeyPatch):
    """Confirm a fresh MCTS instance is built for every game."""
    cfg = _make_cfg()
    eval_cfg = cfg.evaluation

    build_calls = {"n": 0}

    def _build(*args, **kwargs):
        build_calls["n"] += 1
        return DummyMCTS(action_size=9, col_count=3, q_values=None)

    monkeypatch.setattr(arena, "_build_eval_mcts", _build)
    monkeypatch.setattr(arena, "play_match", lambda *args, **kwargs: (1, 0, 2))
    monkeypatch.setattr(arena, "Gomoku", lambda *args, **kwargs: DummyGame())

    metrics = arena._run_duel(
        cfg=cfg,
        eval_cfg=eval_cfg,
        inference_a=MagicMock(),
        inference_b=MagicMock(),
        games=3,
        opening_turns=0,
        temperature=1.0,
        blunder_th=0.5,
    )

    print(f"[recreate] build_calls={build_calls['n']}, games={metrics.total_games}")
    assert build_calls["n"] == 6  # 두 플레이어 × 게임 수
    assert metrics.total_games == 3


def test_arena_swaps_colors_fairly(monkeypatch: pytest.MonkeyPatch):
    """Verify colors swap between games so wins/losses balance by turn order."""
    cfg = _make_cfg()
    eval_cfg = cfg.evaluation

    monkeypatch.setattr(
        arena, "_build_eval_mcts", lambda *args, **kwargs: DummyMCTS(9, 3, None)
    )
    monkeypatch.setattr(arena, "play_match", lambda *args, **kwargs: (1, 0, 1))
    monkeypatch.setattr(arena, "Gomoku", lambda *args, **kwargs: DummyGame())
    metrics = arena._run_duel(
        cfg,
        eval_cfg,
        MagicMock(),
        MagicMock(),
        games=2,
        opening_turns=0,
        temperature=1.0,
        blunder_th=0.5,
    )

    print(f"[swap] wins={metrics.wins}, losses={metrics.losses}")
    assert metrics.wins == 1
    assert metrics.losses == 1


def test_sprt_calculation_includes_draws_correctly(monkeypatch: pytest.MonkeyPatch):
    """Ensure draws are included correctly in SPRT calculations."""
    cfg = _make_cfg()
    eval_cfg = cfg.evaluation.model_copy(
        update={
            "use_sprt": True,
            "sprt": SPRTConfig(
                p0=0.45,
                p1=0.55,
                alpha=0.05,
                beta=0.05,
                max_games=3,
                ignore_draws=False,
            ),
        }
    )

    calls: list[tuple[int, int, int]] = []

    def _spy_sprt(cfg_obj, wins, losses, draws):
        calls.append((wins, losses, draws))
        return "continue"

    monkeypatch.setattr(arena, "check_sprt", _spy_sprt)
    monkeypatch.setattr(
        arena, "_build_eval_mcts", lambda *args, **kwargs: DummyMCTS(9, 3, None)
    )
    monkeypatch.setattr(arena, "play_match", lambda *args, **kwargs: (0, 0, 1))
    monkeypatch.setattr(arena, "Gomoku", lambda *args, **kwargs: DummyGame())

    arena._run_duel(
        cfg,
        eval_cfg,
        MagicMock(),
        MagicMock(),
        games=2,
        opening_turns=0,
        temperature=1.0,
        blunder_th=0.5,
    )

    print(f"[sprt_draw] calls={calls}")
    assert calls
    wins, losses, draws = calls[-1]
    assert draws > 0 and wins == 0 and losses == 0


def test_eval_handles_illegal_move_as_forfeit(monkeypatch: pytest.MonkeyPatch):
    """Ensure illegal moves are treated as immediate forfeits."""
    dummy_game = DummyGame(terminal_after=1, illegal_forfeit=True)
    mcts = DummyMCTS(action_size=dummy_game.action_size, col_count=3, q_values=None)

    winner, blunders, moves = arena.play_match(
        game=dummy_game,
        p1=mcts,
        p2=mcts,
        opening_turns=0,
        temperature=1.0,
        blunder_threshold=0.5,
    )

    print(f"[illegal] winner={winner}, blunders={blunders}, moves={moves}")
    assert winner == -1
    assert blunders == 0
    assert moves == 1
