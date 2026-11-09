"""Clean self-play targets, tie-breaking, root noise and evaluation settings."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gomoku.alphazero.agent import AlphaZeroAgent
from gomoku.alphazero.eval import arena
from gomoku.alphazero.eval.match import _random_opening
from gomoku.alphazero.runners.common import SelfPlaySample, build_game_record, sample_action
from gomoku.alphazero.runners.selfplay import SelfPlayRunner
from gomoku.alphazero.runners.vectorize_runner import VectorizeRunner
from gomoku.alphazero.types import greedy_action
from gomoku.core.gomoku import Gomoku
from gomoku.cpp_ext import gomoku_cpp
from gomoku.pvmcts.pvmcts import PVMCTS
import gomoku.pvmcts.search.engine as search_engine
from gomoku.utils.config.loader import (
    BoardConfig,
    EvaluationConfig,
    MctsConfig,
    TrainingConfig,
)

SIZE = 9


def _board(lines: int = SIZE) -> BoardConfig:
    return BoardConfig(
        num_lines=lines,
        enable_doublethree=False,
        enable_capture=False,
        capture_goal=5,
        gomoku_goal=5,
        history_length=1,
    )


def _mcts(**overrides) -> MctsConfig:
    settings = {
        "C": 2.0,
        "num_searches": 16,
        "exploration_turns": 0,
        "dirichlet_epsilon": 0.5,
        "dirichlet_alpha": 0.03,
        "batch_infer_size": 1,
        "max_batch_wait_ms": 0,
        "min_batch_size": 1,
        "use_native": True,
        "fpu_from_parent": True,
    }
    return MctsConfig(**{**settings, **overrides})


class _UniformInference:
    device = torch.device("cpu")

    def infer(self, states, native_payload=None, model_slot=None):
        batch = states.unsqueeze(0) if states.dim() == 3 else states
        return torch.zeros(batch.size(0), SIZE * SIZE), torch.zeros(batch.size(0), 1)


class _FirstMoveAgent:
    """Agent stub that always plays the first legal move with a flat search policy."""

    def __init__(self, game: Gomoku) -> None:
        self.game = game

    def reset(self) -> None: ...

    def update_root(self, action: int) -> None: ...

    def get_action_probs(self, state, temperature, add_noise=False):
        self.game.get_legal_moves(state)
        pi = np.zeros(self.game.action_size, dtype=np.float32)
        pi[state.legal_indices_cache.astype(int)[0]] = 1.0
        return pi


def _train_cfg() -> TrainingConfig:
    return TrainingConfig(
        num_iterations=1,
        num_selfplay_iterations=1,
        num_epochs=1,
        batch_size=1,
        learning_rate=0.1,
        weight_decay=0.0,
        temperature=1.0,
        replay_buffer_size=10,
        min_samples_to_train=1,
    )


def test_random_opening_moves_are_not_training_targets() -> None:
    game = Gomoku(_board())
    runner = SelfPlayRunner(game, _mcts(use_native=False), _train_cfg())
    record = runner.play_one_game(
        _FirstMoveAgent(game), temperature=1.0, add_noise=False, random_opening_turns=4
    )
    stones = [int((state.board != 0).sum()) for state in record.states_raw]
    assert stones[0] == 4
    assert stones == sorted(stones)
    assert len(record.policies) == len(record.outcomes) == len(stones)


def test_a_game_with_no_training_targets_gives_an_empty_record() -> None:
    game = Gomoku(_board())
    state = game.get_initial_state()
    pi = np.zeros(game.action_size, dtype=np.float32)
    record = build_game_record(
        [SelfPlaySample(state, pi, 0, 1, is_training_target=False)], 1.0, 1
    )
    assert record.states_raw == []
    assert len(record.outcomes) == 0


def test_visit_ties_are_not_resolved_by_lowest_index() -> None:
    pi = np.array([0.0, 0.5, 0.0, 0.5], dtype=np.float32)
    np.random.seed(0)
    chosen = {sample_action(pi, turn=5, exploration_turns=1) for _ in range(50)}
    assert chosen == {1, 3}


def test_greedy_action_keeps_a_clear_maximum_and_all_zero_fallback() -> None:
    assert greedy_action(np.array([0.1, 0.9, 0.0])) == 1
    assert greedy_action(np.zeros(3)) == 0


@pytest.mark.parametrize("batch", [1, 8])
def test_root_noise_reaches_native_search(batch: int) -> None:
    game = Gomoku(_board(), use_native=True)
    search = PVMCTS(game, _mcts(batch_infer_size=batch), _UniformInference(), mode="sequential")
    np.random.seed(0)
    torch.manual_seed(0)
    search.run_search([search.create_root(game.get_initial_state())], add_noise=True)
    priors = [prior for _, prior in search.engine._native_engine.root_priors()]
    assert max(priors) > 10 * (1 / len(priors))


def test_no_noise_was_requested_keeps_the_uniform_root() -> None:
    game = Gomoku(_board(), use_native=True)
    search = PVMCTS(game, _mcts(batch_infer_size=8), _UniformInference(), mode="sequential")
    search.run_search([search.create_root(game.get_initial_state())], add_noise=False)
    priors = [prior for _, prior in search.engine._native_engine.root_priors()]
    assert max(priors) == pytest.approx(min(priors))


def _top_visits_lost_position(batch: int) -> tuple[int, int]:
    core = gomoku_cpp.GomokuCore(SIZE, True, False, 0, 5, 0)
    engine = gomoku_cpp.MctsEngine(core, 2.0, True)

    def values(features: np.ndarray) -> np.ndarray:
        first_to_move = features[:, 6, 0, 0] > 0
        has_marker = features[:, 0, 1, 1] + features[:, 1, 1, 1] > 0
        magnitude = np.where(has_marker, 0.2, 0.8)
        return np.where(first_to_move, -magnitude, magnitude).reshape(-1, 1)

    def evaluator(batch_features: np.ndarray):
        return np.zeros((len(batch_features), core.action_size)), values(batch_features)

    visits = engine.run_mcts_batch_encoded(core.initial_state(), 200, batch, evaluator)
    return max(visits, key=lambda item: item[1])


def test_batched_search_concentrates_on_the_better_move() -> None:
    move, count = _top_visits_lost_position(8)
    assert move == 1 + 1 * SIZE
    assert count > 50


def test_random_openings_repeat_for_the_same_seed() -> None:
    game = Gomoku(_board(), use_native=True)
    first = _random_opening(game, 4, seed=3)
    second = _random_opening(game, 4, seed=3)
    other = _random_opening(game, 4, seed=4)
    assert int((first.board != 0).sum()) == 4
    assert np.array_equal(first.board, second.board)
    assert not np.array_equal(first.board, other.board)


def test_evaluation_search_keeps_the_serving_settings() -> None:
    cfg = SimpleNamespace(board=_board(), mcts=_mcts(num_searches=8))
    eval_cfg = EvaluationConfig(
        num_eval_games=2,
        eval_every_iters=1,
        promotion_win_rate=0.55,
        num_baseline_games=0,
        blunder_threshold=0.5,
        initial_blunder_rate=0,
        initial_baseline_win_rate=0,
        blunder_increase_limit=1,
        baseline_wr_min=0,
        random_play_ratio=0.0,
        eval_num_searches=32,
        eval_dirichlet_epsilon=0.0,
    )
    search = arena._build_eval_mcts(cfg, _UniformInference(), eval_cfg)
    assert search.params.use_native is True
    assert search.params.fpu_from_parent is True
    assert search.params.num_searches == 32
    assert search.params.dirichlet_epsilon == 0.0
    assert isinstance(search, PVMCTS)


def _top_visits_with_equal_values(fpu_reduction: float) -> int:
    core = gomoku_cpp.GomokuCore(SIZE, True, False, 0, 5, 0)
    engine = gomoku_cpp.MctsEngine(core, 2.0, True, fpu_reduction)

    def evaluator(features, legal):
        return [0.0] * core.action_size, 0.0

    visits = engine.run_mcts_encoded(core.initial_state(), 60, evaluator)
    return max(count for _, count in visits)


def test_without_fpu_reduction_equal_values_spread_the_search() -> None:
    assert _top_visits_with_equal_values(0.0) == 1


def test_fpu_reduction_revisits_tried_moves_when_values_are_equal() -> None:
    assert _top_visits_with_equal_values(0.2) > 1


def _count_noise_draws(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    draws = [0]
    original = search_engine._safe_dirichlet_noise

    def counted(*args, **kwargs):
        draws[0] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(search_engine, "_safe_dirichlet_noise", counted)
    return draws


def _native_agent(game: Gomoku) -> AlphaZeroAgent:
    return AlphaZeroAgent(
        game, _mcts(batch_infer_size=8), _UniformInference(), engine_type="sequential"
    )


def _stones(state) -> int:
    return int((state.board != 0).sum())


def test_batched_self_play_draws_root_noise_on_every_searched_move(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    game = Gomoku(_board(), use_native=True)
    draws = _count_noise_draws(monkeypatch)
    runner = VectorizeRunner(game, _mcts(batch_infer_size=8), _train_cfg())

    records = runner.play_batch_games(
        _native_agent(game), batch_size=1, add_noise=True, random_opening_turns=2
    )

    searched = len(records[0].moves)
    assert searched > 2
    assert _stones(records[0].states_raw[0]) == 2
    assert draws[0] == searched


def _spy_on_searches(agent: AlphaZeroAgent) -> list[int]:
    """Record the stone count of every position the agent is asked to search."""
    searched: list[int] = []
    original = agent.get_action_probs_batch

    def spy(states_raw, temperature, add_noise_flags, active_indices=None):
        active = range(len(states_raw)) if active_indices is None else active_indices
        searched.extend(_stones(states_raw[i]) for i in active)
        return original(states_raw, temperature, add_noise_flags, active_indices)

    agent.get_action_probs_batch = spy
    return searched


@pytest.mark.parametrize("agent_first", [True, False])
def test_random_opening_covers_both_sides_and_is_not_searched(
    agent_first: bool,
) -> None:
    game = Gomoku(_board(), use_native=True)
    agent, opponent = _native_agent(game), _native_agent(game)
    agent_searches = _spy_on_searches(agent)
    opponent_searches = _spy_on_searches(opponent)
    runner = VectorizeRunner(game, _mcts(batch_infer_size=8), _train_cfg())

    record = runner.play_batch_games(
        agent,
        batch_size=1,
        add_noise=False,
        random_opening_turns=4,
        opponent=opponent,
        agent_first=agent_first,
        opponent_tag="external_bot",
    )[0]

    assert agent_searches and opponent_searches
    assert min(agent_searches + opponent_searches) == 4
    assert _stones(record.states_raw[0]) == 4
    assert record.opponent == "external_bot"
    assert record.agent_player == (1 if agent_first else 2)


def test_sequential_random_opening_covers_the_opponent_too() -> None:
    game = Gomoku(_board())
    asked: list[int] = []

    class _RecordingOpponent(_FirstMoveAgent):
        def get_action_probs(self, state, temperature, add_noise=False):
            asked.append(_stones(state))
            return super().get_action_probs(state, temperature, add_noise)

    runner = SelfPlayRunner(game, _mcts(use_native=False), _train_cfg())
    record = runner.play_one_game(
        _FirstMoveAgent(game),
        temperature=1.0,
        add_noise=False,
        random_opening_turns=4,
        opponent=_RecordingOpponent(game),
        opponent_tag="prev_bot",
    )

    assert asked and min(asked) >= 4
    assert _stones(record.states_raw[0]) == 4
    assert (record.opponent, record.agent_player) == ("prev_bot", 1)


@pytest.mark.parametrize(
    ("final_value", "last_player", "winner"), [(1.0, 2, 2), (-1.0, 2, 1), (0.0, 1, 0)]
)
def test_game_record_names_the_winner_and_the_opponent(
    final_value: float, last_player: int, winner: int
) -> None:
    game = Gomoku(_board())
    pi = np.zeros(game.action_size, dtype=np.float32)
    memory = [SelfPlaySample(game.get_initial_state(), pi, 0, last_player)]

    record = build_game_record(
        memory, final_value, last_player, opponent="external_bot", agent_player=1
    )

    assert record.winner == winner
    assert (record.opponent, record.agent_player) == ("external_bot", 1)


def test_self_play_records_carry_no_opponent() -> None:
    game = Gomoku(_board())
    pi = np.zeros(game.action_size, dtype=np.float32)
    memory = [SelfPlaySample(game.get_initial_state(), pi, 0, 1)]
    record = build_game_record(memory, 1.0, 1)
    assert (record.opponent, record.agent_player, record.winner) == ("self", 0, 1)
