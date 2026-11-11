"""Ladder ratings, ladder games and checkpoint averaging."""

from __future__ import annotations

import pytest
import torch

from gomoku.alphazero.eval import reference_match as rm
from gomoku.core.gomoku import Gomoku
from gomoku.model.policy_value_net import PolicyValueNet
from gomoku.alphazero.eval.ladder import fit_ratings, play_ladder_game, summarize_ladder
from gomoku.utils.state_dict_utils import average_state_dicts
from gomoku.utils.config.loader import BoardConfig, MctsConfig, ModelConfig


def test_ratings_order_players_by_results_and_centre_on_zero() -> None:
    games = (
        [("strong", "middle", 1.0)] * 15
        + [("strong", "middle", 0.0)] * 5
        + [("middle", "weak", 1.0)] * 15
        + [("middle", "weak", 0.0)] * 5
        + [("strong", "weak", 1.0)] * 18
        + [("strong", "weak", 0.0)] * 2
    )

    ratings = fit_ratings(games)

    assert ratings["strong"] > ratings["middle"] > ratings["weak"]
    assert sum(ratings.values()) == pytest.approx(0.0, abs=1e-6)


def test_equal_results_give_equal_ratings_and_a_sweep_stays_finite() -> None:
    even = fit_ratings([("a", "b", 1.0), ("a", "b", 0.0), ("a", "b", 0.5)])
    assert even["a"] == pytest.approx(even["b"], abs=1e-6)

    sweep = fit_ratings([("a", "b", 1.0)] * 10)
    assert sweep["a"] > sweep["b"]
    assert all(abs(value) < 2000 for value in sweep.values())


def test_averaging_takes_the_mean_of_floats_and_the_last_integer() -> None:
    first = {"w": torch.tensor([1.0, 3.0]), "n": torch.tensor(5)}
    second = {"w": torch.tensor([3.0, 5.0]), "n": torch.tensor(9)}

    averaged = average_state_dicts([first, second])

    assert torch.equal(averaged["w"], torch.tensor([2.0, 4.0]))
    assert averaged["w"].dtype == torch.float32
    assert int(averaged["n"]) == 9


def test_averaging_rejects_checkpoints_with_different_tensors() -> None:
    with pytest.raises(ValueError):
        average_state_dicts([{"w": torch.zeros(1)}, {"v": torch.zeros(1)}])


def _side(game: Gomoku, model_cfg: ModelConfig, seed: int):
    torch.manual_seed(seed)
    weights = PolicyValueNet(game, model_cfg, torch.device("cpu")).state_dict()
    mcts = MctsConfig(
        C=2.0,
        num_searches=8,
        exploration_turns=0,
        dirichlet_epsilon=0.0,
        dirichlet_alpha=0.03,
        batch_infer_size=1,
        max_batch_wait_ms=0,
        min_batch_size=1,
        use_native=True,
        fpu_from_parent=True,
    )
    return model_cfg, rm.match_mcts(mcts, 8), weights


def test_a_ladder_game_names_both_sides_and_stops_at_the_ply_limit() -> None:
    board = BoardConfig(
        num_lines=9,
        enable_doublethree=False,
        enable_capture=False,
        capture_goal=5,
        gomoku_goal=5,
        history_length=1,
    )
    game = Gomoku(board, use_native=True)
    planes = int(game.get_encoded_state(game.get_initial_state()).shape[1])
    model_cfg = ModelConfig(num_planes=planes, num_hidden=8, num_resblocks=1)
    sides = {"old": _side(game, model_cfg, 1), "new": _side(game, model_cfg, 2)}

    result = play_ladder_game(board, sides, "new", 3, opening_plies=2, max_plies=6)

    assert (result["black"], result["white"]) == ("new", "old")
    assert result["plies"] <= 6
    assert result["winner"] in ("old", "new", None)
    capped = play_ladder_game(board, sides, "old", 3, opening_plies=2, max_plies=3)
    assert capped["plies"] == 3 and capped["winner"] is None


def test_ladder_summary_counts_pairs_points_and_ratings() -> None:
    games = [
        {"a": "new", "b": "old", "winner": "new", "plies": 20},
        {"a": "new", "b": "old", "winner": "new", "plies": 30},
        {"a": "new", "b": "old", "winner": None, "plies": 400},
        {"a": "new", "b": "old", "winner": "old", "plies": 10},
    ]

    summary = summarize_ladder(games)

    assert summary["games"] == 4
    assert summary["pairs"] == {"new vs old": [2.5, 4.0]}
    assert summary["points"] == {"new": [2.5, 4.0], "old": [1.5, 4.0]}
    assert summary["ratings"]["new"] > summary["ratings"]["old"]
    assert summary["mean_plies"] == 115.0
