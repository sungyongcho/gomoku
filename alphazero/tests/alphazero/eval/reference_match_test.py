"""Champion versus fixed reference model match used to track absolute strength across a run."""

import pytest
import torch
import yaml

from gomoku.alphazero.eval import reference_match as rm
from gomoku.core.gomoku import Gomoku
from gomoku.model.policy_value_net import PolicyValueNet
from gomoku.utils.config.loader import (
    BoardConfig,
    MctsConfig,
    ModelConfig,
    ReferenceMatchConfig,
    parse_config,
)
from tests.helpers import minimal_raw_config

SIZE = 9


def _board() -> BoardConfig:
    return BoardConfig(
        num_lines=SIZE,
        enable_doublethree=True,
        enable_capture=True,
        capture_goal=5,
        gomoku_goal=5,
        history_length=1,
    )


def _side(seed: int):
    game = Gomoku(_board())
    planes = int(game.get_encoded_state(game.get_initial_state()).shape[1])
    model_cfg = ModelConfig(num_planes=planes, num_hidden=8, num_resblocks=1)
    torch.manual_seed(seed)
    weights = PolicyValueNet(game, model_cfg, torch.device("cpu")).state_dict()
    mcts = rm.match_mcts(
        MctsConfig(
            C=2.0,
            num_searches=4,
            exploration_turns=0,
            dirichlet_epsilon=0.0,
            dirichlet_alpha=0.03,
            batch_infer_size=1,
            min_batch_size=1,
            max_batch_wait_ms=0,
            use_native=True,
        ),
        8,
    )
    return model_cfg, mcts, weights


def test_match_search_has_a_fixed_budget_without_noise():
    mcts = MctsConfig(
        C=2.0,
        num_searches=100000,
        time_limit_s=2.5,
        exploration_turns=8,
        dirichlet_epsilon=0.25,
        dirichlet_alpha=0.03,
        batch_infer_size=8,
        min_batch_size=1,
        max_batch_wait_ms=0,
    )
    out = rm.match_mcts(mcts, 400)
    assert out.num_searches == 400
    assert out.time_limit_s is None
    assert out.batch_infer_size == 1
    assert out.dirichlet_epsilon == 0.0
    assert out.exploration_turns == 0


def test_game_ends_with_a_result_from_the_champion_side():
    sides = {rm.CHAMPION: _side(0), rm.REFERENCE: _side(1)}
    result = rm.play_game(_board(), sides, rm.CHAMPION, opening_seed=3, opening_plies=2)
    assert result["black"] == rm.CHAMPION
    assert result["winner"] in (rm.CHAMPION, rm.REFERENCE, None)
    assert result["plies"] > 2


def test_same_opening_seed_repeats_the_game():
    sides = {rm.CHAMPION: _side(0), rm.REFERENCE: _side(1)}
    first = rm.play_game(_board(), sides, rm.REFERENCE, opening_seed=5, opening_plies=2)
    second = rm.play_game(_board(), sides, rm.REFERENCE, opening_seed=5, opening_plies=2)
    assert first == second


def test_summary_counts_colours_and_half_point_draws():
    results = [
        {"black": rm.CHAMPION, "winner": rm.CHAMPION, "plies": 10},
        {"black": rm.REFERENCE, "winner": rm.REFERENCE, "plies": 20},
        {"black": rm.CHAMPION, "winner": None, "plies": 30},
        {"black": rm.REFERENCE, "winner": rm.CHAMPION, "plies": 40},
    ]
    s = rm.summarize(results)
    assert (s["wins"], s["losses"], s["draws"]) == (2, 1, 1)
    assert s["score"] == pytest.approx(2.5 / 4)
    assert (s["wins_as_black"], s["games_as_black"]) == (1, 2)
    assert (s["wins_as_white"], s["games_as_white"]) == (1, 2)
    assert s["mean_plies"] == pytest.approx(25.0)


def test_reference_with_other_board_rules_is_rejected(tmp_path):
    raw = minimal_raw_config(num_lines=15)
    path = tmp_path / "reference.yaml"
    path.write_text(yaml.safe_dump(raw))
    model_cfg, mcts, weights = _side(0)
    cfg = ReferenceMatchConfig(checkpoint="unused.pt", config=str(path))
    with pytest.raises(ValueError):
        rm.run_reference_match(_board(), model_cfg, mcts, weights, cfg, iteration=10)


def test_evaluation_config_accepts_a_reference_block():
    raw = minimal_raw_config()
    raw["evaluation"]["reference"] = {
        "name": "v4",
        "checkpoint": "gs://b/v4.pt",
        "config": "configs/deploy.yaml",
    }
    parsed = parse_config(raw).evaluation.reference
    assert parsed.name == "v4" and parsed.every_iters == 10 and parsed.num_games == 40


def test_reference_can_search_on_its_own_budget(tmp_path, monkeypatch):
    raw = minimal_raw_config()
    raw["board"] = _board().model_dump()
    path = tmp_path / "reference.yaml"
    path.write_text(yaml.safe_dump(raw))
    model_cfg, mcts, weights = _side(0)
    seen = {}

    class _Task:
        def options(self, **_options):
            return self

        def remote(self, _board_cfg, sides, black, *_rest):
            seen["searches"] = {name: side[1].num_searches for name, side in sides.items()}
            return {"black": black, "winner": rm.CHAMPION, "plies": 9}

    class _Ray:
        put = staticmethod(lambda value: value)
        get = staticmethod(lambda values: values)
        cluster_resources = staticmethod(dict)

    monkeypatch.setattr(rm, "ray", _Ray)
    monkeypatch.setattr(rm, "_play_game_task", _Task())
    monkeypatch.setattr(rm, "load_state_dict", lambda _path: weights)
    cfg = ReferenceMatchConfig(
        checkpoint="unused.pt", config=str(path), num_games=2, num_searches=6, reference_searches=15
    )

    summary = rm.run_reference_match(_board(), model_cfg, mcts, weights, cfg, iteration=5)

    assert seen["searches"] == {rm.CHAMPION: 6.0, rm.REFERENCE: 15.0}
    assert summary["wins"] == 2


def test_reference_block_defaults_to_the_champion_on_equal_searches():
    cfg = ReferenceMatchConfig(checkpoint="unused.pt", config="unused.yaml")
    assert cfg.player == "champion" and cfg.reference_searches is None
