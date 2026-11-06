"""Static quantization backend, noise-free teacher labels and local evaluation."""

import argparse
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

pytest.importorskip("pyarrow")
pytest.importorskip("onnxruntime")

from gomoku.core.gomoku import Gomoku
from gomoku.cpp_ext import gomoku_cpp
from gomoku.inference.onnx_inference import OnnxInference
from gomoku.model.policy_value_net import PolicyValueNet
from gomoku.scripts import distill, eval_local
from gomoku.utils.config.loader import load_and_parse_config
from server.engine import AlphaZeroEngine

SMALL_BOARD = 9
COLOR_PLANE = 6


@pytest.fixture
def small_setup(tmp_path: Path) -> tuple[str, str]:
    raw = yaml.safe_load(Path("configs/small-6x96.yaml").read_text())
    raw["board"]["num_lines"] = SMALL_BOARD
    raw["model"].update(num_hidden=8, num_resblocks=1)
    raw["mcts"]["time_limit_s"] = None
    raw["mcts"]["num_searches"] = 8
    raw["training"]["batch_size"] = 4
    config_path = tmp_path / "small.yaml"
    config_path.write_text(yaml.safe_dump(raw))
    config = load_and_parse_config(str(config_path))
    torch.manual_seed(0)
    model = PolicyValueNet(Gomoku(config.board), config.model, torch.device("cpu"))
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint)
    return str(config_path), str(checkpoint)


def _engine(small_setup: tuple[str, str]) -> AlphaZeroEngine:
    return AlphaZeroEngine(small_setup[0], small_setup[1], "cpu")


def test_static_backend_serves_a_calibrated_qdq_model(
    small_setup: tuple[str, str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import onnx

    monkeypatch.setenv("ALPHAZERO_INFER_BACKEND", "onnx-static")
    monkeypatch.setenv("ONNX_CACHE_DIR", str(tmp_path / "onnx"))
    engine = _engine(small_setup)
    assert isinstance(engine.inference_client, OnnxInference)
    graph = onnx.load(tmp_path / "onnx" / "model_int8.onnx").graph
    assert any(node.op_type == "QuantizeLinear" for node in graph.node)
    action = engine.get_best_move(engine.game.get_initial_state())
    assert 0 <= action < SMALL_BOARD**2


def _top_visits_when_every_position_looks_lost(fpu_from_parent: bool) -> int:
    core = gomoku_cpp.GomokuCore(9, True, False, 0, 5, 0)
    engine = gomoku_cpp.MctsEngine(core, 2.0, fpu_from_parent)

    def evaluator(features, legal):
        # The first player, who moves at the root, looks lost unless a stone is on (1, 1).
        first_player_to_move = features[COLOR_PLANE, 0, 0] > 0
        magnitude = 0.2 if features[0, 1, 1] + features[1, 1, 1] > 0 else 0.8
        return [0.0] * core.action_size, -magnitude if first_player_to_move else magnitude

    visits = engine.run_mcts_encoded(core.initial_state(), 60, evaluator)
    return max(count for _, count in visits)


def test_zero_fpu_spreads_a_losing_search_across_all_moves() -> None:
    assert _top_visits_when_every_position_looks_lost(False) == 1


def test_parent_fpu_lets_a_losing_search_go_deep() -> None:
    assert _top_visits_when_every_position_looks_lost(True) > 1


def test_labels_are_noise_free_and_repeatable(small_setup: tuple[str, str]) -> None:
    engine = _engine(small_setup)
    first = distill.label_game(engine, 16, 0, random.Random(0))
    second = distill.label_game(engine, 16, 0, random.Random(0))
    assert len(first) == len(second)
    for one, other in zip(first, second, strict=True):
        np.testing.assert_array_equal(one.policy_probs, other.policy_probs)
    assert first[0].policy_probs.sum() == pytest.approx(1.0, abs=1e-5)


def test_label_values_are_the_game_result_for_the_player_to_move(
    small_setup: tuple[str, str],
) -> None:
    game = distill.label_game(_engine(small_setup), 4, 0, random.Random(0))
    assert game[-1].value == 1.0
    assert game[-2].value == -1.0
    assert int(game[-1].state.next_player) != int(game[-2].state.next_player)


def test_labels_train_a_student_the_server_can_load(
    small_setup: tuple[str, str], tmp_path: Path
) -> None:
    config_path, _ = small_setup
    engine = _engine(small_setup)
    samples = distill.collect_labels(engine, 24, 4, 2, random.Random(0))
    assert len(samples) == 24
    labels = tmp_path / "labels"
    labels.mkdir()
    shard = labels / "shard.parquet"
    from gomoku.alphazero.learning.dataset import save_records_to_parquet_shard

    save_records_to_parquet_shard([distill._to_record(samples)], str(shard))
    out = tmp_path / "student.pt"
    distill.run_train(
        argparse.Namespace(
            config=config_path, labels=str(labels), rounds=1, out=str(out), seed=0
        )
    )
    student = AlphaZeroEngine(config_path, str(out), "cpu")
    assert 0 <= student.get_best_move(student.game.get_initial_state()) < SMALL_BOARD**2


def test_opening_report_counts_top_visit_ties(small_setup: tuple[str, str]) -> None:
    report = eval_local.opening_report(_engine(small_setup), (4, 4), sims=8)
    assert report.top_visits >= 1
    assert report.tied_top >= 1
    assert report.uniform_entropy == pytest.approx(np.log(SMALL_BOARD**2 - 1))
    assert 0 < report.entropy <= report.uniform_entropy + 1e-6


def test_duel_game_finishes_with_a_scored_result(small_setup: tuple[str, str]) -> None:
    a, b = _engine(small_setup), _engine(small_setup)
    result = eval_local.play_game(a, b, eval_local.GameTask(1, True), opening_plies=2)
    assert result.a_score in (0.0, 0.5, 1.0)
    assert result.plies > 0


def test_player_specs_default_to_the_deploy_config() -> None:
    assert eval_local.parse_player("old=a.pt").config == eval_local.DEFAULT_CONFIG
    custom = eval_local.parse_player("new=b.pt,c.yaml,onnx-static")
    assert (custom.name, custom.checkpoint, custom.config, custom.backend) == (
        "new", "b.pt", "c.yaml", "onnx-static",
    )
    with pytest.raises(argparse.ArgumentTypeError):
        eval_local.parse_player("missing-checkpoint")


def test_elo_is_zero_for_an_even_score() -> None:
    assert eval_local.elo_difference(0.5) == pytest.approx(0.0)
    assert eval_local.elo_difference(0.75) > 0
