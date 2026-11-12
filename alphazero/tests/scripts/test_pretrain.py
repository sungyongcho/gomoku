"""Pretraining a fresh network on an earlier run's replay shards."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import yaml

pytest.importorskip("pyarrow")

from gomoku.alphazero.learning.dataset import save_records_to_parquet_shard
from gomoku.alphazero.runners.common import SelfPlaySample, build_game_record
from gomoku.core.gomoku import Gomoku
from gomoku.model.policy_value_net import PolicyValueNet
from gomoku.scripts import pretrain
from gomoku.utils.config.loader import load_and_parse_config

SMALL_BOARD = 9


def _small_config(tmp_path: Path, hidden: int, name: str) -> str:
    raw = yaml.safe_load(Path("configs/small-6x96.yaml").read_text())
    raw["board"]["num_lines"] = SMALL_BOARD
    raw["model"].update(num_hidden=hidden, num_resblocks=1)
    raw["training"].update(batch_size=4, dataloader_num_workers=0)
    path = tmp_path / name
    path.write_text(yaml.safe_dump(raw))
    return str(path)


def _write_shard(replay: Path, game: Gomoku, iteration: int, positions: int) -> None:
    rng = np.random.default_rng(iteration)
    state = game.get_initial_state()
    memory = []
    for _ in range(positions):
        pi = np.zeros(game.action_size, dtype=np.float32)
        game.get_legal_moves(state)
        action = int(rng.choice(state.legal_indices_cache.astype(int)))
        pi[action] = 1.0
        memory.append(SelfPlaySample(state, pi, action, int(state.next_player)))
        xy = (action % game.col_count, action // game.col_count)
        state = game.get_next_state(state, xy, state.next_player)
    record = build_game_record(memory, 1.0, int(memory[-1].player))
    record.config_snapshot = {"source": "test"}  # an empty snapshot cannot be written
    save_records_to_parquet_shard(
        [record], str(replay / f"shard-iter{iteration:04d}-1-test.parquet")
    )


def test_shards_are_grouped_by_the_iteration_in_their_name(tmp_path: Path) -> None:
    config = load_and_parse_config(_small_config(tmp_path, 8, "small.yaml"))
    game = Gomoku(config.board)
    for iteration in (3, 4, 4):
        _write_shard(tmp_path, game, iteration, 2)
    (tmp_path / "notes.parquet").write_bytes(b"")

    grouped = pretrain.shards_by_iteration(str(tmp_path))

    assert sorted(grouped) == [3, 4]
    assert len(grouped[3]) == 1
    assert all(path.startswith("file://") for paths in grouped.values() for path in paths)


def test_pretraining_saves_weights_of_the_new_size_and_reports_the_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    small = _small_config(tmp_path, 8, "small.yaml")
    large = _small_config(tmp_path, 16, "large.yaml")
    config = load_and_parse_config(small)
    game = Gomoku(config.board)
    replay = tmp_path / "replay"
    replay.mkdir()
    for iteration in (1, 2, 3, 4):
        _write_shard(replay, game, iteration, 6)
    torch.manual_seed(0)
    baseline = tmp_path / "baseline.pt"
    torch.save(PolicyValueNet(game, config.model, torch.device("cpu")).state_dict(), baseline)
    out = tmp_path / "v6" / "ckpt" / "seed_test.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pretrain", "--config", large, "--replay", str(replay), "--first", "1",
            "--last", "3", "--holdout", "4", "--lr", "0.01,0.001",
            "--baseline", f"{baseline}@{small}", "--cache", str(tmp_path / "cache"),
            "--out", str(out),
        ],
    )  # fmt: skip

    pretrain.main()

    printed = capsys.readouterr().out
    assert "18 samples from iterations 1-3, 6 held out from 4" in printed
    assert "baseline holdout policy=" in printed
    assert "pass 2/2 lr=0.001" in printed
    large_model = PolicyValueNet(game, load_and_parse_config(large).model, torch.device("cpu"))
    large_model.load_state_dict(torch.load(out))
    assert len(list((tmp_path / "cache").iterdir())) == 4

    # A second call can go on from the saved weights.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pretrain", "--config", large, "--replay", str(replay), "--first", "2",
            "--last", "3", "--holdout", "4", "--lr", "0.001", "--init", str(out),
            "--out", str(out),
        ],
    )  # fmt: skip
    pretrain.main()
    assert "pass 1/1 lr=0.001" in capsys.readouterr().out


def test_the_holdout_iteration_must_not_be_trained_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pretrain", "--config", "unused.yaml", "--replay", str(tmp_path), "--first", "1",
            "--last", "5", "--holdout", "3", "--lr", "0.01", "--out", str(tmp_path / "x.pt"),
        ],
    )  # fmt: skip
    with pytest.raises(SystemExit):
        pretrain.main()
