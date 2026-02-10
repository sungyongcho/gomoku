from __future__ import annotations

import json
import os
from pathlib import Path

import fsspec
import numpy as np
import pytest
import torch
import torch.nn as nn

from gomoku.core.gomoku import Gomoku
from gomoku.model.policy_value_net import PolicyValueNet
from gomoku.scripts.pipelines import common
from gomoku.scripts.pipelines.run_mp import run_mp
from gomoku.scripts.pipelines.run_sequential import run_sequential
from gomoku.scripts.pipelines.run_vectorize import run_vectorize
from gomoku.utils.config.loader import (
    BoardConfig,
    EvaluationConfig,
    IoConfig,
    MctsConfig,
    ModelConfig,
    ParallelConfig,
    PathsConfig,
    RunnerParams,
    TrainingConfig,
)
from gomoku.utils.io_helpers import save_as_parquet_shard
from gomoku.utils.paths import format_path, manifest_path, new_replay_shard_path
from gomoku.utils.serialization import to_savable_sample


class DummyModel(nn.Module):
    """Minimal model with a single parameter for checkpoint tests."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))
        self.device = torch.device("cpu")


def _make_runner_params(tmpdir: str) -> RunnerParams:
    paths = PathsConfig(
        use_gcs=False,
        run_prefix=tmpdir,
        run_id="test_run",
        replay_dir="{run_prefix}/{run_id}/replay",
        ckpt_dir="{run_prefix}/{run_id}/ckpt",
        evaluation_logs_dir="{run_prefix}/{run_id}/eval_logs",
        manifest="{run_prefix}/{run_id}/manifest.json",
    )

    board_cfg = BoardConfig(
        num_lines=3,
        enable_doublethree=False,
        enable_capture=False,
        capture_goal=5,
        gomoku_goal=3,
        history_length=1,
    )
    model_cfg = ModelConfig(
        num_planes=8 + board_cfg.history_length, num_hidden=8, num_resblocks=1
    )
    mcts_cfg = MctsConfig(
        C=1.0,
        num_searches=1,
        exploration_turns=0,
        dirichlet_epsilon=0.0,
        dirichlet_alpha=0.0,
        batch_infer_size=1,
        max_batch_wait_ms=0,
        min_batch_size=1,
    )
    training_cfg = TrainingConfig(
        num_iterations=1,
        num_selfplay_iterations=1,
        num_epochs=1,
        batch_size=1,
        learning_rate=0.1,
        weight_decay=0.0,
        temperature=1.0,
        replay_buffer_size=4,
        min_samples_to_train=1,
        random_play_ratio=0.0,
        priority_replay=None,
        dataloader_num_workers=0,
        dataloader_prefetch_factor=1,
        enable_tf32=False,
        use_channels_last=False,
    )
    eval_cfg = EvaluationConfig(
        num_eval_games=0,
        eval_every_iters=1,
        promotion_win_rate=0.5,
        num_baseline_games=0,
        blunder_threshold=0.0,
        initial_blunder_rate=0.0,
        initial_baseline_win_rate=0.0,
        blunder_increase_limit=1.0,
        baseline_wr_min=0.0,
        random_play_ratio=0.0,
        eval_opening_turns=0,
        eval_temperature=0.0,
        eval_dirichlet_epsilon=0.0,
        eval_num_searches=0,
        baseline_num_searches=0,
        use_sprt=False,
        sprt=None,
        fast_eval=None,
        adjudication_win_prob=None,
        adjudication_min_turns=None,
    )
    parallel_cfg = ParallelConfig(
        num_parallel_games=1, mp_num_workers=1, ray_local_num_workers=1
    )
    io_cfg = IoConfig(
        initial_replay_shards=None,
        initial_replay_iters=None,
        max_samples_per_shard=None,
        local_replay_cache=None,
    )

    return RunnerParams(
        board=board_cfg,
        model=model_cfg,
        mcts=mcts_cfg,
        training=training_cfg,
        evaluation=eval_cfg,
        parallel=parallel_cfg,
        paths=paths,
        io=io_cfg,
        runtime=None,
        device_type="cpu",
        run_id="test_run",
    )


def test_save_and_load_checkpoint(tmp_path: Path) -> None:
    fs = fsspec.filesystem("file")
    params = _make_runner_params(tmp_path.as_posix())
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Save checkpoint at iteration 0
    ckpt_path = common.save_checkpoint(
        fs, params, iteration=0, model=model, optimizer=optimizer
    )
    assert fs.exists(ckpt_path)
    assert fs.exists(ckpt_path + ".optim")

    # Create manifest pointing to the saved checkpoint
    mpath = manifest_path(params.paths)
    fs.makedirs(os.path.dirname(mpath), exist_ok=True)
    manifest = {
        "champion_model_path": ckpt_path,
        "progress": {"completed_iterations": 0},
        "status": "running",
    }
    with fs.open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    # Mutate model to ensure load overwrites it
    with torch.no_grad():
        model.weight.fill_(5.0)

    completed, champion, shards = common.load_resume_state(params, fs, model, optimizer)
    assert completed == 0
    assert champion == ckpt_path
    assert shards == []
    assert torch.allclose(model.weight, torch.tensor([1.0]))


def test_update_manifest(tmp_path: Path) -> None:
    fs = fsspec.filesystem("file")
    params = _make_runner_params(tmp_path.as_posix()).model_copy(
        update={"config_name": "cfgA"}
    )
    mpath = manifest_path(params.paths)
    fs.makedirs(os.path.dirname(mpath), exist_ok=True)
    with fs.open(mpath, "w", encoding="utf-8") as f:
        json.dump({"progress": {"completed_iterations": 1}, "status": "running"}, f)

    common.update_manifest(
        params,
        fs,
        completed_iterations=3,
        status="completed",
        champion_path="ckpt.pt",
        eval_summary={"win_rate": 0.8, "promotion_win_rate": 0.6, "promote": True},
    )

    with fs.open(mpath, "r", encoding="utf-8") as f:
        updated = json.load(f)
    assert updated["progress"]["completed_iterations"] == 3
    assert updated["status"] == "completed"
    assert updated["champion_model_path"] == "ckpt.pt"
    assert updated["configs"][-1]["name"] == "cfgA"
    assert isinstance(updated.get("evaluations"), list)
    assert updated["evaluations"][-1]["promote"] is True
    assert isinstance(updated.get("promotions"), list)


def test_warm_start_replay_buffer_loads_shard(tmp_path: Path) -> None:
    fs = fsspec.filesystem("file")
    params = _make_runner_params(tmp_path.as_posix())
    game = Gomoku(params.board)

    # Save one shard with a single sample
    state = game.get_initial_state()
    # Inject a move to populate history and last_move_idx.
    state = game.get_next_state(state, (0, 0), state.next_player)
    policy = np.full(game.action_size, 1.0 / game.action_size, dtype=np.float32)
    row = to_savable_sample(state, policy, 0.5)
    shard_path = new_replay_shard_path(params.paths, iteration=1)
    save_as_parquet_shard(fs, shard_path, [row])

    buffer, loaded = common.warm_start_replay_buffer(params, fs, game)

    assert shard_path in loaded
    assert len(buffer) == 1
    sample = buffer[0]
    assert np.isclose(sample.value, 0.5)
    # History and last_move_idx should round-trip from shard.
    assert sample.state.history == state.history
    assert sample.state.last_move_idx == state.last_move_idx


@pytest.mark.parametrize(
    "runner_fn",
    [
        run_sequential,
        run_vectorize,
        run_mp,
    ],
)
def test_learning_rate_schedule_applied(tmp_path: Path, runner_fn) -> None:
    fs = fsspec.filesystem("file")
    params = _make_runner_params(tmp_path.as_posix())
    # Apply LR schedule: iteration 1 uses 0.123
    train_override = params.training.model_copy(
        update={
            "learning_rate": [{"until": 1, "value": 0.123}],
            "num_iterations": 1,
            "num_selfplay_iterations": 0,  # skip heavy self-play
            "random_play_ratio": [{"until": 1, "value": 0.5}],
        }
    )
    params = params.model_copy(update={"training": train_override})

    game = Gomoku(params.board)
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    # Run a single iteration; self-play/training skipped but LR should update.
    runner_fn(
        model=model,  # type: ignore[arg-type]
        optimizer=optimizer,
        game=game,
        params=params,
        fs=fs,
        completed_offset=0,
        replay_shards=[],
        champion_path=None,
        manifest_updater=None,
    )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.123)


def test_resume_flow_updates_manifest_and_ckpt(tmp_path: Path, monkeypatch) -> None:
    """1회 실행 후 manifest/ckpt/shard가 갱신되는지 소규모 통합 검증."""
    torch.manual_seed(0)
    fs = fsspec.filesystem("file")
    params = _make_runner_params(tmp_path.as_posix())
    game = Gomoku(params.board)
    model = PolicyValueNet(game, params.model, device="cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    ckpt0 = common.save_checkpoint(
        fs,
        params,
        iteration=0,
        model=model,
        optimizer=optimizer,
        prefix="[TestResume] ",
    )
    mpath = manifest_path(params.paths)
    fs.makedirs(os.path.dirname(mpath), exist_ok=True)
    with fs.open(mpath, "w", encoding="utf-8") as f:
        json.dump(
            {
                "champion_model_path": ckpt0,
                "progress": {"completed_iterations": 0},
                "status": "running",
            },
            f,
        )

    run_params = params.model_copy(
        update={
            "training": params.training.model_copy(
                update={
                    "num_iterations": 1,
                    "num_selfplay_iterations": 1,
                    "batch_size": 1,
                    "num_epochs": 1,
                }
            )
        }
    )

    print("[ResumeFlow] Starting run_sequential for 1 iteration")
    run_sequential(
        model,
        optimizer,
        game,
        run_params,
        fs,
        completed_offset=0,
        champion_path=ckpt0,
        replay_shards=None,
        manifest_updater=common.update_manifest,
    )

    with fs.open(mpath, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    print(f"[ResumeFlow] Manifest after run: {manifest}")

    assert manifest["status"] == "completed"
    assert manifest["progress"]["completed_iterations"] == 1
    ckpt_path = manifest["champion_model_path"]
    assert ckpt_path and fs.exists(ckpt_path)

    replay_dir = format_path(params.paths, params.paths.replay_dir)
    shards = fs.glob(replay_dir.rstrip("/") + "/*.parquet")
    print(f"[ResumeFlow] Shards: {shards}")
    assert len(shards) >= 1


def test_evaluate_and_promote(tmp_path: Path, monkeypatch) -> None:
    """평가가 승격을 true로 반환하면 챔피언 경로를 교체한다."""
    fs = fsspec.filesystem("file")
    params = _make_runner_params(tmp_path.as_posix())
    game = Gomoku(params.board)
    model = PolicyValueNet(game, params.model, device="cpu")

    # Seed champion and current ckpt
    champ_ckpt = common.save_checkpoint(
        fs, params, iteration=0, model=model, optimizer=None
    )
    baseline_ckpt = common.save_checkpoint(
        fs, params, iteration=0, model=model, optimizer=None
    )
    new_ckpt = common.save_checkpoint(
        fs, params, iteration=1, model=model, optimizer=None
    )

    mpath = manifest_path(params.paths)
    fs.makedirs(os.path.dirname(mpath), exist_ok=True)
    with fs.open(mpath, "w", encoding="utf-8") as f:
        json.dump(
            {
                "champion_model_path": champ_ckpt,
                "baseline_model_path": baseline_ckpt,
                "progress": {"completed_iterations": 0},
                "status": "running",
            },
            f,
        )

    # Enable evaluation
    params_eval = params.model_copy(
        update={
            "evaluation": params.evaluation.model_copy(
                update={
                    "num_eval_games": 1,
                    "eval_every_iters": 1,
                    "promotion_win_rate": 0.5,
                }
            )
        }
    )

    def fake_run_arena(cfg, inf_new, inf_best, **kwargs):
        return {"win_rate": 1.0, "promotion_win_rate": 0.5, "promote": True}

    monkeypatch.setattr(common, "run_arena", fake_run_arena)

    new_champ, summary = common.evaluate_and_maybe_promote(
        game,
        model,
        params_eval,
        fs,
        champion_path=champ_ckpt,
        ckpt_path=new_ckpt,
        iteration=1,
    )
    print(f"[EvalPromote] summary={summary}")
    assert new_champ == new_ckpt
    assert summary is not None and summary.get("promote") is True
