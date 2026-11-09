"""Order of the bookkeeping at the end of a Ray training iteration.

A run stopped during the reference match must not repeat the iteration it had finished.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import fsspec
import numpy as np
import pytest
import torch

from gomoku.alphazero.runners.common import SelfPlaySample, build_game_record
from gomoku.core.gomoku import Gomoku
from gomoku.scripts.pipelines import run_ray as run_ray_mod
from gomoku.utils.config.loader import ReferenceMatchConfig
from gomoku.utils.paths import ensure_run_dirs, manifest_path
from tests.scripts.pipelines.test_common import DummyModel, _make_runner_params


def _fake_ray() -> SimpleNamespace:
    """Stand in for Ray: no cluster, and remote tasks run in place."""

    def remote(**_options: Any):
        return lambda fn: SimpleNamespace(remote=lambda *args: fn(*args))

    return SimpleNamespace(
        init=lambda **_kwargs: None,
        shutdown=lambda: None,
        nodes=list,
        cluster_resources=dict,
        available_resources=dict,
        remote=remote,
    )


def test_iteration_is_recorded_before_the_reference_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fs = fsspec.filesystem("file")
    params = _make_runner_params(tmp_path.as_posix())
    reference = ReferenceMatchConfig(
        checkpoint="v4.pt", config="v4.yaml", every_iters=1, num_games=2, num_searches=1
    )
    params = params.model_copy(
        update={"evaluation": params.evaluation.model_copy(update={"reference": reference})}
    )
    ensure_run_dirs(params, fs)
    manifest = Path(manifest_path(params.paths))
    manifest.write_text(json.dumps({"progress": {"completed_iterations": 0}}))

    game = Gomoku(params.board)
    pi = np.full(game.action_size, 1.0 / game.action_size, dtype=np.float32)
    record = build_game_record(
        [SelfPlaySample(game.get_initial_state(), pi, 0, 1)],
        1.0,
        1,
        opponent="external_bot",
        agent_player=1,
    )

    class _Runner:
        actors: list = []

        def __init__(self, **_kwargs: Any) -> None: ...

        def run(self, **_kwargs: Any) -> list:
            return [record]

        def stop_workers(self) -> None: ...

    class _Trainer:
        def __init__(self, **_kwargs: Any) -> None: ...

        def train_one_iteration(self, samples: list, progress_label: str = "") -> dict:
            return {"loss": 0.0}

    recorded_when_the_match_ran: list[int] = []

    def reference_match(*_args: Any) -> dict:
        progress = json.loads(manifest.read_text())["progress"]
        recorded_when_the_match_ran.append(progress["completed_iterations"])
        return {
            "wins": 1,
            "losses": 1,
            "draws": 0,
            "score": 0.5,
            "wins_as_black": 1,
            "games_as_black": 1,
            "wins_as_white": 0,
            "games_as_white": 1,
            "mean_plies": 9.0,
            "seconds": 0.0,
        }

    monkeypatch.setattr(run_ray_mod, "ray", _fake_ray())
    monkeypatch.setattr(run_ray_mod, "RayAsyncRunner", _Runner)
    monkeypatch.setattr(run_ray_mod, "AlphaZeroTrainer", _Trainer)
    monkeypatch.setattr(run_ray_mod, "run_reference_match", reference_match)
    monkeypatch.setattr(
        run_ray_mod,
        "distributed_evaluate_and_promote",
        lambda *_args, **kwargs: (kwargs["ckpt_path"], None),
    )

    model = DummyModel()
    run_ray_mod.run_ray(
        model=model,  # type: ignore[arg-type]
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        game=game,
        params=params,
        fs=fs,
        replay_shards=[],
    )

    assert recorded_when_the_match_ran == [1]
    opponent_log = manifest.parent / "eval_logs" / "selfplay_opponent_log.jsonl"
    entries = [json.loads(line) for line in opponent_log.read_text().splitlines()]
    assert [entry["iteration"] for entry in entries] == [1]
    assert entries[0]["opponents"]["external_bot"]["wins_as_black"] == 1
