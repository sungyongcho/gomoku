import os
import random

import fsspec
import pytest

from gomoku.scripts.pipelines.common import select_past_checkpoint
from gomoku.utils.config.loader import PathsConfig


def test_select_past_checkpoint_picks_latest_within_window(
    tmp_path: "os.PathLike[str]", monkeypatch: pytest.MonkeyPatch
) -> None:
    """윈도우 내 과거 체크포인트 중 가장 최근(정렬 상위)을 선택한다."""
    base = tmp_path / "ckpts"
    base.mkdir()
    ckpt_dir = base / "runs/run/ckpt"
    ckpt_dir.mkdir(parents=True)
    # create ckpt files
    for idx in [1, 2, 3]:
        (ckpt_dir / f"iteration_{idx:04d}.pt").write_bytes(b"x")

    paths = PathsConfig(
        use_gcs=False,
        run_prefix=str(base / "runs"),
        run_id="run",
        replay_dir="{run_prefix}/{run_id}/replay",
        ckpt_dir="{run_prefix}/{run_id}/ckpt",
        evaluation_logs_dir="{run_prefix}/{run_id}/eval_logs",
        manifest="{run_prefix}/{run_id}/manifest.json",
    )
    fs = fsspec.filesystem("file")

    # deterministic choice: pick first candidate
    monkeypatch.setattr(random, "choice", lambda seq: seq[0])

    selected = select_past_checkpoint(
        fs=fs, paths=paths, current_iter=5, window=2, champion_path="champ.pt"
    )
    assert selected.endswith("iteration_0003.pt")


def test_select_past_checkpoint_falls_back_to_champion_when_none(
    tmp_path: "os.PathLike[str]",
) -> None:
    """과거 체크포인트가 없으면 챔피언 경로를 반환한다."""
    base = tmp_path / "ckpts_empty"
    base.mkdir()
    paths = PathsConfig(
        use_gcs=False,
        run_prefix=str(base / "runs"),
        run_id="run",
        replay_dir="{run_prefix}/{run_id}/replay",
        ckpt_dir="{run_prefix}/{run_id}/ckpt",
        evaluation_logs_dir="{run_prefix}/{run_id}/eval_logs",
        manifest="{run_prefix}/{run_id}/manifest.json",
    )
    fs = fsspec.filesystem("file")

    selected = select_past_checkpoint(
        fs=fs, paths=paths, current_iter=3, window=3, champion_path="champ.pt"
    )
    assert selected == "champ.pt"
