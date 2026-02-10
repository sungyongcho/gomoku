import fsspec
import torch

from gomoku.core.gomoku import Gomoku
from gomoku.model.policy_value_net import PolicyValueNet
from gomoku.scripts.pipelines.common import update_manifest
from gomoku.scripts.pipelines.run_mp import run_mp
from gomoku.scripts.pipelines.run_ray import run_ray
from gomoku.scripts.pipelines.run_sequential import run_sequential
from gomoku.scripts.pipelines.run_vectorize import run_vectorize
from gomoku.utils.config.loader import RunnerParams


def run_training(
    mode: str,
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    game: Gomoku,
    params: RunnerParams,
    fs: fsspec.AbstractFileSystem,
    *,
    async_inflight_limit: int | None = None,
    completed_offset: int = 0,
    replay_shards: list[str] | None = None,
    champion_path: str | None = None,
    manifest_updater=None,
) -> None:
    """Run training pipeline by mode."""
    if manifest_updater is None:
        def manifest_updater(params, fs, **kwargs):
            return update_manifest(params, fs, **kwargs)

    mode = mode.lower()
    if mode == "sequential":
        run_sequential(
            model,
            optimizer,
            game,
            params,
            fs,
            completed_offset=completed_offset,
            replay_shards=replay_shards,
            champion_path=champion_path,
            manifest_updater=manifest_updater,
        )
        return
    if mode == "vectorize":
        run_vectorize(
            model,
            optimizer,
            game,
            params,
            fs,
            completed_offset=completed_offset,
            replay_shards=replay_shards,
            champion_path=champion_path,
            manifest_updater=manifest_updater,
        )
        return
    if mode == "mp":
        run_mp(
            model,
            optimizer,
            game,
            params,
            fs,
            completed_offset=completed_offset,
            replay_shards=replay_shards,
            champion_path=champion_path,
            manifest_updater=manifest_updater,
        )
        return
    if mode == "ray":
        run_ray(
            model,
            optimizer,
            game,
            params,
            fs,
            async_inflight_limit=async_inflight_limit,
            completed_offset=completed_offset,
            replay_shards=replay_shards,
            champion_path=champion_path,
            manifest_updater=manifest_updater,
        )
        return
    raise ValueError(f"Unsupported mode: {mode}")
