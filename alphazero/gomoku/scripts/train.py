import argparse
import os
from typing import Any

import fsspec
import torch

from gomoku.core.gomoku import Gomoku
from gomoku.model.policy_value_net import PolicyValueNet
from gomoku.scripts.pipelines.common import load_resume_state
from gomoku.scripts.pipelines.run_loop import run_training
from gomoku.utils.config.loader import (
    RootConfig,
    RunnerParams,
    assemble_runner_params,
    load_and_parse_config,
)
from gomoku.utils.config.schedule_param import get_scheduled_value
from gomoku.utils.io_helpers import atomic_write_json
from gomoku.utils.manifest import create_new_manifest, load_manifest
from gomoku.utils.paths import (
    ensure_run_dirs,
    format_path,
    iteration_ckpt_path,
    load_state_dict_from_fs,
    manifest_path,
    save_state,
)
from gomoku.utils.state_dict_utils import align_state_dict_to_model


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments for training."""
    parser = argparse.ArgumentParser(description="Train a Gomoku AlphaZero model.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config. Required.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["sequential", "vectorize", "mp", "ray"],
        help="Execution mode.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu"],
        help="Device to use.",
    )
    return parser.parse_args()


def build_model_and_optimizer(
    game: Gomoku,
    cfg: RootConfig,
    device: torch.device,
) -> tuple[PolicyValueNet, torch.optim.Optimizer]:
    """Build model/optimizer with initial LR from schedule."""
    model = PolicyValueNet(game, cfg.model, device=str(device))
    if getattr(cfg.training, "use_channels_last", False):
        model = model.to(memory_format=torch.channels_last)
    init_lr = get_scheduled_value(cfg.training.learning_rate, 0)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=init_lr, weight_decay=cfg.training.weight_decay
    )
    return model, optimizer


def _choose_fs(paths_cfg: Any) -> fsspec.AbstractFileSystem:
    protocol = "file"
    if getattr(paths_cfg, "use_gcs", False):
        protocol = "gcs"
    return fsspec.filesystem(protocol)


def _find_seed_checkpoint(fs: fsspec.AbstractFileSystem, paths_cfg) -> str | None:
    ckpt_dir_path = format_path(paths_cfg, paths_cfg.ckpt_dir)
    seed_glob = os.path.join(ckpt_dir_path.rstrip("/"), "seed_*.pt")
    try:
        pretrained_seeds = sorted(fs.glob(seed_glob))
    except FileNotFoundError:
        pretrained_seeds = []
    return pretrained_seeds[-1] if pretrained_seeds else None


def _strip_model_invariants(manifest: dict) -> None:
    model_section = manifest.get("model")
    if isinstance(model_section, dict):
        for k in ("num_planes", "policy_channels", "value_channels"):
            model_section.pop(k, None)


def _format_eval_schedule(eval_cfg) -> str:
    """Summarize evaluation cadence for logging."""
    if not eval_cfg or getattr(eval_cfg, "num_eval_games", 0) <= 0:
        return "[Eval] Disabled (num_eval_games <= 0)"
    every = max(1, int(getattr(eval_cfg, "eval_every_iters", 1)))
    num_games = int(getattr(eval_cfg, "num_eval_games", 0))
    base_games = int(getattr(eval_cfg, "num_baseline_games", 0))
    return f"[Eval] Every {every} iter(s): {num_games} vs champion" + (
        f", {base_games} vs baseline" if base_games > 0 else ""
    )


def main() -> None:
    args = parse_arguments()
    cfg: RootConfig = load_and_parse_config(args.config)
    run_id = cfg.paths.run_id
    paths_cfg = cfg.paths
    fs = _choose_fs(paths_cfg)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.device == "auto" else "cpu"
    )
    print(f"[Main] Using device: {device.type}")
    use_native = bool(getattr(cfg.mcts, "use_native", False))
    runtime_cfg = getattr(cfg, "runtime", None)
    inference_rt = getattr(runtime_cfg, "inference", None) if runtime_cfg else None
    use_local_infer = bool(getattr(inference_rt, "use_local_inference", False))
    print(
        f"[Main] Inference flags | use_native={use_native} | use_local_inference={use_local_infer}"
    )
    params: RunnerParams = assemble_runner_params(
        cfg, device.type, run_id, config_name=os.path.basename(args.config)
    )

    if device.type == "cuda" and getattr(params.training, "enable_tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    ensure_run_dirs(params, fs)

    game = Gomoku(cfg.board, use_native=use_native)
    model, optimizer = build_model_and_optimizer(game=game, cfg=cfg, device=device)

    manifest = load_manifest(paths_cfg, fs)
    manifest_dirty = False
    config_name = os.path.basename(args.config)
    completed_before = 0

    if manifest is None:
        print(f"No manifest found for run '{run_id}'. Creating a new one.")
        manifest = create_new_manifest(run_id, cfg, device.type, config_name)
        manifest["modes"] = [{"begin_iteration": 1, "mode": args.mode}]

        initial_champion_path = iteration_ckpt_path(paths_cfg, 0)
        seed_path = _find_seed_checkpoint(fs, paths_cfg)
        if seed_path:
            seeded_state = load_state_dict_from_fs(fs, seed_path, device)
            resized, missing, dropped = align_state_dict_to_model(
                seeded_state, model.state_dict()
            )
            if resized:
                print(
                    "[Init Champion] Resized tensors:",
                    ", ".join(resized[:10]) + ("..." if len(resized) > 10 else ""),
                )
            if missing:
                print(f"[Init Champion] Missing keys: {len(missing)}")
            if dropped:
                print(f"[Init Champion] Dropped keys: {len(dropped)}")
            model.load_state_dict(seeded_state, strict=False)
        else:
            print(
                f"Saving randomly initialized model as initial champion to {initial_champion_path}."
            )
        save_state(fs, model.state_dict(), initial_champion_path)
        manifest["champion_model_path"] = initial_champion_path
        manifest_dirty = True
    else:
        last_config = manifest.get("configs", [{}])[-1]
        if last_config.get("name") != config_name:
            current_iter = (
                manifest.get("progress", {}).get("completed_iterations", 0) + 1
            )
            manifest.setdefault("configs", []).append(
                {
                    "revision": len(manifest["configs"]),
                    "name": config_name,
                    "begin_iteration": current_iter,
                }
            )
            manifest_dirty = True

    # Load champion/model/optimizer and completed iterations
    completed_before, champion_path, replay_shards = load_resume_state(
        params, fs, model, optimizer
    )
    if champion_path:
        print(f"Loading model state from {champion_path}")

    remaining = int(params.training.num_iterations) - completed_before

    # Update manifest status before deciding to continue.
    manifest_status = "running" if remaining > 0 else "completed"
    manifest["status"] = manifest_status
    manifest_dirty = True

    if manifest_dirty:
        _strip_model_invariants(manifest)
        atomic_write_json(fs, manifest_path(paths_cfg), manifest)

    if remaining <= 0:
        print(
            f"No iterations remaining (completed={completed_before}, configured={params.training.num_iterations})."
        )
        return

    mode = args.mode.lower()
    print(f"[Main] Mode: {mode}")
    print(_format_eval_schedule(cfg.evaluation))
    params = params.model_copy(
        update={
            "training": params.training.model_copy(update={"num_iterations": remaining})
        }
    )
    try:
        run_training(
            mode,
            model,
            optimizer,
            game,
            params,
            fs,
            completed_offset=completed_before,
            champion_path=champion_path,
            replay_shards=replay_shards,
        )
    except KeyboardInterrupt:
        print("Ctrl+C detected. Saving manifest before exit...")
        manifest_current = load_manifest(paths_cfg, fs) or {}
        progress = manifest_current.get("progress", {})
        completed = int(progress.get("completed_iterations", completed_before))
        manifest_current["status"] = "stopped"
        manifest_current.setdefault("progress", {})["completed_iterations"] = completed
        atomic_write_json(fs, manifest_path(paths_cfg), manifest_current)
        print("Manifest saved. Exiting.")
        raise


if __name__ == "__main__":
    main()
