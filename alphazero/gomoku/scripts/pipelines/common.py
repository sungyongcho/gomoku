from collections import deque
from collections.abc import Sequence
import json
import os
import random
import re
import shutil
from typing import Any

import fsspec
import torch

from gomoku.alphazero.eval.arena import run_arena
from gomoku.alphazero.learning.dataset import (
    GameSample,
    decode_policy,
    reconstruct_state,
)
from gomoku.core.gomoku import Gomoku
from gomoku.inference.local import LocalInference
from gomoku.model.policy_value_net import PolicyValueNet
from gomoku.utils.config.loader import RunnerParams
from gomoku.utils.config.schedule_param import get_scheduled_value
from gomoku.utils.io_helpers import (
    atomic_write_json,
    list_replay_shards,
    read_parquet_shard,
)
from gomoku.utils.paths import (
    format_path,
    iteration_ckpt_path,
    load_state_dict_from_fs,
    manifest_path,
    save_state,
)
from gomoku.utils.state_dict_utils import align_state_dict_to_model


def load_resume_state(
    params: RunnerParams,
    fs: fsspec.AbstractFileSystem,
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, str, list[str]]:
    """
    Load model/optimizer from manifest and return completed iteration offset.

    Parameters
    ----------
    params : RunnerParams
        Aggregated configuration for the current run.
    fs : fsspec.AbstractFileSystem
        Filesystem abstraction for checkpoints and manifest.
    model : PolicyValueNet
        Model instance to load weights into.
    optimizer : torch.optim.Optimizer
        Optimizer instance to load state into (if available).

    Returns
    -------
    tuple[int, str, list[str]]
        ``(completed_iterations, champion_path, replay_shards)`` where
        `completed_iterations` is the number of iterations already finished,
        `champion_path` is the current champion checkpoint path, and
        `replay_shards` is the list of existing replay shard paths (may be empty).

    """
    mpath = manifest_path(params.paths)
    manifest = _safe_load_manifest(fs, mpath)
    completed_before = 0
    champion_path = None
    replay_shards: list[str] = []

    if manifest:
        completed_before = int(
            manifest.get("progress", {}).get("completed_iterations", 0)
        )
        champion_path = manifest.get("champion_model_path")
    else:
        print(f"[Resume] Manifest not found at {mpath}; will try seed checkpoint.")

    if not champion_path:
        # Fall back to iteration 0 checkpoint if specified
        seed_path = iteration_ckpt_path(params.paths, 0)
        if fs.exists(seed_path):
            champion_path = seed_path

    if not champion_path or not fs.exists(champion_path):
        print(
            "[Resume] Checkpoint missing. Either create an initial checkpoint"
            f" at {iteration_ckpt_path(params.paths, 0)} or start a fresh run."
        )
        raise FileNotFoundError(
            f"Checkpoint to load not found. Checked manifest and: {champion_path}"
        )

    state_dict = load_state_dict_from_fs(fs, champion_path, torch.device(model.device))
    resized, missing, dropped = align_state_dict_to_model(
        state_dict, model.state_dict()
    )
    if resized or missing or dropped:
        print(
            "[StateDict] Resized tensors:",
            ", ".join(resized[:10]) + ("..." if len(resized) > 10 else "")
            if resized
            else "none",
        )
        if missing:
            print(f"[StateDict] Missing keys: {len(missing)}")
        if dropped:
            print(f"[StateDict] Dropped keys: {len(dropped)}")
    model.load_state_dict(state_dict, strict=False)

    # Try loading optimizer state if present
    optim_path = champion_path + ".optim"
    if fs.exists(optim_path):
        try:
            optim_state = load_state_dict_from_fs(
                fs, optim_path, torch.device(model.device)
            )
            optimizer.load_state_dict(optim_state)
        except Exception as exc:  # noqa: BLE001
            print(f"[Resume] Failed to load optimizer state from {optim_path}: {exc}")

    # Enumerate existing replay shards for optional warm-start
    replay_dir = format_path(params.paths, params.paths.replay_dir)
    replay_shards = list_replay_shards(fs, replay_dir) if fs.exists(replay_dir) else []

    return completed_before, champion_path, replay_shards


def warm_start_replay_buffer(
    params: RunnerParams,
    fs: fsspec.AbstractFileSystem,
    game,
    *,
    replay_shards: list[str] | None = None,
) -> tuple[deque[GameSample], list[str]]:
    """
    Hydrate an in-memory replay buffer from existing shards.

    Returns a bounded deque (maxlen=replay_buffer_size) and the list of shard
    paths actually loaded.
    """
    max_len = max(1, int(getattr(params.training, "replay_buffer_size", 1)))
    buffer: deque[GameSample] = deque(maxlen=max_len)

    replay_dir = format_path(params.paths, params.paths.replay_dir)
    shard_paths = replay_shards or (
        list_replay_shards(fs, replay_dir) if fs.exists(replay_dir) else []
    )
    if not shard_paths:
        return buffer, []

    io_cfg = getattr(params, "io", None)
    max_iters = getattr(io_cfg, "initial_replay_iters", None)
    max_shards = getattr(io_cfg, "initial_replay_shards", None)
    max_samples_per_shard = getattr(io_cfg, "max_samples_per_shard", None)
    cache_dir = getattr(io_cfg, "local_replay_cache", None)

    def parse_iter(path: str) -> int | None:
        m = re.search(r"shard-iter(\d+)-", os.path.basename(path))
        return int(m.group(1)) if m else None

    shard_infos: list[tuple[int | None, str]] = [
        (parse_iter(p), p) for p in shard_paths
    ]
    shard_infos.sort(key=lambda x: (x[0] if x[0] is not None else -1, x[1]))

    # Apply iteration-based limit if present (keep latest N iterations)
    if max_iters is not None and max_iters > 0:
        iter_order: list[int] = []
        for iter_no, _ in shard_infos:
            if iter_no is None:
                continue
            if iter_no not in iter_order:
                iter_order.append(iter_no)
        keep_iters = set(iter_order[-max_iters:]) if iter_order else set()
        shard_infos = [
            (iter_no, p)
            for iter_no, p in shard_infos
            if (iter_no in keep_iters) or (iter_no is None)
        ]

    if max_shards is not None and max_shards > 0:
        shard_infos = shard_infos[-max_shards:]

    shards_to_load = [p for _, p in shard_infos]
    cache_hits = downloads = 0

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        local_fs = fsspec.filesystem("file")
    else:
        local_fs = None

    for shard in shards_to_load:
        read_fs = fs
        read_path = shard
        if cache_dir and local_fs is not None:
            cache_path = os.path.join(cache_dir, os.path.basename(shard))
            if not local_fs.exists(cache_path):
                with (
                    fs.open(shard, "rb") as src,
                    local_fs.open(cache_path, "wb") as dst,
                ):
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
                downloads += 1
            else:
                cache_hits += 1
            read_fs = local_fs
            read_path = cache_path

        rows = read_parquet_shard(read_fs, read_path)
        if max_samples_per_shard is not None and max_samples_per_shard > 0:
            rows = rows[:max_samples_per_shard]

        for row in rows:
            if "last_move_idx" not in row and "last_move" in row:
                row = dict(row)
                row["last_move_idx"] = row["last_move"]
            state = reconstruct_state(row, game)
            policy = decode_policy(row["policy_probs"])
            value = float(row.get("value", 0.0))
            priority = float(row.get("priority", 1.0))
            buffer.append(GameSample(state, policy, value, priority))

    if cache_dir and local_fs is not None:
        print(
            f"[ReplayWarmStart] cache hits={cache_hits}, downloads={downloads}, loaded_shards={len(shards_to_load)}"
        )
    else:
        print(f"[ReplayWarmStart] loaded_shards={len(shards_to_load)}")

    return buffer, shards_to_load


def save_checkpoint(
    fs: fsspec.AbstractFileSystem,
    params: RunnerParams,
    iteration: int,
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    prefix: str = "",
) -> str:
    """Save model (and optionally optimizer) checkpoint for the given iteration."""
    ckpt_path = iteration_ckpt_path(params.paths, iteration)
    save_state(fs, model.state_dict(), ckpt_path)
    if optimizer is not None:
        opt_path = ckpt_path + ".optim"
        save_state(fs, optimizer.state_dict(), opt_path)
    print(f"{prefix}Saved checkpoint: {ckpt_path}")
    return ckpt_path


def update_manifest(
    params: RunnerParams,
    fs: fsspec.AbstractFileSystem,
    *,
    completed_iterations: int,
    status: str,
    champion_path: str | None = None,
    eval_summary: dict[str, Any] | None = None,
    prefix: str = "",
) -> None:
    """Update manifest progress/status and optional champion path."""
    mpath = manifest_path(params.paths)
    manifest = _safe_load_manifest(fs, mpath)
    if manifest is None:
        print(f"{prefix}Manifest not found at {mpath}; skipping update.")
        return

    cfg_name = getattr(params, "config_name", None)
    if cfg_name:
        cfgs = manifest.get("configs")
        if not isinstance(cfgs, list):
            cfgs = []
        last_name = cfgs[-1]["name"] if cfgs else None
        if last_name != cfg_name:
            cfgs.append(
                {
                    "revision": len(cfgs),
                    "name": cfg_name,
                    "begin_iteration": int(completed_iterations) + 1,
                }
            )
            manifest["configs"] = cfgs

    progress = manifest.get("progress")
    if not isinstance(progress, dict):
        progress = {}
    progress["completed_iterations"] = int(completed_iterations)
    manifest["progress"] = progress
    manifest["status"] = status
    if champion_path:
        manifest["champion_model_path"] = champion_path
    if eval_summary:
        evals = manifest.get("evaluations")
        if not isinstance(evals, list):
            evals = []
        evals.append({"iteration": completed_iterations, **eval_summary})
        manifest["evaluations"] = evals
        champ_elo_decision = eval_summary.get("champion_elo_after")
        if champ_elo_decision is not None:
            manifest["elo"] = {"champion": float(champ_elo_decision)}
        if eval_summary.get("promote"):
            promos = manifest.get("promotions")
            if not isinstance(promos, list):
                promos = []
            promos.append(
                {
                    "iteration": completed_iterations,
                    "win_rate": eval_summary.get("win_rate"),
                    "baseline_win_rate": eval_summary.get("baseline_win_rate"),
                }
            )
            manifest["promotions"] = promos

    atomic_write_json(fs, mpath, manifest)


def _safe_load_manifest(
    fs: fsspec.AbstractFileSystem, path: str
) -> dict[str, Any] | None:
    """Load manifest JSON; return None on errors."""
    if not fs.exists(path):
        return None
    try:
        with fs.open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        return manifest if isinstance(manifest, dict) else None
    except Exception:
        return None


def evaluate_and_maybe_promote(
    game: Gomoku,
    model: PolicyValueNet,
    params: RunnerParams,
    fs: fsspec.AbstractFileSystem,
    *,
    champion_path: str | None,
    ckpt_path: str,
    iteration: int,
) -> tuple[str | None, dict[str, Any] | None]:
    """Run arena evaluation and decide promotion."""
    manifest_obj = _safe_load_manifest(fs, manifest_path(params.paths)) or {}
    eval_cfg = getattr(params, "evaluation", None)
    if not eval_cfg or getattr(eval_cfg, "num_eval_games", 0) <= 0:
        return champion_path or ckpt_path, None

    every = max(1, int(getattr(eval_cfg, "eval_every_iters", 1)))
    if iteration % every != 0:
        return champion_path or ckpt_path, None

    if not champion_path or not fs.exists(champion_path):
        print(
            f"[Eval] Champion checkpoint missing at {champion_path}; skipping evaluation."
        )
        return champion_path or ckpt_path, None

    device_str = str(model.device)
    try:
        champ_model = PolicyValueNet(game, params.model, device=device_str)
        champ_state = load_state_dict_from_fs(
            fs, champion_path, torch.device(device_str)
        )
        champ_model.load_state_dict(champ_state, strict=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[Eval] Failed to load champion model from {champion_path}: {exc}")
        return champion_path or ckpt_path, None

    baseline_model = None
    baseline_path = manifest_obj.get("baseline_model_path")
    if baseline_path and fs.exists(baseline_path):
        try:
            baseline_model = PolicyValueNet(game, params.model, device=device_str)
            base_state = load_state_dict_from_fs(
                fs, baseline_path, torch.device(device_str)
            )
            baseline_model.load_state_dict(base_state, strict=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[Eval] Failed to load baseline model from {baseline_path}: {exc}")
            baseline_model = None

    inference_new = LocalInference(model)
    inference_best = LocalInference(champ_model)
    baseline_inference = LocalInference(baseline_model) if baseline_model else None

    last_eval = None
    prev_evals = manifest_obj.get("evaluations")
    if isinstance(prev_evals, list) and prev_evals:
        last_eval = prev_evals[-1]

    summary = run_arena(
        params,
        inference_new,
        inference_best,
        eval_cfg=eval_cfg,
        matches=int(eval_cfg.num_eval_games),
        baseline_inference=baseline_inference,
    )

    # Elo-like score tracking (lightweight, not persisted elsewhere)
    wins = int(summary.get("wins", 0))
    losses = int(summary.get("losses", 0))
    draws = int(summary.get("draws", 0))
    games_played = max(1, wins + losses + draws)
    score_rate = (wins + 0.5 * draws) / games_played

    elo_state = manifest_obj.get("elo") or {}
    champion_elo_before = elo_state.get("champion", 1500.0)
    if isinstance(champion_elo_before, dict):
        champion_elo_before = champion_elo_before.get("elo", 1500.0)
    champion_elo_before = float(champion_elo_before)
    challenger_elo_before = champion_elo_before
    k_factor = float(getattr(eval_cfg, "elo_k_factor", 2.0))
    # k_factor = 32.0
    expected_challenger = 1.0 / (
        1.0 + 10 ** ((champion_elo_before - challenger_elo_before) / 400.0)
    )
    challenger_elo_after = challenger_elo_before + k_factor * games_played * (
        score_rate - expected_challenger
    )
    champion_elo_after = champion_elo_before + k_factor * games_played * (
        (1.0 - score_rate) - (1.0 - expected_challenger)
    )

    summary.update(
        {
            "score_rate": score_rate,
            "elo_k": k_factor,
            "champion_elo_before": champion_elo_before,
            "champion_elo_after": champion_elo_after,
            "challenger_elo_before": challenger_elo_before,
            "challenger_elo_after": challenger_elo_after,
            "baseline_win_rate_prev": last_eval.get("baseline_win_rate")
            if isinstance(last_eval, dict)
            else None,
            "blunder_rate_prev": last_eval.get("blunder_rate")
            if isinstance(last_eval, dict)
            else None,
        }
    )
    print()
    print(format_eval_summary(summary))

    promoted_path = champion_path
    if summary.get("promote"):
        promoted_path = ckpt_path
        # 승급 시: 도전자의 점수(상승된 점수)를 챔피언 점수로 반영
        summary["champion_elo_after"] = summary.get("challenger_elo_after")
    else:
        # 승급 실패 시: 챔피언 점수를 변동시키지 않음 (원상 복구)
        summary["champion_elo_after"] = champion_elo_before

    decision, guard_parts = _build_guardrail_parts(summary)
    decision_line = f"[Eval] Iteration {iteration} decision={decision}"
    if guard_parts:
        decision_line += " | " + " | ".join(guard_parts)
    print(decision_line)

    return promoted_path, summary


def select_past_checkpoint(
    fs: fsspec.AbstractFileSystem,
    paths,
    current_iter: int,
    window: int,
    champion_path: str | None,
) -> str | None:
    """Pick a past checkpoint path within the given window before current_iter."""
    ckpt_dir = format_path(paths, paths.ckpt_dir)
    if not fs.exists(ckpt_dir):
        return champion_path

    try:
        entries = fs.ls(ckpt_dir)
    except Exception:  # noqa: BLE001
        return champion_path

    candidates: list[tuple[int, str]] = []
    for entry in entries:
        name = os.path.basename(entry)
        m = re.match(r"iteration_(\d+).pt$", name)
        if not m:
            continue
        idx = int(m.group(1))
        if idx < current_iter:
            candidates.append((idx, entry))

    if not candidates:
        return champion_path

    candidates.sort(key=lambda x: x[0], reverse=True)
    if window > 0:
        candidates = candidates[:window]
    return random.choice(candidates)[1]


def log_iteration_header(current: int, total: int) -> None:
    """Print a standardized iteration header."""
    print(f"\n--- Iteration {current} / {total} ---")


def format_schedule_line(
    *,
    lr: float,
    temp: float,
    epsilon: float,
    mcts_searches: float,
    total_games: int,
    random_ratio: float,
    random_bot_rate: float | None = None,
    prev_bot_rate: float | None = None,
    eval_searches: float | None = None,
    dir_alpha: float | None = None,
    explore_turns: int | None = None,
) -> str:
    """Format a schedule summary line."""
    searches = int(mcts_searches)
    parts = [
        f"LR={lr:.6f}",
        f"Temp={temp:.2f}",
        f"Epsilon={epsilon:.3f}",
    ]
    if dir_alpha is not None:
        parts.append(f"DirAlpha={dir_alpha:.3f}")
    parts.append(f"MCTS Searches={searches}")
    if explore_turns is not None:
        parts.append(f"ExploreTurns={explore_turns}")
    if eval_searches is not None:
        parts.append(f"Eval Searches={int(eval_searches)}")
    parts.append(f"Total Games={int(total_games)}")
    parts.append(f"RandomRatio={random_ratio:.2f}")
    if random_bot_rate is not None or prev_bot_rate is not None:
        rnd_bot = 0.0 if random_bot_rate is None else random_bot_rate
        prev_bot = 0.0 if prev_bot_rate is None else prev_bot_rate
        parts.append(f"Opp(Random)={rnd_bot:.2f}")
        parts.append(f"Opp(Prev)={prev_bot:.2f}")
    return "Scheduled params: " + ", ".join(parts)


def compute_iteration_schedule(
    params: RunnerParams, schedule_idx: int
) -> dict[str, float]:
    """Compute scheduled values for a given iteration index."""
    temp = get_scheduled_value(params.training.temperature, schedule_idx)
    dir_eps = get_scheduled_value(params.mcts.dirichlet_epsilon, schedule_idx)
    dir_alpha = get_scheduled_value(params.mcts.dirichlet_alpha, schedule_idx)
    num_searches = get_scheduled_value(params.mcts.num_searches, schedule_idx)
    explore_turns = int(
        get_scheduled_value(params.mcts.exploration_turns, schedule_idx)
    )
    eval_searches = None
    if getattr(params.evaluation, "eval_num_searches", None) is not None:
        eval_searches = get_scheduled_value(
            params.evaluation.eval_num_searches, schedule_idx
        )
    num_games = get_scheduled_value(
        params.training.num_selfplay_iterations, schedule_idx
    )
    lr_val = get_scheduled_value(params.training.learning_rate, schedule_idx)
    rnd_ratio = (
        get_scheduled_value(params.training.random_play_ratio, schedule_idx)
        if params.training.random_play_ratio is not None
        else 0.0
    )
    rnd_opening = int(
        get_scheduled_value(params.training.random_opening_turns, schedule_idx)
        if params.training.random_opening_turns is not None
        else 0
    )
    opp_cfg = getattr(params.training, "opponent_rates", None)
    rnd_bot_rate = (
        get_scheduled_value(opp_cfg.random_bot_ratio, schedule_idx)
        if opp_cfg is not None
        else 0.0
    )
    prev_bot_rate = (
        get_scheduled_value(opp_cfg.prev_bot_ratio, schedule_idx)
        if opp_cfg is not None
        else 0.0
    )
    rnd_bot_rate = max(0.0, float(rnd_bot_rate))
    prev_bot_rate = max(0.0, float(prev_bot_rate))
    total_rate = rnd_bot_rate + prev_bot_rate
    if total_rate > 1.0:
        scale = 1.0 / total_rate
        rnd_bot_rate *= scale
        prev_bot_rate *= scale

    return {
        "temp": float(temp),
        "dir_eps": float(dir_eps),
        "dir_alpha": float(dir_alpha),
        "num_searches": float(num_searches),
        "explore_turns": explore_turns,
        "eval_searches": float(eval_searches) if eval_searches is not None else None,
        "num_games": int(num_games),
        "lr": float(lr_val),
        "rnd_ratio": float(rnd_ratio),
        "rnd_opening": rnd_opening,
        "rnd_bot_rate": rnd_bot_rate,
        "prev_bot_rate": prev_bot_rate,
    }


def build_selfplay_summary(
    samples: Sequence[GameSample],
) -> dict[str, float] | None:
    """
    Summarize self-play outcomes for logging.

    Parameters
    ----------
    samples : Sequence[GameSample]
        Flattened samples from self-play games.

    Returns
    -------
    dict[str, float] | None
        Summary fields (num_samples, avg_result, pos_ratio, optional game stats)
        or ``None`` when no samples are present.

    """
    if not samples:
        return None

    total_samples = len(samples)
    total_value = 0.0
    positive = 0
    root_games = 0
    root_value_sum = 0.0

    for sample in samples:
        value = float(getattr(sample, "value", 0.0))
        total_value += value
        if value > 0:
            positive += 1
        state = getattr(sample, "state", None)
        last_move_idx = getattr(state, "last_move_idx", None)
        if last_move_idx is None or int(last_move_idx) < 0:
            root_games += 1
            root_value_sum += value

    summary = {
        "num_samples": total_samples,
        "avg_result": total_value / total_samples,
        "pos_ratio": positive / total_samples,
    }
    if root_games > 0:
        avg_game_result = root_value_sum / root_games
        first_win_rate = 0.5 * (avg_game_result + 1.0)
        summary.update(
            {
                "game_count": root_games,
                "avg_game_result": avg_game_result,
                "first_player_win_rate": first_win_rate,
                "second_player_win_rate": 1.0 - first_win_rate,
            }
        )
    return summary


def format_training_metrics(
    metrics: dict,
    *,
    prefix: str = "[Training] ",
    iteration: int | None = None,
    duration_sec: float | None = None,
) -> str:
    """
    Format training metrics to mirror gmk console output.

    Parameters
    ----------
    metrics : dict
        Metrics dictionary returned by the trainer.
    prefix : str, optional
        Prefix inserted in front of the message.
    iteration : int, optional
        Iteration index for display; when omitted the label is skipped.
    duration_sec : float, optional
        Elapsed training seconds; when omitted the duration field is skipped.

    Returns
    -------
    str
        Human-readable metrics line.

    """
    head_parts: list[str] = []
    if iteration is not None:
        head_parts.append(f"Iteration {iteration}")
    if duration_sec is not None:
        head_parts.append(f"finished in {duration_sec:.1f}s")

    last_loss = float(metrics.get("last_batch_loss", metrics.get("loss", 0.0)))
    last_policy = float(
        metrics.get("last_policy_loss", metrics.get("policy_loss", 0.0))
    )
    last_value = float(metrics.get("last_value_loss", metrics.get("value_loss", 0.0)))

    base_prefix = prefix.rstrip()
    head = f"{base_prefix} "
    if head_parts:
        head = f"{base_prefix} {' '.join(head_parts)}. "

    metric_str = (
        f"Last batch metrics: loss={last_loss:.4f}, "
        f"policy={last_policy:.4f}, value={last_value:.4f}"
    )
    avg_present = {"loss", "policy_loss", "value_loss"} <= set(metrics.keys())
    if avg_present and any(k.startswith("last_") for k in metrics):
        avg_loss = float(metrics.get("loss", 0.0))
        avg_policy = float(metrics.get("policy_loss", 0.0))
        avg_value = float(metrics.get("value_loss", 0.0))
        metric_str += f" | Avg: loss={avg_loss:.4f}, policy={avg_policy:.4f}, value={avg_value:.4f}"

    return f"{head}{metric_str}"


def _percent_str(value: float | None) -> str:
    """Return percentage string or placeholder."""
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def _float_str(value: float | None, digits: int = 4) -> str:
    """Return float string with fixed precision or placeholder."""
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _build_guardrail_parts(summary: dict[str, Any]) -> tuple[str, list[str]]:
    """Build guardrail decision details for promotion logging."""
    decision = "PROMOTE" if summary.get("promote") else "REJECT"
    parts: list[str] = []

    wr = summary.get("win_rate")
    promo_target = summary.get("promotion_win_rate")
    pass_promo = summary.get("pass_promotion")
    if promo_target is not None and wr is not None:
        status = "PASS" if pass_promo else "FAIL"
        cmp_symbol = ">=" if pass_promo else "<"
        parts.append(
            f"promo {status}: WR {_percent_str(wr)} {cmp_symbol} "
            f"target {_percent_str(promo_target)}"
        )

    base_wr = summary.get("baseline_win_rate")
    base_target = summary.get("baseline_win_rate_required")
    pass_base = summary.get("pass_baseline")
    if base_target is not None and base_wr is not None:
        status = "PASS" if pass_base else "FAIL"
        cmp_symbol = ">=" if pass_base else "<"
        parts.append(
            f"baseline {status}: WR {_percent_str(base_wr)} {cmp_symbol} "
            f"target {_percent_str(base_target)}"
        )

    blunder = summary.get("blunder_rate")
    blunder_limit = summary.get("blunder_rate_limit")
    pass_blunder = summary.get("pass_blunder")
    if blunder_limit is not None and blunder is not None:
        status = "PASS" if pass_blunder else "FAIL"
        cmp_symbol = "<=" if pass_blunder else ">"
        parts.append(
            f"blunder {status}: {_float_str(blunder)} {cmp_symbol} "
            f"limit {_float_str(blunder_limit)}"
        )

    return decision, parts


def format_eval_summary(summary: dict[str, Any], *, prefix: str = "[Eval] ") -> str:
    """Format evaluation summary metrics."""
    wins = int(summary.get("wins", 0) or 0)
    losses = int(summary.get("losses", 0) or 0)
    draws = int(summary.get("draws", 0) or 0)
    games = int(summary.get("games", wins + losses + draws) or 0)
    wr = summary.get("win_rate")
    score = summary.get("score_rate")
    blunder = summary.get("blunder_rate")
    base_wr = summary.get("baseline_win_rate")
    base_blunder = summary.get("baseline_blunder_rate")
    base_prev = summary.get("baseline_win_rate_prev")
    blunder_prev = summary.get("blunder_rate_prev")
    base_target = summary.get("baseline_win_rate_required")
    blunder_limit = summary.get("blunder_rate_limit")
    k_factor = summary.get("elo_k")
    champ_before = summary.get("champion_elo_before")
    champ_after = summary.get("champion_elo_after")
    chall_before = summary.get("challenger_elo_before")
    chall_after = summary.get("challenger_elo_after")
    decision, guard_parts = _build_guardrail_parts(summary)

    lines = [
        (
            f"{prefix}H2H | WR={_percent_str(wr)} | SR={_percent_str(score)} | "
            f"Blunder={_float_str(blunder)} | Games={games} "
            f"(W{wins}/L{losses}/D{draws})"
        )
    ]
    if (
        k_factor is not None
        and champ_before is not None
        and champ_after is not None
        and chall_before is not None
        and chall_after is not None
    ):
        champ_delta = float(champ_after) - float(champ_before)
        chall_delta = float(chall_after) - float(chall_before)
        lines.append(
            f"{prefix}Elo | K={float(k_factor):.1f} | "
            f"Champion: {float(champ_before):.1f}->{float(champ_after):.1f} "
            f"({champ_delta:+.1f}) | "
            f"Challenger: {float(chall_before):.1f}->{float(chall_after):.1f} "
            f"({chall_delta:+.1f})"
        )
    if blunder is not None:
        if blunder_prev is not None:
            lines.append(
                f"{prefix}Stability | Blunder={_float_str(blunder)} "
                f"(prev {_float_str(blunder_prev)})"
            )
        elif blunder_limit is not None:
            lines.append(
                f"{prefix}Stability | Blunder={_float_str(blunder)} "
                f"(limit {_float_str(blunder_limit)})"
            )
        else:
            lines.append(f"{prefix}Stability | Blunder={_float_str(blunder)}")
    if base_wr is not None:
        base_line = f"{prefix}Baseline | WR={_percent_str(base_wr)}"
        if base_target is not None:
            base_line += f" (target {_percent_str(base_target)})"
        if base_prev is not None:
            base_line += f", Prev={_percent_str(base_prev)}"
        if base_blunder is not None:
            base_line += f", Blunder={_float_str(base_blunder)}"
        lines.append(base_line)

    decision_line = f"{prefix}Decision | {decision}"
    if guard_parts:
        decision_line += " | " + " | ".join(guard_parts)
    lines.append(decision_line)

    return "\n".join(lines)


def build_evaluation_log_entry(
    *,
    iteration: int,
    promoted: bool,
    eval_summary: dict[str, Any] | None,
    training_metrics: dict[str, float] | None,
    scheduled_params: dict[str, float],
    selfplay_summary: dict[str, float] | None,
    timing_sec: dict[str, float],
    eval_games_requested: int,
    priority_replay_active: bool,
    priority_trigger_reason: str | None = None,
) -> dict[str, Any]:
    """
    Assemble an evaluation log entry aligned with gmk schema.

    Parameters
    ----------
    iteration : int
        1-based iteration index.
    promoted : bool
        Whether the challenger was promoted.
    eval_summary : dict[str, Any] | None
        Evaluation summary produced by ``evaluate_and_maybe_promote``.
    training_metrics : dict[str, float] | None
        Training metrics for the iteration.
    scheduled_params : dict[str, float]
        Snapshot of scheduled parameters (lr, temperature, searches, etc.).
    selfplay_summary : dict[str, float] | None
        Summary of self-play samples for the iteration.
    timing_sec : dict[str, float]
        Durations for self-play/training/evaluation in seconds.
    eval_games_requested : int
        Requested number of evaluation games from the config.
    priority_replay_active : bool
        Whether priority replay (PER) was active.
    priority_trigger_reason : str | None, optional
        Reason for enabling priority replay (if applicable).

    Returns
    -------
    dict[str, Any]
        Log entry ready to append to ``evaluation_log.jsonl``.

    """
    summary = eval_summary or {}

    def _as_float(value: Any, default: float = 0.0) -> float:
        if value is None:
            return float(default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _as_int(value: Any, default: int = 0) -> int:
        if value is None:
            return int(default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    wins = int(summary.get("wins", 0) or 0)
    losses = int(summary.get("losses", 0) or 0)
    draws = int(summary.get("draws", 0) or 0)
    h2h_games = int(summary.get("games", wins + losses + draws) or 0)
    eval_games_requested = max(0, int(eval_games_requested))
    h2h_games_missing = max(0, eval_games_requested - h2h_games)

    timing = {
        "selfplay": _as_float(timing_sec.get("selfplay", 0.0)),
        "training": _as_float(timing_sec.get("training", 0.0)),
        "evaluation": _as_float(timing_sec.get("evaluation", 0.0)),
    }

    elo_after_eval = _as_float(summary.get("champion_elo_after", 0.0))
    challenger_elo_after_eval = _as_float(summary.get("challenger_elo_after", 0.0))
    champion_after_decision = _as_float(
        summary.get("champion_elo_after", elo_after_eval), elo_after_eval
    )

    return {
        "iteration": iteration,
        "promoted": bool(promoted),
        "elo_after_eval": elo_after_eval,
        "champion_elo_after_decision": champion_after_decision,
        "challenger_elo_after_eval": challenger_elo_after_eval,
        "training_metrics": training_metrics or {},
        "scheduled_params": scheduled_params,
        "selfplay_summary": selfplay_summary,
        "h2h_wins": wins,
        "h2h_losses": losses,
        "h2h_draws": draws,
        "h2h_games": h2h_games,
        "eval_games_requested": eval_games_requested,
        "h2h_games_missing": h2h_games_missing,
        "score_rate": _as_float(summary.get("score_rate", 0.0)),
        "blunder_rate": _as_float(summary.get("blunder_rate", 0.0)),
        "baseline_win_rate": _as_float(summary.get("baseline_win_rate", 0.0)),
        "timing_sec": timing,
        "priority_replay_active": priority_replay_active,
        "priority_trigger_reason": priority_trigger_reason,
        "evaluation_details": {
            "avg_game_length": _as_float(summary.get("avg_length", 0.0)),
            "max_loss_streak": _as_int(summary.get("max_loss_streak", 0)),
        },
    }
