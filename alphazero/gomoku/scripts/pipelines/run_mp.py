from collections.abc import Callable
import functools
import time

import fsspec
import torch

from gomoku.alphazero.learning.dataset import flatten_game_records
from gomoku.alphazero.learning.trainer import AlphaZeroTrainer
from gomoku.alphazero.runners.multiprocess_runner import MultiprocessRunner
from gomoku.core.gomoku import Gomoku
from gomoku.model.policy_value_net import PolicyValueNet
from gomoku.scripts.pipelines.common import (
    build_evaluation_log_entry,
    build_selfplay_summary,
    compute_iteration_schedule,
    evaluate_and_maybe_promote,
    format_schedule_line,
    format_training_metrics,
    log_iteration_header,
    save_checkpoint,
    select_past_checkpoint,
    update_manifest,
    warm_start_replay_buffer,
)
from gomoku.utils.config.loader import RunnerParams
from gomoku.utils.io_helpers import atomic_append_jsonl, save_as_parquet_shard
from gomoku.utils.paths import evaluation_log_path, new_replay_shard_path
from gomoku.utils.serialization import to_savable_sample


def _build_model_with_state(
    game: Gomoku,
    cfg,
    *,
    state_dict: dict[str, torch.Tensor],
) -> PolicyValueNet:
    """Factory to build a model on CPU and load provided state dict."""
    model = PolicyValueNet(game, cfg.model, device="cpu")
    model.load_state_dict(state_dict, strict=False)
    return model


def run_mp(
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    game: Gomoku,
    params: RunnerParams,
    fs: fsspec.AbstractFileSystem,
    *,
    completed_offset: int = 0,
    champion_path: str | None = None,
    replay_shards: list[str] | None = None,
    manifest_updater: Callable | None = None,
) -> None:
    """
    Run vectorized self-play and training using the mp search engine.

    Parameters
    ----------
    model : PolicyValueNet
        Policy/value network to train.
    optimizer : torch.optim.Optimizer
        Optimizer handling model updates.
    game : Gomoku
        Game instance carrying board configuration.
    params : RunnerParams
        Aggregated config for self-play, MCTS, training, and paths.
    fs : fsspec.AbstractFileSystem
        Filesystem abstraction for replay/ckpt writes.
    completed_offset : int, optional
        Iterations already completed (for resume). Defaults to 0.
    manifest_updater : callable, optional
        Hook to update manifest progress; falls back to local updater when absent.

    Notes
    -----
    Uses ``MultiprocessRunner`` to spawn self-play workers that delegate inference
    to a dedicated IPC inference server, then runs a training epoch.

    """
    trainer = AlphaZeroTrainer(
        train_cfg=params.training, model=model, optimizer=optimizer, game=game
    )

    num_iters = int(params.training.num_iterations)
    total_done_before = int(completed_offset)
    replay_buffer, loaded_shards = warm_start_replay_buffer(
        params, fs, game, replay_shards=replay_shards
    )
    if loaded_shards:
        print(f"[MP] Warm-started replay buffer with {len(loaded_shards)} shard(s).")

    for it in range(num_iters):
        global_iter = total_done_before + it + 1
        log_iteration_header(global_iter, total_done_before + num_iters)

        schedule_idx = completed_offset + it
        sched = compute_iteration_schedule(params, schedule_idx)
        print(
            format_schedule_line(
                lr=sched["lr"],
                temp=sched["temp"],
                epsilon=sched["dir_eps"],
                mcts_searches=sched["num_searches"],
                total_games=sched["num_games"],
                random_ratio=sched["rnd_ratio"],
                random_bot_rate=sched["rnd_bot_rate"],
                prev_bot_rate=sched["prev_bot_rate"],
                eval_searches=sched["eval_searches"],
            )
        )
        scheduled_params_snapshot = {
            "learning_rate": sched["lr"],
            "temperature": sched["temp"],
            "dirichlet_epsilon": sched["dir_eps"],
            "random_play_ratio": sched["rnd_ratio"],
            "random_bot_rate": sched["rnd_bot_rate"],
            "prev_bot_rate": sched["prev_bot_rate"],
            "num_searches": sched["num_searches"],
            "eval_num_searches": sched["eval_searches"],
            "num_selfplay_iterations": sched["num_games"],
        }
        # Apply scheduled learning rate to optimizer before training.
        for group in optimizer.param_groups:
            group["lr"] = float(sched["lr"])
        priority_cfg = getattr(params.training, "priority_replay", None)
        if sched["num_games"] <= 0:
            print("[MP] No self-play games scheduled; skipping iteration.")
            updater_fn = manifest_updater or update_manifest
            updater_fn(
                params,
                fs,
                completed_iterations=global_iter,
                status="running" if (it + 1) < num_iters else "completed",
                prefix="[MP] ",
            )
            continue

        mcts_cfg = params.mcts.model_copy(
            update={
                "dirichlet_epsilon": sched["dir_eps"],
                "dirichlet_alpha": sched["dir_alpha"],
                "num_searches": sched["num_searches"],
                "exploration_turns": sched["explore_turns"],
            }
        )
        opp_cfg = getattr(params.training, "opponent_rates", None)
        opp_cfg = (
            opp_cfg.model_copy(
                update={
                    "random_bot_ratio": sched["rnd_bot_rate"],
                    "prev_bot_ratio": sched["prev_bot_rate"],
                }
            )
            if opp_cfg is not None
            else None
        )
        train_cfg = params.training.model_copy(
            update={
                "temperature": sched["temp"],
                "random_play_ratio": sched["rnd_ratio"],
                "opponent_rates": opp_cfg,
            }
        )
        iter_cfg = params.model_copy(update={"training": train_cfg, "mcts": mcts_cfg})

        num_workers = max(1, int(params.parallel.mp_num_workers or 1))
        base = sched["num_games"] // num_workers
        rem = sched["num_games"] % num_workers
        games_per_worker = [base + (1 if i < rem else 0) for i in range(num_workers)]
        games_per_worker = [g for g in games_per_worker if g > 0]

        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

        past_ckpt = None
        opp_cfg = getattr(params.training, "opponent_rates", None)
        if opp_cfg is not None and sched["prev_bot_rate"] > 0.0:
            window = getattr(opp_cfg, "past_model_window", 5) or 5
            past_ckpt = select_past_checkpoint(
                fs=fs,
                paths=params.paths,
                current_iter=global_iter,
                window=int(window),
                champion_path=champion_path,
            )

        runner = MultiprocessRunner(cfg=iter_cfg)
        selfplay_start = time.monotonic()
        records = runner.run(
            num_workers=len(games_per_worker),
            games_per_worker=games_per_worker,
            model_factory=functools.partial(
                _build_model_with_state, state_dict=state_dict
            ),
            progress_desc="[MP] Self-Playing vs Champion",
            past_opponent_path=past_ckpt,
        )
        selfplay_sec = time.monotonic() - selfplay_start
        if len(records) > sched["num_games"]:
            records = records[: sched["num_games"]]

        samples = flatten_game_records(records, game)
        shard_path = new_replay_shard_path(params.paths, iteration=global_iter)
        rows = [to_savable_sample(s.state, s.policy_probs, s.value) for s in samples]
        save_as_parquet_shard(fs, shard_path, rows)
        print(
            f"Saved replay shard: {shard_path} (games={len(records)}, samples={len(samples)})"
        )
        selfplay_summary = build_selfplay_summary(samples)
        print(f"[SelfPlay] Iteration {global_iter}: completed in {selfplay_sec:.1f}s")

        replay_buffer.extend(samples)
        if not replay_buffer:
            print("[MP] Replay buffer empty; skipping training.")
            continue

        total_samples = len(replay_buffer)
        print(
            f"[Training] Iteration {global_iter}: starting optimisation on {total_samples} samples."
        )
        train_start = time.monotonic()
        metrics = trainer.train_one_iteration(
            list(replay_buffer), progress_label="[Training] Batches"
        )
        train_sec = time.monotonic() - train_start
        print(
            format_training_metrics(
                metrics,
                iteration=global_iter,
                duration_sec=train_sec,
            )
        )

        ckpt_path = save_checkpoint(
            fs,
            params,
            iteration=global_iter,
            model=model,
            optimizer=optimizer,
            prefix="[MP] ",
        )
        eval_cfg = getattr(params, "evaluation", None)
        eval_every = (
            max(1, int(getattr(eval_cfg, "eval_every_iters", 1))) if eval_cfg else 1
        )
        will_eval = bool(
            eval_cfg
            and getattr(eval_cfg, "num_eval_games", 0) > 0
            and global_iter % eval_every == 0
        )
        if will_eval:
            print("Evaluating new model...")
        eval_start = time.monotonic()
        champion_path, eval_summary = evaluate_and_maybe_promote(
            game,
            model,
            params,
            fs,
            champion_path=champion_path,
            ckpt_path=ckpt_path,
            iteration=global_iter,
        )
        eval_sec = time.monotonic() - eval_start
        if will_eval and eval_summary is not None:
            print(f"[Eval] Iteration {global_iter} finished in {eval_sec:.1f}s.")
        if eval_summary is not None:
            log_entry = build_evaluation_log_entry(
                iteration=global_iter,
                promoted=bool(eval_summary.get("promote")),
                eval_summary=eval_summary,
                training_metrics=metrics,
                scheduled_params=scheduled_params_snapshot,
                selfplay_summary=selfplay_summary,
                timing_sec={
                    "selfplay": selfplay_sec,
                    "training": train_sec,
                    "evaluation": eval_sec,
                },
                eval_games_requested=int(
                    getattr(eval_cfg, "num_eval_games", 0) if eval_cfg else 0
                ),
                priority_replay_active=bool(getattr(priority_cfg, "enabled", False)),
                priority_trigger_reason=None,
            )
            atomic_append_jsonl(fs, evaluation_log_path(params.paths), log_entry)
        status = "completed" if (it + 1) == num_iters else "running"
        updater_fn = manifest_updater or update_manifest
        updater_fn(
            params,
            fs,
            completed_iterations=global_iter,
            status=status,
            champion_path=champion_path or ckpt_path,
            eval_summary=eval_summary,
            prefix="[MP] ",
        )
