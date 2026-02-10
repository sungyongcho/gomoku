import random
import time

import fsspec
import torch

from gomoku.alphazero.agent import AlphaZeroAgent
from gomoku.alphazero.agents import RandomBot
from gomoku.alphazero.learning.dataset import flatten_game_records
from gomoku.alphazero.learning.trainer import AlphaZeroTrainer
from gomoku.alphazero.runners.selfplay import SelfPlayRunner
from gomoku.core.gomoku import Gomoku
from gomoku.inference.local import LocalInference
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
from gomoku.utils.paths import (
    evaluation_log_path,
    load_state_dict_from_fs,
    new_replay_shard_path,
)
from gomoku.utils.progress import make_progress
from gomoku.utils.serialization import to_savable_sample


def run_sequential(
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    game: Gomoku,
    params: RunnerParams,
    fs: fsspec.AbstractFileSystem,
    *,
    completed_offset: int = 0,
    champion_path: str | None = None,
    replay_shards: list[str] | None = None,
    manifest_updater=None,
) -> None:
    """Sequential self-play → replay shard 저장 → 학습 루프 (단순 버전)."""
    inference = LocalInference(model)
    agent = AlphaZeroAgent(
        game=game,
        mcts_cfg=params.mcts,
        inference_client=inference,
        engine_type="sequential",
    )
    selfplay = SelfPlayRunner(
        game=game, mcts_cfg=params.mcts, train_cfg=params.training
    )
    trainer = AlphaZeroTrainer(
        train_cfg=params.training, model=model, optimizer=optimizer, game=game
    )
    random_bot = RandomBot(game)
    past_agent: AlphaZeroAgent | None = None

    num_iters = int(params.training.num_iterations)
    total_done_before = int(completed_offset)
    replay_buffer, loaded_shards = warm_start_replay_buffer(
        params, fs, game, replay_shards=replay_shards
    )
    if loaded_shards:
        print(
            f"[Sequential] Warm-started replay buffer with {len(loaded_shards)} shard(s)."
        )
    print("Using Non-Batched MCTS search (CPU Environment).")
    print(f"Loaded {len(replay_buffer)} samples into replay buffer.")

    for it in range(num_iters):
        global_iter = total_done_before + it + 1
        log_iteration_header(global_iter, total_done_before + num_iters)

        # 스케줄 적용
        schedule_idx = completed_offset + it
        sched = compute_iteration_schedule(params, schedule_idx)
        mcts_cfg = params.mcts.model_copy(
            update={
                "dirichlet_epsilon": sched["dir_eps"],
                "dirichlet_alpha": sched["dir_alpha"],
                "num_searches": sched["num_searches"],
                "exploration_turns": sched["explore_turns"],
            }
        )
        agent.mcts_cfg = mcts_cfg
        agent.mcts.params = mcts_cfg
        if hasattr(agent.mcts, "engine"):
            agent.mcts.engine.params = mcts_cfg

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
            print("[Sequential] No self-play games scheduled; skipping iteration.")
            if manifest_updater:
                manifest_updater(
                    params,
                    fs,
                    completed_iterations=global_iter,
                    status="running" if (it + 1) < num_iters else "completed",
                    champion_path=None,
                    prefix="[Sequential] ",
                )
            continue

        selfplay_start = time.monotonic()
        progress = make_progress(
            total=sched["num_games"], desc="Self-Playing vs Champion", unit="game"
        )
        records: list = []
        total_games = int(sched["num_games"])
        rnd_bot_games = int(total_games * sched["rnd_bot_rate"])
        prev_bot_games = int(total_games * sched["prev_bot_rate"])
        self_games = total_games - (rnd_bot_games + prev_bot_games)
        past_agent = None
        if prev_bot_games > 0:
            past_ckpt = select_past_checkpoint(
                fs=fs,
                paths=params.paths,
                current_iter=global_iter,
                window=getattr(
                    getattr(params.training, "opponent_rates", None),
                    "past_model_window",
                    5,
                ),
                champion_path=champion_path,
            )
            if past_ckpt:
                try:
                    past_model = PolicyValueNet(
                        game, params.model, device=str(model.device)
                    )
                    past_state = load_state_dict_from_fs(
                        fs, past_ckpt, torch.device(model.device)
                    )
                    past_model.load_state_dict(past_state, strict=False)
                    past_infer = LocalInference(past_model)
                    past_agent = AlphaZeroAgent(
                        game=game,
                        mcts_cfg=params.mcts,
                        inference_client=past_infer,
                        engine_type="sequential",
                    )
                    print(f"[Sequential] Loaded past opponent from {past_ckpt}")
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[Sequential] Failed to load past opponent at {past_ckpt}: {exc}"
                    )
                    past_agent = None
            else:
                print("[Sequential] No past checkpoint found; falling back to self-play.")

        game_tasks = (
            ["random_bot"] * rnd_bot_games
            + ["prev"] * prev_bot_games
            + ["self"] * self_games
        )
        random.shuffle(game_tasks)
        for idx, task in enumerate(game_tasks):
            if task == "random_bot":
                agent_first = (idx % 2) == 0
                record = selfplay.play_one_game(
                    agent,
                    temperature=sched["temp"],
                    add_noise=True,
                    random_ratio=sched["rnd_ratio"],
                    random_opening_turns=sched["rnd_opening"],
                    opponent=random_bot,
                    agent_first=agent_first,
                )
            elif task == "prev" and past_agent is not None:
                agent_first = (idx % 2) == 0
                record = selfplay.play_one_game(
                    agent,
                    temperature=sched["temp"],
                    add_noise=True,
                    random_ratio=sched["rnd_ratio"],
                    random_opening_turns=sched["rnd_opening"],
                    opponent=past_agent,
                    agent_first=agent_first,
                )
            else:
                record = selfplay.play_one_game(
                    agent,
                    temperature=sched["temp"],
                    add_noise=True,
                    random_ratio=sched["rnd_ratio"],
                    random_opening_turns=sched["rnd_opening"],
                )
            records.append(record)
            progress.update(1)
        progress.close()
        samples = flatten_game_records(records, game)
        selfplay_sec = time.monotonic() - selfplay_start
        print(f"[SelfPlay] Iteration {global_iter}: completed in {selfplay_sec:.1f}s")
        selfplay_summary = build_selfplay_summary(samples)

        # 샤드 저장
        shard_path = new_replay_shard_path(params.paths, iteration=global_iter)
        rows = [to_savable_sample(s.state, s.policy_probs, s.value) for s in samples]
        save_as_parquet_shard(fs, shard_path, rows)
        print(f"Saved replay shard: {shard_path}")

        replay_buffer.extend(samples)
        if not replay_buffer:
            print("[Sequential] Replay buffer empty; skipping training.")
            continue

        # 단순 학습 한 에폭
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
            prefix="[Sequential] ",
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
        is_last = (it + 1) == num_iters
        if manifest_updater:
            manifest_updater(
                params,
                fs,
                completed_iterations=global_iter,
                status="completed" if is_last else "running",
                champion_path=champion_path or ckpt_path,
                eval_summary=eval_summary,
                prefix="[Sequential] ",
            )
        else:
            update_manifest(
                params,
                fs,
                completed_iterations=global_iter,
                status="completed" if is_last else "running",
                champion_path=champion_path or ckpt_path,
                eval_summary=eval_summary,
                prefix="[Sequential] ",
            )
