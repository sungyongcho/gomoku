from collections.abc import Callable
from dataclasses import dataclass
import logging

import ray
import torch

from gomoku.alphazero.runners.workers.ray_worker import RaySelfPlayWorker
from gomoku.alphazero.types import GameRecord
from gomoku.core.gomoku import Gomoku
from gomoku.inference.ray_client import RayInferenceActor
from gomoku.model.policy_value_net import PolicyValueNet
from gomoku.utils.config.loader import MctsConfig, RootConfig, TrainingConfig
from gomoku.utils.config.schedule_param import get_scheduled_value
from gomoku.utils.progress import make_progress

logger = logging.getLogger(__name__)


def _build_model_fn(cfg: RootConfig, device: str) -> Callable[[], PolicyValueNet]:
    """Create a factory that builds a PolicyValueNet on the given device."""

    def _fn() -> PolicyValueNet:
        game = Gomoku(cfg.board)
        model = PolicyValueNet(game, cfg.model, device=device)
        return model

    return _fn


@dataclass(slots=True)
class RayAsyncRunner:
    """Ray-backed asynchronous self-play orchestrator."""

    cfg: RootConfig
    num_actors: int = 1
    num_workers: int = 1
    set_weights_fn: Callable | None = None  # optional: hook to broadcast weights
    past_weights_fn: Callable | None = None  # optional: hook for champion weights
    _actors: list[ray.actor.ActorHandle] = None
    _workers: list[ray.actor.ActorHandle] = None

    def _init_ray(self) -> None:
        """Initialize Ray if not already initialized."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

    def _resource_tag(self, name: str) -> dict | None:
        """Return a custom resource tag only if it exists in the cluster."""
        try:
            resources = ray.cluster_resources()
        except Exception:
            return None
        if name in resources:
            # multiple actors share the resource (fractional)
            return {name: 1.0 / max(1, self.num_actors)}
        return None

    def _create_actors(self) -> list[ray.actor.ActorHandle]:
        """Create inference actors on available resources."""
        runtime = getattr(self.cfg, "runtime", None)
        inference_rt = getattr(runtime, "inference", None) if runtime else None
        num_cpus_cfg = getattr(inference_rt, "actor_num_cpus", None)
        num_gpus_cfg = getattr(inference_rt, "actor_num_gpus", None)

        def _safe_num(value: float | int | None) -> float | None:
            try:
                return float(value) if value is not None else None
            except Exception:  # noqa: BLE001
                return None

        resources = ray.cluster_resources() if ray.is_initialized() else {}
        available_gpus = float(resources.get("GPU", 0.0) or 0.0)
        requested_gpus = _safe_num(num_gpus_cfg)
        if requested_gpus is None:
            requested_gpus = 1.0 if torch.cuda.is_available() else 0.0

        if requested_gpus > 0.0:
            # We skip the driver-side torch.cuda.is_available() check.
            # If the Ray cluster has GPUs but the driver (job runner) container does not,
            # checking is_available() here would incorrectly downgrade actors to CPU.
            # We trust the config/cluster resources. If requested > available, Ray will queue it.
            if available_gpus < requested_gpus:
                logger.warning(
                    f"Requested GPU resources ({requested_gpus}) exceed currently available ({available_gpus}). "
                    "Actors may stay PENDING until Autoscaler provisions nodes."
                )
            device = "cuda"
            num_gpus_total = requested_gpus
            # For GPU inference with server-side batching, fewer actors (1 per GPU) is better
            # to maximize batch size. Over-subscribing actors fragments the batch.
            # However, self.num_actors is determined by caller (run_ray).
            # We should respect it, but logging a warning if it looks high for batching
            num_gpus_per_actor = num_gpus_total / max(1, self.num_actors)
        else:
            device = "cpu"
            num_gpus_per_actor = 0

        model_fn = _build_model_fn(self.cfg, device)

        # Extract batching params
        mcts_cfg = getattr(self.cfg, "mcts", None)
        batch_size = getattr(mcts_cfg, "batch_infer_size", 32) or 32
        min_batch_size = getattr(mcts_cfg, "min_batch_size", 1) or 1
        wait_ms = getattr(mcts_cfg, "max_batch_wait_ms", 10) or 10

        actors: list[ray.actor.ActorHandle] = []
        head_resource = self._resource_tag("head")
        for _ in range(self.num_actors):
            options = {
                "num_cpus": _safe_num(num_cpus_cfg) or 1,
                "num_gpus": num_gpus_per_actor,
                "max_concurrency": 256,
            }
            if head_resource:
                options["resources"] = head_resource
            # Pass batching config to Actor
            actors.append(
                RayInferenceActor.options(**options).remote(
                    model_fn,
                    batch_size=int(batch_size),
                    min_batch_size=int(min_batch_size),
                    max_wait_ms=int(wait_ms),
                )
            )
        return actors

    def _create_workers(
        self, inference_actors: list[ray.actor.ActorHandle]
    ) -> list[ray.actor.ActorHandle]:
        """Create self-play workers that delegate inference to provided actors."""
        runtime = getattr(self.cfg, "runtime", None)
        selfplay_rt = getattr(runtime, "selfplay", None) if runtime else None
        num_cpus_cfg = getattr(selfplay_rt, "actor_num_cpus", None)
        num_gpus_cfg = getattr(selfplay_rt, "actor_num_gpus", None)

        def _safe_num(value: float | int | None) -> float | None:
            try:
                return float(value) if value is not None else None
            except Exception:  # noqa: BLE001
                return None

        workers: list[ray.actor.ActorHandle] = []
        worker_resource = self._resource_tag("selfplay")
        for _ in range(max(1, self.num_workers)):
            options = {
                "num_cpus": _safe_num(num_cpus_cfg) or 1,
                "num_gpus": _safe_num(num_gpus_cfg) or 0,
            }
            if worker_resource:
                options["resources"] = worker_resource
            workers.append(
                RaySelfPlayWorker.options(**options).remote(self.cfg, inference_actors)
            )
        return workers

    def _broadcast_weights(
        self,
        actors: list[ray.actor.ActorHandle],
        workers: list[ray.actor.ActorHandle] | None = None,
    ) -> None:
        """Synchronize model weights to inference actors and workers."""
        # 1. Broadcast Main Weights (slot="current")
        if self.set_weights_fn:
            try:
                weights = self.set_weights_fn()
                # Broadcast to Actors (GPU/Remote Inference)
                futures = [
                    actor.set_weights.remote(weights, slot="current")
                    for actor in actors
                ]
                # Broadcast to Workers (Local Inference) - workers default to current
                if workers:
                    futures.extend(
                        [worker.set_weights.remote(weights) for worker in workers]
                    )
                ray.get(futures)
            except Exception as exc:
                logger.error("Ray main weight broadcast failed: %s", exc)
                raise RuntimeError("Failed to broadcast main weights") from exc

        # 2. Broadcast Past Weights (slot="champion")
        if self.past_weights_fn:
            try:
                logger.info("Loading past weights (champion)...")
                past_weights = self.past_weights_fn()
                logger.info("Broadcasting past weights to actors...")
                # Broadcast to Actors ONLY (Workers use actors via SlotInferenceClient)
                futures = [
                    actor.set_weights.remote(past_weights, slot="champion")
                    for actor in actors
                ]
                ray.get(futures)
            except Exception as exc:
                logger.error("Ray past weight broadcast failed: %s", exc)
                # Don't crash if past weights fail loading, maybe just warn?
                # But safer to raise if we expect them.
                raise RuntimeError("Failed to broadcast past weights") from exc

    def _resolve_iteration_configs(
        self,
        iteration_idx: int | None,
        training_cfg: TrainingConfig | None,
        mcts_cfg: MctsConfig | None,
    ) -> tuple[TrainingConfig | None, MctsConfig | None]:
        """Resolve scheduled configs for a given iteration (0-based)."""
        train = training_cfg or self.cfg.training
        mcts = mcts_cfg or self.cfg.mcts
        if iteration_idx is None:
            return train, mcts

        train_updates: dict[str, float] = {
            "temperature": get_scheduled_value(train.temperature, iteration_idx),
        }
        if train.random_play_ratio is not None:
            train_updates["random_play_ratio"] = get_scheduled_value(
                train.random_play_ratio, iteration_idx
            )
        if train.random_opening_turns is not None:
            train_updates["random_opening_turns"] = int(
                get_scheduled_value(train.random_opening_turns, iteration_idx)
            )
        if getattr(train, "opponent_rates", None) is not None:
            train_updates["opponent_rates"] = train.opponent_rates.model_copy(
                update={
                    "random_bot_ratio": get_scheduled_value(
                        train.opponent_rates.random_bot_ratio, iteration_idx
                    ),
                    "prev_bot_ratio": get_scheduled_value(
                        train.opponent_rates.prev_bot_ratio, iteration_idx
                    ),
                }
            )

        mcts_updates = {
            "dirichlet_epsilon": get_scheduled_value(
                mcts.dirichlet_epsilon, iteration_idx
            ),
            "dirichlet_alpha": get_scheduled_value(
                mcts.dirichlet_alpha, iteration_idx
            ),
            "num_searches": get_scheduled_value(mcts.num_searches, iteration_idx),
            "exploration_turns": int(
                get_scheduled_value(mcts.exploration_turns, iteration_idx)
            ),
        }

        return (
            train.model_copy(update=train_updates),
            mcts.model_copy(update=mcts_updates),
        )

    def _broadcast_worker_params(
        self,
        workers: list[ray.actor.ActorHandle],
        training_cfg: TrainingConfig | None,
        mcts_cfg: MctsConfig | None,
        async_inflight_limit: int | None = None,
        past_opponent_path: str | None = None,
    ) -> None:
        """Push updated training/MCTS configs to self-play workers."""
        if not workers or (
            training_cfg is None
            and mcts_cfg is None
            and async_inflight_limit is None
            and past_opponent_path is None
        ):
            return
        try:
            ray.get(
                [
                    worker.update_params.remote(
                        training_cfg=training_cfg,
                        mcts_cfg=mcts_cfg,
                        async_inflight_limit=async_inflight_limit,
                        past_opponent_path=past_opponent_path,
                    )
                    for worker in workers
                ]
            )
        except Exception as exc:
            logger.error("Ray worker param broadcast failed: %s", exc)
            raise RuntimeError(
                "Failed to broadcast parameters to Ray self-play workers"
            ) from exc

    def ensure_started(self):
        """Ensure Ray actors/workers are created."""
        self._init_ray()
        if not self._actors:
            logger.info("Creating actors...")
            self._actors = self._create_actors()
            logger.info(f"Started {len(self._actors)} actors.")

        if not self._workers:
            # We want persistent workers too?
            # Workers might need to be recreated if iteration params change?
            # Ideally params are pushed via broadcast.
            logger.info("Creating workers...")
            self._workers = self._create_workers(self._actors)
            logger.info(f"Started {len(self._workers)} workers.")

    def stop_workers(self):
        """Stop only the self-play workers to release 'selfplay' resources."""
        if self._workers:
            logger.info(f"Stopping {len(self._workers)} self-play workers...")
            for w in self._workers:
                ray.kill(w)
            self._workers = None

    def shutdown(self):
        """Kill all managed actors."""
        self.stop_workers()
        if self._actors:
            for a in self._actors:
                ray.kill(a)
            self._actors = None

    @property
    def actors(self):
        return self._actors or []

    def run(
        self,
        batch_size: int,
        games: int,
        *,
        training_cfg: TrainingConfig | None = None,
        mcts_cfg: MctsConfig | None = None,
        iteration_idx: int | None = None,
        async_inflight_limit: int | None = None,
        progress_desc: str | None = None,
        past_opponent_path: str | None = None,
    ) -> list[GameRecord]:
        """
        Execute asynchronous batch self-play using Ray actors.

        Note: This method now EXPECTS actors to be managed externally or via ensure_started().
        """
        self.ensure_started()

        resolved_training, resolved_mcts = self._resolve_iteration_configs(
            iteration_idx, training_cfg, mcts_cfg
        )
        runtime = getattr(self.cfg, "runtime", None)
        selfplay_rt = getattr(runtime, "selfplay", None) if runtime else None
        async_limit = (
            async_inflight_limit
            if async_inflight_limit is not None
            else getattr(selfplay_rt, "inflight_per_actor", None)
        )

        actors = self._actors
        workers = self._workers

        # Broadcast weights to both Actors and Workers
        self._broadcast_weights(actors, workers)

        self._broadcast_worker_params(
            workers=workers,
            training_cfg=resolved_training,
            mcts_cfg=resolved_mcts,
            async_inflight_limit=async_limit,
            past_opponent_path=past_opponent_path,
        )

        total_workers = len(workers)
        base = games // total_workers
        remainder = games % total_workers

        progress = make_progress(
            total=games,
            desc=progress_desc,
            unit="game",
            disable=progress_desc is None,
            dynamic_ncols=True,
        )
        futures = []
        # Increase chunk size to keep workers busy and allow VectorizeRunner to fill its batch slots.
        # We want at least a few chunks per worker for progress visibility, but each chunk
        # must be at least 'batch_size' to avoid wasting vectorized slots.
        # [MODIFIED] Use configured chunk size if available, else default to 16 for responsiveness.
        max_chunk_conf = getattr(selfplay_rt, "max_chunk_size", None)
        if max_chunk_conf is not None:
            chunk_size = int(max_chunk_conf)
        else:
            chunk_size = 16  # Default small chunk for UI responsiveness
        rnd_bot_rate = float(
            getattr(
                getattr(resolved_training, "opponent_rates", None),
                "random_bot_ratio",
                0.0,
            )
            or 0.0
        )
        prev_bot_rate = float(
            getattr(
                getattr(resolved_training, "opponent_rates", None),
                "prev_bot_ratio",
                0.0,
            )
            or 0.0
        )
        for idx, w in enumerate(workers):
            target = base + (1 if idx < remainder else 0)
            if target <= 0:
                continue
            remaining = target
            while remaining > 0:
                this_chunk = min(chunk_size, remaining)
                futures.append(
                    w.run_games.remote(
                        batch_size,
                        this_chunk,
                        getattr(resolved_training, "random_play_ratio", 0.0)
                        if resolved_training is not None
                        else 0.0,
                        rnd_bot_rate,
                        prev_bot_rate,
                        past_opponent_path,
                        int(
                            getattr(resolved_training, "random_opening_turns", 0)
                            if resolved_training is not None
                            else 0
                        ),
                    )
                )
                remaining -= this_chunk

        records: list[GameRecord] = []
        pending = list(futures)
        while pending:
            ready, pending = ray.wait(pending, num_returns=1)
            for handle in ready:
                batch = ray.get(handle)
                progress.update(len(batch))
                records.extend(batch)
        progress.close()
        return records
        # Finally block removed: we keep actors alive. Call shutdown() if needed.
