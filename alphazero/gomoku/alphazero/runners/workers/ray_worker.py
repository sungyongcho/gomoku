"""Ray worker that executes self-play games."""

from collections.abc import Mapping
import random
import time
from typing import Any

import ray
import torch

from gomoku.alphazero.agent import AlphaZeroAgent
from gomoku.alphazero.agents import RandomBot
from gomoku.alphazero.runners.vectorize_runner import VectorizeRunner
from gomoku.alphazero.types import GameRecord
from gomoku.core.gomoku import Gomoku
from gomoku.inference.local import LocalInference
from gomoku.inference.ray_client import RayInferenceClient, SlotInferenceClient
from gomoku.model.policy_value_net import PolicyValueNet
from gomoku.pvmcts.pvmcts import PVMCTS
from gomoku.utils.config.loader import MctsConfig, RootConfig, TrainingConfig
from gomoku.utils.ray.ray_logger import setup_actor_logging


@ray.remote
class RaySelfPlayWorker:
    """CPU self-play worker that delegates inference to Ray actors."""

    def __init__(self, cfg: RootConfig, inference_actors: list[ray.actor.ActorHandle]):
        """Initialize worker with its own game, client, and agent."""
        self.logger = setup_actor_logging(self.__class__.__name__)
        self.cfg = cfg
        use_native = getattr(cfg.mcts, "use_native", False)
        self.game = Gomoku(cfg.board, use_native=use_native)

        # Check for local inference
        runtime = getattr(cfg, "runtime", None)
        inference_rt = getattr(runtime, "inference", None) if runtime else None
        use_local_inference = getattr(inference_rt, "use_local_inference", False)

        if use_local_inference:
            # Workers run inference locally on CPU
            # Optimization: Restrict torch threads to 1 to avoid contention in many-worker scenario
            torch.set_num_threads(1)
            device = "cpu"
            self.model = PolicyValueNet(self.game, cfg.model, device=device)
            self.client = LocalInference(self.model)
            self.logger.info("Using LocalInference (CPU) on worker.")
        else:
            self.client = RayInferenceClient(
                actors=inference_actors, max_batch_size=cfg.mcts.batch_infer_size
            )
            self.model = None

        selfplay_rt = getattr(runtime, "selfplay", None) if runtime else None
        async_limit = getattr(selfplay_rt, "inflight_per_actor", None)
        if async_limit is None:
            async_limit = getattr(cfg.mcts, "async_inflight_limit", None)
        engine_type = "ray"
        # if use_native:
        #    engine_type = "sequential"
        self.agent = AlphaZeroAgent(
            game=self.game,
            mcts_cfg=cfg.mcts,
            inference_client=self.client,
            engine_type=engine_type,
            async_inflight_limit=async_limit,
        )
        self.runner = VectorizeRunner(
            game=self.game, mcts_cfg=cfg.mcts, train_cfg=cfg.training
        )
        self.random_bot = RandomBot(self.game)
        self.past_opponent_path: str | None = None
        self.past_agent: AlphaZeroAgent | None = None
        self.logger.info("RaySelfPlayWorker initialized.")

    def set_weights(self, weights: dict[str, Any]) -> None:
        """Update local model weights if using local inference."""
        if self.model is not None:
            self.model.load_state_dict(weights)

    def update_params(
        self,
        training_cfg: TrainingConfig | Mapping[str, Any] | None = None,
        mcts_cfg: MctsConfig | Mapping[str, Any] | None = None,
        async_inflight_limit: int | None = None,
        past_opponent_path: str | None = None,
    ) -> None:
        """Update training/MCTS configs and async inflight limit for future games."""
        if mcts_cfg:
            new_val = (
                getattr(mcts_cfg, "use_native", "N/A")
                if not isinstance(mcts_cfg, Mapping)
                else mcts_cfg.get("use_native", "N/A")
            )
            # self.logger.info(
            #     "RaySelfPlayWorker update_params called. mcts.use_native=%s", new_val
            # )

        cfg = self.cfg

        def _coerce_training(
            cfg_candidate: TrainingConfig | Mapping[str, Any] | None,
        ) -> TrainingConfig | None:
            if cfg_candidate is None:
                return None
            if isinstance(cfg_candidate, TrainingConfig):
                return cfg_candidate
            if isinstance(cfg_candidate, Mapping):
                return self.cfg.training.model_copy(update=dict(cfg_candidate))
            raise TypeError("training_cfg must be TrainingConfig or mapping.")

        def _coerce_mcts(
            cfg_candidate: MctsConfig | Mapping[str, Any] | None,
        ) -> MctsConfig | None:
            if cfg_candidate is None:
                return None
            if isinstance(cfg_candidate, MctsConfig):
                return cfg_candidate
            if isinstance(cfg_candidate, Mapping):
                return self.cfg.mcts.model_copy(update=dict(cfg_candidate))
            raise TypeError("mcts_cfg must be MctsConfig or mapping.")

        new_training = _coerce_training(training_cfg)
        new_mcts = _coerce_mcts(mcts_cfg)

        if new_training is not None:
            cfg = cfg.model_copy(update={"training": new_training})
            self.runner.train_cfg = new_training

        if new_mcts is not None:
            cfg = cfg.model_copy(update={"mcts": new_mcts})
            self.client.max_batch_size = new_mcts.batch_infer_size

        resolved_async = async_inflight_limit
        if resolved_async is None and new_mcts is not None:
            resolved_async = getattr(new_mcts, "async_inflight_limit", None)
        if resolved_async is None:
            runtime = getattr(self.cfg, "runtime", None)
            selfplay_rt = getattr(runtime, "selfplay", None) if runtime else None
            resolved_async = getattr(selfplay_rt, "inflight_per_actor", None)
        if resolved_async is not None:
            self.agent.async_inflight_limit = resolved_async

        if new_mcts is not None or resolved_async is not None:
            mcts_params = new_mcts or self.agent.mcts_cfg
            self.agent.mcts_cfg = mcts_params

            # Check native usage and update game if needed
            use_native = getattr(mcts_params, "use_native", False)
            if use_native != self.game.use_native:
                self.game = Gomoku(self.cfg.board, use_native=use_native)
                self.agent.game = self.game

            engine_type = "ray"
            if use_native:
                engine_type = "sequential"
            self.agent.mcts = PVMCTS(
                self.game,
                mcts_params,
                self.client,
                mode="ray",
                async_inflight_limit=self.agent.async_inflight_limit,
            )
            self.agent.engine_type = engine_type
            self.runner.mcts_cfg = mcts_params
            self.agent.reset()

        if past_opponent_path is not None:
            if past_opponent_path != self.past_opponent_path:
                # self.logger.info(
                #     "Updating past opponent path: %s -> %s",
                #     self.past_opponent_path,
                #     past_opponent_path,
                # )
                self.past_opponent_path = past_opponent_path
                self.past_agent = None  # force reload only if path changed

        self.cfg = cfg

    def _ensure_past_agent(self) -> None:
        """Lazily load past opponent model if a path is set."""
        if self.past_agent is not None:
            return
        if not self.past_opponent_path:
            return

        # Check for local inference mode
        runtime = getattr(self.cfg, "runtime", None)
        inference_rt = getattr(runtime, "inference", None) if runtime else None
        use_local_inference = getattr(inference_rt, "use_local_inference", False)

        if use_local_inference:
            self.logger.info(
                f"Loading past opponent locally (CPU): {self.past_opponent_path}"
            )
            try:
                # Load past model locally
                past_model = PolicyValueNet(self.game, self.cfg.model, device="cpu")

                # We need to load weights from the path.
                # Ideally, weights are broadcasted or available in FS.
                # Here we assume standard torch load or fs helper if available,
                # but RayAsyncRunner broadcasts weights to workers?
                # Actually, RayAsyncRunner broadcasts 'current' weights.
                # For 'past' weights, we might need to load from disk or receive them.
                # Given update_params receives paths, let's try to load from FS if we can't find them in memory?
                # Wait, RayWorker doesn't have the weights dict for past op in update_params args directly usually?
                # RayAsyncRunner broadcasts past weights to ACTORS, but not WORKERS in local mode by default?
                # Let's check RayAsyncRunner._broadcast_weights.
                # If local inference is on, we need to load it.

                # Simple fix: Use fsspec or standard torch load if path is local/gcs
                # But worker might not have GCS creds or bandwidth.
                # HOWEVER, RayAsyncRunner broadcasts weights to workers if they are local!
                # But update_params doesn't take weights.
                # set_weights takes main weights.

                # Let's assume for now we load from the path since we have it.
                # Or better: check if we can reuse the logic from local.py or similar?
                # Actually, simply loading from the path is the most robust way if the worker has access.

                # But wait, checking the structure:
                pass
                # Let's rely on set_weights hook to populate something?
                # No, past agent is separate.

                # Re-reading: RayAsyncRunner.run() -> broadcasts weights.
                # If we are in local mode, we should receive weights there?
                # _broadcast_weights calls update_params? No, it calls set_weights.

                # Let's look at how we get the past weights.
                # RayAsyncRunner._broadcast_weights calls actor.set_weights(past_weights, slot="champion").
                # It does NOT call worker.set_weights for past weights.

                # So we MUST load it here from path.
                from gomoku.utils.paths import load_state_dict_from_path

                past_state = load_state_dict_from_path(
                    self.past_opponent_path, device="cpu"
                )
                past_model.load_state_dict(past_state)
                past_client = LocalInference(past_model)

                self.past_agent = AlphaZeroAgent(
                    game=self.game,
                    mcts_cfg=self.cfg.mcts,
                    inference_client=past_client,
                    engine_type="sequential",  # local inference implies sequential/native usually
                )
                return
            except Exception as e:
                self.logger.error(f"Failed to load past opponent locally: {e}")
                return

        # Optimization: Use "champion" slot on Ray actors instead of local CPU loading
        if self.client:
            self.logger.info(
                f"Using 'champion' slot for past opponent: {self.past_opponent_path}"
            )
            past_infer = SlotInferenceClient(self.client, slot="champion")
            engine_type = "ray"
            # Note: Weights are broadcasted by RayAsyncRunner, so we assume they are ready.

            self.past_agent = AlphaZeroAgent(
                game=self.game,
                mcts_cfg=self.cfg.mcts,
                inference_client=past_infer,
                engine_type=engine_type,
            )
            return

        # Fallback (unlikely in Ray worker)
        try:
            past_model = PolicyValueNet(self.game, self.cfg.model, device="cpu")
            # ... (rest of simple fallback or just fail)
            # For brevity, let's keep it simple or minimal.
            # If no client, we are in trouble anyway for a RayWorker.
            self.logger.warning(
                "No inference client found, skipping past opponent load."
            )
        except Exception as exc:
            self.logger.error(f"Failed to setup past opponent: {exc}")

    def run_games(
        self,
        batch_size: int,
        target_games: int,
        random_ratio: float = 0.0,
        random_bot_rate: float = 0.0,
        prev_bot_rate: float = 0.0,
        past_opponent_path: str | None = None,
        random_opening_turns: int = 0,
    ) -> list[GameRecord]:
        """Execute the requested number of self-play games."""
        start_wall = time.monotonic()
        seed = int(time.time_ns()) % (2**32)
        random.seed(seed)
        try:
            import numpy as np

            np.random.seed(seed)
        except Exception:
            pass
        torch.manual_seed(seed)

        per_batch = max(1, batch_size)
        remaining = target_games
        records: list[GameRecord] = []
        # self.logger.info(
        #     "Starting self-play batch: games=%d, batch_size=%d", target_games, per_batch
        # )
        warn_prev = False
        rnd_rate = max(0.0, float(random_bot_rate))
        prev_rate = max(0.0, float(prev_bot_rate))
        total_rate = rnd_rate + prev_rate
        if total_rate > 1.0:
            scale = 1.0 / total_rate
            rnd_rate *= scale
            prev_rate *= scale

        if (
            past_opponent_path is not None
            and past_opponent_path != self.past_opponent_path
        ):
            self.past_opponent_path = past_opponent_path
            self.past_agent = None
        self._ensure_past_agent()
        tasks: list[str] = []
        for _ in range(target_games):
            r = random.random()
            if r < rnd_rate:
                tasks.append("random_bot")
            elif r < rnd_rate + prev_rate:
                tasks.append("prev_bot")
            else:
                tasks.append("self")

        idx = 0
        while idx < len(tasks):
            opponent_type = tasks[idx]
            batch_tasks = []
            while idx < len(tasks) and len(batch_tasks) < per_batch:
                if tasks[idx] != opponent_type:
                    break
                batch_tasks.append(tasks[idx])
                idx += 1
            opp = None
            agent_first = True
            if opponent_type == "random_bot":
                opp = self.random_bot
                agent_first = random.choice([True, False])
            elif opponent_type == "prev_bot":
                if self.past_agent is not None:
                    opp = self.past_agent
                    agent_first = random.choice([True, False])
                else:
                    if not warn_prev:
                        self.logger.info(
                            "prev_bot_rate requested but past opponent unavailable; using self-play."
                        )
                        warn_prev = True
                    opp = None
            records.extend(
                self.runner.play_batch_games(
                    agent=self.agent,
                    batch_size=len(batch_tasks),
                    temperature=getattr(self.runner.train_cfg, "temperature", 1.0),
                    add_noise=True,
                    target_games=len(batch_tasks),
                    random_ratio=random_ratio,
                    random_opening_turns=int(random_opening_turns),
                    opponent=opp,
                    agent_first=agent_first,
                )
            )
        elapsed = time.monotonic() - start_wall
        # self.logger.info(
        #     "Completed self-play batch: games=%d, elapsed=%.1fs", target_games, elapsed
        # )
        return records
