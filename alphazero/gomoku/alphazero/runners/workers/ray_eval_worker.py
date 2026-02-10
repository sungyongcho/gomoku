"""Ray worker that executes evaluation duels."""

import ray

from gomoku.alphazero.eval.arena import _run_duel
from gomoku.core.gomoku import Gomoku
from gomoku.inference.local import LocalInference
from gomoku.inference.ray_client import RayInferenceClient, SlotInferenceClient
from gomoku.model.policy_value_net import PolicyValueNet
from gomoku.utils.config.loader import EvaluationConfig, RootConfig
from gomoku.utils.ray.ray_logger import setup_actor_logging


@ray.remote
class RayEvalWorker:
    """Worker that runs evaluation games (duels) between two models."""

    def __init__(
        self,
        cfg: RootConfig,
        challenger_actors: list[ray.actor.ActorHandle],
        champion_actors: list[ray.actor.ActorHandle],
        challenger_slot: str | None = None,
        champion_slot: str | None = None,
        challenger_weights: dict | None = None,
        champion_weights: dict | None = None,
    ):
        self.logger = setup_actor_logging(self.__class__.__name__)
        self.cfg = cfg
        use_native = getattr(cfg.mcts, "use_native", False)
        self.game = Gomoku(cfg.board, use_native=use_native)

        # Check for local inference
        runtime = getattr(cfg, "runtime", None)
        inference_rt = getattr(runtime, "inference", None) if runtime else None
        use_local_inference = getattr(inference_rt, "use_local_inference", False)

        if use_local_inference:
            import torch

            torch.set_num_threads(1)
            self.logger.info("Using LocalInference (CPU) for EvalWorker.")

            # Challenger
            self.challenger_model = PolicyValueNet(self.game, cfg.model, device="cpu")
            if challenger_weights:
                self.challenger_model.load_state_dict(challenger_weights)
            self.challenger_client = LocalInference(self.challenger_model)

            # Champion
            self.champion_model = PolicyValueNet(self.game, cfg.model, device="cpu")
            if champion_weights:
                self.champion_model.load_state_dict(champion_weights)
            self.champion_client = LocalInference(self.champion_model)

        else:
            # Initialize inference clients (Remote)
            self.challenger_client = RayInferenceClient(
                challenger_actors, max_batch_size=cfg.mcts.batch_infer_size or 32
            )
            if challenger_slot:
                self.challenger_client = SlotInferenceClient(
                    self.challenger_client, challenger_slot
                )

            self.champion_client = RayInferenceClient(
                champion_actors, max_batch_size=cfg.mcts.batch_infer_size or 32
            )
            if champion_slot:
                self.champion_client = SlotInferenceClient(
                    self.champion_client, champion_slot
                )
        # self.logger.info("RayEvalWorker initialized.")

    def run_duel(
        self,
        games: int,
        eval_cfg: EvaluationConfig,
        opening_turns: int = 0,
        temperature: float = 0.0,
        blunder_threshold: float = 0.5,
    ) -> dict:
        """Run a batch of duel games and return H2H metrics summary."""
        # self.logger.info(f"Starting duel batch of {games} games.")

        # Determine async inflight limit from runtime config (same as selfplay)
        inflight_limit = 0
        if self.cfg.runtime and self.cfg.runtime.selfplay:
            inflight_limit = (
                getattr(self.cfg.runtime.selfplay, "inflight_per_actor", 0) or 0
            )

        metrics = _run_duel(
            cfg=self.cfg,
            eval_cfg=eval_cfg,
            inference_a=self.challenger_client,
            inference_b=self.champion_client,
            games=games,
            opening_turns=opening_turns,
            temperature=temperature,
            blunder_th=blunder_threshold,
            progress_desc=None,
            async_inflight_limit=inflight_limit,
        )

        summary = metrics.summary()
        # self.logger.info(f"Duel batch finished: {summary}") # Moved to runner via tqdm.write
        return summary
