"""Ray runner for distributed evaluation."""

from collections.abc import Callable
import logging

import ray

from gomoku.alphazero.runners.ray_runner import RayAsyncRunner
from gomoku.alphazero.runners.workers.ray_eval_worker import RayEvalWorker
from gomoku.utils.config.loader import RootConfig

logger = logging.getLogger(__name__)


class RayEvalRunner(RayAsyncRunner):
    """Orchestrates distributed evaluation using separate actor pools."""

    def __init__(
        self,
        cfg: RootConfig,
        num_actors: int = 1,
        num_workers: int = 1,
        challenger_weights_fn: Callable[[], dict] | None = None,
        champion_weights_fn: Callable[[], dict] | None = None,
    ):
        # We inherit from RayAsyncRunner to reuse actor creation logic helpers if needed,
        # but we need two pools of actors and specialized workers.
        # So we override run() completely.
        self.cfg = cfg
        self.num_actors = max(1, num_actors)
        self.num_workers = max(1, num_workers)
        self.challenger_weights_fn = challenger_weights_fn
        self.champion_weights_fn = champion_weights_fn
        self._init_ray()

    def run_evaluation(
        self,
        num_games: int,
        eval_cfg,
        shared_actors: list[ray.actor.ActorHandle] | None = None,
    ) -> dict:
        """Run the distributed evaluation."""
        # logger.info(f"Starting distributed evaluation: {num_games} games")
        logger.info("[Eval] Initializing evaluation resources...")

        challenger_actors = []
        champion_actors = []
        challenger_slot = None
        champion_slot = None

        # Check Local Inference
        runtime = getattr(self.cfg, "runtime", None)
        inference_rt = getattr(runtime, "inference", None) if runtime else None
        use_local_inference = getattr(inference_rt, "use_local_inference", False)

        challenger_w_ref = None
        champion_w_ref = None

        if use_local_inference:
            logger.info(
                "Evaluation using Local Inference (CPU). Skipping Actor creation."
            )
            if self.challenger_weights_fn:
                challenger_w_ref = ray.put(self.challenger_weights_fn())
            if self.champion_weights_fn:
                champion_w_ref = ray.put(self.champion_weights_fn())
        else:
            # 1. Resource Management
            if shared_actors:
                # Reusing existing self-play actors via slots
                logger.info(
                    f"Reusing {len(shared_actors)} shared actors for evaluation."
                )
                challenger_actors = shared_actors
                champion_actors = shared_actors
                challenger_slot = "challenger"
                champion_slot = "champion"
            else:
                # Fallback: Create dedicated actors (Resource intensive!)
                pool_size = max(1, self.num_actors // 2)
                original_num = self.num_actors
                self.num_actors = pool_size
                try:
                    logger.info(f"[Eval] Creating {pool_size} challenger actors...")
                    challenger_actors = self._create_actors()
                    logger.info(f"[Eval] Creating {pool_size} champion actors...")
                    champion_actors = self._create_actors()
                finally:
                    self.num_actors = original_num

            # 2. Broadcast Weights
            # Note: If reusing shared actors, we must be careful not to overwrite "current" slot used by self-play
            # if self-play runs concurrently. But here execution is sequential.
            # We target specific slots.

            target_challenger_slot = challenger_slot or "current"
            # If not shared, default slot is "current" for both (since they are separate actors)
            target_champion_slot = champion_slot or "current"

            if self.challenger_weights_fn:
                w = self.challenger_weights_fn()
                ray.get(
                    [
                        a.set_weights.remote(w, slot=target_challenger_slot)
                        for a in challenger_actors
                    ]
                )

            if self.champion_weights_fn:
                w = self.champion_weights_fn()
                ray.get(
                    [
                        a.set_weights.remote(w, slot=target_champion_slot)
                        for a in champion_actors
                    ]
                )

        # 3. Create Workers
        # Auto-scale workers if default is 1
        if self.num_workers <= 1:
            try:
                cluster_resources = ray.cluster_resources()
                # Estimate available CPUs for eval workers (total - head - existing actors)
                # This is rough, so we just try to spawn as many as games if games < cluster cpus
                total_cpus = cluster_resources.get("CPU", 1)
                # Reserve some for actors if they are new
                # Simple heuristic: 1 worker per game if we have enough CPUs, up to a limit
                self.num_workers = min(num_games, int(total_cpus), 64)
                logger.info(f"Auto-scaled eval workers to {self.num_workers}")
            except Exception:
                pass

        logger.info(f"[Eval] Creating {self.num_workers} eval workers...")
        workers = []
        worker_resource = self._resource_tag("selfplay")
        for _ in range(self.num_workers):
            workers.append(
                RayEvalWorker.options(resources=worker_resource).remote(
                    self.cfg,
                    challenger_actors,
                    champion_actors,
                    challenger_slot=challenger_slot,
                    champion_slot=champion_slot,
                    challenger_weights=challenger_w_ref,
                    champion_weights=champion_w_ref,
                )
            )

        try:
            # 4. Distribute Games
            base = num_games // self.num_workers
            remainder = num_games % self.num_workers
            futures = []

            for idx, w in enumerate(workers):
                count = base + (1 if idx < remainder else 0)
                if count > 0:
                    futures.append(
                        w.run_duel.remote(
                            games=count,
                            eval_cfg=eval_cfg,
                            opening_turns=eval_cfg.eval_opening_turns,
                            temperature=eval_cfg.eval_temperature,
                            blunder_threshold=eval_cfg.blunder_threshold,
                        )
                    )

            # 5. Collect Results with tqdm
            from tqdm import tqdm

            total_wins = 0
            total_losses = 0
            total_draws = 0
            total_blunders = 0
            total_moves = 0
            total_games = 0

            pending = list(futures)
            # Use dynamic_ncols=True to fill terminal, or default to standard width
            with tqdm(
                total=num_games, desc="[Eval] Games", unit="game", dynamic_ncols=True
            ) as pbar:
                while pending:
                    done, pending = ray.wait(pending, timeout=1.0)
                    for ref in done:
                        r = ray.get(ref)
                        wins = r.get("wins", 0)
                        losses = r.get("losses", 0)
                        draws = r.get("draws", 0)
                        batch_count = wins + losses + draws

                        total_wins += wins
                        total_losses += losses
                        total_draws += draws
                        total_blunders += r.get("blunders", 0)
                        total_moves += r.get("total_moves", 0)
                        total_games += batch_count

                        # Print result using tqdm.write to avoid breaking the bar
                        tqdm.write(f"Duel finished: {r}")

                        pbar.update(batch_count)

            return {
                "wins": total_wins,
                "losses": total_losses,
                "draws": total_draws,
                "games": total_games,
                "blunders": total_blunders,
                "total_moves": total_moves,
            }

        finally:
            # 7. Shutdown
            # Only kill workers and actors we created.
            logger.info("Cleaning up Eval workers...")
            for w in workers:
                ray.kill(w)

            if not shared_actors:
                logger.info("Cleaning up Eval actors...")
                for a in challenger_actors + champion_actors:
                    ray.kill(a)
