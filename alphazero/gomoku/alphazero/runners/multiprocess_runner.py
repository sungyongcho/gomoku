from collections.abc import Callable, Sequence
from dataclasses import dataclass
import multiprocessing as mp

from gomoku.alphazero.runners.workers.mp_worker import _worker_selfplay_loop
from gomoku.alphazero.types import GameRecord
from gomoku.core.gomoku import Gomoku
from gomoku.inference.mp_server import BatchInferenceServer
from gomoku.model.policy_value_net import PolicyValueNet
from gomoku.utils.config.loader import RootConfig
from gomoku.utils.progress import make_progress


def _run_server_process(
    cfg: RootConfig,
    request_q: mp.Queue,
    result_queues: list[mp.Queue],
    model_factory: Callable[[Gomoku, RootConfig], PolicyValueNet] | None = None,
) -> None:
    """Entry point for the inference server process.

    Parameters
    ----------
    cfg : RootConfig
        Root configuration used to build the game/model.
    request_q : mp.Queue
        Incoming inference requests from workers.
    result_queues : list[mp.Queue]
        Outgoing result queues, one per worker.
    model_factory : Callable[[Gomoku, RootConfig], PolicyValueNet] | None
        Optional model factory; defaults to PolicyValueNet when None.
    """
    server = BatchInferenceServer(
        cfg=cfg,
        request_q=request_q,
        result_queues=result_queues,
        model_factory=model_factory,
    )
    server.run()


@dataclass(slots=True)
class MultiprocessRunner:
    """Orchestrate self-play across multiple processes."""

    cfg: RootConfig
    ctx_name: str = "spawn"

    def run(
        self,
        num_workers: int,
        games_per_worker: int | Sequence[int],
        model_factory: Callable[[Gomoku, RootConfig], PolicyValueNet] | None = None,
        progress_desc: str | None = None,
        past_opponent_path: str | None = None,
    ) -> list[GameRecord]:
        """Spawn workers to generate the requested number of games in parallel.

        Parameters
        ----------
        num_workers :
            Number of worker processes to launch.
        games_per_worker :
            Number of games each worker should play (int or per-worker sequence).
        model_factory :
            Optional factory used by the inference server; defaults to
            ``PolicyValueNet`` when None.
        progress_desc :
            Optional tqdm label for progress display.

        Returns
        -------
        list[GameRecord]
            Collected self-play game records.
        """
        per_worker: list[int]
        if isinstance(games_per_worker, Sequence) and not isinstance(
            games_per_worker, (str, bytes)
        ):
            per_worker = [int(g) for g in games_per_worker]
            if num_workers <= 0:
                num_workers = len(per_worker)
        else:
            if num_workers <= 0 or int(games_per_worker) <= 0:
                return []
            per_worker = [int(games_per_worker)] * num_workers

        # Drop workers with zero/negative load
        per_worker = [g for g in per_worker if g > 0]
        num_workers = len(per_worker)
        if num_workers == 0:
            return []

        ctx = mp.get_context(self.ctx_name)
        # Queue is used (not SimpleQueue) because MPInferenceClient expects
        # timeout support on get().
        request_q: mp.Queue = ctx.Queue()
        result_queues: list[mp.Queue] = [ctx.Queue() for _ in range(num_workers)]
        record_q: mp.Queue = ctx.Queue()

        server = ctx.Process(
            target=_run_server_process,
            args=(self.cfg, request_q, result_queues, model_factory),
            daemon=True,
        )
        server.start()

        workers: list[mp.Process] = []
        for worker_id, games_for_worker in enumerate(per_worker):
            p = ctx.Process(
                target=_worker_selfplay_loop,
                args=(
                    worker_id,
                    self.cfg,
                    request_q,
                    result_queues[worker_id],
                    record_q,
                    games_for_worker,
                    past_opponent_path,
                ),
                daemon=True,
            )
            p.start()
            workers.append(p)

        total_games = sum(per_worker)
        records: list[GameRecord] = []
        progress = make_progress(
            total=total_games,
            desc=progress_desc,
            unit="game",
            disable=progress_desc is None,
        )

        try:
            while len(records) < total_games:
                # Monitor server/worker health
                if not server.is_alive():
                    raise RuntimeError("Inference server died unexpectedly.")
                dead_workers = [p for p in workers if not p.is_alive()]
                if dead_workers:
                    raise RuntimeError("One or more worker processes died.")

                try:
                    rec = record_q.get(timeout=0.5)
                    records.append(rec)
                    progress.update(1)
                except Exception:
                    continue
        finally:
            # Signal server shutdown
            try:
                request_q.put(None)
            except Exception:
                pass

            for p in workers:
                p.join(timeout=5)
                if p.is_alive():
                    p.terminate()

            server.join(timeout=5)
            if server.is_alive():
                server.terminate()

            progress.close()

        return records
