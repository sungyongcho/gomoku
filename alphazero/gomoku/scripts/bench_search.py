"""Measure full serving search speed: simulations completed within the move budget.

Pin the process to the target core count, for example:
    ALPHAZERO_INFER_THREADS=2 ALPHAZERO_INFER_BACKEND=onnx-static \\
    taskset -c 0,1 python -m gomoku.scripts.bench_search
"""

import argparse
import os
import statistics
import tempfile
import time

from gomoku.pvmcts.pvmcts import PVMCTS
from server.engine import AlphaZeroEngine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/deploy.yaml")
    parser.add_argument("--checkpoint", default="models/champion.pt")
    parser.add_argument("--budget-s", type=float, default=2.5)
    parser.add_argument("--positions", type=int, default=12)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as cache:
        os.environ["ONNX_CACHE_DIR"] = cache
        engine = AlphaZeroEngine(args.config, args.checkpoint, "cpu")
    states = _sample_states(engine, args.positions)
    cfg = engine.mcts_config.model_copy(
        update={"num_searches": 100_000.0, "time_limit_s": args.budget_s}
    )
    sims, elapsed = [], []
    for state in states:
        search = PVMCTS(
            game=engine.game,
            mcts_params=cfg,
            inference_client=engine.inference_client,
            mode="sequential",
        )
        root = search.create_root(engine._ensure_native_state(state))
        started = time.perf_counter()
        search.run_search([root], add_noise=False)
        elapsed.append(time.perf_counter() - started)
        sims.append(root.visit_count)
    print(
        f"backend={os.getenv('ALPHAZERO_INFER_BACKEND', 'local')} "
        f"threads={os.getenv('ALPHAZERO_INFER_THREADS', '0')} "
        f"affinity={len(os.sched_getaffinity(0))} "
        f"sims median={statistics.median(sims):.0f} min={min(sims)} max={max(sims)} "
        f"elapsed max={max(elapsed):.2f}s"
    )


def _sample_states(engine: AlphaZeroEngine, count: int) -> list:
    """Evenly spaced positions from one game played by the raw network."""
    states = []
    state = engine.game.get_initial_state()
    while len(states) < count:
        states.append(state)
        action = engine.get_best_move(state, num_searches_override=1)
        state, _ = engine.apply_move(engine._ensure_native_state(state), action)
    return states


if __name__ == "__main__":
    main()
