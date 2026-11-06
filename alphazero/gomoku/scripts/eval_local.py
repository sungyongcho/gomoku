"""Local model evaluation: opening behaviour and equal-time duels.

The opening report only measures models. Nothing here feeds back into the engine.

Players are written as ``name=checkpoint[,config[,backend]]``:
    python -m gomoku.scripts.eval_local opening old=runs/a.pt new=runs/b.pt,configs/small.yaml
    python -m gomoku.scripts.eval_local duel \\
        static=models/champion.pt,configs/deploy.yaml,onnx-static \\
        dynamic=models/champion.pt,configs/deploy.yaml,onnx-int8 --games 20 --budget-s 2.5
"""

import argparse
from dataclasses import dataclass
import math
import multiprocessing as mp
import os
import random
import tempfile
import time

import numpy as np
import torch

from server.engine import AlphaZeroEngine

DEFAULT_CONFIG = "configs/deploy.yaml"
FIRST_STONES = ((9, 9), (3, 3), (15, 9), (0, 0))


@dataclass(frozen=True)
class Player:
    name: str
    checkpoint: str
    config: str = DEFAULT_CONFIG
    backend: str = "local"


def parse_player(text: str) -> Player:
    name, _, rest = text.partition("=")
    if not rest:
        raise argparse.ArgumentTypeError(f"expected name=checkpoint[,config[,backend]]: {text}")
    parts = rest.split(",")
    return Player(name, parts[0], *(part for part in parts[1:] if part))


def load_engine(player: Player, cache_dir: str, threads: int) -> AlphaZeroEngine:
    os.environ["ALPHAZERO_INFER_BACKEND"] = player.backend
    os.environ["ALPHAZERO_INFER_THREADS"] = str(threads)
    os.environ["ONNX_CACHE_DIR"] = os.path.join(cache_dir, player.name)
    return AlphaZeroEngine(player.config, player.checkpoint, "cpu")


@dataclass(frozen=True)
class OpeningReport:
    first_stone: tuple[int, int]
    reply: tuple[int, int]
    distance: int
    value: float
    entropy: float
    uniform_entropy: float
    top_visits: int
    tied_top: int


def opening_report(
    engine: AlphaZeroEngine, first_stone: tuple[int, int], sims: int = 200
) -> OpeningReport:
    """Measure the network and search answer to a single first stone."""
    game = engine.game
    action = first_stone[0] + first_stone[1] * game.col_count
    state, _ = engine.apply_move(
        engine._ensure_native_state(game.get_initial_state()), action
    )
    encoded = np.asarray(game.get_encoded_state(state), dtype=np.float32)
    encoded = encoded[0] if encoded.ndim == 4 else encoded
    logits, value = engine.inference_client.infer(torch.as_tensor(encoded))
    logits = logits.reshape(-1).detach().cpu().numpy().astype(np.float64)
    legal = state.board.reshape(-1) == 0
    logits[~legal] = -np.inf
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    support = probs[probs > 0]

    root, policy = engine.search(state, sims)
    visits = [child.visit_count for child in root.children.values()]
    top = max(visits)
    move = engine._most_visited_action(root, policy)
    reply = (move % game.col_count, move // game.col_count)
    return OpeningReport(
        first_stone=first_stone,
        reply=reply,
        distance=max(abs(reply[0] - first_stone[0]), abs(reply[1] - first_stone[1])),
        value=float(value.reshape(-1)[0]),
        entropy=float(-(support * np.log(support)).sum()),
        uniform_entropy=math.log(int(legal.sum())),
        top_visits=top,
        tied_top=visits.count(top),
    )


def run_opening(players: list[Player], sims: int) -> None:
    print("player        stone    reply    dist  value   entropy/uniform  top_visits  tied_top")
    with tempfile.TemporaryDirectory() as cache:
        for player in players:
            engine = load_engine(player, cache, threads=0)
            engine.mcts_config = engine.mcts_config.model_copy(update={"time_limit_s": None})
            for stone in FIRST_STONES:
                r = opening_report(engine, stone, sims)
                print(
                    f"{player.name:13s} {str(r.first_stone):8s} {str(r.reply):8s} {r.distance:4d}"
                    f" {r.value:+.2f}   {r.entropy:.2f}/{r.uniform_entropy:.2f}"
                    f"        {r.top_visits:6d}  {r.tied_top:8d}",
                    flush=True,
                )


@dataclass(frozen=True)
class GameTask:
    opening_seed: int
    a_is_first: bool


@dataclass(frozen=True)
class GameResult:
    a_score: float
    plies: int
    first_player_won: bool
    longest_move_s: float


def _random_opening(engine: AlphaZeroEngine, seed: int, plies: int):
    """Place random stones so paired games leave the deterministic main line."""
    rng = random.Random(seed)
    game = engine.game
    state = engine._ensure_native_state(game.get_initial_state())
    for _ in range(rng.randint(0, plies)):
        root, _ = engine.search(state, 2)
        action = rng.choice(sorted(root.children))
        state, _ = engine.apply_move(state, action[0] + action[1] * game.col_count)
        if game.get_value_and_terminated(state, action)[1]:
            return game.get_initial_state()
    return state


def play_game(
    a: AlphaZeroEngine,
    b: AlphaZeroEngine,
    task: GameTask,
    opening_plies: int,
) -> GameResult:
    """Play one game with both engines on their own configured time budget."""
    game = a.game
    state = _random_opening(a, task.opening_seed, opening_plies)
    a_player = 1 if task.a_is_first else 2
    longest = 0.0
    plies = 0
    for plies in range(1, 2 * game.action_size + 1):
        mover = int(state.next_player)
        engine = a if mover == a_player else b
        started = time.perf_counter()
        action = engine.get_best_move(state)
        longest = max(longest, time.perf_counter() - started)
        xy = (action % game.col_count, action // game.col_count)
        state, _ = engine.apply_move(engine._ensure_native_state(state), action)
        value, terminal = game.get_value_and_terminated(state, xy)
        if terminal:
            if value == 0:
                return GameResult(0.5, plies, False, longest)
            winner = mover if value > 0 else 3 - mover
            return GameResult(float(winner == a_player), plies, winner == 1, longest)
    return GameResult(0.5, plies, False, longest)


def _duel_worker(args: tuple) -> list[GameResult]:
    a_spec, b_spec, tasks, cpus, budget_s, opening_plies = args
    os.sched_setaffinity(0, cpus)
    torch.set_num_threads(len(cpus))
    with tempfile.TemporaryDirectory() as cache:
        engines = [load_engine(spec, cache, threads=len(cpus)) for spec in (a_spec, b_spec)]
        for engine in engines:
            engine.mcts_config = engine.mcts_config.model_copy(
                update={"num_searches": 100_000.0, "time_limit_s": budget_s}
            )
        return [play_game(*engines, task, opening_plies) for task in tasks]


def elo_difference(score: float) -> float:
    clipped = min(max(score, 1e-3), 1 - 1e-3)
    return -400 * math.log10(1 / clipped - 1)


def run_duel(
    a: Player,
    b: Player,
    games: int,
    budget_s: float,
    workers: int,
    cpus_per_worker: int,
    opening_plies: int,
    seed: int,
) -> None:
    # Each opening is played twice with the colours swapped.
    tasks = [
        GameTask(seed + pair, a_first)
        for pair in range(math.ceil(games / 2))
        for a_first in (True, False)
    ]
    cpus = sorted(os.sched_getaffinity(0))
    workers = min(workers, len(cpus) // cpus_per_worker)
    jobs = [
        (a, b, tasks[index::workers], cpus[index * cpus_per_worker:(index + 1) * cpus_per_worker],
         budget_s, opening_plies)
        for index in range(workers)
    ]
    with mp.get_context("spawn").Pool(workers) as pool:
        results = [result for batch in pool.map(_duel_worker, jobs) for result in batch]
    wins = sum(r.a_score == 1 for r in results)
    draws = sum(r.a_score == 0.5 for r in results)
    losses = len(results) - wins - draws
    score = sum(r.a_score for r in results) / len(results)
    margin = 1.96 * float(np.std([r.a_score for r in results])) / math.sqrt(len(results))
    print(
        f"{a.name} vs {b.name}: +{wins} ={draws} -{losses} of {len(results)} "
        f"score={score:.3f} 95%=[{max(0, score - margin):.3f},{min(1, score + margin):.3f}] "
        f"elo={elo_difference(score):+.0f}"
    )
    print(
        f"budget={budget_s}s cpus_per_worker={cpus_per_worker} "
        f"first_player_wins={sum(r.first_player_won for r in results)} "
        f"mean_plies={np.mean([r.plies for r in results]):.0f} "
        f"longest_move={max(r.longest_move_s for r in results):.2f}s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    opening = commands.add_parser("opening")
    opening.add_argument("players", nargs="+", type=parse_player)
    opening.add_argument("--sims", type=int, default=200)
    duel = commands.add_parser("duel")
    duel.add_argument("a", type=parse_player)
    duel.add_argument("b", type=parse_player)
    duel.add_argument("--games", type=int, default=20)
    duel.add_argument("--budget-s", type=float, default=2.5)
    duel.add_argument("--workers", type=int, default=4)
    duel.add_argument("--cpus-per-worker", type=int, default=2)
    duel.add_argument("--opening-plies", type=int, default=4)
    duel.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    if args.command == "opening":
        run_opening(args.players, args.sims)
    else:
        run_duel(
            args.a, args.b, args.games, args.budget_s, args.workers,
            args.cpus_per_worker, args.opening_plies, args.seed,
        )


if __name__ == "__main__":
    main()
