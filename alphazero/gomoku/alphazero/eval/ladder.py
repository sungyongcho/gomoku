"""Round-robin ladder between models: who is strongest, with search and without noise.

The promotion gate compares the latest model with one champion only, and a fixed
outside model stops telling models apart once they all beat it. A ladder plays every model against
every other from the same random openings with colours swapped, then fits one rating per
model, so progress over a run can be read from one scale.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from itertools import combinations
import math
import random
from typing import Any

import ray
import torch

from gomoku.alphazero.eval import reference_match as rm
from gomoku.core.gomoku import Gomoku
from gomoku.utils.config.loader import BoardConfig, MctsConfig, ModelConfig

# One player: its model config, its search settings and its weights.
Player = tuple[ModelConfig, MctsConfig, dict]


def fit_ratings(
    results: list[tuple[str, str, float]], iterations: int = 500
) -> dict[str, float]:
    """Elo-scale ratings from games given as (a, b, score of a), mean rating 0.

    Bradley-Terry strengths fitted by the usual minorisation steps. One virtual draw
    between every pair of players keeps an unbeaten player's rating finite.
    """
    players = sorted({name for a, b, _ in results for name in (a, b)})
    wins: dict[str, float] = defaultdict(float)
    games: dict[tuple[str, str], float] = defaultdict(float)
    for a, b, score in results:
        wins[a] += score
        wins[b] += 1.0 - score
        games[tuple(sorted((a, b)))] += 1.0
    for a, b in combinations(players, 2):
        wins[a] += 0.5
        wins[b] += 0.5
        games[(a, b)] += 1.0
    strength = dict.fromkeys(players, 1.0)
    for _ in range(iterations):
        updated = {}
        for player in players:
            denominator = sum(
                count / (strength[a] + strength[b])
                for (a, b), count in games.items()
                if player in (a, b)
            )
            updated[player] = wins[player] / denominator if denominator else 1.0
        scale = math.exp(sum(math.log(v) for v in updated.values()) / len(updated))
        strength = {player: value / scale for player, value in updated.items()}
    return {player: 400.0 * math.log10(value) for player, value in strength.items()}


def play_ladder_game(
    board: BoardConfig,
    sides: dict[str, Player],
    black: str,
    opening_seed: int,
    opening_plies: int,
    max_plies: int,
) -> dict[str, Any]:
    """Play one game between the two models in ``sides``.

    A game that reaches ``max_plies`` counts as a draw.
    """
    torch.set_num_threads(1)
    game = Gomoku(board, use_native=True)
    searches = {name: rm._player(game, *spec) for name, spec in sides.items()}  # noqa: SLF001
    white = next(name for name in sides if name != black)
    seat = {1: black, 2: white}
    rng = random.Random(opening_seed)
    state = game.get_initial_state()
    winner = None
    plies = 0
    while plies < max_plies:
        mover = int(state.next_player)
        if plies < opening_plies:
            game.get_legal_moves(state)
            action = int(rng.choice(sorted(int(i) for i in state.legal_indices_cache)))
        else:
            action = rm._most_visited(game, searches[seat[mover]], state)  # noqa: SLF001
        xy = (action % game.col_count, action // game.col_count)
        state = game.get_next_state(state, xy, mover)
        plies += 1
        value, terminal = game.get_value_and_terminated(state, xy)
        if terminal:
            winner = seat[mover] if value != 0 else None
            break
    return {"black": black, "white": white, "winner": winner, "plies": plies}


@ray.remote(num_cpus=1)
def _ladder_game_task(
    board: BoardConfig,
    side_a: Player,
    side_b: Player,
    a: str,
    b: str,
    black: str,
    opening_seed: int,
    opening_plies: int,
    max_plies: int,
) -> dict[str, Any]:
    sides = {a: side_a, b: side_b}
    return play_ladder_game(board, sides, black, opening_seed, opening_plies, max_plies)


def summarize_ladder(games: list[dict[str, Any]]) -> dict[str, Any]:
    """Pair scores and ratings from finished games (each with a, b, winner)."""
    outcomes = [
        (g["a"], g["b"], 0.5 if g["winner"] is None else float(g["winner"] == g["a"]))
        for g in games
    ]
    pairs: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    points: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for a, b, score in outcomes:
        pairs[f"{a} vs {b}"][0] += score
        pairs[f"{a} vs {b}"][1] += 1
        points[a][0] += score
        points[a][1] += 1
        points[b][0] += 1.0 - score
        points[b][1] += 1
    ratings = fit_ratings(outcomes) if outcomes else {}
    return {
        "games": len(games),
        "ratings": {name: round(value, 1) for name, value in ratings.items()},
        "points": {name: value for name, value in points.items()},
        "pairs": dict(pairs),
        "mean_plies": sum(g["plies"] for g in games) / len(games) if games else 0.0,
    }


def run_ladder(
    board: BoardConfig,
    players: dict[str, Player],
    *,
    pairs: int,
    opening_plies: int,
    max_plies: int,
    seed: int,
    on_game: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Play every pairing on Ray and return pair scores and ratings.

    ``pairs`` openings per pairing, each played twice with colours swapped. ``on_game``
    receives every finished game at once, so a caller can store games as they arrive.
    """
    if len(players) < 2:
        raise ValueError("a ladder needs at least two players")
    refs = {name: ray.put(side) for name, side in players.items()}
    on_cluster = "selfplay" in ray.cluster_resources()
    options = {"resources": {"selfplay": 1}} if on_cluster else {}
    pending = {}
    for a, b in combinations(list(players), 2):
        for pair in range(pairs):
            for black in (a, b):
                task = _ladder_game_task.options(**options)
                ref = task.remote(
                    board, refs[a], refs[b], a, b, black, seed + pair,
                    opening_plies, max_plies,
                )
                pending[ref] = (a, b, pair)
    games = []
    while pending:
        done, _ = ray.wait(list(pending), num_returns=1)
        a, b, pair = pending.pop(done[0])
        game = {"a": a, "b": b, "pair": pair, **ray.get(done[0])}
        games.append(game)
        if on_game is not None:
            on_game(game)
    return summarize_ladder(games)
