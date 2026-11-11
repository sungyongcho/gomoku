"""Rank checkpoints against each other with a round-robin ladder on Ray.

    python gomoku/scripts/ladder.py --searches 800 --pairs 20 --out ladder.jsonl \
        it084=gs://bucket/run/ckpt/iteration_0084.pt \
        it130=gs://bucket/run/ckpt/iteration_0130.pt \
        v4=gs://bucket/reference/v4.pt@configs/deploy.yaml

Each model is ``name=checkpoint`` with an optional ``@config`` (default ``--config``)
naming the YAML that holds its model and search settings. Every finished game is
appended to ``--out`` at once, so a stopped run keeps what it played. The workers need the same
code as the driver, as for training.
"""

from __future__ import annotations

import argparse
import json
import time

import ray

from gomoku.alphazero.eval import reference_match as rm
from gomoku.alphazero.eval.ladder import run_ladder
from gomoku.utils.config.loader import load_and_parse_config


def _parse_model(spec: str, default_config: str) -> tuple[str, str, str]:
    name, _, rest = spec.partition("=")
    checkpoint, _, config = rest.partition("@")
    if not name or not checkpoint:
        raise SystemExit(f"model must be name=checkpoint[@config], got {spec!r}")
    return name, checkpoint, config or default_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("models", nargs="+", help="name=checkpoint[@config]")
    parser.add_argument("--config", default="configs/v5-small-6x96-cont5.yaml")
    parser.add_argument("--searches", type=int, default=800)
    parser.add_argument("--pairs", type=int, default=20, help="openings per pairing")
    parser.add_argument("--opening-plies", type=int, default=4)
    parser.add_argument("--max-plies", type=int, default=400)
    parser.add_argument("--seed", type=int, default=70000)
    parser.add_argument("--out", required=True, help="JSON lines, one game per line")
    args = parser.parse_args()

    specs = [_parse_model(spec, args.config) for spec in args.models]
    if len({name for name, _, _ in specs}) != len(specs) or len(specs) < 2:
        raise SystemExit("give at least two models with distinct names")

    ray.init(address="auto", logging_level="ERROR")
    board = None
    players = {}
    for name, checkpoint, config_path in specs:
        cfg = load_and_parse_config(config_path)
        if board is None:
            board = cfg.board
        elif cfg.board.model_dump() != board.model_dump():
            raise SystemExit(f"{name}: board rules differ from the first model's")
        mcts = rm.match_mcts(cfg.mcts, args.searches)
        players[name] = (cfg.model, mcts, rm.load_state_dict(checkpoint))

    start = time.monotonic()
    played = [0]
    with open(args.out, "a", encoding="utf-8") as handle:

        def store(game: dict) -> None:
            handle.write(json.dumps(game) + "\n")
            handle.flush()
            played[0] += 1
            if played[0] % 20 == 0:
                print(f"[{time.monotonic() - start:6.0f}s] {played[0]} games", flush=True)

        summary = run_ladder(
            board,
            players,
            pairs=args.pairs,
            opening_plies=args.opening_plies,
            max_plies=args.max_plies,
            seed=args.seed,
            on_game=store,
        )

    print(f"searches {args.searches}, {args.pairs} openings per pairing")
    for pairing, (score, count) in sorted(summary["pairs"].items()):
        print(f"PAIR {pairing}: {score:.1f}/{count:.0f} = {score / count:.3f}")
    for name, rating in sorted(summary["ratings"].items(), key=lambda item: -item[1]):
        points, count = summary["points"][name]
        print(f"RATING {name}: {rating:+7.1f}  ({points:.1f}/{count:.0f})")


if __name__ == "__main__":
    main()
