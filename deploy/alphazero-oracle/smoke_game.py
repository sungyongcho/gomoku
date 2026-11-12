"""Play one full game against a running engine through its WebSocket, both sides.

The engine answers one move per request, so asking it for every move makes it play itself.
This client keeps its own copy of the game with the repository's rules and checks each
answer against it: the move is legal, the returned board and capture scores match, and the
move came back within the time limit. Run it from alphazero/ with the project virtualenv:

    ssh -N -L 18080:127.0.0.1:8080 ubuntu@<host> &        # the port is not public
    python ../deploy/alphazero-oracle/smoke_game.py --url ws://127.0.0.1:18080/ws

With ``mcts.reuse_tree`` the answers also say how many searches stand behind each move:
those carried over in the kept tree plus the new ones. Asking for both colours on one
connection keeps the whole subtree of the engine's own move, which no real game does. To
measure what a game against a person gets, use two connections, one per colour, so each
sees its own move followed by the opponent's reply, and give the opponent a thinking time
for ``mcts.ponder`` to use:

    python ../deploy/alphazero-oracle/smoke_game.py --url ... --connections 2 --think-s 5

Exits non-zero when an answer is wrong or late.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import AsyncExitStack
import json
import statistics
import time

import websockets

from gomoku.core.gomoku import Gomoku
from gomoku.utils.config.loader import load_and_parse_config

STONE = {0: ".", 1: "X", 2: "O"}


def request_for(state, game: Gomoku, moves: list[int]) -> dict:
    """The frontend's move request for ``state``."""
    size = game.col_count
    payload = {
        "type": "move",
        "board": [[STONE[int(cell)] for cell in row] for row in state.board],
        "scores": [
            {"player": "X", "score": int(state.p1_pts)},
            {"player": "O", "score": int(state.p2_pts)},
        ],
        "nextPlayer": STONE[int(state.next_player)],
        "goal": 5,
        "enableCapture": True,
        "enableDoubleThreeRestriction": True,
    }
    if moves:
        last = moves[-1]
        payload["lastPlay"] = {
            "coordinate": {"x": last % size, "y": last // size},
            "stone": STONE[3 - int(state.next_player)],
        }
        payload["recentMoves"] = [
            {"x": move % size, "y": move // size} for move in moves[-game.history_length :]
        ]
    return payload


async def play(
    url: str,
    config: str,
    limit_s: float,
    max_plies: int,
    connections: int = 1,
    think_s: float = 0.0,
) -> int:
    cfg = load_and_parse_config(config)
    game = Gomoku(cfg.board, use_native=False)
    state = game.get_initial_state()
    moves: list[int] = []
    times: list[float] = []
    searches: list[dict] = []
    problems: list[str] = []
    result = "no result"
    async with AsyncExitStack() as stack:
        sockets = [
            await stack.enter_async_context(websockets.connect(url, open_timeout=20))
            for _ in range(connections)
        ]
        for ply in range(max_plies):
            mover = int(state.next_player)
            socket = sockets[ply % connections]
            if ply and think_s > 0:
                await asyncio.sleep(think_s)  # the side to move "thinks"
            start = time.monotonic()
            await socket.send(json.dumps(request_for(state, game, moves)))
            answer = json.loads(await asyncio.wait_for(socket.recv(), timeout=limit_s + 10))
            elapsed = time.monotonic() - start
            times.append(elapsed)
            if answer.get("type") != "move":
                problems.append(f"ply {ply + 1}: unexpected answer {answer}")
                break
            if "search" in answer:
                searches.append(answer["search"])
            coordinate = answer["lastPlay"]["coordinate"]
            x, y = int(coordinate["x"]), int(coordinate["y"])
            action = x + y * game.col_count
            game.get_legal_moves(state)
            if action not in {int(i) for i in state.legal_indices_cache}:
                problems.append(f"ply {ply + 1}: illegal move ({x}, {y})")
                break
            state = game.get_next_state(state, (x, y), mover)
            moves.append(action)
            board = [[STONE[int(cell)] for cell in row] for row in state.board]
            scores = {item["player"]: item["score"] for item in answer["scores"]}
            if answer["board"] != board:
                problems.append(f"ply {ply + 1}: returned board differs from the rules' board")
            if scores != {"X": int(state.p1_pts), "O": int(state.p2_pts)}:
                problems.append(f"ply {ply + 1}: capture scores differ: {scores}")
            if elapsed > limit_s:
                problems.append(f"ply {ply + 1}: answered in {elapsed:.2f}s, limit {limit_s}s")
            value, terminal = game.get_value_and_terminated(state, (x, y))
            if terminal:
                result = f"{STONE[mover]} wins" if value != 0 else "draw"
                break
    print(f"{len(moves)} plies, {result}")
    if times:
        print(
            f"answer time: mean {sum(times) / len(times):.2f}s, max {max(times):.2f}s, "
            f"first move {times[0]:.2f}s"
        )
    print(f"captures: X {int(state.p1_pts)}, O {int(state.p2_pts)}")
    # A connection's first answer has no kept tree yet, so it is left out. Medians, because
    # near the end of a game most searches stop at finished positions without evaluating
    # the network and the counts grow a hundredfold.
    kept = searches[connections:]
    if kept:

        def median(key: str) -> float:
            return statistics.median(item[key] for item in kept)

        print(
            f"searches per move after each connection's first: median {median('total'):.0f} "
            f"({median('carried'):.0f} carried over, {median('new'):.0f} new), "
            f"min {min(item['total'] for item in kept)}; "
            f"searched while waiting: median {median('pondered'):.0f}"
        )
        for key in ("carried", "new", "pondered"):
            print(f"{key} per move:", " ".join(str(item[key]) for item in searches))
    for problem in problems:
        print("PROBLEM", problem)
    return 1 if problems else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", required=True, help="for example ws://127.0.0.1:18080/ws")
    parser.add_argument("--config", default="configs/deploy-v5-pruned.yaml", help="board rules")
    parser.add_argument("--limit-s", type=float, default=3.0, help="longest answer allowed")
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument(
        "--connections",
        type=int,
        choices=(1, 2),
        default=1,
        help="2 gives each colour its own connection, as in a game against an opponent",
    )
    parser.add_argument(
        "--think-s", type=float, default=0.0, help="wait this long before asking for each move"
    )
    args = parser.parse_args()
    raise SystemExit(
        asyncio.run(
            play(
                args.url,
                args.config,
                args.limit_s,
                args.max_plies,
                connections=args.connections,
                think_s=args.think_s,
            )
        )
    )


if __name__ == "__main__":
    main()
