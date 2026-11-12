"""Champion versus a fixed reference model (for example the v4 champion), run on Ray CPU workers.

The promotion gate only compares a run with its own previous champion, so it cannot show
whether the run has caught up with an older, differently sized model. This match does:
both sides search ``num_searches`` times per move with no time limit, root noise or root
batching, and each random opening is played twice with colours swapped.
"""

from __future__ import annotations

import io
import logging
import random
import time
from typing import Any

import fsspec
import numpy as np
import ray
import torch

from gomoku.core.gomoku import Gomoku
from gomoku.inference.local import LocalInference
from gomoku.model.policy_value_net import PolicyValueNet
from gomoku.pvmcts.pvmcts import PVMCTS
from gomoku.utils.config.loader import (
    BoardConfig,
    MctsConfig,
    ModelConfig,
    ReferenceMatchConfig,
    load_and_parse_config,
)
from gomoku.utils.state_dict_utils import align_state_dict_to_model

logger = logging.getLogger(__name__)

CHAMPION = "champion"
REFERENCE = "reference"


def load_state_dict(path: str) -> dict[str, torch.Tensor]:
    """Read a checkpoint from a local path or fsspec URL and return its CPU state dict."""
    with fsspec.open(path, "rb") as handle:
        raw = torch.load(io.BytesIO(handle.read()), map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        raw = raw.get("model_state_dict", raw.get("state_dict", raw))
    return {k: v.cpu() for k, v in raw.items()}


def match_mcts(mcts: MctsConfig, num_searches: int) -> MctsConfig:
    """Deterministic search with a fixed budget: no time limit, noise or root batching."""
    return mcts.model_copy(
        update={
            "num_searches": float(num_searches),
            "time_limit_s": None,
            "batch_infer_size": 1,
            "dirichlet_epsilon": 0.0,
            "exploration_turns": 0,
        }
    )


def _player(game: Gomoku, model_cfg: ModelConfig, mcts: MctsConfig, state_dict) -> PVMCTS:
    model = PolicyValueNet(game, model_cfg, torch.device("cpu"))
    align_state_dict_to_model(state_dict, model.state_dict())
    model.load_state_dict(state_dict)
    return PVMCTS(game, mcts, LocalInference(model.eval()), mode="sequential")


def _most_visited(game: Gomoku, search: PVMCTS, state) -> int:
    root = search.create_root(state)
    [(policy, _)] = search.run_search([root], add_noise=False)
    if not root.children:
        return int(np.argmax(policy))
    x, y = max(
        root.children,
        key=lambda move: (root.children[move].visit_count, root.children[move].prior),
    )
    return int(x + y * game.col_count)


def play_game(
    board: BoardConfig,
    sides: dict[str, tuple[ModelConfig, MctsConfig, dict]],
    black: str,
    opening_seed: int,
    opening_plies: int,
) -> dict[str, Any]:
    """Play one game; ``sides`` maps CHAMPION/REFERENCE to (model config, mcts config, weights)."""
    torch.set_num_threads(1)
    # Weights decayed to denormals make float32 inference about 20 times slower on Intel CPUs.
    torch.set_flush_denormal(True)
    game = Gomoku(board, use_native=True)
    searches = {name: _player(game, *spec) for name, spec in sides.items()}
    seat = {1: black, 2: REFERENCE if black == CHAMPION else CHAMPION}
    rng = random.Random(opening_seed)
    state = game.get_initial_state()
    winner = None
    plies = 0
    for _ in range(game.action_size * 2):
        mover = int(state.next_player)
        if plies < opening_plies:
            game.get_legal_moves(state)
            action = int(rng.choice(sorted(int(i) for i in state.legal_indices_cache)))
        else:
            action = _most_visited(game, searches[seat[mover]], state)
        xy = (action % game.col_count, action // game.col_count)
        state = game.get_next_state(state, xy, mover)
        plies += 1
        value, terminal = game.get_value_and_terminated(state, xy)
        if terminal:
            winner = seat[mover] if value != 0 else None
            break
    return {"black": black, "winner": winner, "plies": plies}


@ray.remote(num_cpus=1)
def _play_game_task(board, sides, black, opening_seed, opening_plies):
    return play_game(board, sides, black, opening_seed, opening_plies)


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Score from the champion's side; draws count half."""
    games = len(results)
    wins = sum(r["winner"] == CHAMPION for r in results)
    losses = sum(r["winner"] == REFERENCE for r in results)
    draws = games - wins - losses
    as_black = [r for r in results if r["black"] == CHAMPION]
    as_white = [r for r in results if r["black"] == REFERENCE]
    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "score": (wins + 0.5 * draws) / games if games else 0.0,
        "wins_as_black": sum(r["winner"] == CHAMPION for r in as_black),
        "games_as_black": len(as_black),
        "wins_as_white": sum(r["winner"] == CHAMPION for r in as_white),
        "games_as_white": len(as_white),
        "mean_plies": float(np.mean([r["plies"] for r in results])) if results else 0.0,
    }


def run_reference_match(
    board: BoardConfig,
    model_cfg: ModelConfig,
    mcts: MctsConfig,
    champion_state: dict,
    ref_cfg: ReferenceMatchConfig,
    iteration: int,
) -> dict[str, Any]:
    """Play the champion against the reference on Ray and return the summary."""
    reference = load_and_parse_config(ref_cfg.config)
    if reference.board.model_dump() != board.model_dump():
        raise ValueError(f"Reference {ref_cfg.name} uses different board rules")
    sides = {
        CHAMPION: (model_cfg, match_mcts(mcts, ref_cfg.num_searches), champion_state),
        REFERENCE: (
            reference.model,
            match_mcts(reference.mcts, ref_cfg.reference_searches or ref_cfg.num_searches),
            load_state_dict(ref_cfg.checkpoint),
        ),
    }
    sides_ref = ray.put(sides)
    options = {"resources": {"selfplay": 1}} if "selfplay" in ray.cluster_resources() else {}
    pairs = max(1, ref_cfg.num_games // 2)
    start = time.monotonic()
    futures = [
        _play_game_task.options(**options).remote(
            board, sides_ref, black, iteration * 1000 + pair, ref_cfg.opening_plies
        )
        for pair in range(pairs)
        for black in (CHAMPION, REFERENCE)
    ]
    summary = summarize(ray.get(futures))
    summary.update(
        {
            "iteration": iteration,
            "reference": ref_cfg.name,
            "num_searches": ref_cfg.num_searches,
            "seconds": round(time.monotonic() - start, 1),
        }
    )
    return summary
