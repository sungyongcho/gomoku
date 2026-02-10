"""Multiprocessing self-play worker loop."""

from __future__ import annotations

import multiprocessing as mp
import random
import time

import torch

from gomoku.alphazero.agent import AlphaZeroAgent
from gomoku.alphazero.agents import RandomBot
from gomoku.alphazero.runners.selfplay import SelfPlayRunner
from gomoku.core.gomoku import Gomoku
from gomoku.inference.mp_client import MPInferenceClient
from gomoku.utils.config.loader import RootConfig


def _worker_selfplay_loop(
    worker_id: int,
    cfg: RootConfig,
    request_q: mp.Queue,
    result_q: mp.Queue,
    record_q: mp.Queue,
    games_per_worker: int,
    past_opponent_path: str | None = None,
) -> None:
    """Run self-play games in a worker process and push results."""
    seed = worker_id + int(time.time())
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    game = Gomoku(cfg.board)
    inference_client = MPInferenceClient(worker_id, request_q, result_q)
    engine_type = "mp"
    if getattr(cfg.mcts, "use_native", False):
        engine_type = "sequential"
    agent = AlphaZeroAgent(
        game=game,
        mcts_cfg=cfg.mcts,
        inference_client=inference_client,
        engine_type=engine_type,
    )
    runner = SelfPlayRunner(game=game, mcts_cfg=cfg.mcts, train_cfg=cfg.training)
    random_bot = RandomBot(game)
    past_agent = None
    if past_opponent_path:
        try:
            from gomoku.inference.local import LocalInference
            from gomoku.model.policy_value_net import PolicyValueNet

            past_model = PolicyValueNet(game, cfg.model, device="cpu")
            past_state = torch.load(past_opponent_path, map_location="cpu")
            past_model.load_state_dict(past_state, strict=False)
            past_infer = LocalInference(past_model)
            past_engine_type = "mp"
            if getattr(cfg.mcts, "use_native", False):
                past_engine_type = "sequential"
            past_agent = AlphaZeroAgent(
                game=game,
                mcts_cfg=cfg.mcts,
                inference_client=past_infer,
                engine_type=past_engine_type,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[MP] Failed to load past opponent at {past_opponent_path}: {exc}")
            past_agent = None
    opp_cfg = getattr(cfg.training, "opponent_rates", None)
    rnd_bot_rate = float(getattr(opp_cfg, "random_bot_ratio", 0.0)) if opp_cfg else 0.0
    prev_bot_rate = float(getattr(opp_cfg, "prev_bot_ratio", 0.0)) if opp_cfg else 0.0
    total_rate = rnd_bot_rate + prev_bot_rate
    if total_rate > 1.0:
        scale = 1.0 / total_rate
        rnd_bot_rate *= scale
        prev_bot_rate *= scale
    warned_prev = False

    for _ in range(games_per_worker):
        r = random.random()
        opponent = None
        agent_first = True
        if r < rnd_bot_rate:
            opponent = random_bot
            agent_first = random.choice([True, False])
        elif r < rnd_bot_rate + prev_bot_rate:
            if past_agent is not None:
                opponent = past_agent
                agent_first = random.choice([True, False])
            else:
                if not warned_prev:
                    print(
                        "[MP] prev_bot_rate requested but past opponent unavailable; using self-play."
                    )
                    warned_prev = True
                opponent = None

        record = runner.play_one_game(
            agent,
            temperature=cfg.training.temperature,
            random_ratio=getattr(cfg.training, "random_play_ratio", 0.0),
            random_opening_turns=getattr(cfg.training, "random_opening_turns", 0),
            opponent=opponent,
            agent_first=agent_first,
        )
        record_q.put(record)
