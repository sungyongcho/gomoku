from __future__ import annotations

import numpy as np
import pytest

from gomoku.alphazero.agent import AlphaZeroAgent
from gomoku.alphazero.agents import RandomBot
from gomoku.alphazero.runners.selfplay import SelfPlayRunner
from gomoku.core.gomoku import Gomoku
from gomoku.inference.local import LocalInference
from gomoku.model.policy_value_net import PolicyValueNet
from gomoku.utils.config.loader import (
    BoardConfig,
    MctsConfig,
    ModelConfig,
    TrainingConfig,
)


@pytest.mark.parametrize("random_ratio", [1.0])
def test_selfplay_random_ratio_picks_uniform_legal(random_ratio: float) -> None:
    """When random_ratio=1.0, actions should be uniformly sampled from legal moves."""
    board_cfg = BoardConfig(
        num_lines=3,
        enable_doublethree=False,
        enable_capture=False,
        capture_goal=5,
        gomoku_goal=3,
        history_length=1,
    )
    game = Gomoku(board_cfg)
    mcts_cfg = MctsConfig(
        C=1.0,
        num_searches=1,
        exploration_turns=0,
        dirichlet_epsilon=0.0,
        dirichlet_alpha=0.3,
        batch_infer_size=1,
        max_batch_wait_ms=0,
        min_batch_size=1,
    )
    train_cfg = TrainingConfig(
        num_iterations=1,
        num_selfplay_iterations=1,
        num_epochs=1,
        batch_size=1,
        learning_rate=0.1,
        weight_decay=0.0,
        temperature=1.0,
        replay_buffer_size=10,
        min_samples_to_train=1,
        random_play_ratio=0.0,
        priority_replay=None,
        dataloader_num_workers=0,
        dataloader_prefetch_factor=1,
        enable_tf32=False,
        use_channels_last=False,
    )

    model_cfg = ModelConfig(num_planes=9, num_hidden=8, num_resblocks=1)
    model = PolicyValueNet(game, config=model_cfg, device="cpu")
    inference = LocalInference(model)
    agent = AlphaZeroAgent(
        game, mcts_cfg, inference_client=inference, engine_type="sequential"
    )
    runner = SelfPlayRunner(game, mcts_cfg, train_cfg)

    # First move with empty board: legal moves are all cells (uniform expected)
    samples = []
    for _ in range(50):
        record = runner.play_one_game(
            agent, temperature=1.0, add_noise=False, random_ratio=random_ratio
        )
        samples.append(record.moves[0])
    counts = np.bincount(samples, minlength=game.action_size)
    # Uniformity check: every move should be chosen at least once
    assert np.all(counts > 0), f"Some legal moves were never chosen: {counts}"


def test_selfplay_vs_random_bot_legal_moves() -> None:
    """Agent가 랜덤 봇과 대국할 때 두 플레이어 모두 합법 수만 두는지 확인."""
    board_cfg = BoardConfig(
        num_lines=3,
        enable_doublethree=False,
        enable_capture=False,
        capture_goal=5,
        gomoku_goal=3,
        history_length=1,
    )
    game = Gomoku(board_cfg)
    mcts_cfg = MctsConfig(
        C=1.0,
        num_searches=1,
        exploration_turns=0,
        dirichlet_epsilon=0.0,
        dirichlet_alpha=0.3,
        batch_infer_size=1,
        max_batch_wait_ms=0,
        min_batch_size=1,
    )
    train_cfg = TrainingConfig(
        num_iterations=1,
        num_selfplay_iterations=1,
        num_epochs=1,
        batch_size=1,
        learning_rate=0.1,
        weight_decay=0.0,
        temperature=1.0,
        replay_buffer_size=10,
        min_samples_to_train=1,
        random_play_ratio=0.0,
        priority_replay=None,
        dataloader_num_workers=0,
        dataloader_prefetch_factor=1,
        enable_tf32=False,
        use_channels_last=False,
    )
    model_cfg = ModelConfig(num_planes=9, num_hidden=8, num_resblocks=1)
    model = PolicyValueNet(game, config=model_cfg, device="cpu")
    inference = LocalInference(model)
    agent = AlphaZeroAgent(
        game, mcts_cfg, inference_client=inference, engine_type="sequential"
    )
    runner = SelfPlayRunner(game, mcts_cfg, train_cfg)
    rnd_bot = RandomBot(game)

    record = runner.play_one_game(
        agent,
        temperature=1.0,
        add_noise=False,
        random_ratio=0.0,
        opponent=rnd_bot,
        agent_first=True,
    )
    for move in record.moves:
        assert 0 <= move < game.action_size
