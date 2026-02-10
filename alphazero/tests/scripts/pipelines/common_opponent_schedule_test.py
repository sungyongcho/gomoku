import types

import pytest

from gomoku.scripts.pipelines import common
from gomoku.utils.config.loader import OpponentConfig, RunnerParams, TrainingConfig
from gomoku.utils.config.schedule_param import ScheduledFloat


class _DummyParams(types.SimpleNamespace):
    """Minimal params stub for schedule computation."""


@pytest.mark.parametrize(
    "rnd,prev,expected_rnd,expected_prev",
    [
        (0.2, 0.3, 0.2, 0.3),
        (0.7, 0.7, 0.5, 0.5),  # normalize when sum>1
        (0.0, 0.0, 0.0, 0.0),
    ],
)
def test_opponent_rates_normalized(rnd: float, prev: float, expected_rnd: float, expected_prev: float):
    """opponent_rates 스케줄 합이 1을 넘으면 정규화되어야 한다."""
    opp = OpponentConfig(
        random_bot_ratio=float(rnd),
        prev_bot_ratio=float(prev),
        past_model_window=5,
    )
    train = TrainingConfig(
        num_iterations=1,
        num_selfplay_iterations=1.0,
        num_epochs=1,
        batch_size=1,
        learning_rate=0.1,
        weight_decay=0.0,
        temperature=1.0,
        replay_buffer_size=1,
        min_samples_to_train=1,
        random_play_ratio=None,
        opponent_rates=opp,
        priority_replay=None,
        dataloader_num_workers=0,
        dataloader_prefetch_factor=1,
        enable_tf32=False,
        use_channels_last=False,
    )
    params = _DummyParams(
        training=train,
        mcts=types.SimpleNamespace(
            dirichlet_epsilon=0.25,
            dirichlet_alpha=0.15,
            num_searches=1.0,
            exploration_turns=30,
        ),
        evaluation=types.SimpleNamespace(
            eval_num_searches=None,
            random_play_ratio=0.0,
            promotion_win_rate=0.5,
        ),
    )

    sched = common.compute_iteration_schedule(params, 0)
    assert pytest.approx(sched["rnd_bot_rate"], rel=1e-6) == expected_rnd
    assert pytest.approx(sched["prev_bot_rate"], rel=1e-6) == expected_prev
