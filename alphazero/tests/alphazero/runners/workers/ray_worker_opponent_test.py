from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gomoku.alphazero.runners.workers.ray_worker import RaySelfPlayWorker
from gomoku.utils.config.loader import OpponentConfig, load_and_parse_config


@pytest.mark.parametrize(
    "rnd_rate, prev_rate, expected_warn",
    [
        (0.5, 0.0, False),
        (0.0, 0.5, True),
    ],
)
def test_ray_worker_prev_bot_warning(
    monkeypatch: pytest.MonkeyPatch,
    rnd_rate: float,
    prev_rate: float,
    expected_warn: bool,
) -> None:
    """prev_bot_rate가 주어지면 self-play 대체 시 경고 로그가 한 번만 찍힌다."""
    cfg = load_and_parse_config("configs/config_alphazero_vectorize_test.yaml")
    opp_cfg = OpponentConfig(
        random_bot_ratio=rnd_rate,
        prev_bot_ratio=prev_rate,
        past_model_window=3,
    )
    cfg = cfg.model_copy(
        update={"training": cfg.training.model_copy(update={"opponent_rates": opp_cfg})}
    )

    # 로거를 모킹해 호출 여부 확인
    dummy_logger = MagicMock()
    monkeypatch.setattr(
        "gomoku.alphazero.runners.workers.ray_worker.setup_actor_logging",
        lambda *args, **kwargs: dummy_logger,
    )

    def _unwrap_ray_actor(actor_cls):
        meta = getattr(actor_cls, "__ray_metadata__", None)
        if meta is None:
            return actor_cls
        for attr in ("modified_class", "class_ref", "cls", "decorated_class"):
            if hasattr(meta, attr):
                return getattr(meta, attr)
        return actor_cls

    # Ray를 실제로 띄우지 않고 메서드 호출만 검증
    worker_cls = _unwrap_ray_actor(RaySelfPlayWorker)
    worker = worker_cls.__new__(worker_cls)
    worker.cfg = cfg
    worker.logger = dummy_logger
    worker.random_bot = MagicMock()  # 실제 행동은 필요 없음
    worker.agent = MagicMock()  # 메인 에이전트
    worker.past_agent = None
    worker.past_opponent_path = None
    worker._prev_bot_warned = False
    worker.runner = MagicMock()
    worker.runner.play_batch_games.return_value = []

    # run_games 호출
    worker.run_games(
        batch_size=1,
        target_games=3,
        random_ratio=0.0,
        random_bot_rate=rnd_rate,
        prev_bot_rate=prev_rate,
    )

    warn_called = any(
        "prev_bot_rate requested but past opponent unavailable" in str(call.args[0])
        for call in dummy_logger.info.call_args_list
    )
    assert warn_called == expected_warn
