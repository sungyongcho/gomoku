"""Self-play games against a fixed outside model.

They bring the network positions its own games never reach.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

import gomoku.alphazero.runners.ray_runner as ray_runner_mod
from gomoku.alphazero.runners.workers.ray_worker import RaySelfPlayWorker
from gomoku.core.gomoku import Gomoku
from gomoku.model.policy_value_net import PolicyValueNet
from gomoku.utils.config.loader import (
    BoardConfig,
    ExternalOpponentConfig,
    MctsConfig,
    ModelConfig,
    OpponentConfig,
    TrainingConfig,
    parse_config,
)
from gomoku.utils.config.schedule_param import SchedulePoint, get_scheduled_value
from tests.helpers import minimal_raw_config


def _raw_with_external(ratio: Any) -> dict:
    raw = minimal_raw_config()
    raw["training"]["opponent_rates"] = {
        "external": {
            "name": "v4",
            "checkpoint": "gs://b/v4.pt",
            "config": "configs/deploy.yaml",
            "ratio": ratio,
            "num_searches": 400,
        }
    }
    return raw


def test_config_accepts_an_external_opponent_with_a_scheduled_share():
    raw = _raw_with_external([{"until": 5, "value": 0.0}, {"until": 10, "value": 0.25}])
    external = parse_config(raw).training.opponent_rates.external
    assert (external.name, external.num_searches) == ("v4", 400)
    assert get_scheduled_value(external.ratio, 2) == 0.0
    assert get_scheduled_value(external.ratio, 7) == 0.25


def test_external_share_schedule_must_cover_the_whole_run():
    with pytest.raises(ValueError):
        parse_config(_raw_with_external([{"until": 5, "value": 0.25}]))


def _mcts(num_searches: float = 10) -> MctsConfig:
    return MctsConfig(
        C=1.0,
        num_searches=num_searches,
        exploration_turns=1,
        dirichlet_epsilon=0.25,
        dirichlet_alpha=0.3,
        batch_infer_size=2,
        max_batch_wait_ms=0,
        min_batch_size=1,
    )


def _training(external: ExternalOpponentConfig | None) -> TrainingConfig:
    return TrainingConfig(
        num_iterations=2,
        num_selfplay_iterations=1.0,
        num_epochs=1,
        batch_size=1,
        learning_rate=0.1,
        weight_decay=0.0,
        temperature=1.0,
        replay_buffer_size=1,
        min_samples_to_train=1,
        random_opening_turns=0,
        opponent_rates=OpponentConfig(external=external),
    )


def test_runner_resolves_the_external_share_for_the_iteration():
    external = ExternalOpponentConfig(
        checkpoint="v4.pt",
        config="v4.yaml",
        ratio=[SchedulePoint(until=1, value=0.0), SchedulePoint(until=2, value=0.25)],
    )
    runner = ray_runner_mod.RayAsyncRunner(
        cfg=MagicMock(training=_training(external), mcts=_mcts()), num_actors=0
    )

    first, _ = runner._resolve_iteration_configs(0, None, None)
    second, _ = runner._resolve_iteration_configs(1, None, None)

    assert first.opponent_rates.external.ratio == 0.0
    assert second.opponent_rates.external.ratio == 0.25


class _Ref:
    def __init__(self, value: Any):
        self.value = value


def _get(refs: Any) -> Any:
    return [r.value for r in refs] if isinstance(refs, list) else refs.value


def _board() -> BoardConfig:
    return BoardConfig(
        num_lines=9,
        enable_doublethree=True,
        enable_capture=True,
        capture_goal=5,
        gomoku_goal=5,
    )


def test_runner_sends_the_outside_model_to_new_workers_and_passes_its_share(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    board = _board()
    other_model = ModelConfig(num_planes=13, num_hidden=8, num_resblocks=1)
    weights = {"w": torch.zeros(1)}
    monkeypatch.setattr(
        ray_runner_mod,
        "load_and_parse_config",
        lambda path: SimpleNamespace(board=board, model=other_model),
    )
    monkeypatch.setattr(ray_runner_mod, "load_state_dict", lambda path: weights)
    monkeypatch.setattr(
        ray_runner_mod,
        "ray",
        SimpleNamespace(
            put=_Ref,
            get=_get,
            wait=lambda refs, num_returns=1: (list(refs)[:num_returns], []),
            init=lambda **k: None,
            is_initialized=lambda: True,
            cluster_resources=lambda: {},
        ),
    )
    received: dict[str, Any] = {"external": [], "shares": []}

    class FakeWorker:
        def __init__(self):
            self.update_params = SimpleNamespace(remote=lambda **k: _Ref(None))
            self.set_external_opponent = SimpleNamespace(remote=self._set_external)
            self.run_games = SimpleNamespace(remote=self._run_games)

        def _set_external(self, model_cfg, num_searches, weights_ref):
            received["external"].append((model_cfg, num_searches, weights_ref.value))
            return _Ref(None)

        def _run_games(self, *args):
            received["shares"].append(args[-1])
            return _Ref([])

    class WorkerCtor:
        @classmethod
        def options(cls, **kwargs):
            return cls

        @classmethod
        def remote(cls, *args, **kwargs):
            return FakeWorker()

    monkeypatch.setattr(ray_runner_mod, "RaySelfPlayWorker", WorkerCtor)
    external = ExternalOpponentConfig(
        checkpoint="v4.pt", config="v4.yaml", ratio=0.25, num_searches=400
    )
    cfg = SimpleNamespace(
        board=board, training=_training(external), mcts=_mcts(), runtime=None
    )
    runner = ray_runner_mod.RayAsyncRunner(cfg=cfg, num_actors=0, num_workers=2)

    runner.run(batch_size=1, games=2, iteration_idx=0)

    assert received["external"] == [(other_model, 400, weights)] * 2
    assert received["shares"] == [0.25]  # both games fit in one chunk


def test_runner_rejects_an_outside_model_with_other_board_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    board = _board()
    monkeypatch.setattr(
        ray_runner_mod,
        "load_and_parse_config",
        lambda path: SimpleNamespace(board=board.model_copy(update={"num_lines": 15})),
    )
    external = ExternalOpponentConfig(checkpoint="v4.pt", config="v4.yaml", ratio=0.25)
    runner = ray_runner_mod.RayAsyncRunner(
        cfg=SimpleNamespace(board=board, training=_training(external)), num_actors=0
    )

    with pytest.raises(ValueError):
        runner._external_opponent()


def _bare_worker() -> Any:
    meta = RaySelfPlayWorker.__ray_metadata__
    worker_cls = meta.modified_class
    worker = worker_cls.__new__(worker_cls)
    worker.logger = MagicMock()
    worker.random_bot = MagicMock()
    worker.agent = MagicMock()
    worker.past_agent = None
    worker.past_opponent_path = None
    worker.external_agent = None
    worker._external = None
    worker.runner = MagicMock()
    worker.runner.play_batch_games.return_value = []
    return worker


def test_worker_plays_the_external_share_against_the_outside_model_on_both_colours():
    worker = _bare_worker()
    worker.external_agent = MagicMock(name="outside model")

    worker.run_games(batch_size=1, target_games=24, external_bot_rate=1.0)

    calls = worker.runner.play_batch_games.call_args_list
    assert len(calls) == 24
    assert all(c.kwargs["opponent"] is worker.external_agent for c in calls)
    assert {c.kwargs["agent_first"] for c in calls} == {True, False}


def test_worker_falls_back_to_self_play_without_the_outside_model_and_says_so_once():
    worker = _bare_worker()

    worker.run_games(batch_size=1, target_games=3, external_bot_rate=1.0)

    calls = worker.runner.play_batch_games.call_args_list
    assert len(calls) == 3 and all(c.kwargs["opponent"] is None for c in calls)
    notes = [str(c.args[0]) for c in worker.logger.info.call_args_list]
    assert sum("external opponent unavailable" in n for n in notes) == 1


def _game_and_planes() -> tuple[Gomoku, int]:
    game = Gomoku(
        BoardConfig(
            num_lines=9,
            enable_doublethree=True,
            enable_capture=True,
            capture_goal=5,
            gomoku_goal=5,
            history_length=1,
        )
    )
    return game, int(game.get_encoded_state(game.get_initial_state()).shape[1])


def test_worker_builds_the_outside_model_from_its_own_config_without_root_noise():
    game, planes = _game_and_planes()
    worker = _bare_worker()
    worker.game = game
    worker.cfg = SimpleNamespace(runtime=None)
    worker.model = None
    worker.agent = SimpleNamespace(
        mcts_cfg=_mcts(10), engine_type="sequential", async_inflight_limit=None
    )
    other = ModelConfig(num_planes=planes, num_hidden=8, num_resblocks=1)
    torch.manual_seed(1)
    weights = PolicyValueNet(game, other, torch.device("cpu")).state_dict()

    worker.set_external_opponent(other, 4, weights)

    agent = worker.external_agent
    assert agent.mcts_cfg.num_searches == 4
    assert agent.mcts_cfg.dirichlet_epsilon == 0.0
    loaded = agent.inference._model.state_dict()
    assert all(torch.equal(loaded[k], weights[k]) for k in weights)


def test_worker_follows_the_new_search_budget_when_the_outside_model_has_none():
    game, planes = _game_and_planes()
    worker = _bare_worker()
    worker.game = game
    worker.model = None
    worker.client = SimpleNamespace(max_batch_size=2)
    training = _training(None)
    worker.cfg = SimpleNamespace(
        training=training,
        mcts=_mcts(10),
        runtime=None,
        model_copy=lambda update: worker.cfg,
    )
    worker.runner = SimpleNamespace(train_cfg=training, mcts_cfg=_mcts(10))
    worker.agent = SimpleNamespace(
        mcts_cfg=_mcts(10),
        engine_type="sequential",
        async_inflight_limit=None,
        reset=lambda: None,
    )
    other = ModelConfig(num_planes=planes, num_hidden=8, num_resblocks=1)
    worker.set_external_opponent(
        other, None, PolicyValueNet(game, other, torch.device("cpu")).state_dict()
    )
    assert worker.external_agent.mcts_cfg.num_searches == 10

    worker.update_params(mcts_cfg=_mcts(20))

    assert worker.external_agent.mcts_cfg.num_searches == 20
    assert worker.external_agent.mcts_cfg.dirichlet_epsilon == 0.0
