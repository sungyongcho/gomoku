from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

import gomoku.alphazero.runners.ray_runner as ray_runner_mod
import gomoku.alphazero.runners.workers.ray_worker as ray_worker_mod
import gomoku.inference.ray_client as ray_client_mod
import gomoku.pvmcts.search.ray.batch_inference_manager as bim_mod


class FakeRef:
    """Minimal Ray ObjectRef stand-in."""

    def __init__(self, value: Any):
        self.value = value


def _patch_fake_ray(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch ray APIs (put/get/wait/cancel) with lightweight fakes."""

    def fake_put(obj: Any) -> FakeRef:
        return FakeRef(obj)

    def fake_get(ref: Any) -> Any:
        return ref.value if isinstance(ref, FakeRef) else ref

    def fake_wait(refs, num_returns=None, timeout=None, fetch_local=True):
        ready = list(refs)
        if num_returns is not None:
            ready = ready[:num_returns]
        return ready, []

    def fake_cancel(ref, force=True):
        return True

    monkeypatch.setattr(
        ray_client_mod,
        "ray",
        SimpleNamespace(put=fake_put, get=fake_get, ObjectRef=FakeRef),
    )
    monkeypatch.setattr(
        bim_mod,
        "ray",
        SimpleNamespace(
            put=fake_put,
            get=fake_get,
            wait=fake_wait,
            cancel=fake_cancel,
        ),
    )
    monkeypatch.setattr(
        ray_runner_mod,
        "ray",
        SimpleNamespace(
            put=fake_put,
            get=fake_get,
            wait=fake_wait,
            cancel=fake_cancel,
            init=lambda **k: None,
            is_initialized=lambda: True,
            shutdown=lambda: None,
            cluster_resources=lambda: {},
        ),
    )


def test_ray_client_roundtrip_single_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """RayInferenceClient should preserve shapes/dtypes through fake round-trip."""
    _patch_fake_ray(monkeypatch)
    action_size = 7

    class FakeActor:
        def __init__(self):
            self.infer = SimpleNamespace(remote=self._infer_remote)

        def _infer_remote(
            self, inputs: np.ndarray, native_payload=None, model_slot=None
        ) -> FakeRef:
            batch = inputs.shape[0] if inputs.ndim == 4 else 1
            policy = torch.arange(action_size, dtype=torch.float32).repeat(batch, 1)
            value = torch.ones(batch, 1, dtype=torch.float32)
            return FakeRef((policy, value))

    client = ray_client_mod.RayInferenceClient([FakeActor()], max_batch_size=8)

    state = torch.zeros(3, 2, 2, dtype=torch.float32)
    policy, value = client.infer(state)

    print(
        f"[debug] roundtrip shapes policy={tuple(policy.shape)} value={tuple(value.shape)}"
    )
    assert tuple(policy.shape) == (1, action_size)
    assert tuple(value.shape) in {(1, 1), (1,)}
    assert policy.dtype == torch.float32
    assert value.dtype == torch.float32


def test_batch_manager_timeout_flush(monkeypatch: pytest.MonkeyPatch) -> None:
    """Partial batch should flush after timeout."""
    _patch_fake_ray(monkeypatch)
    action_size = 5

    class _DummyClient:
        def infer_async(self, batch: torch.Tensor, native_payload=None) -> FakeRef:
            policy = torch.ones(batch.size(0), action_size, dtype=torch.float32)
            value = torch.zeros(batch.size(0), 1, dtype=torch.float32)
            return FakeRef((policy, value))

    manager = bim_mod.BatchInferenceManager(
        client=_DummyClient(),
        batch_size=8,
        max_wait_ms=1,
        max_inflight_batches=None,
    )

    dummy_tensor = torch.zeros(1, 2, 2, dtype=torch.float32)
    manager.enqueue(
        bim_mod.PendingNodeInfo(node=None, is_start_node=True), dummy_tensor
    )

    time.sleep(0.01)  # exceed timeout
    results = manager.drain_ready(timeout_s=0.0)

    print(
        f"[debug] timeout flush results={len(results)} mapping={len(results[0].mapping) if results else 0}"
    )
    assert len(results) == 1
    batch = results[0]
    assert len(batch.mapping) == 1
    assert tuple(batch.policy_logits.shape) == (1, action_size)
    assert tuple(batch.values.shape) in {(1, 1), (1,)}


def test_inflight_limit_backpressure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inflight cap should defer dispatch to the queue."""
    _patch_fake_ray(monkeypatch)

    class _MockClient:
        def infer_async(self, batch: torch.Tensor, native_payload=None) -> FakeRef:
            return FakeRef((torch.zeros(1, 1), torch.zeros(1, 1)))

    manager = bim_mod.BatchInferenceManager(
        client=_MockClient(),
        batch_size=1,
        max_wait_ms=0,
        max_inflight_batches=1,
    )

    node = bim_mod.PendingNodeInfo(node=None, is_start_node=True)
    tensor = torch.zeros(1, 2, 2)

    manager.enqueue(node, tensor)
    assert len(manager._inflight_refs) == 1
    assert len(manager._queue) == 0

    manager.enqueue(node, tensor)
    assert len(manager._inflight_refs) == 1
    assert len(manager._queue) == 1

    results = manager.drain_ready()
    print(
        f"[debug] backpressure queue_after={len(manager._queue)} "
        f"inflight_after={len(manager._inflight_refs)} results={len(results)}"
    )
    assert len(results) >= 1
    manager.dispatch_ready(force=True)
    assert len(manager._queue) == 0
    assert len(manager._inflight_refs) == 1


def test_inflight_limit_applied_in_enqueue(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enqueue should defer when inflight batches exceed limit."""
    _patch_fake_ray(monkeypatch)

    class _MockClient:
        def infer_async(self, batch: torch.Tensor, native_payload=None) -> FakeRef:
            return FakeRef((torch.zeros(1, 1), torch.zeros(1, 1)))

    manager = bim_mod.BatchInferenceManager(
        client=_MockClient(),
        batch_size=1,
        max_wait_ms=0,
        max_inflight_batches=1,
    )

    node = bim_mod.PendingNodeInfo(node=None, is_start_node=True)
    tensor = torch.zeros(1, 2, 2)

    manager.enqueue(node, tensor)
    assert len(manager._inflight_refs) == 1
    assert len(manager._queue) == 0

    manager.enqueue(node, tensor)
    assert len(manager._inflight_refs) == 1
    assert len(manager._queue) == 1
    print(
        f"[debug] enqueue backpressure inflight={len(manager._inflight_refs)} "
        f"queued={len(manager._queue)}"
    )


def test_ray_async_runner_smoke_generates_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RayAsyncRunner should run loop using mocked actors and return records."""
    from gomoku.alphazero.types import GameRecord

    _patch_fake_ray(monkeypatch)

    dummy_rec = GameRecord(
        states_raw=[None],
        policies=np.zeros((1, 1), dtype=np.float32),
        moves=np.zeros(1, dtype=np.int32),
        players=np.zeros(1, dtype=np.int8),
        outcomes=np.zeros(1, dtype=np.int8),
    )

    class FakeWorkerActor:
        def __init__(self):
            self.update_params = SimpleNamespace(remote=self._update_params_remote)
            self.run_games = SimpleNamespace(remote=self._run_games_remote)

        def _update_params_remote(self, **_kwargs) -> None:
            return None

        def _run_games_remote(
            self,
            batch_size: int,
            target_games: int,
            random_ratio: float = 0.0,
            random_bot_rate: float = 0.0,
            prev_bot_rate: float = 0.0,
            past_opponent_path: str | None = None,
            random_opening_turns: int = 0,
        ) -> list[Any]:
            return [dummy_rec]

    class RemoteCtor:
        @classmethod
        def options(cls, **kwargs):
            return cls

        @classmethod
        def remote(cls, *args, **kwargs) -> FakeWorkerActor:
            return FakeWorkerActor()

    monkeypatch.setattr(ray_runner_mod, "RaySelfPlayWorker", RemoteCtor)
    monkeypatch.setattr(
        ray_runner_mod, "_build_model_fn", lambda cfg, device: lambda: None
    )
    monkeypatch.setattr(ray_runner_mod, "Gomoku", MagicMock())
    monkeypatch.setattr(ray_runner_mod, "PolicyValueNet", MagicMock())
    monkeypatch.setattr(
        ray_runner_mod,
        "RayInferenceActor",
        type(
            "FakeActorClass",
            (),
            {
                "options": classmethod(lambda cls, **opt: cls),
                "remote": lambda *a, **k: SimpleNamespace(),
            },
        ),
    )

    cfg = MagicMock()
    cfg.mcts.batch_infer_size = 1
    cfg.training.temperature = 1.0
    cfg.runtime = SimpleNamespace(inference=None, selfplay=None)
    runner = ray_runner_mod.RayAsyncRunner(cfg=cfg, num_actors=1, num_workers=1)

    records = runner.run(batch_size=1, games=1)
    print(f"[debug] smoke records count={len(records)}")
    assert records and isinstance(records[0], GameRecord)


def test_ray_async_runner_broadcasts_worker_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RayAsyncRunner should push provided configs to workers via update_params."""
    _patch_fake_ray(monkeypatch)

    received = {"train": None, "mcts": None}

    class FakeWorkerActor:
        def __init__(self, *_args, **_kwargs):
            self.update_params = SimpleNamespace(remote=self._update_params_remote)
            self.run_games = SimpleNamespace(remote=self._run_games_remote)

        def _update_params_remote(
            self,
            *,
            training_cfg=None,
            mcts_cfg=None,
            async_inflight_limit=None,
            past_opponent_path=None,
        ) -> None:
            received["train"] = training_cfg
            received["mcts"] = mcts_cfg

        def _run_games_remote(
            self,
            batch_size: int,
            target_games: int,
            random_ratio: float = 0.0,
            random_bot_rate: float = 0.0,
            prev_bot_rate: float = 0.0,
            past_opponent_path: str | None = None,
            random_opening_turns: int = 0,
        ) -> list[Any]:
            return []

    class WorkerCtor:
        @classmethod
        def options(cls, **kwargs):
            return cls

        @classmethod
        def remote(cls, *args, **kwargs) -> FakeWorkerActor:
            return FakeWorkerActor()

    monkeypatch.setattr(ray_runner_mod, "RaySelfPlayWorker", WorkerCtor)
    monkeypatch.setattr(
        ray_runner_mod, "_build_model_fn", lambda cfg, device: lambda: None
    )
    monkeypatch.setattr(ray_runner_mod, "Gomoku", MagicMock())
    monkeypatch.setattr(ray_runner_mod, "PolicyValueNet", MagicMock())
    monkeypatch.setattr(
        ray_runner_mod,
        "RayInferenceActor",
        type(
            "FakeActorClass",
            (),
            {
                "options": classmethod(lambda cls, **opt: cls),
                "remote": lambda *a, **k: SimpleNamespace(
                    set_weights=SimpleNamespace(remote=lambda *_: None)
                ),
            },
        ),
    )

    cfg = MagicMock()
    cfg.mcts.batch_infer_size = 1
    cfg.training.temperature = 1.0
    cfg.runtime = SimpleNamespace(inference=None, selfplay=None)

    train_cfg = MagicMock(name="train_cfg")
    mcts_cfg = MagicMock(name="mcts_cfg")

    runner = ray_runner_mod.RayAsyncRunner(cfg=cfg, num_actors=1, num_workers=1)
    _ = runner.run(
        batch_size=1,
        games=1,
        training_cfg=train_cfg,
        mcts_cfg=mcts_cfg,
        iteration_idx=None,
    )

    print(
        f"[debug] worker training_cfg={received['train']} mcts_cfg={received['mcts']}"
    )
    assert received["train"] is train_cfg
    assert received["mcts"] is mcts_cfg


def test_resolve_iteration_configs_applies_schedule() -> None:
    """_resolve_iteration_configs should return per-iteration scalars."""
    from gomoku.utils.config.schedule_param import SchedulePoint

    train_cfg = ray_runner_mod.TrainingConfig(
        num_iterations=2,
        num_selfplay_iterations=1.0,
        num_epochs=1,
        batch_size=1,
        learning_rate=0.1,
        weight_decay=0.0,
        temperature=[
            SchedulePoint(until=1, value=1.5),
            SchedulePoint(until=2, value=0.5),
        ],
        replay_buffer_size=1,
        min_samples_to_train=1,
    )
    mcts_cfg = ray_runner_mod.MctsConfig(
        C=1.0,
        num_searches=[
            SchedulePoint(until=1, value=100),
            SchedulePoint(until=2, value=10),
        ],
        exploration_turns=1,
        dirichlet_epsilon=[
            SchedulePoint(until=1, value=0.25),
            SchedulePoint(until=2, value=0.05),
        ],
        dirichlet_alpha=0.3,
        batch_infer_size=2,
        max_batch_wait_ms=0,
        min_batch_size=1,
    )
    runner = ray_runner_mod.RayAsyncRunner(
        cfg=MagicMock(training=train_cfg, mcts=mcts_cfg), num_actors=1, num_workers=1
    )

    resolved_train, resolved_mcts = runner._resolve_iteration_configs(
        iteration_idx=1, training_cfg=None, mcts_cfg=None
    )

    print(
        f"[debug] schedule resolved temp={resolved_train.temperature} "
        f"num_searches={resolved_mcts.num_searches} "
        f"dir_eps={resolved_mcts.dirichlet_epsilon}"
    )
    assert pytest.approx(resolved_train.temperature, rel=1e-6) == 0.5
    assert pytest.approx(resolved_mcts.num_searches, rel=1e-6) == 10
    assert pytest.approx(resolved_mcts.dirichlet_epsilon, rel=1e-6) == 0.05


def test_worker_update_params_accepts_mapping_and_rebuilds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RaySelfPlayWorker.update_params should accept mappings and rebuild PVMCTS."""
    created: dict[str, Any] = {}

    class _FakePVMCTS:
        def __init__(self, game, mcts_params, client, mode, async_inflight_limit):
            created["args"] = (game, mcts_params, client, mode, async_inflight_limit)

    monkeypatch.setattr(ray_worker_mod, "PVMCTS", _FakePVMCTS)

    training = ray_worker_mod.TrainingConfig(
        num_iterations=1,
        num_selfplay_iterations=1.0,
        num_epochs=1,
        batch_size=1,
        learning_rate=0.1,
        weight_decay=0.0,
        temperature=1.0,
        replay_buffer_size=8,
        min_samples_to_train=1,
    )
    mcts = ray_worker_mod.MctsConfig(
        C=1.0,
        num_searches=10,
        exploration_turns=1,
        dirichlet_epsilon=0.25,
        dirichlet_alpha=0.3,
        batch_infer_size=2,
        max_batch_wait_ms=0,
        min_batch_size=1,
    )

    class _Cfg:
        def __init__(self, training_cfg, mcts_cfg):
            self.training = training_cfg
            self.mcts = mcts_cfg

        def model_copy(self, update):
            training_cfg = update.get("training", self.training)
            mcts_cfg = update.get("mcts", self.mcts)
            return _Cfg(training_cfg, mcts_cfg)

    def _unwrap_ray_actor(actor_cls):
        meta = getattr(actor_cls, "__ray_metadata__", None)
        if meta is None:
            return actor_cls
        for attr in ("modified_class", "class_ref", "cls", "decorated_class"):
            if hasattr(meta, attr):
                return getattr(meta, attr)
        return actor_cls

    worker_cls = _unwrap_ray_actor(ray_worker_mod.RaySelfPlayWorker)
    worker = worker_cls.__new__(worker_cls)
    worker.cfg = _Cfg(training, mcts)
    worker.game = SimpleNamespace(use_native=False)
    worker.client = SimpleNamespace(max_batch_size=mcts.batch_infer_size)
    worker.runner = SimpleNamespace(train_cfg=training, mcts_cfg=mcts)
    worker.agent = SimpleNamespace(
        mcts_cfg=mcts,
        async_inflight_limit=3,
        reset=lambda: created.setdefault("reset_called", True),
    )

    worker.update_params(
        training_cfg={"temperature": 0.5},
        mcts_cfg={
            "C": 1.0,
            "num_searches": 20,
            "exploration_turns": 1,
            "dirichlet_epsilon": 0.1,
            "dirichlet_alpha": 0.3,
            "batch_infer_size": 4,
            "max_batch_wait_ms": 5,
            "min_batch_size": 2,
        },
        async_inflight_limit=7,
    )

    print(
        f"[debug] worker updated: temp={worker.runner.train_cfg.temperature}, "
        f"batch_size={worker.runner.mcts_cfg.batch_infer_size}, "
        f"async_limit={worker.agent.async_inflight_limit}, "
        f"pvmcts_args={created.get('args')}"
    )
    assert worker.runner.train_cfg.temperature == 0.5
    assert worker.runner.mcts_cfg.batch_infer_size == 4
    assert worker.client.max_batch_size == 4
    assert worker.agent.async_inflight_limit == 7
    assert created.get("args") == (
        worker.game,
        worker.runner.mcts_cfg,
        worker.client,
        "ray",
        7,
    )
    assert created.get("reset_called") is True


def test_broadcast_weights_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Broadcast errors should propagate."""
    _patch_fake_ray(monkeypatch)

    def _boom():
        raise RuntimeError("fail-weight")

    # Keep real runner but inject failing set_weights_fn via monkeypatch
    monkeypatch.setattr(
        ray_runner_mod, "_build_model_fn", lambda cfg, device: lambda: None
    )
    monkeypatch.setattr(ray_runner_mod, "Gomoku", MagicMock())
    monkeypatch.setattr(ray_runner_mod, "PolicyValueNet", MagicMock())
    monkeypatch.setattr(
        ray_runner_mod,
        "RayInferenceActor",
        type(
            "FakeActorClass",
            (),
            {
                "options": classmethod(lambda cls, **opt: cls),
                "remote": lambda *a, **k: SimpleNamespace(),
            },
        ),
    )
    monkeypatch.setattr(
        ray_runner_mod,
        "RaySelfPlayWorker",
        type(
            "FakeWorkerCtor",
            (),
            {
                "options": classmethod(lambda cls, **opt: cls),
                "remote": lambda *a, **k: SimpleNamespace(
                    run_games=SimpleNamespace(remote=lambda b, t: FakeRef([]))
                ),
            },
        ),
    )

    cfg = MagicMock()
    cfg.mcts.batch_infer_size = 1
    cfg.training.temperature = 1.0
    runner = ray_runner_mod.RayAsyncRunner(
        cfg=cfg, num_actors=1, num_workers=1, set_weights_fn=_boom
    )

    with pytest.raises(RuntimeError):
        runner.run(batch_size=1, games=1)


def test_pending_flush_prevents_deadlock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated single requests should flush after timeout."""
    _patch_fake_ray(monkeypatch)

    class _SlowClient:
        def __init__(self) -> None:
            self.calls = 0

        def infer_async(self, batch: torch.Tensor, native_payload=None) -> FakeRef:
            self.calls += 1
            time.sleep(0.001)
            return FakeRef(
                (torch.zeros(batch.size(0), 1), torch.zeros(batch.size(0), 1))
            )

    client = _SlowClient()
    manager = bim_mod.BatchInferenceManager(
        client=client,
        batch_size=4,
        max_wait_ms=1,
        max_inflight_batches=2,
    )

    node = bim_mod.PendingNodeInfo(node=None, is_start_node=True)
    tensor = torch.zeros(1, 2, 2)

    for _ in range(3):
        manager.enqueue(node, tensor)
        time.sleep(0.002)
        _ = manager.drain_ready(timeout_s=0.0)

    print(f"[debug] pending flush calls={client.calls} queue_len={len(manager._queue)}")
    assert client.calls == 3
    assert len(manager._queue) < manager.batch_size


def test_batch_result_mapping_integrity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Batch results should map back to the correct nodes in order."""
    _patch_fake_ray(monkeypatch)
    action_size = 5

    class _EchoClient:
        def infer_async(self, batch: torch.Tensor, native_payload=None) -> FakeRef:
            vals = batch.view(batch.size(0), -1).mean(dim=1, keepdim=True)
            policy = torch.zeros(batch.size(0), action_size, dtype=torch.float32)
            return FakeRef((policy, vals))

    manager = bim_mod.BatchInferenceManager(
        client=_EchoClient(),
        batch_size=2,
        max_wait_ms=0,
    )

    tensor_a = torch.full((1, 2, 2), 10.0)
    tensor_b = torch.full((1, 2, 2), 20.0)

    node_a = bim_mod.PendingNodeInfo(node="NodeA", is_start_node=True)
    node_b = bim_mod.PendingNodeInfo(node="NodeB", is_start_node=False)

    manager.enqueue(node_a, tensor_a)
    manager.enqueue(node_b, tensor_b)

    results = manager.drain_ready(timeout_s=1.0)
    assert len(results) == 1
    batch = results[0]
    print(
        f"[debug] batch mapping order={[m.node for m in batch.mapping]} "
        f"values={[v.item() for v in batch.values]}"
    )

    assert len(batch.mapping) == 2
    assert batch.mapping[0] == node_a
    assert batch.mapping[1] == node_b
    assert pytest.approx(batch.values[0].item()) == 10.0
    assert pytest.approx(batch.values[1].item()) == 20.0


def test_ray_client_distributes_requests_round_robin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requests should distribute round-robin across actors."""
    _patch_fake_ray(monkeypatch)

    class DummyActorHandle:
        def __init__(self) -> None:
            self.count = 0
            self.infer = SimpleNamespace(remote=self._infer_remote)
            self.get_count = SimpleNamespace(remote=self._get_count_remote)

        def _infer_remote(self, inputs, native_payload=None, model_slot=None):
            self.count += 1
            batch = inputs.shape[0] if hasattr(inputs, "shape") else 1
            policy = np.zeros((batch, 1), dtype=np.float32)
            value = np.zeros((batch, 1), dtype=np.float32)
            return FakeRef((policy, value))

        def _get_count_remote(self):
            return FakeRef(self.count)

    actors = [DummyActorHandle() for _ in range(3)]
    client = ray_client_mod.RayInferenceClient(actors, max_batch_size=8)

    state = torch.zeros(3, 2, 2, dtype=torch.float32)
    for _ in range(6):
        client.infer_async(state)

    counts = [actor.get_count.remote().value for actor in actors]
    print(f"[debug] rr counts={counts}")
    assert counts == [2, 2, 2]


def test_manager_cleanup_cancels_inflight_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup should cancel all inflight refs via ray.cancel."""
    cancel_calls = []

    def _cancel_spy(ref, force=True):
        cancel_calls.append((ref, force))
        return True

    def fake_put(obj):
        return FakeRef(obj)

    _patch_fake_ray(monkeypatch)
    monkeypatch.setattr(
        bim_mod,
        "ray",
        SimpleNamespace(
            put=fake_put,
            get=lambda r: r.value,
            wait=lambda r, **k: (r, []),
            cancel=_cancel_spy,
        ),
    )

    class _MockClient:
        def infer_async(self, batch: torch.Tensor, native_payload=None) -> FakeRef:
            return FakeRef((torch.zeros(1, 1), torch.zeros(1, 1)))

    manager = bim_mod.BatchInferenceManager(
        client=_MockClient(),
        batch_size=1,
        max_wait_ms=0,
        max_inflight_batches=2,
    )

    node = bim_mod.PendingNodeInfo(node=None, is_start_node=True)
    tensor = torch.zeros(1, 2, 2)

    manager.enqueue(node, tensor)
    manager.enqueue(node, tensor)
    assert len(manager._inflight_refs) == 2

    manager.cleanup()

    print(
        f"[debug] cleanup cancels={len(cancel_calls)} "
        f"inflight={len(manager._inflight_refs)} queue={len(manager._queue)}"
    )
    assert len(cancel_calls) == 2
    assert len(manager._inflight_refs) == 0
    assert len(manager._queue) == 0


def test_ray_integration_real_init() -> None:
    """Integration-like smoke with fake Ray to catch serialization path issues."""
    _patch_fake_ray(monkeypatch=pytest.MonkeyPatch())

    class FakeActor:
        def __init__(self) -> None:
            self.infer = SimpleNamespace(
                remote=lambda inputs, native_payload=None, model_slot=None: FakeRef(
                    (
                        np.zeros(
                            (inputs.shape[0] if hasattr(inputs, "shape") else 1, 1),
                            dtype=np.float32,
                        ),
                        np.zeros(
                            (inputs.shape[0] if hasattr(inputs, "shape") else 1, 1),
                            dtype=np.float32,
                        ),
                    )
                )
            )

    actor = FakeActor()
    client = ray_client_mod.RayInferenceClient([actor], max_batch_size=2)
    state = torch.zeros(1, 2, 2, dtype=torch.float32)
    policy, value = client.infer(state)
    print(
        f"[debug] integration policy_shape={tuple(policy.shape)} "
        f"value_shape={tuple(value.shape)}"
    )
    assert policy.shape[0] == 1 and value.shape[0] == 1
