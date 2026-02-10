from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("ray")

from gomoku.pvmcts.search.ray.batch_inference_manager import (  # noqa: E402
    BatchResult,
    PendingNodeInfo,
)
from gomoku.pvmcts.search.ray.ray_async import RayAsyncEngine  # noqa: E402
from gomoku.pvmcts.treenode import TreeNode  # noqa: E402
from tests.pvmcts.search.conftest import make_game_and_params  # noqa: E402


def _dummy_manager(action_size: int):
    """Return a minimal manager stub that satisfies RayAsyncEngine calls."""

    class DummyManager:
        def __init__(self):
            self.cleaned = False
            self._queue: list[PendingNodeInfo] = []

        def pending_count(self) -> int:
            return len(self._queue)

        def enqueue(self, mapping: PendingNodeInfo, tensor: torch.Tensor) -> None:  # noqa: ARG002
            self._queue.append(mapping)

        def dispatch_ready(self, *, force: bool = False) -> bool:  # noqa: ARG002
            # No-op dispatch: drain_ready will emit immediately.
            return False

        def drain_ready(self, timeout_s: float = 0.0) -> list[BatchResult]:  # noqa: ARG002
            if not self._queue:
                return []
            mapping = self._queue[:]
            self._queue.clear()
            logits = torch.zeros((len(mapping), action_size), dtype=torch.float32)
            values = torch.zeros((len(mapping), 1), dtype=torch.float32)
            return [BatchResult(mapping=mapping, policy_logits=logits, values=values)]

        def cleanup(self) -> None:
            self.cleaned = True

    return DummyManager()


def test_ensure_root_noise_only_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """_ensure_root_noise는 add_noise=True일 때만 호출된다."""
    game, params = make_game_and_params()
    params = params.model_copy(update={"num_searches": 0.0, "dirichlet_epsilon": 0.25})
    engine = RayAsyncEngine(
        game,
        params,
        MagicMock(),
        batch_size=1,
        min_batch_size=1,
        max_wait_ms=0,
        async_inflight_limit=1,
    )
    engine.manager = _dummy_manager(game.action_size)

    root = TreeNode(game.get_initial_state())
    root.children[(0, 0)] = TreeNode(
        state=game.get_initial_state(), parent=root, action_taken=(0, 0), prior=1.0
    )

    calls = {"count": 0}

    def fake_noise(_root: TreeNode) -> None:
        calls["count"] += 1
        _root.noise_applied = True

    monkeypatch.setattr(engine, "_ensure_root_noise", fake_noise)

    engine.search([root], add_noise=True)
    assert calls["count"] == 1

    calls["count"] = 0
    root.noise_applied = False
    engine.search([root], add_noise=False)
    assert calls["count"] == 0


def test_cleanup_on_select_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Selection 단계 예외 시 cleanup이 호출된다."""
    game, params = make_game_and_params()
    params = params.model_copy(update={"num_searches": 1.0, "dirichlet_epsilon": 0.0})
    engine = RayAsyncEngine(
        game,
        params,
        MagicMock(),
        batch_size=1,
        min_batch_size=1,
        max_wait_ms=0,
        async_inflight_limit=1,
    )
    manager = _dummy_manager(game.action_size)
    engine.manager = manager

    root = TreeNode(game.get_initial_state())

    def boom(_root: TreeNode):  # noqa: ARG001
        raise RuntimeError("select failed")

    monkeypatch.setattr(engine, "_select_path", boom)

    with pytest.raises(RuntimeError):
        engine.search([root], add_noise=False)

    assert manager.cleaned is True


def test_pending_visits_rolled_back_on_expand_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expand 실패 시 pending_visits와 큐가 정리된다."""
    game, params = make_game_and_params()
    params = params.model_copy(update={"num_searches": 1.0, "dirichlet_epsilon": 0.0})
    engine = RayAsyncEngine(
        game,
        params,
        MagicMock(),
        batch_size=1,
        min_batch_size=1,
        max_wait_ms=0,
        async_inflight_limit=1,
    )

    class OneShotManager:
        def __init__(self) -> None:
            self.cleaned = False
            self._emitted = False
            self._mappings: list[PendingNodeInfo] = []

        def pending_count(self) -> int:
            return len(self._mappings)

        def enqueue(
            self,
            mapping: PendingNodeInfo,
            tensor: torch.Tensor,
            native_state: object | None = None,  # noqa: ARG002
        ) -> None:
            self._mappings.append(mapping)

        def dispatch_ready(self, *, force: bool = False) -> bool:  # noqa: ARG002
            return False

        def drain_ready(self, timeout_s: float = 0.0) -> list[BatchResult]:  # noqa: ARG002
            if self._emitted or not self._mappings:
                return []
            self._emitted = True
            mapping = self._mappings[:]
            self._mappings.clear()
            logits = torch.zeros((len(mapping), game.action_size), dtype=torch.float32)
            values = torch.zeros((len(mapping), 1), dtype=torch.float32)
            return [BatchResult(mapping=mapping, policy_logits=logits, values=values)]

        def cleanup(self) -> None:
            self.cleaned = True

    manager = OneShotManager()
    engine.manager = manager

    root = TreeNode(game.get_initial_state())

    def fake_select_path(_root: TreeNode):
        return _root, [_root]

    def fake_eval(node: TreeNode):
        return 0.0, False

    monkeypatch.setattr(engine, "_select_path", fake_select_path)
    monkeypatch.setattr(engine, "_evaluate_terminal", fake_eval)

    def boom(*args, **kwargs):  # noqa: ARG002
        raise RuntimeError("expand failed")

    monkeypatch.setattr(engine, "_expand_and_backup", boom)

    with pytest.raises(RuntimeError):
        engine.search([root], add_noise=False)

    assert root.pending_visits == 0
    assert manager.cleaned is True
