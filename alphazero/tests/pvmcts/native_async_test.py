"""Native search through the asynchronous inference path used by Ray inference actors."""

import numpy as np
import pytest
import torch

ray = pytest.importorskip("ray")

from gomoku.core.gomoku import Gomoku
from gomoku.pvmcts.pvmcts import PVMCTS
from gomoku.utils.config.loader import BoardConfig, MctsConfig

SIZE = 9


@pytest.fixture(scope="module", autouse=True)
def _ray():
    ray.init(num_cpus=1, include_dashboard=False, ignore_reinit_error=True)
    yield
    ray.shutdown()


class _AsyncInference:
    """Returns uniform bfloat16 outputs, as a CPU inference actor under autocast does."""

    device = torch.device("cpu")

    def __init__(self, broken: bool = False) -> None:
        self.broken = broken

    def infer(self, states, native_payload=None, model_slot=None):
        raise AssertionError("the batched native search must use infer_async")

    def infer_async(self, states, native_payload=None, model_slot=None):
        if self.broken:
            return ray.put("not a (policy, value) pair")
        batch = states.shape[0] if states.dim() == 4 else 1
        policy = torch.zeros(batch, SIZE * SIZE, dtype=torch.bfloat16)
        value = torch.zeros(batch, 1, dtype=torch.bfloat16)
        return ray.put((policy, value))


def _peaked_outputs(batch: int, size: int = SIZE):
    policy = torch.zeros(batch, size * size)
    policy[:, (size // 2) * size + size // 2] = 4.0
    return policy, torch.zeros(batch, 1)


class _SlowAsyncInference(_AsyncInference):
    """Answers arrive long after the search could select every leaf, like a remote actor."""

    def __init__(self) -> None:
        super().__init__()

        # Self-contained local function, so Ray ships it by value instead of importing
        # this test module on the worker.
        def slow_outputs(batch: int, size: int):
            import time

            import torch

            time.sleep(0.05)
            policy = torch.zeros(batch, size * size)
            policy[:, (size // 2) * size + size // 2] = 4.0
            return policy, torch.zeros(batch, 1)

        self._remote = ray.remote(num_cpus=0)(slow_outputs)

    def infer_async(self, states, native_payload=None, model_slot=None):
        return self._remote.remote(states.shape[0], SIZE)


class _PeakedSyncInference:
    """The same outputs through the sync batch search (no infer_async)."""

    device = torch.device("cpu")

    def infer(self, states, native_payload=None, model_slot=None):
        return _peaked_outputs(states.shape[0])


def _search(inference: _AsyncInference, add_noise: bool, **mcts_overrides):
    """Search the empty board once and return the search and its root."""
    board = BoardConfig(
        num_lines=SIZE,
        enable_doublethree=False,
        enable_capture=False,
        capture_goal=5,
        gomoku_goal=5,
        history_length=1,
    )
    game = Gomoku(board, use_native=True)
    settings = {
        "C": 2.0,
        "num_searches": 24,
        "exploration_turns": 0,
        "dirichlet_epsilon": 0.5,
        "dirichlet_alpha": 0.03,
        "batch_infer_size": 8,
        "max_batch_wait_ms": 0,
        "min_batch_size": 1,
        "use_native": True,
        "fpu_from_parent": True,
    }
    params = MctsConfig(**{**settings, **mcts_overrides})
    search = PVMCTS(game, params, inference, mode="sequential")
    np.random.seed(0)
    torch.manual_seed(0)
    root = search.create_root(game.get_initial_state())
    search.run_search([root], add_noise=add_noise)
    return search, root


def _priors(inference: _AsyncInference, add_noise: bool) -> list[float]:
    search, _ = _search(inference, add_noise)
    return [prior for _, prior in search.engine._native_engine.root_priors()]


def _root_visits(inference, **mcts_overrides) -> dict:
    _, root = _search(inference, add_noise=False, num_searches=48, **mcts_overrides)
    return {action: child.visit_count for action, child in root.children.items()}


def _width(visits: dict) -> int:
    return sum(1 for count in visits.values() if count > 0)


def test_async_search_reads_bfloat16_outputs_and_mixes_root_noise() -> None:
    priors = _priors(_AsyncInference(), add_noise=True)
    assert max(priors) > 10 * (1 / len(priors))


def test_async_search_without_noise_keeps_the_uniform_root() -> None:
    priors = _priors(_AsyncInference(), add_noise=False)
    assert max(priors) == pytest.approx(min(priors))


def test_async_inference_errors_stop_the_search_instead_of_hanging() -> None:
    with pytest.raises(Exception):
        _priors(_AsyncInference(broken=True), add_noise=False)


def test_slow_remote_inference_keeps_the_search_concentrated_by_default() -> None:
    # One batch in flight waits for each answer, so the search equals the sync batch search;
    # with many batches in flight it picks leaves blind and visits far more moves.
    sync = _root_visits(_PeakedSyncInference())
    capped = _root_visits(_SlowAsyncInference())
    uncapped = _root_visits(_SlowAsyncInference(), max_inflight_batches=8)
    assert capped == sync
    assert _width(uncapped) > _width(capped) + 10
