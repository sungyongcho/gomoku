"""The native async search must not flood a slow evaluator with every simulation at once.

A remote inference actor answers a batch only after several polls. Without a cap on in-flight
batches the search dispatches all simulations before the first answer, so virtual loss and the
parent-value FPU spread the visits over almost every legal move.
"""

import numpy as np
import pytest

from gomoku.cpp_ext import gomoku_cpp

SIZE = 9
SIMS = 200
BATCH = 8
POLLS_UNTIL_READY = 20


def _engine():
    core = gomoku_cpp.GomokuCore(SIZE, False, False, 0, 5, 1)
    return core, gomoku_cpp.MctsEngine(core, 2.0, True, 0.1)


def _evaluate(batch_features: np.ndarray, action_size: int):
    """Peaked policy (one strong move, two fair ones) with neutral values."""
    n = len(batch_features)
    logits = np.zeros((n, action_size), dtype=np.float32)
    center = (SIZE // 2) * SIZE + SIZE // 2
    logits[:, center] = 4.0
    logits[:, center + 1] = 2.0
    logits[:, center - 1] = 2.0
    return logits, np.zeros((n, 1), dtype=np.float32)


def _sync_visits() -> dict[int, int]:
    core, engine = _engine()
    visits = engine.run_mcts_batch_encoded(
        core.initial_state(), SIMS, BATCH, lambda f: _evaluate(f, core.action_size)
    )
    return dict(visits)


def _delayed_async_visits(**kwargs) -> tuple[dict[int, int], int]:
    """Run the async path with results that become visible only after several polls."""
    core, engine = _engine()
    pending: dict[int, list] = {}
    peak_inflight = 0

    def dispatcher(batch_features: np.ndarray) -> int:
        nonlocal peak_inflight
        handle = dispatcher.next_handle
        dispatcher.next_handle += 1
        pending[handle] = [POLLS_UNTIL_READY, _evaluate(batch_features, core.action_size)]
        peak_inflight = max(peak_inflight, len(pending))
        return handle

    dispatcher.next_handle = 0

    def checker(handles, timeout_s):
        ready = []
        for handle in handles:
            entry = pending[handle]
            entry[0] -= 1
            if entry[0] <= 0:
                ready.append((handle, pending.pop(handle)[1]))
        return ready

    visits = engine.run_mcts_async_encoded(
        core.initial_state(), SIMS, BATCH, dispatcher, checker, **kwargs
    )
    return dict(visits), peak_inflight


def _width(visits: dict[int, int]) -> int:
    return sum(1 for count in visits.values() if count > 0)


def test_default_async_search_matches_the_sync_batch_search_on_a_slow_evaluator() -> None:
    sync = _sync_visits()
    visits, peak_inflight = _delayed_async_visits()

    # Uncapped, every one of the 81 moves got visits; the sync search visits about 30.
    assert peak_inflight == 1
    assert _width(sync) < 0.5 * SIZE * SIZE
    assert visits == sync


def test_a_larger_inflight_cap_allows_more_batches_in_flight() -> None:
    visits, peak_inflight = _delayed_async_visits(max_inflight_batches=4)
    assert 1 < peak_inflight <= 4
    assert sum(visits.values()) > 0


def test_the_inflight_cap_must_be_positive() -> None:
    with pytest.raises(ValueError):
        _delayed_async_visits(max_inflight_batches=0)
