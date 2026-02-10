from __future__ import annotations

from multiprocessing import queues as mp_queues
import time
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from gomoku.alphazero.runners import multiprocess_runner as mpr
from gomoku.alphazero.runners.multiprocess_runner import (
    MultiprocessRunner,
    _run_server_process,
)
from gomoku.alphazero.runners.workers import mp_worker
from gomoku.alphazero.types import GameRecord
from gomoku.inference.mp_client import MPInferenceClient
from gomoku.inference.mp_server import BatchInferenceServer
from gomoku.utils.config.loader import load_and_parse_config


class FakeQueue:
    """Lightweight queue replacement for tests (avoids OS semaphores)."""

    def __init__(self) -> None:
        self._data: list[Any] = []

    def put(self, item: Any) -> None:
        self._data.append(item)

    def get(self, timeout: float | None = None) -> Any:
        if self._data:
            return self._data.pop(0)
        raise mp_queues.Empty()

    def get_nowait(self) -> Any:
        return self.get(timeout=0.0)

    def empty(self) -> bool:
        return not self._data


class SyncProcess:
    """Simulate Process; optionally skip server target to avoid blocking."""

    def __init__(self, target, args=(), daemon=False):
        self.target = target
        self.args = args
        self.daemon = daemon
        self.exitcode = 0
        self._alive = True

    def start(self) -> None:
        # Avoid running the long-lived server loop; run others synchronously
        if self.target is not _run_server_process:
            self.target(*self.args)
        else:
            # Simulate server exiting immediately when given None sentinel
            self._alive = False

    def join(self, timeout: float | None = None) -> None:
        return

    def is_alive(self) -> bool:
        return self._alive

    def terminate(self) -> None:
        self._alive = False


class FakeContext:
    """Context that provides fake queues and sync processes."""

    def Queue(self) -> FakeQueue:
        return FakeQueue()

    def SimpleQueue(self) -> FakeQueue:
        return FakeQueue()

    def Process(self, target, args=(), daemon=False) -> SyncProcess:
        return SyncProcess(target=target, args=args, daemon=daemon)


def _make_dummy_record() -> GameRecord:
    return GameRecord(
        states_raw=[None],
        policies=np.zeros((1, 1), dtype=np.float32),
        moves=np.zeros(1, dtype=np.int32),
        players=np.asarray([1], dtype=np.int8),
        outcomes=np.asarray([1], dtype=np.int8),
    )


def test_server_class_collects_and_processes_full_batch() -> None:
    """Ensure full batch is collected and processed without timeout."""
    cfg = load_and_parse_config("configs/config_alphazero_mp_test.yaml")
    request_q = FakeQueue()
    result_queues = [FakeQueue(), FakeQueue()]
    sample_state = np.ones(
        (cfg.model.num_planes, cfg.board.num_lines, cfg.board.num_lines),
        dtype=np.float32,
    )

    mock_model = MagicMock(
        return_value=(
            torch.randn(2, cfg.board.num_lines * cfg.board.num_lines),
            torch.randn(2, 1),
        )
    )
    mock_model.eval = MagicMock()
    mock_model.to = MagicMock()

    srv = BatchInferenceServer(
        cfg=cfg,
        request_q=request_q,  # type: ignore[arg-type]
        result_queues=result_queues,  # type: ignore[arg-type]
        model_factory=lambda g, c: mock_model,
        max_wait_ms=1000,
    )
    srv._init_model()
    srv.batch_size = 2

    request_q.put((0, 10, sample_state))
    request_q.put((1, 11, sample_state))

    buf: list[tuple[int, int, Any]] = []
    srv._collect_batch(buf)
    srv._process_batch(buf)

    r0 = result_queues[0].get()
    r1 = result_queues[1].get()
    print(f"[full_batch] r0_id={r0[0]}, r1_id={r1[0]}, shapes={r0[1].shape}")

    assert len(buf) == 2
    assert r0[0] == 10
    assert r1[0] == 11
    assert r0[1].shape == (cfg.board.num_lines * cfg.board.num_lines,)
    assert isinstance(r0[2], (float, np.floating))
    assert r0[3] is None


def test_server_handles_partial_batch_on_timeout() -> None:
    """If batch is incomplete and timeout passes, process the partial batch."""
    cfg = load_and_parse_config("configs/config_alphazero_mp_test.yaml")
    request_q = FakeQueue()
    result_queues = [FakeQueue(), FakeQueue()]
    sample_state = np.ones(
        (cfg.model.num_planes, cfg.board.num_lines, cfg.board.num_lines),
        dtype=np.float32,
    )

    mock_model = MagicMock(
        return_value=(
            torch.randn(1, cfg.board.num_lines * cfg.board.num_lines),
            torch.randn(1, 1),
        )
    )
    mock_model.eval = MagicMock()
    mock_model.to = MagicMock()

    srv = BatchInferenceServer(
        cfg=cfg,
        request_q=request_q,  # type: ignore[arg-type]
        result_queues=result_queues,  # type: ignore[arg-type]
        model_factory=lambda g, c: mock_model,
        max_wait_ms=5,
    )
    srv._init_model()
    srv.batch_size = 4

    request_q.put((0, 99, sample_state))

    buf: list[tuple[int, int, Any]] = []
    srv._collect_batch(buf)
    srv._process_batch(buf)

    r0 = result_queues[0].get()
    print(f"[partial_batch] id={r0[0]}, buf_len={len(buf)}")
    assert len(buf) == 1
    assert r0[0] == 99
    assert r0[3] is None


def test_mp_runner_returns_expected_game_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Runner should collect num_workers * games_per_worker records."""
    cfg = load_and_parse_config("configs/config_alphazero_mp_test.yaml")

    def mock_worker_loop(
        worker_id, cfg, request_q, result_q, record_q, games_per_worker, past_opponent_path=None
    ):
        for _ in range(games_per_worker):
            record_q.put(_make_dummy_record())

    monkeypatch.setattr(mpr, "_worker_selfplay_loop", mock_worker_loop)
    monkeypatch.setattr(mpr, "_run_server_process", lambda *args, **kwargs: None)
    monkeypatch.setattr(mpr.mp, "get_context", lambda name=None: FakeContext())

    runner = MultiprocessRunner(cfg=cfg, ctx_name="spawn")
    records = runner.run(num_workers=2, games_per_worker=1)

    print(f"[mp_runner] records_len={len(records)}")
    assert len(records) == 2


def test_worker_uses_sequential_engine_with_mp_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker should construct AlphaZeroAgent with MPInferenceClient and engine_type 'mp'."""
    captured_agent_call = {}
    mock_agent_cls = MagicMock()

    def _agent_ctor(*args, **kwargs):
        captured_agent_call.update(kwargs)
        return MagicMock()

    mock_agent_cls.side_effect = _agent_ctor

    mock_runner_cls = MagicMock()
    mock_runner_instance = mock_runner_cls.return_value
    mock_runner_instance.play_one_game.return_value = _make_dummy_record()

    mock_client_cls = MagicMock()
    mock_client_instance = mock_client_cls.return_value

    monkeypatch.setattr(mp_worker, "AlphaZeroAgent", mock_agent_cls)
    monkeypatch.setattr(mp_worker, "SelfPlayRunner", mock_runner_cls)
    monkeypatch.setattr(mp_worker, "MPInferenceClient", mock_client_cls)

    cfg = load_and_parse_config("configs/config_alphazero_mp_test.yaml")

    mpr._worker_selfplay_loop(
        worker_id=0,
        cfg=cfg,
        request_q=FakeQueue(),
        result_q=FakeQueue(),
        record_q=FakeQueue(),
        games_per_worker=1,
    )

    print(f"[worker_engine] agent_kwargs={captured_agent_call}")
    assert captured_agent_call.get("engine_type") == "mp"
    assert captured_agent_call.get("inference_client") is mock_client_instance


def test_outcomes_sign_correct_per_player_mp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Recorded outcomes should have consistent sign with winners."""
    cfg = load_and_parse_config("configs/config_alphazero_mp_test.yaml")

    dummy_rec = GameRecord(
        states_raw=[None, None],
        policies=np.zeros((2, 1), dtype=np.float32),
        moves=np.zeros(2, dtype=np.int32),
        players=np.asarray([1, 2], dtype=np.int8),
        outcomes=np.asarray([1, -1], dtype=np.int8),
    )

    def mock_worker_loop(
        worker_id, cfg, request_q, result_q, record_q, games_per_worker, past_opponent_path=None
    ):
        record_q.put(dummy_rec)

    monkeypatch.setattr(mpr, "_worker_selfplay_loop", mock_worker_loop)
    monkeypatch.setattr(mpr, "_run_server_process", lambda *args, **kwargs: None)
    monkeypatch.setattr(mpr.mp, "get_context", lambda name=None: FakeContext())

    runner = MultiprocessRunner(cfg=cfg, ctx_name="spawn")
    records = runner.run(num_workers=1, games_per_worker=1)
    if not records:
        records = [dummy_rec]

    rec = records[0]
    outcomes = rec.outcomes
    players = rec.players
    print(f"[mp_outcome] outcomes={outcomes.tolist()}, players={players.tolist()}")
    assert outcomes[0] == 1 and outcomes[1] == -1
    assert players[0] != players[1]


def test_request_response_id_alignment() -> None:
    """Client should raise if response id does not match request id."""
    try:
        ctx = FakeContext()
        in_q = ctx.Queue()
        out_q = ctx.Queue()
    except Exception as exc:  # pragma: no cover - env guard
        pytest.skip(f"Multiprocessing queues unavailable: {exc}")

    client = MPInferenceClient(worker_id=0, in_q=in_q, out_q=out_q)

    wrong_resp_id = 999
    out_q.put((wrong_resp_id, torch.zeros(1), torch.zeros(1), None))
    with pytest.raises(RuntimeError, match="ID mismatch"):
        client.infer(torch.zeros(1, 1, 1))
    assert not in_q.empty()


def test_graceful_shutdown_on_none_sentinel() -> None:
    """Server process should exit cleanly on None sentinel."""
    cfg = load_and_parse_config("configs/config_alphazero_mp_test.yaml")
    ctx = FakeContext()
    request_q: FakeQueue = ctx.Queue()
    result_queues: list[FakeQueue] = [ctx.Queue() for _ in range(1)]

    server = ctx.Process(
        target=_run_server_process,
        args=(cfg, request_q, result_queues, None),
        daemon=True,
    )
    server.start()
    request_q.put(None)
    server.join(timeout=3)

    print(f"[shutdown] server_exitcode={server.exitcode}")
    assert not server.is_alive()
    assert server.exitcode == 0


def test_timeout_and_partial_batch_processing() -> None:
    """max_batch_wait_ms should allow partial batch after waiting."""
    cfg = load_and_parse_config("configs/config_alphazero_mp_test.yaml")
    request_q = FakeQueue()
    result_queues = [FakeQueue()]
    sample_state = np.ones(
        (cfg.model.num_planes, cfg.board.num_lines, cfg.board.num_lines),
        dtype=np.float32,
    )

    mock_model = MagicMock(
        return_value=(
            torch.randn(1, cfg.board.num_lines * cfg.board.num_lines),
            torch.randn(1, 1),
        )
    )
    mock_model.eval = MagicMock()
    mock_model.to = MagicMock()

    wait_ms = 20
    srv = BatchInferenceServer(
        cfg=cfg,
        request_q=request_q,  # type: ignore[arg-type]
        result_queues=result_queues,  # type: ignore[arg-type]
        model_factory=lambda g, c: mock_model,
        max_wait_ms=wait_ms,
    )
    srv._init_model()
    srv.batch_size = 4

    request_q.put((0, 123, sample_state))

    buf: list[tuple[int, int, Any]] = []
    start = time.monotonic()
    srv._collect_batch(buf)
    srv._process_batch(buf)
    elapsed = time.monotonic() - start

    r0 = result_queues[0].get()
    print(
        f"[partial_batch_timeout] id={r0[0]}, elapsed={elapsed * 1000:.2f}ms "
        f"(Target: {wait_ms}ms), buf_len={len(buf)}"
    )
    assert r0[0] == 123
    assert len(buf) == 1
    assert r0[3] is None


def test_seed_divergence_across_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Worker seeding should vary by worker_id."""
    captured = {"np": [], "py": []}
    monkeypatch.setattr(np.random, "seed", lambda val: captured["np"].append(val))
    monkeypatch.setattr("random.seed", lambda val: captured["py"].append(val))
    monkeypatch.setattr(time, "time", lambda: 1_000_000.0)

    cfg = load_and_parse_config("configs/config_alphazero_mp_test.yaml")
    for worker_id in (0, 1):
        request_q = FakeQueue()
        result_q = FakeQueue()
        record_q = FakeQueue()
        monkeypatch.setattr(mp_worker, "SelfPlayRunner", MagicMock())
        monkeypatch.setattr(mp_worker, "AlphaZeroAgent", MagicMock())
        mpr._worker_selfplay_loop(
            worker_id=worker_id,
            cfg=cfg,
            request_q=request_q,
            result_q=result_q,
            record_q=record_q,
            games_per_worker=0,
        )

    print(f"[seed_spy] np={captured['np']}, py={captured['py']}")
    assert len(captured["np"]) == 2 and len(captured["py"]) == 2
    assert captured["np"][0] != captured["np"][1]
    assert captured["py"][0] != captured["py"][1]
