"""Sessions on the serving path: tree reuse per connection and search while waiting."""

import asyncio
import time

from fastapi import WebSocketDisconnect
import numpy as np
import pytest
import torch

from gomoku.core.gomoku import GameState, Gomoku
from gomoku.utils.config.loader import BoardConfig, MctsConfig
from server import websocket as ws
from server.engine import AlphaZeroEngine
from server.session import Ponderer, SearchSession

SIZE = 9
STONE = {0: ".", 1: "X", 2: "O"}


class _CentreInference:
    """Stands in for the network: prefers the centre, optionally slow."""

    device = torch.device("cpu")

    def __init__(self, delay_s: float = 0.0) -> None:
        self.delay_s = delay_s
        self.calls = 0
        centre = SIZE // 2
        self._logits = torch.tensor(
            [
                -1.5 * (abs(i % SIZE - centre) + abs(i // SIZE - centre))
                for i in range(SIZE * SIZE)
            ]
        ).reshape(1, -1)

    def infer(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls += 1
        if self.delay_s:
            time.sleep(self.delay_s)
        return self._logits, torch.tensor([[0.1]])


def _engine(
    searches: int = 80,
    ponder: bool = False,
    delay_s: float = 0.0,
    time_limit_s: float | None = None,
    ponder_max_nodes: int = 5_000,
    ponder_time_limit_s: float = 30.0,
) -> AlphaZeroEngine:
    """An AlphaZeroEngine without a model: only what the sessions use."""
    engine = object.__new__(AlphaZeroEngine)
    board = BoardConfig(
        num_lines=SIZE,
        enable_doublethree=True,
        enable_capture=True,
        capture_goal=5,
        gomoku_goal=5,
        history_length=5,
    )
    engine.game = Gomoku(board, use_native=True)
    assert engine.game.use_native
    engine.inference_client = _CentreInference(delay_s)
    engine.device = torch.device("cpu")
    engine.mcts_config = MctsConfig(
        C=2.0,
        num_searches=searches,
        exploration_turns=0,
        dirichlet_epsilon=0.0,
        dirichlet_alpha=0.0,
        batch_infer_size=1,
        max_batch_wait_ms=0,
        min_batch_size=1,
        use_native=True,
        fpu_from_parent=True,
        fpu_reduction=0.1,
        time_limit_s=time_limit_s,
        reuse_tree=True,
        ponder=ponder,
        ponder_max_nodes=ponder_max_nodes,
        ponder_time_limit_s=ponder_time_limit_s,
    )
    engine.reuse_tree = True
    engine.ponderer = (
        Ponderer(ponder_max_nodes, ponder_time_limit_s) if ponder else None
    )
    return engine


def _request_state(state: GameState, **changes) -> GameState:
    """The state as a request delivers it: plain fields, no native state attached."""
    fields = {
        "board": np.array(state.board, dtype=np.int8),
        "p1_pts": np.int16(state.p1_pts),
        "p2_pts": np.int16(state.p2_pts),
        "next_player": np.int8(state.next_player),
        "last_move_idx": np.int16(state.last_move_idx),
        "empty_count": np.int16(state.empty_count),
        "history": tuple(int(move) for move in state.history),
    }
    fields.update(changes)
    return GameState(**fields)


def _play(engine: AlphaZeroEngine, state: GameState, action: int) -> GameState:
    return engine.apply_move(state, action)[0]


def _opening(engine: AlphaZeroEngine) -> GameState:
    state = engine.game.get_initial_state()
    return _play(engine, _play(engine, state, 40), 41)


def _replies(session: SearchSession) -> dict[int, int]:
    """Searches below each reply at the kept root."""
    return dict(session.native.continue_mcts_encoded(0, lambda *_: None))


def _wait_until(condition, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        time.sleep(0.005)


# --- tree reuse -----------------------------------------------------------------------


def test_first_request_starts_a_new_tree() -> None:
    engine = _engine(searches=80)
    session = engine.open_session()
    action, stats = session.best_move(_request_state(_opening(engine)))
    assert (stats.carried, stats.new, stats.total, stats.pondered) == (0, 80, 80, 0)
    assert 0 <= action < SIZE * SIZE


def test_reply_request_continues_from_the_matching_subtree() -> None:
    engine = _engine(searches=300)
    session = engine.open_session()
    state = _opening(engine)
    action, _ = session.best_move(_request_state(state))
    session.commit_move(action)
    replies = _replies(session)
    reply = max(replies, key=replies.get)
    assert replies[reply] > 0

    after = _play(engine, _play(engine, state, action), reply)
    _, stats = session.best_move(_request_state(after))
    assert stats.carried == replies[reply]
    assert stats.new == 300
    assert stats.total == replies[reply] + 300


def test_same_move_as_a_fresh_search_of_equal_size() -> None:
    """Carried-over searches are the same searches a longer fresh search would run."""
    engine = _engine(searches=150)
    state = _request_state(_opening(engine))
    kept = engine.open_session()
    kept.best_move(state)
    reused_action, stats = kept.best_move(state)  # same position: the whole tree is kept
    assert (stats.carried, stats.new) == (150, 150)
    fresh_action, _ = engine.open_session().best_move(state, num_searches_override=300)
    assert reused_action == fresh_action


def test_engine_playing_both_sides_keeps_the_subtree_of_its_own_move() -> None:
    engine = _engine(searches=200)
    session = engine.open_session()
    state = _opening(engine)
    action, _ = session.best_move(_request_state(state))
    session.commit_move(action)
    kept = session.native.root_visits()
    assert kept > 0
    _, stats = session.best_move(_request_state(_play(engine, state, action)))
    assert stats.carried == kept


@pytest.mark.parametrize("change", ["capture_points", "board", "history"])
def test_unexpected_position_falls_back_to_a_new_tree(change: str) -> None:
    engine = _engine(searches=300)
    session = engine.open_session()
    state = _opening(engine)
    action, _ = session.best_move(_request_state(state))
    session.commit_move(action)
    replies = _replies(session)
    reply = max(replies, key=replies.get)
    after = _play(engine, _play(engine, state, action), reply)

    if change == "capture_points":
        request = _request_state(after, p2_pts=np.int16(2))
    elif change == "board":
        board = np.array(after.board, dtype=np.int8)
        board[0, 0] = 1
        request = _request_state(
            after, board=board, empty_count=np.int16(after.empty_count - 1)
        )
    else:
        request = _request_state(after, history=(int(after.last_move_idx),))

    _, stats = session.best_move(request)
    assert (stats.carried, stats.new) == (0, 300)


def test_reset_forgets_the_tree() -> None:
    engine = _engine(searches=100)
    session = engine.open_session()
    state = _request_state(_opening(engine))
    session.best_move(state)
    session.reset()
    _, stats = session.best_move(state)
    assert stats.carried == 0


def test_spent_time_comes_out_of_the_search_budget() -> None:
    engine = _engine(searches=100_000, delay_s=0.002, time_limit_s=0.3)
    session = engine.open_session()
    state = _request_state(_opening(engine))
    started = time.perf_counter()
    _, full = session.best_move(state, started=started)
    assert 0.25 < time.perf_counter() - started < 0.6

    session.reset()
    started = time.perf_counter()
    _, late = session.best_move(state, started=started - 0.25)  # 0.25 s already gone
    assert time.perf_counter() - started < 0.2
    assert 1 <= late.new < full.new / 2

    # Budget already used up: a new tree still gets the one search a move needs.
    session.reset()
    started = time.perf_counter()
    _, spent = session.best_move(state, started=started - 5.0)
    assert spent.new == 1
    assert time.perf_counter() - started < 0.1


def test_tree_reuse_is_off_without_the_switch() -> None:
    engine = _engine()
    engine.reuse_tree = False
    assert engine.open_session() is None


def test_ponder_needs_tree_reuse_in_the_config() -> None:
    base = _engine().mcts_config.model_dump()
    assert MctsConfig(**{**base, "reuse_tree": False}).ponder is False
    with pytest.raises(ValueError, match="reuse_tree"):
        MctsConfig(**{**base, "reuse_tree": False, "ponder": True})
    with pytest.raises(ValueError, match="positive"):
        MctsConfig(**{**base, "ponder": True, "ponder_max_nodes": 0})


# --- search while waiting -------------------------------------------------------------


def _pondering_session(engine: AlphaZeroEngine) -> tuple[SearchSession, GameState, int]:
    session = engine.open_session()
    state = _opening(engine)
    action, _ = session.best_move(_request_state(state))
    session.commit_move(action)
    return session, state, action


def test_ponder_is_off_by_default() -> None:
    engine = _engine(ponder=False)
    session, _, _ = _pondering_session(engine)
    assert engine.ponderer is None
    calls = engine.inference_client.calls
    time.sleep(0.05)
    assert engine.inference_client.calls == calls


def test_ponder_searches_until_the_next_request_and_is_carried_over() -> None:
    engine = _engine(searches=50, ponder=True, delay_s=0.001)
    session, state, action = _pondering_session(engine)
    assert engine.ponderer.active_session is session
    _wait_until(lambda: engine.inference_client.calls > 250)

    after_own = _play(engine, state, action)
    centre_reply = max(
        (m for m in range(SIZE * SIZE) if after_own.board.reshape(-1)[m] == 0),
        key=lambda m: -(abs(m % SIZE - 4) + abs(m // SIZE - 4)),
    )
    requested = time.perf_counter()
    _, stats = session.best_move(_request_state(_play(engine, after_own, centre_reply)))
    assert engine.ponderer.active_session is None
    assert stats.pondered >= 150
    # The reply the search expected most holds a real share of the pondered searches.
    assert stats.carried > 20
    assert stats.new == 50
    # Stopping costs about one evaluation, not the rest of the ponder budget.
    assert time.perf_counter() - requested < 0.5


def test_stop_does_not_wait_for_the_ponder_limits() -> None:
    engine = _engine(searches=20, ponder=True, delay_s=0.02, ponder_time_limit_s=60.0)
    session, _, _ = _pondering_session(engine)
    _wait_until(lambda: engine.ponderer.active_session is session)
    time.sleep(0.05)
    requested = time.perf_counter()
    engine.stop_pondering()
    assert time.perf_counter() - requested < 0.2
    assert engine.ponderer.active_session is None
    calls = engine.inference_client.calls
    time.sleep(0.1)
    assert engine.inference_client.calls == calls


def test_ponder_stops_at_the_node_cap() -> None:
    engine = _engine(searches=20, ponder=True, ponder_max_nodes=120)
    session, _, _ = _pondering_session(engine)
    _wait_until(lambda: engine.ponderer.active_session is None)
    assert session.node_count == 120
    calls = engine.inference_client.calls
    time.sleep(0.05)
    assert engine.inference_client.calls == calls


def test_ponder_stops_at_the_time_limit() -> None:
    engine = _engine(searches=20, ponder=True, delay_s=0.002, ponder_time_limit_s=0.15)
    session, _, _ = _pondering_session(engine)
    started = time.perf_counter()
    _wait_until(lambda: engine.ponderer.active_session is None)
    assert 0.1 < time.perf_counter() - started < 1.0
    assert session.node_count < 200


def test_closing_the_session_stops_its_ponder() -> None:
    engine = _engine(searches=20, ponder=True, delay_s=0.002)
    session, _, _ = _pondering_session(engine)
    assert engine.ponderer.active_session is session
    session.close()
    assert engine.ponderer.active_session is None
    assert not session.native.has_root()
    calls = engine.inference_client.calls
    time.sleep(0.05)
    assert engine.inference_client.calls == calls


def test_only_one_session_ponders_at_a_time() -> None:
    engine = _engine(searches=20, ponder=True, delay_s=0.002)
    first, _, _ = _pondering_session(engine)
    assert engine.ponderer.active_session is first
    second, _, _ = _pondering_session(engine)  # its request ended the first one's ponder
    assert engine.ponderer.active_session is second
    second.close()
    assert engine.ponderer.active_session is None
    first.close()


def test_node_cap_counts_every_kept_tree() -> None:
    engine = _engine(searches=20, ponder=True, ponder_max_nodes=150)
    first, _, _ = _pondering_session(engine)
    _wait_until(lambda: engine.ponderer.active_session is None)
    assert first.node_count == 150
    second, _, _ = _pondering_session(engine)
    # The first tree fills the cap, so the second connection gets no background search.
    assert engine.ponderer.active_session is None
    assert second.node_count <= 20
    first.close()
    second.commit_move(_most_searched_reply(second))
    assert engine.ponderer.active_session is second
    second.close()


def _most_searched_reply(session: SearchSession) -> int:
    session.native.continue_mcts_encoded(30, session._evaluate)  # noqa: SLF001
    replies = _replies(session)
    return max(replies, key=replies.get)


def test_failed_ponder_rebuilds_the_tree() -> None:
    engine = _engine(searches=30, ponder=True)
    session = engine.open_session()
    state = _opening(engine)
    action, _ = session.best_move(_request_state(state))

    def broken(tensor):
        raise RuntimeError("inference failed")

    healthy = engine.inference_client.infer
    engine.inference_client.infer = broken
    session.commit_move(action)
    _wait_until(lambda: engine.ponderer.active_session is None)
    engine.inference_client.infer = healthy
    _, stats = session.best_move(_request_state(_play(engine, state, action)))
    assert (stats.carried, stats.new) == (0, 30)


# --- through the WebSocket handler ----------------------------------------------------


class _Socket:
    """A connection that delivers the given messages, then disconnects."""

    def __init__(self, engine: AlphaZeroEngine, messages) -> None:
        self.app = type("App", (), {"state": type("State", (), {"engine": engine})()})()
        self.url = type("Url", (), {"path": "/ws"})()
        self._messages = iter(messages)
        self.sent: list[dict] = []

    async def accept(self) -> None:
        pass

    async def receive_json(self) -> dict:
        message = next(self._messages, None)
        if message is None:
            raise WebSocketDisconnect()
        if callable(message):
            message = message(self)
        return message

    async def send_json(self, payload: dict) -> None:
        self.sent.append(payload)


def _move_request(state: GameState, moves: list[int]) -> dict:
    payload = {
        "type": "move",
        "board": [[STONE[int(cell)] for cell in row] for row in state.board],
        "scores": [
            {"player": "X", "score": int(state.p1_pts)},
            {"player": "O", "score": int(state.p2_pts)},
        ],
        "nextPlayer": STONE[int(state.next_player)],
    }
    if moves:
        payload["lastPlay"] = {
            "coordinate": {"x": moves[-1] % SIZE, "y": moves[-1] // SIZE},
            "stone": STONE[3 - int(state.next_player)],
        }
        payload["recentMoves"] = [
            {"x": move % SIZE, "y": move // SIZE} for move in moves[-5:]
        ]
    return payload


def _answer_action(answer: dict) -> int:
    coordinate = answer["lastPlay"]["coordinate"]
    return int(coordinate["x"]) + int(coordinate["y"]) * SIZE


def test_requests_on_one_connection_reuse_the_tree_and_closing_ends_the_ponder() -> None:
    engine = _engine(searches=60, ponder=True, delay_s=0.001)
    state = _opening(engine)
    moves = [40, 41]
    seen = {}

    def second_request(socket: _Socket) -> dict:
        # The engine has answered and searches on; play the reply it expects most.
        assert engine.ponderer.active_session is not None
        _wait_until(lambda: engine.inference_client.calls > 200)
        own = _answer_action(socket.sent[-1])
        after_own = _play(engine, state, own)
        reply = max(
            (m for m in range(SIZE * SIZE) if after_own.board.reshape(-1)[m] == 0),
            key=lambda m: -(abs(m % SIZE - 4) + abs(m // SIZE - 4)),
        )
        seen["after"] = _play(engine, after_own, reply)
        return _move_request(seen["after"], [*moves, own, reply])

    def before_close(socket: _Socket) -> dict:
        seen["pondering_before_close"] = engine.ponderer.active_session is not None
        return {"type": "ping"}

    socket = _Socket(engine, [_move_request(state, moves), second_request, before_close])
    asyncio.run(ws._serve(socket))  # noqa: SLF001

    first, second, pong = socket.sent
    assert first["search"] == {"carried": 0, "new": 60, "total": 60, "pondered": 0}
    assert second["search"]["carried"] > 0
    assert second["search"]["pondered"] > 100
    assert second["search"]["total"] == second["search"]["carried"] + 60
    assert pong == {"type": "pong"}
    assert seen["pondering_before_close"]
    # The connection is gone: nothing searches any more.
    assert engine.ponderer.active_session is None
    calls = engine.inference_client.calls
    time.sleep(0.05)
    assert engine.inference_client.calls == calls


def test_reset_message_stops_the_ponder_and_drops_the_tree() -> None:
    engine = _engine(searches=40, ponder=True, delay_s=0.001)
    state = _opening(engine)
    seen = {}

    def after_reset(socket: _Socket) -> dict:
        seen["pondering_after_reset"] = engine.ponderer.active_session is not None
        return _move_request(state, [40, 41])

    socket = _Socket(
        engine, [_move_request(state, [40, 41]), {"type": "reset"}, after_reset]
    )
    asyncio.run(ws._serve(socket))  # noqa: SLF001
    assert seen["pondering_after_reset"] is False
    assert socket.sent[1]["search"]["carried"] == 0


def test_answers_carry_no_search_field_without_tree_reuse() -> None:
    engine = _engine(searches=20)
    engine.reuse_tree = False
    state = _opening(engine)
    socket = _Socket(engine, [_move_request(state, [40, 41])])
    asyncio.run(ws._serve(socket))  # noqa: SLF001
    assert socket.sent[0]["type"] == "move"
    assert "search" not in socket.sent[0]
