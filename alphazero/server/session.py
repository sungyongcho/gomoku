"""Search trees kept between the moves of a game, and search during the opponent's turn.

A WebSocket connection is one game as far as the server can tell: requests carry the whole
position and no game id. Each connection therefore gets a ``SearchSession`` holding a native
search tree. After the engine answers, the tree's root is moved to the position it played
into; when the next request arrives its position is compared with that root and with the
positions one move below it, and on a match the search goes on from that subtree. Anything
else (a new game, a taken-back move, a position edited by the client) starts a new tree.

With ``mcts.ponder`` the session also searches that root while the opponent thinks. The
``Ponderer`` runs one such search at a time for the whole process and ends it before any
timed search starts, because a timed search needs every core to itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
import time
from typing import TYPE_CHECKING

from gomoku.core.gomoku import GameState
from gomoku.pvmcts.pvmcts import PVMCTS
from gomoku.pvmcts.search.sequential.cpp_strategy import make_encoded_evaluator

if TYPE_CHECKING:
    from server.engine import AlphaZeroEngine

logger = logging.getLogger("uvicorn.error")

# The limits that end a background search are the node cap, the time cap and the stop
# request; this only has to be larger than any of them allows.
_UNBOUNDED_SEARCHES = 2_000_000_000
# A timed search whose budget is already spent still runs with a limit: 0 means "no limit".
_MIN_TIME_LIMIT_S = 0.001


@dataclass(frozen=True)
class SearchStats:
    """Where the searches behind one answer came from."""

    carried: int  # searches already in the reused subtree when the request arrived
    new: int  # searches run for this request
    pondered: int  # searches run while waiting for this request, over all replies

    @property
    def total(self) -> int:
        return self.carried + self.new

    def as_payload(self) -> dict[str, int]:
        return {
            "carried": self.carried,
            "new": self.new,
            "total": self.total,
            "pondered": self.pondered,
        }


class Ponderer:
    """Runs at most one background search and stops it on request.

    Every method is called from the server's event loop thread; only the search itself runs
    on the background thread.
    """

    def __init__(self, max_nodes: int, time_limit_s: float) -> None:
        self.max_nodes = int(max_nodes)
        self.time_limit_s = float(time_limit_s)
        self._sessions: set[SearchSession] = set()
        self._thread: threading.Thread | None = None
        self._owner: SearchSession | None = None

    def register(self, session: SearchSession) -> None:
        self._sessions.add(session)

    def unregister(self, session: SearchSession) -> None:
        self.stop(session)
        self._sessions.discard(session)

    @property
    def active_session(self) -> SearchSession | None:
        """The session being searched in the background right now, if any."""
        if self._thread is not None and self._thread.is_alive():
            return self._owner
        return None

    def start(self, session: SearchSession) -> bool:
        """Search ``session``'s tree in the background; False when the node cap leaves no room."""
        self.stop()
        others = sum(s.node_count for s in self._sessions if s is not session)
        node_limit = self.max_nodes - others
        if session.node_count >= node_limit:
            return False
        session.native.clear_stop()
        thread = threading.Thread(
            target=session.ponder,
            args=(node_limit, self.time_limit_s),
            name="alphazero-ponder",
            daemon=True,
        )
        self._thread, self._owner = thread, session
        thread.start()
        return True

    def stop(self, session: SearchSession | None = None) -> None:
        """End the background search and wait for it; with ``session``, only if it is that one's."""
        thread, owner = self._thread, self._owner
        if thread is None or owner is None:
            return
        if session is not None and owner is not session:
            return
        owner.native.request_stop()
        thread.join()
        self._thread = self._owner = None


class SearchSession:
    """One connection's search tree."""

    def __init__(self, engine: AlphaZeroEngine, ponderer: Ponderer | None) -> None:
        self._engine = engine
        self._ponderer = ponderer
        search = PVMCTS(
            game=engine.game,
            mcts_params=engine.mcts_config,
            inference_client=engine.inference_client,
            mode="sequential",
        ).engine
        self.native = search._native_engine
        if self.native is None:
            raise RuntimeError("Tree reuse needs the native search (mcts.use_native).")
        self._evaluate = make_encoded_evaluator(search)  # no root noise in serving
        self._pondered = 0
        self._ponder_failed = False
        if ponderer is not None:
            ponderer.register(self)

    @property
    def node_count(self) -> int:
        """Nodes in the kept tree; read it only while this session is not being searched."""
        return int(self.native.node_count())

    def best_move(
        self,
        state: GameState,
        started: float | None = None,
        num_searches_override: int | None = None,
    ) -> tuple[int, SearchStats]:
        """Search ``state`` and return the move with the most searches.

        ``started`` is the ``time.perf_counter()`` reading taken when the request arrived; the
        time already spent since then (stopping a background search, matching the tree) comes
        out of the search's time limit, so the answer is not later than without a kept tree.
        """
        if self._ponderer is not None:
            self._ponderer.stop()
        if self._ponder_failed:
            self.native.clear_tree()
            self._ponder_failed = False
        pondered, self._pondered = self._pondered, 0

        cfg = self._engine.mcts_config
        native_state = self._engine._ensure_native_state(state).native_state
        carried = int(self.native.align_root(native_state))
        if carried < 0:
            self.native.reset_root(native_state)
            carried = 0

        searches = int(
            cfg.num_searches if num_searches_override is None else num_searches_override
        )
        time_limit_s = float(cfg.time_limit_s or 0.0)
        if time_limit_s > 0.0 and started is not None:
            spent = time.perf_counter() - started
            time_limit_s = max(time_limit_s - spent, _MIN_TIME_LIMIT_S)
        visits = self.native.continue_mcts_encoded(
            searches, self._evaluate, time_limit_s=time_limit_s
        )
        if not visits:
            raise ValueError("The position has no move to search.")
        priors = dict(self.native.root_priors())
        # Most searched move, ties broken by the network prior: the rule of
        # AlphaZeroEngine._most_visited_action.
        action, _ = max(visits, key=lambda move: (move[1], priors[move[0]]))
        stats = SearchStats(
            carried=carried,
            new=int(self.native.last_simulations()),
            pondered=pondered,
        )
        return int(action), stats

    def commit_move(self, action: int, ponder: bool = True) -> None:
        """Move the tree to the position after the engine's ``action`` and, if on, ponder it."""
        if not self.native.advance_root(int(action)):
            self.native.clear_tree()
            return
        if ponder and self._ponderer is not None:
            self._ponderer.start(self)

    def ponder(self, max_nodes: int, time_limit_s: float) -> None:
        """Body of the background search thread."""
        try:
            self.native.continue_mcts_encoded(
                _UNBOUNDED_SEARCHES,
                self._evaluate,
                time_limit_s=time_limit_s,
                max_nodes=max_nodes,
                stoppable=True,
            )
            self._pondered = int(self.native.last_simulations())
        except Exception:
            logger.exception("Background search failed; the tree will be rebuilt.")
            self._ponder_failed = True

    def reset(self) -> None:
        """Forget the tree, for a new game on the same connection."""
        if self._ponderer is not None:
            self._ponderer.stop(self)
        self.native.clear_tree()
        self._pondered = 0
        self._ponder_failed = False

    def close(self) -> None:
        """Stop searching for this connection and free its tree."""
        if self._ponderer is not None:
            self._ponderer.unregister(self)
        self.native.clear_tree()
