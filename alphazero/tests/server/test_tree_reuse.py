"""Keeping the search tree between moves: subtree matching, fallback and the limits."""

import threading
import time

import pytest

from gomoku.cpp_ext import gomoku_cpp

SIZE = 9


def _core(capture: bool = True):
    return gomoku_cpp.GomokuCore(SIZE, True, capture, 5 if capture else 0, 5, 5)


def _evaluator(core, delay_s: float = 0.0):
    """Prefer moves near the centre, so the searches concentrate on a few replies."""
    size = core.board_size
    centre = size // 2
    logits = [
        -1.5 * (abs(i % size - centre) + abs(i // size - centre))
        for i in range(core.action_size)
    ]

    def evaluate(features, legal):
        if delay_s:
            time.sleep(delay_s)
        return logits, 0.1

    return evaluate


def _play(core, state, move: int):
    return core.apply_move(state, move % SIZE, move // SIZE, state.next_player)


def _most_searched(visits) -> int:
    return max(visits, key=lambda move: move[1])[0]


def _searched_engine(searches: int = 200):
    """An engine whose tree was searched from a position with two stones."""
    core = _core()
    engine = gomoku_cpp.MctsEngine(core, 2.0, True, 0.1)
    state = _play(core, _play(core, core.initial_state(), 40), 41)
    engine.reset_root(state)
    visits = engine.continue_mcts_encoded(searches, _evaluator(core))
    return core, engine, state, visits


def test_same_position_keeps_every_search() -> None:
    core, engine, state, _ = _searched_engine(200)
    assert engine.align_root(state) == 200
    assert engine.root_visits() == 200


def test_reply_position_continues_from_its_subtree() -> None:
    core, engine, state, visits = _searched_engine(300)
    own = _most_searched(visits)
    assert engine.advance_root(own)
    replies = dict(engine.continue_mcts_encoded(0, _evaluator(core)))
    reply = max(replies, key=replies.get)
    assert replies[reply] > 0

    after_reply = _play(core, _play(core, state, own), reply)
    assert engine.align_root(after_reply) == replies[reply]
    assert engine.root_visits() == replies[reply]

    engine.continue_mcts_encoded(50, _evaluator(core))
    assert engine.last_simulations() == 50
    assert engine.root_visits() == replies[reply] + 50


def test_unsearched_reply_starts_empty_but_matches() -> None:
    core, engine, state, visits = _searched_engine(60)
    own = _most_searched(visits)
    engine.advance_root(own)
    replies = dict(engine.continue_mcts_encoded(0, _evaluator(core)))
    reply = next(move for move, count in replies.items() if count == 0)
    assert engine.align_root(_play(core, _play(core, state, own), reply)) == 0
    assert engine.node_count() == 1


def test_continuing_equals_one_longer_search() -> None:
    core, engine, state, _ = _searched_engine(120)
    continued = engine.continue_mcts_encoded(180, _evaluator(core))
    fresh = gomoku_cpp.MctsEngine(core, 2.0, True, 0.1)
    assert continued == fresh.run_mcts_encoded(state, 300, _evaluator(core))


def test_subtree_matches_the_search_it_came_from() -> None:
    """Advancing keeps exactly the nodes below the move and nothing else."""
    core, engine, state, visits = _searched_engine(300)
    own = _most_searched(visits)
    before = dict(visits)[own]
    engine.advance_root(own)
    assert engine.root_visits() == before
    # Every search through a node adds at most one node below it.
    assert 1 <= engine.node_count() <= before


@pytest.mark.parametrize(
    "change",
    ["capture_points", "extra_stone", "history", "side_to_move", "two_moves_on"],
)
def test_other_positions_do_not_match(change: str) -> None:
    core, engine, state, visits = _searched_engine(300)
    own = _most_searched(visits)
    engine.advance_root(own)
    replies = dict(engine.continue_mcts_encoded(0, _evaluator(core)))
    reply = max(replies, key=replies.get)
    target = _play(core, _play(core, state, own), reply)
    kept_visits, kept_nodes = engine.root_visits(), engine.node_count()

    if change == "capture_points":
        target.p1_pts = target.p1_pts + 2
    elif change == "extra_stone":
        board = list(target.board)
        board[0] = 1
        target.board = board
        target.empty_count = target.empty_count - 1
    elif change == "history":
        target.history = list(target.history)[:1]
    elif change == "side_to_move":
        target.next_player = 3 - target.next_player
    else:
        target = _play(core, target, 0)

    assert engine.align_root(target) == -1
    # The kept tree is untouched; the caller decides to start a new one.
    assert engine.root_visits() == kept_visits
    assert engine.node_count() == kept_nodes


def test_last_move_alone_is_not_enough() -> None:
    """Same last move and stones on a different game: the full comparison rejects it."""
    core, engine, state, visits = _searched_engine(200)
    own = _most_searched(visits)
    engine.advance_root(own)
    other = _play(core, _play(core, core.initial_state(), 0), 80)
    other = _play(core, other, own)
    assert other.last_move_idx == own
    assert engine.align_root(other) == -1


def test_align_without_a_tree_reports_no_match() -> None:
    core = _core()
    engine = gomoku_cpp.MctsEngine(core, 2.0)
    assert not engine.has_root()
    assert engine.align_root(core.initial_state()) == -1
    assert not engine.advance_root(40)
    with pytest.raises(Exception, match="ResetRoot"):
        engine.continue_mcts_encoded(1, _evaluator(core))


def test_advance_rejects_a_move_the_root_does_not_have() -> None:
    core, engine, state, _ = _searched_engine(50)
    assert not engine.advance_root(40)  # occupied
    assert engine.root_visits() == 50


def test_reset_and_clear_drop_the_tree() -> None:
    core, engine, state, _ = _searched_engine(100)
    assert engine.node_count() > 1
    engine.reset_root(state)
    assert (engine.root_visits(), engine.node_count()) == (0, 1)
    engine.clear_tree()
    assert (engine.has_root(), engine.node_count()) == (False, 0)


def test_node_cap_ends_the_search() -> None:
    core = _core()
    engine = gomoku_cpp.MctsEngine(core, 2.0, True, 0.1)
    engine.reset_root(core.initial_state())
    engine.continue_mcts_encoded(10_000, _evaluator(core), max_nodes=64)
    assert engine.node_count() == 64
    assert engine.last_simulations() < 10_000
    # At the cap already: nothing more is searched.
    engine.continue_mcts_encoded(10_000, _evaluator(core), max_nodes=64)
    assert engine.last_simulations() == 0


def test_time_limit_applies_at_once_to_a_searched_root() -> None:
    core, engine, state, _ = _searched_engine(50)
    engine.continue_mcts_encoded(10_000, _evaluator(core, 0.01), time_limit_s=0.001)
    assert engine.last_simulations() <= 1
    # A root that was never searched still gets its one search.
    engine.reset_root(state)
    visits = engine.continue_mcts_encoded(
        10_000, _evaluator(core, 0.01), time_limit_s=0.001
    )
    assert engine.last_simulations() == 1
    assert visits


def test_stop_request_ends_a_stoppable_search_quickly() -> None:
    core = _core()
    engine = gomoku_cpp.MctsEngine(core, 2.0, True, 0.1)
    engine.reset_root(core.initial_state())
    thread = threading.Thread(
        target=engine.continue_mcts_encoded,
        args=(1_000_000, _evaluator(core, 0.002)),
        kwargs={"stoppable": True},
    )
    thread.start()
    time.sleep(0.1)
    requested = time.perf_counter()
    engine.request_stop()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert time.perf_counter() - requested < 0.5
    assert 1 <= engine.last_simulations() < 1_000_000
    assert engine.root_visits() == engine.last_simulations()


def test_stop_requested_before_the_search_starts_still_ends_it() -> None:
    core, engine, state, _ = _searched_engine(20)
    engine.request_stop()
    engine.continue_mcts_encoded(1_000, _evaluator(core), stoppable=True)
    assert engine.last_simulations() == 0
    engine.clear_stop()
    engine.continue_mcts_encoded(30, _evaluator(core), stoppable=True)
    assert engine.last_simulations() == 30


def test_timed_search_ignores_a_stale_stop_request() -> None:
    core, engine, state, _ = _searched_engine(20)
    engine.request_stop()
    engine.continue_mcts_encoded(40, _evaluator(core))
    assert engine.last_simulations() == 40


def test_finished_game_at_the_root_is_not_searched() -> None:
    core = _core(capture=False)
    state = core.initial_state()
    for move in (0, 9, 1, 10, 2, 11, 3, 12, 4):  # black completes five on the top row
        state = _play(core, state, move)
    engine = gomoku_cpp.MctsEngine(core, 2.0)
    engine.reset_root(state)
    visits = engine.continue_mcts_encoded(1_000, _evaluator(core))
    assert visits == []
    assert engine.last_simulations() == 1
