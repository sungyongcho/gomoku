import numpy as np
import pytest


def _require_gomoku_cpp():
    try:
        from gomoku.cpp_ext import gomoku_cpp  # type: ignore
    except Exception:
        pytest.skip("gomoku_cpp module not available")
    return gomoku_cpp


def _center_index(board_size: int) -> int:
    return (board_size // 2) * board_size + (board_size // 2)


def test_run_mcts_single_child():
    gomoku_cpp = _require_gomoku_cpp()

    board_size = 9
    core = gomoku_cpp.GomokuCore(board_size, True, False, 0, 5, 0)
    engine = gomoku_cpp.MctsEngine(core, 2.0)

    action_size = core.action_size
    state = core.initial_state()  # Use GomokuState instead of list
    sims = 3
    center = _center_index(board_size)

    def evaluator(board_state):
        policy = [0.0 for _ in range(action_size)]
        policy[center] = 1.0
        value = 0.25
        board = board_state.board  # Use board property
        print(
            "evaluator called, nonzero moves:",
            [i for i, v in enumerate(board) if v != 0],
        )
        return policy, value

    visits = engine.run_mcts(state, sims, evaluator)
    print("run_mcts visits:", visits)
    # Filter for moves that were actually visited (count > 0)
    visited_results = [v for v in visits if v[1] > 0]
    assert len(visited_results) == 1
    move_idx, count = visited_results[0]
    assert move_idx == center
    # 시뮬레이션 루프에서 루트는 방문하지 않으므로 자식 방문 수는 sims - 1
    assert count == sims - 1


def test_mcts_follows_evaluator_policy():
    """Ensure the engine prioritizes moves suggested by the evaluator (no masking)."""
    gomoku_cpp = _require_gomoku_cpp()

    board_size = 9
    core = gomoku_cpp.GomokuCore(board_size, True, False, 0, 5, 0)
    engine = gomoku_cpp.MctsEngine(core, 2.0)

    action_size = core.action_size
    center = _center_index(board_size)
    center_row, center_col = center // board_size, center % board_size

    # Place a stone at center using apply_move
    state = core.initial_state()
    state = core.apply_move(state, center_row, center_col, 1)

    corner_idx = 0
    near_idx = center + 1

    def evaluator(board_state):
        policy = [0.0] * action_size
        policy[corner_idx] = 0.99
        policy[near_idx] = 0.01
        return policy, 0.0

    visits = engine.run_mcts(state, 10, evaluator)
    print("policy follow visits:", visits)
    visited_moves = {v[0] for v in visits if v[1] > 0}

    # Since masking is not implemented in the native MctsEngine,
    # it should follow the evaluator's high prior and visit the corner.
    assert corner_idx in visited_moves, "Engine should visit high-prior move."


def test_mcts_winning_logic():
    gomoku_cpp = _require_gomoku_cpp()

    board_size = 9
    core = gomoku_cpp.GomokuCore(board_size, True, False, 0, 5, 0)
    engine = gomoku_cpp.MctsEngine(core, 2.0)

    action_size = core.action_size

    # Build state with 4 black stones at (0,0)-(3,0) and 4 white stones
    state = core.initial_state()
    for i in range(4):
        row, col = i // board_size, i % board_size
        state = core.apply_move(state, row, col, 1)
    # Add white stones to balance parity
    for idx in (40, 41, 42, 43):
        row, col = idx // board_size, idx % board_size
        state = core.apply_move(state, row, col, 2)

    winning_move = 4

    def dummy_evaluator(board_state):
        policy = [1.0 / action_size] * action_size
        return policy, 0.0

    visits = engine.run_mcts(state, 50, dummy_evaluator)
    print("winning visits:", visits)
    visit_map = {m: c for m, c in visits}
    assert winning_move in visit_map, "Winning move was not explored."
    assert visit_map[winning_move] > 0, "Winning move has zero visits."


def test_mcts_value_propagation():
    gomoku_cpp = _require_gomoku_cpp()

    board_size = 9
    core = gomoku_cpp.GomokuCore(board_size, True, False, 0, 5, 0)
    engine = gomoku_cpp.MctsEngine(core, 2.0)
    action_size = core.action_size
    state = core.initial_state()

    def pessimistic_evaluator(board_state):
        policy = [1.0 / action_size] * action_size
        return policy, -0.9

    visits = engine.run_mcts(state, 5, pessimistic_evaluator)
    print("value propagation visits:", visits)
    assert len(visits) > 0


def test_run_mcts_encoded_features():
    gomoku_cpp = _require_gomoku_cpp()

    board_size = 9
    core = gomoku_cpp.GomokuCore(board_size, True, False, 0, 5, 0)
    engine = gomoku_cpp.MctsEngine(core, 2.0)

    action_size = core.action_size
    state = core.initial_state()
    center = _center_index(board_size)

    def evaluator(features, legal_moves):
        print(
            "encoded evaluator features shape:",
            features.shape,
            "legal_moves count:",
            len(legal_moves),
        )
        assert features.shape == (8, board_size, board_size)
        assert features.dtype == np.float32
        policy = np.zeros(action_size, dtype=np.float32)
        policy[center] = 1.0
        return policy.tolist(), 0.1

    visits = engine.run_mcts_encoded(state, 3, evaluator)
    print("run_mcts_encoded visits:", visits)
    # Filter for moves that were actually visited
    visited_results = [v for v in visits if v[1] > 0]
    assert len(visited_results) == 1
    move_idx, count = visited_results[0]
    assert move_idx == center
    assert count == 2
