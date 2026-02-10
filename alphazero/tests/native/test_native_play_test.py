import time

import numpy as np
import torch

from gomoku.core.gomoku import Gomoku
from gomoku.pvmcts.search.sequential import SequentialEngine
from gomoku.pvmcts.treenode import TreeNode
from gomoku.utils.config.loader import BoardConfig, MctsConfig


class _DummyInference:
    """Deterministic inference stub for speed comparison."""

    def __init__(self, action_size: int):
        self.device = torch.device("cpu")
        self.action_size = action_size

    def infer(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if batch.dim() == 3:
            batch = batch.unsqueeze(0)
        bsz = batch.shape[0]
        logits = torch.zeros((bsz, self.action_size), dtype=torch.float32)
        logits[:, 0] = 2.0  # make a stable best move
        values = torch.zeros((bsz, 1), dtype=torch.float32)
        return logits, values


def _make_game(board_size: int, use_native: bool) -> Gomoku:
    cfg = BoardConfig(
        num_lines=board_size,
        enable_doublethree=True,
        enable_capture=True,
        capture_goal=5,
        gomoku_goal=5,
        history_length=5,
    )
    return Gomoku(cfg, use_native=use_native)


def _prep_state(game: Gomoku) -> TreeNode:
    """Create a slightly non-empty state to avoid trivial expansion."""
    state = game.get_initial_state()
    moves = [(4, 4), (4, 5)]
    player = state.next_player
    for move in moves:
        state = game.get_next_state(state, move, player)
        player = 3 - player
    return TreeNode(state=state)


def test_native_vs_python_speed(native_game: Gomoku) -> None:
    """Compare Python vs C++ sequential search speed with print diagnostics."""
    board_size = native_game.row_count
    py_game = _make_game(board_size, use_native=False)

    cfg = MctsConfig(
        C=5.0,
        num_searches=200,
        exploration_turns=1,
        dirichlet_epsilon=0.0,
        dirichlet_alpha=0.3,
        batch_infer_size=1,
        max_batch_wait_ms=0,
        min_batch_size=1,
        use_native=False,
    )

    inference = _DummyInference(action_size=py_game.action_size)

    py_engine = SequentialEngine(py_game, cfg, inference, use_native=False)
    cpp_engine = SequentialEngine(native_game, cfg, inference, use_native=True)

    root_py = _prep_state(py_game)
    root_cpp = _prep_state(native_game)

    print("\n[PlayTest] Running Python engine...")
    start = time.time()
    py_engine.search(root_py, add_noise=False)
    py_time = time.time() - start
    py_visits = root_py.visit_count
    print(f"  Python time: {py_time:.4f}s, visits: {py_visits}")

    print("\n[PlayTest] Running C++ native engine...")
    start = time.time()
    cpp_engine.search(root_cpp, add_noise=False)
    cpp_time = time.time() - start
    cpp_visits = root_cpp.visit_count
    print(f"  C++ time   : {cpp_time:.4f}s, visits: {cpp_visits}")

    if root_py.children:
        py_best = max(root_py.children.items(), key=lambda kv: kv[1].visit_count)[0]
    else:
        py_best = None

    if root_cpp.children:
        cpp_best = max(root_cpp.children.items(), key=lambda kv: kv[1].visit_count)[0]
    else:
        cpp_best = None

    speedup = np.inf if cpp_time == 0 else py_time / cpp_time
    print(f"\n[PlayTest] Speedup (Python/C++): {speedup:.2f}x")
    print(f"[PlayTest] Best move Python: {py_best}, C++: {cpp_best}")

    assert py_visits > 0, "Python engine should visit at least one node."
    assert cpp_visits > 0, "C++ engine should visit at least one node."
