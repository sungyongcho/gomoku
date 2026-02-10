import pytest

from gomoku.core.gomoku import Gomoku
from gomoku.utils.config.loader import BoardConfig

try:
    from gomoku.cpp_ext import gomoku_cpp
except ImportError:
    gomoku_cpp = None


def _maybe_cpp_core(py_game: Gomoku):
    if gomoku_cpp is None or not hasattr(gomoku_cpp, "GomokuCore"):
        pytest.skip("C++ GomokuCore binding not available")
    return gomoku_cpp.GomokuCore(
        py_game.row_count,
        py_game.enable_doublethree,
        py_game.enable_capture,
        py_game.capture_goal,
        py_game.gomoku_goal,
        py_game.history_length,
    )


@pytest.fixture
def py_game() -> Gomoku:
    cfg = BoardConfig(
        num_lines=9,
        enable_doublethree=True,
        enable_capture=True,
        capture_goal=5,
        gomoku_goal=5,
        history_length=5,
    )
    return Gomoku(cfg)


@pytest.fixture
def cpp_core(py_game: Gomoku):
    return _maybe_cpp_core(py_game)


@pytest.fixture
def native_game() -> Gomoku:
    if gomoku_cpp is None or not hasattr(gomoku_cpp, "GomokuCore"):
        pytest.skip("C++ GomokuCore binding not available")
    cfg = BoardConfig(
        num_lines=9,
        enable_doublethree=True,
        enable_capture=True,
        capture_goal=5,
        gomoku_goal=5,
        history_length=5,
    )
    game = Gomoku(cfg, use_native=True)
    if not getattr(game, "use_native", False):
        game._native_core = gomoku_cpp.GomokuCore(  # type: ignore[attr-defined]  # noqa: SLF001
            game.row_count,
            game.enable_doublethree,
            game.enable_capture,
            game.capture_goal,
            game.gomoku_goal,
            game.history_length,
        )
        game.use_native = True
    return game
