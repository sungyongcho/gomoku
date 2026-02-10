from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from gomoku.core.game_config import NUM_LINES
from gomoku.core.gomoku import Gomoku
from gomoku.inference.local import LocalInference
from gomoku.pvmcts.search.sequential import SequentialEngine
from gomoku.pvmcts.search.vectorize import VectorizeEngine
from gomoku.utils.config.loader import BoardConfig, MctsConfig


class DummySequentialModel(nn.Module):
    """Minimal model returning fixed logits/value for sequential engine tests."""

    def __init__(self, logits: torch.Tensor, value: float):
        super().__init__()
        self.logits = logits
        self.value = value

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = x if x.dim() == 4 else x.unsqueeze(0)
        bsz = batch.size(0)
        logits = self.logits.unsqueeze(0).expand(bsz, -1)
        values = torch.full((bsz, 1), self.value, dtype=self.logits.dtype)
        return logits, values


@pytest.fixture
def sequential_engine() -> SequentialEngine:
    """
    Provide a SequentialEngine with deterministic inference for testing.

    The engine is configured with a standard board and specific MCTS parameters
    to verify expansion and noise injection logic.
    """
    board_config = BoardConfig(
        num_lines=NUM_LINES,
        gomoku_goal=5,
        capture_goal=5,
        enable_doublethree=True,
        enable_capture=True,
    )
    game = Gomoku(board_config)

    # Prepare logits matching the actual action size
    # Pattern: High value at index 0, low elsewhere to make behavior predictable.
    logits = torch.ones(game.action_size, dtype=torch.float32)
    logits[0] = 100.0  # Dominant move
    value = 0.5

    mcts_config = MctsConfig(
        C=1.4,
        num_searches=5.0,  # Run 5 simulations
        exploration_turns=1,
        dirichlet_epsilon=0.25,
        dirichlet_alpha=0.3,
        batch_infer_size=1,
        max_batch_wait_ms=0,
        min_batch_size=1,
    )

    model = DummySequentialModel(logits, value)
    inference = LocalInference(model)
    # Expose logits for tests that inspect raw logits (backward compatibility).
    inference.logits = model.logits
    return SequentialEngine(game, mcts_config, inference)


class DummyVectorizeModel(nn.Module):
    """Minimal model returning fixed logits/value for vectorized engine tests."""

    def __init__(self, action_size: int, value: float = 0.0):
        super().__init__()
        self.action_size = action_size
        self.value = float(value)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = x if x.dim() == 4 else x.unsqueeze(0)
        bsz = batch.size(0)
        logits = torch.ones((bsz, self.action_size), dtype=torch.float32)
        values = torch.full((bsz, 1), self.value, dtype=torch.float32)
        return logits, values


@pytest.fixture
def vectorize_engine() -> VectorizeEngine:
    """Provide a VectorizeEngine with deterministic inference for testing."""
    board_config = BoardConfig(
        num_lines=NUM_LINES,
        gomoku_goal=5,
        capture_goal=5,
        enable_doublethree=True,
        enable_capture=True,
    )
    game = Gomoku(board_config)
    mcts_config = MctsConfig(
        C=1.4,
        num_searches=5.0,
        exploration_turns=1,
        dirichlet_epsilon=0.25,
        dirichlet_alpha=0.3,
        batch_infer_size=4,
        max_batch_wait_ms=0,
        min_batch_size=1,
    )
    model = DummyVectorizeModel(action_size=game.action_size, value=0.0)
    inference = LocalInference(model)
    return VectorizeEngine(game, mcts_config, inference)


def make_game_and_params() -> tuple[Gomoku, MctsConfig]:
    """Helper to build a standard Gomoku game and default MCTS params."""
    board_config = BoardConfig(
        num_lines=NUM_LINES,
        gomoku_goal=5,
        capture_goal=5,
        enable_doublethree=True,
        enable_capture=True,
    )
    game = Gomoku(board_config)
    params = MctsConfig(
        C=1.4,
        num_searches=1.0,
        exploration_turns=1,
        dirichlet_epsilon=0.25,
        dirichlet_alpha=0.3,
        batch_infer_size=2,
        max_batch_wait_ms=0,
        min_batch_size=1,
    )
    return game, params


def dummy_mp_inference(action_size: int) -> MagicMock:
    """Return a MagicMock inference client with simple CPU infer."""
    dummy = MagicMock()

    def _infer(batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if batch.dim() == 3:
            batch = batch.unsqueeze(0)
        bsz = batch.size(0)
        logits = torch.zeros((bsz, action_size), dtype=torch.float32)
        values = torch.zeros((bsz, 1), dtype=torch.float32)
        return logits, values

    dummy.infer.side_effect = _infer
    return dummy
