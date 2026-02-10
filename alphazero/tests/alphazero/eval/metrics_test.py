from __future__ import annotations

import numpy as np
import pytest

from gomoku.alphazero.eval.metrics import H2HMetrics


def test_metrics_summary_rates_and_blunders() -> None:
    """H2HMetrics should compute win/loss/draw and blunder rates correctly."""
    m = H2HMetrics()
    m.record_game(winner=1, blunder_count=2, move_count=10)
    m.record_game(winner=-1, blunder_count=1, move_count=8)
    m.record_game(winner=0, blunder_count=0, move_count=6)

    summary = m.summary()
    assert summary["games"] == 3
    assert summary["win_rate"] == 1 / 3
    assert summary["loss_rate"] == 1 / 3
    assert summary["draw_rate"] == 1 / 3
    assert summary["blunder_rate"] == pytest.approx(3 / 24, rel=1e-6)
    assert summary["avg_length"] == pytest.approx(np.mean([10, 8, 6]), rel=1e-6)
