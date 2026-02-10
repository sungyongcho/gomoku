from __future__ import annotations

import pytest

from gomoku.alphazero.eval.sprt import check_sprt
from gomoku.utils.config.loader import SPRTConfig


def test_sprt_accept_h1_when_llr_above_upper() -> None:
    cfg = SPRTConfig(p0=0.45, p1=0.55, alpha=0.05, beta=0.05, max_games=10, ignore_draws=False)
    decision = check_sprt(cfg, wins=100, losses=0, draws=0)
    assert decision == "accept_h1"


def test_sprt_accept_h0_when_llr_below_lower() -> None:
    cfg = SPRTConfig(p0=0.55, p1=0.65, alpha=0.05, beta=0.05, max_games=10, ignore_draws=True)
    decision = check_sprt(cfg, wins=0, losses=100, draws=0)
    assert decision == "accept_h0"


def test_sprt_continue_when_between_bounds() -> None:
    cfg = SPRTConfig(p0=0.45, p1=0.55, alpha=0.05, beta=0.05, max_games=10, ignore_draws=False)
    decision = check_sprt(cfg, wins=1, losses=1, draws=0)
    assert decision == "continue"
