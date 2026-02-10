import math
from typing import Literal

from gomoku.utils.config.loader import SPRTConfig


def check_sprt(
    cfg: SPRTConfig, wins: int, losses: int, draws: int
) -> Literal["accept_h1", "accept_h0", "continue"]:
    """Evaluate SPRT to decide whether to stop early."""
    ignore_draws = cfg.ignore_draws

    if ignore_draws:
        n = wins + losses
        wr = wins / n if n > 0 else 0.0
    else:
        n = wins + losses + draws
        wr = (wins + 0.5 * draws) / n if n > 0 else 0.0

    if n <= 0:
        return "continue"

    # Log-likelihood ratio; add epsilon to guard against division by zero
    llr = n * (
        wr * math.log(cfg.p1 / cfg.p0 + 1e-12)
        + (1 - wr) * math.log((1 - cfg.p1) / (1 - cfg.p0) + 1e-12)
    )

    lower = math.log(cfg.beta / (1 - cfg.alpha))
    upper = math.log((1 - cfg.beta) / cfg.alpha)

    if llr > upper:
        return "accept_h1"
    if llr < lower:
        return "accept_h0"
    return "continue"
