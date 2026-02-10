from dataclasses import dataclass, field

import numpy as np


def calculate_elo_from_win_rate(win_rate: float) -> float:
    """Calculate Elo rating difference from win rate.

    Uses the standard logistic Elo formula:
    Elo_diff ≈ 400 * log10(win_rate / (1 - win_rate))

    Parameters
    ----------
    win_rate : float
        Win rate between 0.0 and 1.0

    Returns
    -------
    float
        Elo rating difference relative to opponent (assumed to be at Elo 0)

    Notes
    -----
    - win_rate = 0.5 → Elo = 0 (equal strength)
    - win_rate = 0.55 → Elo ≈ +35
    - win_rate = 0.75 → Elo ≈ +193
    """
    # Clamp to avoid log(0) or division by zero
    win_rate = max(0.01, min(0.99, win_rate))

    if win_rate >= 0.99:
        return 800.0  # Cap at very high Elo for near-perfect win rate
    if win_rate <= 0.01:
        return -800.0  # Cap at very low Elo

    # Standard Elo formula
    return 400.0 * np.log10(win_rate / (1.0 - win_rate))



@dataclass(slots=True)
class H2HMetrics:
    """Container for head-to-head win/loss/draw and blunder statistics."""

    wins: int = 0
    losses: int = 0
    draws: int = 0

    # Blunder-related stats
    blunders: int = 0
    moves: int = 0
    lengths: list[int] = field(default_factory=list)

    def record_game(self, winner: int, blunder_count: int, move_count: int) -> None:
        """Record the outcome of a single game."""
        if winner > 0:
            self.wins += 1
        elif winner < 0:
            self.losses += 1
        else:
            self.draws += 1

        self.blunders += blunder_count
        self.moves += move_count
        self.lengths.append(move_count)

    @property
    def total_games(self) -> int:
        """Total number of games played."""
        return self.wins + self.losses + self.draws

    def summary(self) -> dict[str, float]:
        """Return accumulated statistics as a dictionary."""
        n = self.total_games
        if n == 0:
            return {}

        score_rate = (self.wins + 0.5 * self.draws) / n

        return {
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "games": n,
            "win_rate": self.wins / n,
            "loss_rate": self.losses / n,
            "draw_rate": self.draws / n,
            "score_rate": score_rate,
            "blunder_rate": (self.blunders / self.moves) if self.moves > 0 else 0.0,
            "avg_length": float(np.mean(self.lengths)) if self.lengths else 0.0,
            "elo_rating": calculate_elo_from_win_rate(self.wins / n),
        }
