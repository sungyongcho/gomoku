"""Replay health report."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from gomoku.alphazero.learning.dataset import save_records_to_parquet_shard
from gomoku.alphazero.types import GameRecord
from gomoku.core.game_config import PLAYER_1, PLAYER_2
from gomoku.core.gomoku import Gomoku
from gomoku.scripts import replay_report
from gomoku.utils.config.loader import BoardConfig


def _game() -> Gomoku:
    return Gomoku(
        BoardConfig(
            num_lines=9,
            enable_doublethree=False,
            enable_capture=False,
            capture_goal=5,
            gomoku_goal=5,
            history_length=1,
        )
    )


def _record(game: Gomoku, winner_first: bool) -> GameRecord:
    states, policies, players, outcomes = [], [], [], []
    state = game.get_initial_state()
    for move in range(4):
        player = int(state.next_player)
        target = np.zeros(game.action_size, dtype=np.float32)
        if player == PLAYER_1:
            target[move] = 1.0
        else:
            target[:] = 1 / game.action_size
        states.append(state)
        policies.append(target)
        players.append(player)
        won = (player == PLAYER_1) == winner_first
        outcomes.append(1 if won else -1)
        state = game.get_next_state(state, (move, 0), player)
    return GameRecord(
        states_raw=states,
        policies=np.stack(policies),
        moves=np.arange(4, dtype=np.int32),
        players=np.asarray(players, dtype=np.int8),
        outcomes=np.asarray(outcomes, dtype=np.int8),
        config_snapshot={"source": "test"},
    )


def test_report_groups_by_iteration_and_measures_targets(tmp_path: Path) -> None:
    game = _game()
    save_records_to_parquet_shard(
        [_record(game, True), _record(game, False)],
        str(tmp_path / "shard-iter0003-1-local.parquet"),
    )
    save_records_to_parquet_shard(
        [_record(game, True)], str(tmp_path / "shard-iter0004-1-local.parquet")
    )
    stats = replay_report.collect(str(tmp_path))
    assert sorted(stats) == [3, 4]
    third = stats[3]
    assert third.games == 2
    assert third.rows == 8
    assert third.first_player_wins == 1
    assert third.early_one_hot / third.early_rows == pytest.approx(0.5)
    assert np.mean(third.entropy[PLAYER_1]) == pytest.approx(0.0, abs=1e-6)
    assert np.mean(third.entropy[PLAYER_2]) == pytest.approx(np.log(81), abs=0.01)
    assert "mean_plies=4.0" in third.summary()
