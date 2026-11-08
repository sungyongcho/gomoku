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
    assert third.draws == 0
    assert np.mean(third.width[PLAYER_1]) == 1
    assert np.mean(third.width[PLAYER_2]) == 81
    assert "mean_plies=4.0" in third.summary()
    # Short test games end with a decisive label and no captures, so they count as five wins.
    assert (third.five_wins, third.capture_wins, third.ended_draws) == (2, 0, 0)
    assert "five_win=1.00" in third.summary()
    # Test games put Black at (0,0) and White at (1,0): the reply touches the first stone.
    assert third.reply_distance == [1, 1]
    assert "reply_near=1.00" in third.summary()
    assert sorted(replay_report.collect(str(tmp_path), last=1)) == [4]


def test_value_saturation_reads_a_checkpoint(tmp_path: Path) -> None:
    import torch
    import yaml

    from gomoku.model.policy_value_net import PolicyValueNet
    from gomoku.utils.config.loader import load_and_parse_config

    raw = yaml.safe_load(Path("configs/small-6x96-selfplay.yaml").read_text())
    raw["board"].update(num_lines=9, enable_doublethree=False, enable_capture=False, history_length=1)
    raw["model"].update(num_hidden=8, num_resblocks=1)
    config = tmp_path / "small.yaml"
    config.write_text(yaml.safe_dump(raw))
    cfg = load_and_parse_config(str(config))
    game = Gomoku(cfg.board)
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_state_dict": PolicyValueNet(game, cfg.model, torch.device("cpu")).state_dict()}, checkpoint)
    replay = tmp_path / "replay"
    save_records_to_parquet_shard([_record(game, True)], str(replay / "shard-iter0001-1-local.parquet"))

    saturated, empty_value = replay_report.value_saturation(str(replay), str(checkpoint), str(config))
    assert 0.0 <= saturated <= 1.0
    assert -1.0 <= empty_value <= 1.0


def _threat_record(game: Gomoku) -> GameRecord:
    """Black builds four on row 4 while White plays row 8; the stored positions include
    Black to move with a win at (4, 4) and White to move facing it."""
    moves = [(0, 4), (0, 8), (1, 4), (2, 8), (2, 4), (4, 8), (3, 4), (6, 8)]
    states, policies, players, outcomes = [], [], [], []
    state = game.get_initial_state()
    for x, y in moves:
        player = int(state.next_player)
        states.append(state)
        policies.append(np.full(game.action_size, 1 / game.action_size, dtype=np.float32))
        players.append(player)
        outcomes.append(1 if player == PLAYER_1 else -1)
        state = game.get_next_state(state, (x, y), player)
    states.append(state)  # Black to move: (4, 4) wins
    policies.append(np.full(game.action_size, 1 / game.action_size, dtype=np.float32))
    players.append(int(state.next_player))
    outcomes.append(1)
    return GameRecord(
        states_raw=states,
        policies=np.stack(policies),
        moves=np.arange(len(states), dtype=np.int32),
        players=np.asarray(players, dtype=np.int8),
        outcomes=np.asarray(outcomes, dtype=np.int8),
        config_snapshot={"source": "test"},
    )


def test_tactics_finds_win_and_block_positions(tmp_path: Path) -> None:
    import torch
    import yaml

    from gomoku.model.policy_value_net import PolicyValueNet
    from gomoku.utils.config.loader import load_and_parse_config

    raw = yaml.safe_load(Path("configs/small-6x96-selfplay.yaml").read_text())
    raw["board"].update(num_lines=9, enable_doublethree=False, enable_capture=False, history_length=1)
    raw["model"].update(num_hidden=8, num_resblocks=1)
    config = tmp_path / "small.yaml"
    config.write_text(yaml.safe_dump(raw))
    cfg = load_and_parse_config(str(config))
    game = Gomoku(cfg.board)
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_state_dict": PolicyValueNet(game, cfg.model, torch.device("cpu")).state_dict()}, checkpoint)
    replay = tmp_path / "replay"
    save_records_to_parquet_shard([_threat_record(game)], str(replay / "shard-iter0001-1-local.parquet"))

    t = replay_report.tactics(str(replay), str(checkpoint), str(config), samples=10)
    assert t["win_positions"] >= 1 and 0.0 <= t["win_top1"] <= 1.0 and 0.0 <= t["win_mass"] <= 1.0
    assert t["block_positions"] >= 1 and 0.0 <= t["block_top1"] <= 1.0
    assert t["forbidden_positions"] == 0  # double-three is off in this config
