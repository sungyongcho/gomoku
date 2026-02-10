from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from gomoku.alphazero.learning.dataset import (
    GameSample,
    ReplayDataset,
    decode_board,
    decode_policy,
    flatten_game_records,
    game_records_to_rows,
    save_records_to_parquet_shard,
)
from gomoku.alphazero.types import GameRecord
from gomoku.core.game_config import PLAYER_1, xy_to_index
from gomoku.core.gomoku import GameState, Gomoku
from gomoku.utils.config.loader import BoardConfig, load_and_parse_config


def _make_game() -> Gomoku:
    cfg = BoardConfig(
        num_lines=3,
        enable_doublethree=False,
        enable_capture=False,
        capture_goal=5,
        gomoku_goal=3,
        history_length=1,
    )
    return Gomoku(cfg)


def _make_record(game: Gomoku) -> GameRecord:
    s0 = game.get_initial_state()
    x0, y0 = 0, 0
    s1 = game.get_next_state(s0, (x0, y0), player=1)
    pi0 = np.zeros(game.action_size, dtype=np.float32)
    pi1 = np.zeros(game.action_size, dtype=np.float32)
    pi0[0] = 1.0
    pi1[1 if game.action_size > 1 else 0] = 1.0
    moves = np.asarray([0, 1], dtype=np.int32)
    players = np.asarray([1, 2], dtype=np.int8)
    outcomes = np.asarray([1.0, -1.0], dtype=np.float32)
    return GameRecord(
        states_raw=[s0, s1],
        policies=np.stack([pi0, pi1]),
        moves=moves,
        players=players,
        outcomes=outcomes,
        config_snapshot={"test": True},
    )


def test_parquet_roundtrip_rows_to_shard_dataset(tmp_path: Path) -> None:
    """Verify GameRecord -> Parquet -> in-memory rows roundtrip."""
    game = _make_game()
    rec = _make_record(game)
    rows = game_records_to_rows([rec])
    shard = tmp_path / "shard.parquet"

    save_records_to_parquet_shard([rec], str(shard))
    assert shard.exists()

    table = pq.read_table(shard)
    loaded = table.to_pylist()
    print(f"[parquet] rows={len(rows)}, loaded={len(loaded)}")
    assert len(loaded) == len(rows)
    # Check a few key fields
    assert loaded[0]["last_move_idx"] == rows[0]["last_move_idx"]
    assert loaded[0]["config_snapshot"] == rows[0]["config_snapshot"]


def test_replay_dataset_priority_weights_exist() -> None:
    """ReplayDataset should return a priority tensor when include_priority=True."""
    game = _make_game()
    rec = _make_record(game)
    samples = flatten_game_records([rec], game)
    ds = ReplayDataset(samples=samples, game=game, include_priority=True)
    state, policy, value, priority = ds[0]
    print(f"[priority] priority={priority}")
    assert priority.shape == ()
    assert priority.item() > 0
    assert state.shape[1:] == (game.row_count, game.col_count)
    assert policy.shape[0] == game.action_size
    assert value.shape == (1,)


def test_replay_dataset_priority_safety() -> None:
    """Priorities with NaN/zero should be clamped to a small positive value."""
    game = _make_game()
    rec = _make_record(game)
    # Inject NaN and zero priorities
    samples = [
        GameSample(
            state=s.state, policy_probs=s.policy_probs, value=s.value, priority=np.nan
        )
        for s in flatten_game_records([rec], game)
    ]
    samples[0] = GameSample(
        state=samples[0].state,
        policy_probs=samples[0].policy_probs,
        value=samples[0].value,
        priority=0.0,
    )
    ds = ReplayDataset(samples=samples, game=game, include_priority=True)
    _, _, _, prio0 = ds[0]
    _, _, _, prio1 = ds[1]
    print(f"[priority_safe] prio0={prio0}, prio1={prio1}")
    assert prio0.item() > 0
    assert prio1.item() > 0


def test_shard_dataset_unknown_keys_raise(tmp_path: Path) -> None:
    """ShardDataset should reject rows containing unknown columns."""
    game = _make_game()
    rec = _make_record(game)
    shard = tmp_path / "shard.parquet"
    save_records_to_parquet_shard([rec], str(shard))

    # Corrupt by adding an unknown column
    table = pq.read_table(shard)
    unknown_col = pa.array([1] * table.num_rows)
    corrupted = table.append_column("unknown_key", unknown_col)
    pq.write_table(corrupted, shard)

    from gomoku.alphazero.learning.dataset import ShardDataset

    with pytest.raises(ValueError, match="Unknown keys"):
        ShardDataset([str(shard)], game=game, include_priority=True)


def test_per_sampler_probability_proportional_to_priority() -> None:
    """PER sampler weights should reflect relative priorities."""
    game = _make_game()
    rec = _make_record(game)
    samples = flatten_game_records([rec], game)
    # Assign different priorities
    samples[0] = GameSample(
        state=samples[0].state,
        policy_probs=samples[0].policy_probs,
        value=samples[0].value,
        priority=10.0,
    )
    samples[1] = GameSample(
        state=samples[1].state,
        policy_probs=samples[1].policy_probs,
        value=samples[1].value,
        priority=1.0,
    )
    ds = ReplayDataset(samples=samples, game=game, include_priority=True)
    weights = torch.tensor(ds.priorities, dtype=torch.float32)
    probs = weights / weights.sum()
    print(f"[per_probs] {probs.tolist()}")
    assert probs[0] > probs[1]
    # Probability ratio should match priority ratio
    assert pytest.approx(probs[0] / probs[1], rel=1e-5) == 10.0


def _make_state_with_marker(game: Gomoku) -> GameState:
    """Create a state with a single marker at (0, 1) to track symmetry transforms."""
    base = game.get_initial_state()
    board = base.board.copy()
    board[1, 0] = PLAYER_1
    last_idx = xy_to_index(0, 1, game.col_count)
    return GameState(
        board=board,
        p1_pts=base.p1_pts,
        p2_pts=base.p2_pts,
        next_player=base.next_player,
        last_move_idx=last_idx,
        empty_count=np.int16(base.empty_count - 1),
        history=(),
        legal_indices_cache=None,
    )


@pytest.mark.parametrize(
    "k_value,expected_idx,expected_coord", [(1, 7, (1, 2)), (4, 5, (2, 1))]
)
def test_replay_dataset_applies_symmetry(
    monkeypatch: pytest.MonkeyPatch,
    k_value: int,
    expected_idx: int,
    expected_coord: tuple[int, int],
) -> None:
    """ReplayDataset should rotate/flip both features and policy consistently."""
    game = _make_game()
    state = _make_state_with_marker(game)
    policy = np.zeros(game.action_size, dtype=np.float32)
    policy[state.last_move_idx] = 1.0
    samples = [GameSample(state=state, policy_probs=policy, value=0.0)]
    ds = ReplayDataset(samples=samples, game=game, include_priority=False)

    monkeypatch.setattr(np.random, "randint", lambda low, high: k_value)
    state_t, policy_t, value_t = ds[0]

    assert int(value_t.item()) == 0
    assert int(policy_t.argmax().item()) == expected_idx

    # last-move plane should move consistently with policy transform
    state_np = state_t.numpy()
    last_plane = state_np[3]
    assert last_plane.sum() == 1.0
    y_exp, x_exp = expected_coord[1], expected_coord[0]
    assert last_plane[y_exp, x_exp] == 1.0


def test_importance_correction_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    """Trainer should apply importance sampling weights when PER is enabled."""
    from gomoku.alphazero.learning.trainer import AlphaZeroTrainer

    game = _make_game()
    rec = _make_record(game)
    samples = flatten_game_records([rec], game)
    # Different priorities to trigger weighting
    samples[0] = GameSample(
        state=samples[0].state,
        policy_probs=samples[0].policy_probs,
        value=samples[0].value,
        priority=1.0,
    )
    samples[1] = GameSample(
        state=samples[1].state,
        policy_probs=samples[1].policy_probs,
        value=samples[1].value,
        priority=100.0,
    )

    # Minimal model that produces fixed outputs but retains grad
    class ConstNet(torch.nn.Module):
        def __init__(self, action_size: int) -> None:
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(action_size))
            self.value = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            batch = x.shape[0]
            return (
                self.logits.unsqueeze(0).repeat(batch, 1),
                self.value.unsqueeze(0).repeat(batch, 1),
            )

    cfg = BoardConfig(
        num_lines=3,
        enable_doublethree=False,
        enable_capture=False,
        capture_goal=5,
        gomoku_goal=3,
        history_length=1,
    )
    train_cfg = load_and_parse_config(
        "configs/config_alphazero_test.yaml"
    ).training.model_copy(
        update={
            "dataloader_num_workers": 0,
            "num_epochs": 1,
            "batch_size": 2,
            "use_channels_last": False,
            "enable_tf32": False,
            "priority_replay": {
                "enabled": True,
                "alpha": 1.0,
                "beta": 1.0,
                "epsilon": 1e-3,
            },
        }
    )
    model = ConstNet(game.action_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    trainer = AlphaZeroTrainer(
        train_cfg=train_cfg, model=model, optimizer=optimizer, game=game
    )
    # Avoid autocast on CPU
    monkeypatch.setattr(
        "gomoku.alphazero.learning.trainer.autocast",
        lambda *args, **kwargs: nullcontext(),
    )
    metrics = trainer.train_one_iteration(samples)
    print(f"[per_loss] loss={metrics['loss']:.6f}")
    assert metrics["loss"] > 0


def test_decode_board_policy_bytes_and_array() -> None:
    """decode_board/policy should accept both bytes and array inputs."""
    game = _make_game()
    board = np.arange(game.row_count * game.col_count, dtype=np.int8).reshape(
        game.row_count, game.col_count
    )
    board_bytes = board.tobytes()
    decoded = decode_board(board_bytes, game)
    decoded_arr = decode_board(board.copy(), game)
    print(f"[decode_board] shape={decoded.shape}, first={decoded.ravel()[0]}")
    assert np.array_equal(decoded, board)
    assert np.array_equal(decoded_arr, board)

    policy = np.linspace(0, 1, game.action_size, dtype=np.float16)
    policy_bytes = policy.tobytes()
    decoded_policy = decode_policy(policy_bytes)
    decoded_policy_arr = decode_policy(policy.copy())
    print(f"[decode_policy] sum={decoded_policy.sum():.4f}")
    assert np.allclose(decoded_policy, policy.astype(np.float32))
    assert np.allclose(decoded_policy_arr, policy.astype(np.float32))
