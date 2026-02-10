from contextlib import nullcontext

import numpy as np
import pytest
import torch
import torch.nn as nn

from gomoku.alphazero.learning.dataset import (
    GameSample,
    ReplayDataset,
    decode_board,
    decode_policy,
    flatten_game_records,
)
from gomoku.alphazero.learning.trainer import AlphaZeroTrainer
from gomoku.alphazero.types import GameRecord
from gomoku.core.gomoku import Gomoku
from gomoku.utils.config.loader import (
    BoardConfig,
    PriorityReplayConfig,
    TrainingConfig,
    load_and_parse_config,
)


def _no_autocast(*args, **kwargs):
    """Return a no-op context manager in place of torch.amp.autocast."""
    return nullcontext()


class TinyNet(nn.Module):
    """Small network for fast CPU-only tests."""

    def __init__(self, game: Gomoku) -> None:
        super().__init__()
        sample_state = game.get_initial_state()
        planes, rows, cols = game.get_encoded_state([sample_state])[0].shape
        hidden = 8
        self.conv = nn.Conv2d(planes, hidden, kernel_size=1)
        self.head_policy = nn.Linear(hidden * rows * cols, game.action_size)
        self.head_value = nn.Linear(hidden * rows * cols, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.conv(x))
        x = torch.flatten(x, start_dim=1)
        policy = self.head_policy(x)
        value = self.head_value(x)
        return policy, value


class FixedNet(nn.Module):
    """Network that returns constant logits/values for deterministic loss checks."""

    def __init__(self, game: Gomoku, policy_value: float = 0.0) -> None:
        super().__init__()
        self.policy_logits = nn.Parameter(
            torch.full((game.action_size,), float(policy_value)), requires_grad=True
        )
        self.value_out = nn.Parameter(
            torch.tensor([[float(policy_value)]], dtype=torch.float32),
            requires_grad=True,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        policy = self.policy_logits.unsqueeze(0).repeat(batch, 1)
        value = self.value_out.repeat(batch, 1)
        return policy, value


def _make_cfg_and_game() -> tuple[TrainingConfig, Gomoku]:
    cfg = load_and_parse_config("configs/config_alphazero_test.yaml")
    train_cfg = cfg.training.model_copy(
        update={"dataloader_num_workers": 0, "num_epochs": 1}
    )
    game = Gomoku(cfg.board)
    return train_cfg, game


def _make_record(game: Gomoku) -> GameRecord:
    """Create a tiny GameRecord with two turns."""
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
    )


def _make_samples(game: Gomoku) -> list[GameSample]:
    rec = _make_record(game)
    return flatten_game_records([rec], game)


def _trainer_with_model(
    model: nn.Module, game: Gomoku, lr: float = 0.1
) -> AlphaZeroTrainer:
    train_cfg = load_and_parse_config(
        "configs/config_alphazero_test.yaml"
    ).training.model_copy(
        update={
            "dataloader_num_workers": 0,
            "num_epochs": 1,
            "batch_size": 2,
            "use_channels_last": False,
            "enable_tf32": False,
        }
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    trainer = AlphaZeroTrainer(
        train_cfg=train_cfg, model=model, optimizer=optimizer, game=game
    )
    return trainer


def test_flatten_length_matches_moves() -> None:
    """Flattened samples count should equal total moves across records."""
    train_cfg, game = _make_cfg_and_game()
    rec = _make_record(game)
    flat = flatten_game_records([rec], game)
    print(f"[flatten] moves={len(rec.moves)}, flat={len(flat)}")
    assert len(flat) == len(rec.moves)
    assert all(isinstance(s, GameSample) for s in flat)


def test_replay_dataset_shapes() -> None:
    """ReplayDataset should return tensors with expected shapes and priority."""
    _, game = _make_cfg_and_game()
    samples = _make_samples(game)
    dataset = ReplayDataset(samples=samples, game=game, include_priority=True)
    state, policy, value, priority = dataset[0]
    print(f"[dataset] state={state.shape}, policy={policy.shape}, value={value.shape}")
    assert state.shape[1:] == (game.row_count, game.col_count)
    assert policy.shape[0] == game.action_size
    assert value.shape == (1,)
    assert priority.shape == ()

    dataset_no_prio = ReplayDataset(samples=samples, game=game, include_priority=False)
    state2, policy2, value2 = dataset_no_prio[0]
    assert state2.shape == state.shape
    assert policy2.shape == policy.shape
    assert value2.shape == value.shape


def test_train_one_iteration_updates_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """Training step should update at least one parameter."""
    torch.manual_seed(0)
    _, game = _make_cfg_and_game()
    model = TinyNet(game)
    trainer = _trainer_with_model(model, game, lr=0.05)

    # Avoid CPU float16 autocast issues
    monkeypatch.setattr("gomoku.alphazero.learning.trainer.autocast", _no_autocast)
    samples = _make_samples(game)

    before = [p.detach().clone() for p in trainer.model.parameters()]
    metrics = trainer.train_one_iteration(samples)
    after = list(trainer.model.parameters())

    changed = any(not torch.allclose(b, a) for b, a in zip(before, after, strict=True))
    print(f"[train_update] loss={metrics['loss']:.4f}, changed={changed}")
    assert changed
    required_keys = {"loss", "policy_loss", "value_loss"}
    assert required_keys.issubset(set(metrics)), f"missing keys: {required_keys - set(metrics)}"


def test_loss_metrics_are_averaged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Loss metrics should equal batch mean when weights are fixed."""
    _, game = _make_cfg_and_game()
    model = FixedNet(game, policy_value=0.0)
    trainer = _trainer_with_model(model, game, lr=0.0)

    monkeypatch.setattr("gomoku.alphazero.learning.trainer.autocast", _no_autocast)
    samples = _make_samples(game)
    metrics = trainer.train_one_iteration(samples)

    # log_softmax of zeros over action_size => -log(A)
    expected_policy = np.log(game.action_size)
    expected_value = 1.0  # targets are ±1, preds are 0
    expected_loss = expected_policy + expected_value
    print(
        f"[avg_loss] loss={metrics['loss']:.4f}, "
        f"policy≈{metrics['policy_loss']:.4f}, value≈{metrics['value_loss']:.4f}"
    )
    assert pytest.approx(metrics["policy_loss"], rel=1e-3) == expected_policy
    assert pytest.approx(metrics["value_loss"], rel=1e-3) == expected_value
    assert pytest.approx(metrics["loss"], rel=1e-3) == expected_loss


def test_single_batch_overfits_loss_drops(monkeypatch: pytest.MonkeyPatch) -> None:
    """Running multiple iterations on identical data should reduce loss."""
    torch.manual_seed(1)
    _, game = _make_cfg_and_game()
    model = TinyNet(game)
    trainer = _trainer_with_model(model, game, lr=0.1)
    monkeypatch.setattr("gomoku.alphazero.learning.trainer.autocast", _no_autocast)
    samples = _make_samples(game)

    loss1 = trainer.train_one_iteration(samples)["loss"]
    loss2 = trainer.train_one_iteration(samples)["loss"]
    print(f"[overfit] loss1={loss1:.4f}, loss2={loss2:.4f}")
    assert loss2 <= loss1


def test_per_priorities_are_updated(monkeypatch: pytest.MonkeyPatch) -> None:
    """When PER is enabled, TD-error should update dataset priorities."""
    torch.manual_seed(0)
    cfg = load_and_parse_config("configs/config_alphazero_test.yaml")
    game = Gomoku(cfg.board)

    # Enable PER with simple hyperparams
    pr_cfg = PriorityReplayConfig(
        enabled=True, start_iteration=0, alpha=0.0, beta=0.0, epsilon=0.1
    )
    train_cfg = cfg.training.model_copy(
        update={
            "dataloader_num_workers": 0,
            "num_epochs": 1,
            "batch_size": 2,
            "priority_replay": pr_cfg,
            "use_channels_last": False,
            "enable_tf32": False,
        }
    )

    samples = _make_samples(game)
    dataset = ReplayDataset(
        samples=samples, game=game, include_priority=True, return_index=True
    )
    assert all(p == 1.0 for p in dataset.priorities)

    model = FixedNet(game, policy_value=0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    trainer = AlphaZeroTrainer(
        train_cfg=train_cfg, model=model, optimizer=optimizer, game=game
    )
    monkeypatch.setattr("gomoku.alphazero.learning.trainer.autocast", _no_autocast)

    trainer.train_one_iteration(dataset)

    # TD-error is |0 - z| = 1; new priorities should reflect epsilon addition
    assert all(p > 1.0 for p in dataset.priorities)


def test_temperature_clamp_prevents_div_zero() -> None:
    """_apply_temperature should clamp near-zero temps to avoid div-by-zero."""
    from gomoku.alphazero.agent import _apply_temperature

    probs = np.array([0.2, 0.3, 0.5], dtype=np.float32)
    # Negative or tiny temperatures should fall back to argmax safely
    for temp in [-1.0, 0.0, 1e-12, 1e-6]:
        out = _apply_temperature(probs, temp)
        print(f"[temp_clamp] temp={temp}, argmax={out.argmax()}, sum={out.sum():.6f}")
        assert np.isclose(out.sum(), 1.0)
        assert out.argmax() == int(np.argmax(probs))
        assert out.max() == 1.0 and out.min() == 0.0


def test_decode_board_policy_bytes_and_array() -> None:
    """decode_board/policy should accept both bytes and array inputs."""
    cfg = BoardConfig(
        num_lines=3,
        enable_doublethree=False,
        enable_capture=False,
        capture_goal=5,
        gomoku_goal=3,
        history_length=1,
    )
    game = Gomoku(cfg)
    board = np.arange(cfg.num_lines * cfg.num_lines, dtype=np.int8).reshape(
        cfg.num_lines, cfg.num_lines
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
