import pytest
import torch

from gomoku.core.game_config import NUM_LINES
from gomoku.core.gomoku import GameState, Gomoku
from gomoku.model.model_helpers import NUM_PLANES, calc_num_hidden, calc_num_resblocks
from gomoku.model.policy_value_net import PolicyValueNet
from gomoku.utils.config.loader import BoardConfig, ModelConfig


def log_section(title: str) -> None:
    """Print a separated test section title."""
    print(f"\n\n{'=' * 20} [TEST: {title}] {'=' * 20}")


def log_state(game: Gomoku, state: GameState, description: str) -> None:
    """Print the current board state with a short description."""
    print(f"\n>>> {description}")
    print("-" * 30)
    game.print_board(state)
    print("-" * 30)


def log_tensor_summary(tensor: torch.Tensor, name: str, top_k: int = 5) -> None:
    """Print basic stats and top-k entries for a tensor."""
    flat = tensor.flatten()
    vals, idxs = torch.topk(flat, k=min(top_k, flat.numel()))
    print(
        f"{name}: shape={tuple(tensor.shape)}, min={float(flat.min())}, max={float(flat.max())}"
    )
    print(f"{name} top{len(vals)} values: {vals.tolist()} at indices {idxs.tolist()}")


def make_game(
    num_lines: int = 19,
    gomoku_goal: int = 5,
    capture_goal: int = 5,
    enable_doublethree: bool = True,
    enable_capture: bool = True,
) -> Gomoku:
    """Generate a Gomoku instance for tests."""
    return Gomoku(
        BoardConfig(
            num_lines=num_lines,
            gomoku_goal=gomoku_goal,
            capture_goal=capture_goal,
            enable_doublethree=enable_doublethree,
            enable_capture=enable_capture,
        )
    )


def make_model(
    num_planes: int | None = 13,
) -> tuple[PolicyValueNet, Gomoku, ModelConfig]:
    """Return a PolicyValueNet, matching Gomoku, and its config for tests."""
    game = make_game()
    derived_num_planes = (
        num_planes
        if num_planes is not None
        else int(game.get_encoded_state(game.get_initial_state()).shape[1])
    )
    config = ModelConfig(
        num_planes=derived_num_planes,
        num_hidden=128,
        num_resblocks=10,
        policy_channels=2,
        value_channels=1,
    )
    model = PolicyValueNet(game, config, device="cpu")
    return model, game, config


def minimal_raw_config(num_lines: int = 19) -> dict:
    """Return a minimal-but-complete config dict for parser tests."""
    return {
        "paths": {
            "run_prefix": "unit",
        },
        "io": {},
        "board": {
            "num_lines": num_lines,
            "enable_doublethree": True,
            "enable_capture": True,
            "capture_goal": 5,
            "gomoku_goal": 5,
        },
        "model": {},
        "training": {
            "num_iterations": 10,
            "num_selfplay_iterations": 1.0,
            "num_epochs": 1,
            "batch_size": 1,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "temperature": 1.0,
            "replay_buffer_size": 10,
            "min_samples_to_train": 1,
            "random_play_ratio": 0.0,
        },
        "mcts": {
            "C": 1.0,
            "num_searches": 16.0,
            "exploration_turns": 1,
            "dirichlet_epsilon": 0.25,
            "dirichlet_alpha": 0.03,
            "batch_infer_size": 1,
            "max_batch_wait_ms": 0,
            "min_batch_size": 1,
        },
        "evaluation": {
            "num_eval_games": 1,
            "eval_every_iters": 1,
            "promotion_win_rate": 0.5,
            "num_baseline_games": 1,
            "blunder_threshold": 0.0,
            "initial_blunder_rate": 0.0,
            "initial_baseline_win_rate": 0.0,
            "blunder_increase_limit": 0.0,
            "baseline_wr_min": 0.0,
            "random_play_ratio": 0.0,
        },
        "parallel": {},
    }


def make_linear_policy_model(
    board_size: int = NUM_LINES, focus_channels: tuple[int, ...] = (3,)
) -> tuple[PolicyValueNet, Gomoku, ModelConfig]:
    """Return a tiny PolicyValueNet with identity-like policy mapping."""
    game = Gomoku(
        BoardConfig(
            num_lines=board_size,
            gomoku_goal=5,
            capture_goal=5,
            enable_doublethree=True,
            enable_capture=True,
        )
    )
    config = ModelConfig(
        num_planes=NUM_PLANES,
        num_hidden=calc_num_hidden(board_size),
        num_resblocks=calc_num_resblocks(board_size),
        policy_channels=2,
        value_channels=1,
    )
    model = PolicyValueNet(game, config, device="cpu")
    model.eval()

    with torch.no_grad():
        for param in model.parameters():
            param.zero_()

        conv = model.start_block[0]
        conv.weight.zero_()
        for ch in focus_channels:
            conv.weight[0, ch, 1, 1] = 1.0

        bn = model.start_block[1]
        bn.weight.fill_(1.0)
        bn.bias.zero_()
        bn.running_mean.zero_()
        bn.running_var.fill_(1.0)
        bn.eps = 0.0

        policy_conv = model.policy_head[0]
        policy_conv.weight.zero_()
        policy_conv.weight[0, 0, 0, 0] = 1.0
        policy_bn = model.policy_head[1]
        policy_bn.weight.fill_(1.0)
        policy_bn.bias.zero_()
        policy_bn.running_mean.zero_()
        policy_bn.running_var.fill_(1.0)
        policy_bn.eps = 0.0

        linear = model.policy_head[4]
        linear.weight.zero_()
        linear.bias.zero_()
        for idx in range(game.action_size):
            linear.weight[idx, idx] = 1.0

        for param in model.value_head.parameters():
            param.zero_()

    return model, game, config


@pytest.fixture
def game_env() -> Gomoku:
    """Pytest fixture providing a 19x19 game with captures and double-three."""
    return Gomoku(
        BoardConfig(
            num_lines=19,
            gomoku_goal=5,
            capture_goal=5,
            enable_doublethree=True,
            enable_capture=True,
        )
    )


@pytest.fixture
def capture_game() -> Gomoku:
    """Game with capture enabled (win by five or by two capture pairs)."""
    return Gomoku(
        BoardConfig(
            num_lines=9,
            gomoku_goal=5,
            capture_goal=2,
            enable_doublethree=False,
            enable_capture=True,
        )
    )


@pytest.fixture
def strict_game() -> Gomoku:
    """Game with double-three forbiddens enabled."""
    return Gomoku(
        BoardConfig(
            num_lines=9,
            gomoku_goal=5,
            capture_goal=0,
            enable_doublethree=True,
            enable_capture=False,
        )
    )
