import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from gomoku.pvmcts.search.mp import MultiprocessEngine
from gomoku.pvmcts.treenode import TreeNode
from tests.pvmcts.search.conftest import dummy_mp_inference, make_game_and_params


@patch("gomoku.pvmcts.search.engine.torch.cuda.is_available", return_value=False)
@patch("gomoku.pvmcts.search.engine.torch.manual_seed")
@patch("gomoku.pvmcts.search.engine.np.random.seed")
def test_mp_engine_reseeds_random_generators(
    mock_np_seed: MagicMock,
    mock_torch_seed: MagicMock,
    mock_cuda_available: MagicMock,
) -> None:
    """RNGs must be reseeded on initialization."""
    game, params = make_game_and_params()
    inference = dummy_mp_inference(game.action_size)

    print("\n[Test] Checking RNG reseeding on initialization...")
    MultiprocessEngine(game, params, inference)

    mock_np_seed.assert_called_once()
    mock_torch_seed.assert_called_once()
    mock_cuda_available.assert_called()
    arg = mock_np_seed.call_args.args[0]
    assert isinstance(arg, int)
    print(f" -> Seeds called with seed={arg}")


@patch("gomoku.pvmcts.search.mp.MultiprocessEngine._seed_random_generators")
def test_mp_engine_pid_assignment(mock_seed: MagicMock) -> None:
    """PID recorded on engine creation matches current process."""
    game, params = make_game_and_params()
    inference = dummy_mp_inference(game.action_size)

    print("\n[Test] Checking PID assignment on engine creation...")
    engine = MultiprocessEngine(game, params, inference)

    assert engine.pid == os.getpid()
    print(f" -> Engine PID={engine.pid}, os.getpid()={os.getpid()}")


def test_mp_engine_batch_mapping() -> None:
    """Batch inference should map values back to matching start nodes."""
    game, params = make_game_and_params()
    roots = [TreeNode(game.get_initial_state()) for _ in range(3)]

    mock_inference = MagicMock()
    action_size = game.action_size
    expected_values = torch.tensor([[0.1], [0.5], [0.9]], dtype=torch.float32)
    mock_policy = torch.ones((3, action_size), dtype=torch.float32) / action_size
    mock_inference.infer.return_value = (mock_policy, expected_values)

    params = params.model_copy(update={"num_searches": 1.0})
    engine = MultiprocessEngine(game, params, mock_inference)
    print("\n[Test] Verifying batch mapping across 3 start nodes...")
    print(
        " -> Before search:",
        [root.value_sum for root in roots],
        [root.visit_count for root in roots],
    )
    engine.search(roots, add_noise=True)

    mock_inference.infer.assert_called_once()
    input_tensor = mock_inference.infer.call_args[0][0]
    assert input_tensor.size(0) == 3
    print(f" -> Inference batch size: {input_tensor.size(0)}")

    assert np.isclose(roots[0].value_sum, 0.1, atol=1e-5)
    assert np.isclose(roots[1].value_sum, 0.5, atol=1e-5)
    assert np.isclose(roots[2].value_sum, 0.9, atol=1e-5)
    print(
        f" -> Values mapped: {[roots[i].value_sum for i in range(3)]} "
        "(expected [0.1, 0.5, 0.9])"
    )


def test_mp_engine_data_integrity_over_ipc() -> None:
    """Ensure IPC batch preserves shape, dtype, and CPU device."""
    game, params = make_game_and_params()
    roots = [TreeNode(game.get_initial_state()) for _ in range(2)]

    captured: dict[str, torch.Tensor] = {}

    def capture_side_effect(batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        captured["input"] = batch
        bsz = batch.size(0)
        logits = torch.zeros((bsz, game.action_size), dtype=torch.float32)
        values = torch.zeros((bsz, 1), dtype=torch.float32)
        return logits, values

    inference = MagicMock()
    inference.infer.side_effect = capture_side_effect
    engine = MultiprocessEngine(game, params, inference)

    print("\n[Test] Verifying IPC data integrity (shape/dtype)...")
    engine.search(roots, add_noise=True)

    batch = captured.get("input")
    assert batch is not None, "Inference was not invoked."
    expected_shape = game.get_encoded_state([roots[0].state])[0].shape
    assert batch.shape[0] == len(roots)
    assert batch.shape[1:] == expected_shape
    assert batch.dtype == torch.float32
    assert batch.device.type == "cpu"
    expected_shape = game.get_encoded_state([roots[0].state])[0].shape
    print(
        f" -> Captured batch shape: {tuple(batch.shape)}, "
        f"dtype: {batch.dtype}, device: {batch.device}"
    )


def test_mp_engine_cpu_payload() -> None:
    """Input tensor sent to worker must be on CPU."""
    game, params = make_game_and_params()
    root = TreeNode(game.get_initial_state())
    mock_inference = MagicMock()
    mock_inference.infer.return_value = (
        torch.zeros((1, game.action_size)),
        torch.zeros((1, 1)),
    )

    engine = MultiprocessEngine(game, params, mock_inference)
    print("\n[Test] Verifying CPU payload enforcement...")
    engine.search([root], add_noise=True)

    input_tensor = mock_inference.infer.call_args[0][0]
    assert input_tensor.device.type == "cpu"
    print(f" -> Payload device: {input_tensor.device}")


def test_mp_engine_result_device_handling() -> None:
    """GPU outputs from client should be brought to CPU safely."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    game, params = make_game_and_params()
    root = TreeNode(game.get_initial_state())
    mock_inference = MagicMock()
    gpu_policy = torch.ones((1, game.action_size), device="cuda")
    gpu_value = torch.ones((1, 1), device="cuda")
    mock_inference.infer.return_value = (gpu_policy, gpu_value)

    engine = MultiprocessEngine(game, params, mock_inference)
    print("\n[Test] Verifying GPU output is handled on CPU...")
    engine.search([root], add_noise=True)

    assert root.visit_count > 0
    print(" -> Search completed without device errors.")


def test_mp_engine_root_dirichlet_only() -> None:
    """Dirichlet noise should apply only at root expansion."""
    np.random.seed(0)
    torch.manual_seed(0)

    game, params = make_game_and_params()
    params = params.model_copy(update={"num_searches": 2.0})
    start_node = TreeNode(state=game.get_initial_state())

    def infer_side_effect(batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = batch.size(0)
        logits = torch.ones((bsz, game.action_size), dtype=torch.float32)
        values = torch.zeros((bsz, 1), dtype=torch.float32)
        return logits, values

    mock_inference = MagicMock()
    mock_inference.infer.side_effect = infer_side_effect

    engine = MultiprocessEngine(game, params, mock_inference)
    print("\n[Test] Verifying root-only Dirichlet in MP engine...")
    engine.search([start_node], add_noise=True)

    root_priors = np.zeros(game.action_size, dtype=np.float32)
    for child in start_node.children.values():
        if child.action_taken:
            idx = child.action_taken[0] + child.action_taken[1] * game.col_count
            root_priors[idx] = child.prior

    expanded_child = next((c for c in start_node.children.values() if c.children), None)
    assert expanded_child is not None, "Expected an expanded child."

    child_priors = np.zeros(game.action_size, dtype=np.float32)
    for grandchild in expanded_child.children.values():
        if grandchild.action_taken:
            idx = (
                grandchild.action_taken[0] + grandchild.action_taken[1] * game.col_count
            )
            child_priors[idx] = grandchild.prior

    legal_mask_root_t = engine._legal_mask_tensor(start_node.state, torch.device("cpu"))
    legal_mask_child_t = engine._legal_mask_tensor(
        expanded_child.state, torch.device("cpu")
    )
    legal_mask_root = legal_mask_root_t.cpu().numpy().astype(bool)
    legal_mask_child = legal_mask_child_t.cpu().numpy().astype(bool)

    pure_policy_root = (
        engine._masked_policy_from_logits(
            torch.ones(game.action_size, dtype=torch.float32),
            legal_mask_root_t,
            apply_dirichlet=False,
        )
        .cpu()
        .numpy()
    )

    pure_policy_child = (
        engine._masked_policy_from_logits(
            torch.ones(game.action_size, dtype=torch.float32),
            legal_mask_child_t,
            apply_dirichlet=False,
        )
        .cpu()
        .numpy()
    )

    print(
        f" -> Root prior sample: sum={root_priors.sum():.3f}, "
        f"max={root_priors.max():.4f}, max_pure={pure_policy_root.max():.4f}"
    )
    print(
        f" -> Child prior sample: sum={child_priors.sum():.3f}, "
        f"max={child_priors.max():.4f}, max_pure={pure_policy_child.max():.4f}"
    )

    assert not np.allclose(
        root_priors[legal_mask_root],
        pure_policy_root[legal_mask_root],
        rtol=1e-3,
        atol=1e-4,
    )
    assert np.allclose(
        child_priors[legal_mask_child],
        pure_policy_child[legal_mask_child],
        atol=1e-5,
    )


def test_mp_engine_empty_input() -> None:
    """Empty input should short-circuit with no inference calls."""
    game, params = make_game_and_params()
    mock_inference = MagicMock()

    engine = MultiprocessEngine(game, params, mock_inference)
    print("\n[Test] Verifying empty input short-circuits...")
    engine.search([], add_noise=True)

    mock_inference.infer.assert_not_called()
    print(" -> Inference not called.")


def test_mp_engine_handles_broken_pipe(capsys: pytest.CaptureFixture[str]) -> None:
    """BrokenPipe/EOF should log and exit without crashing the tree."""
    game, params = make_game_and_params()
    root = TreeNode(game.get_initial_state())
    mock_inference = MagicMock()
    mock_inference.infer.side_effect = BrokenPipeError("Connection lost")

    engine = MultiprocessEngine(game, params, mock_inference)
    print("\n[Test] Verifying BrokenPipe handling...")
    engine.search([root], add_noise=True)

    captured = capsys.readouterr()
    assert "Inference Server connection lost" in captured.err
    print(f" -> stderr captured: {captured.err.strip()}")


def test_mp_engine_mismatched_batch_size_error() -> None:
    """Detect mismatched batch sizes between request and response."""
    game, params = make_game_and_params()
    roots = [TreeNode(game.get_initial_state()) for _ in range(2)]

    def bad_side_effect(batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Return fewer rows than requested to simulate server bug.
        bsz = max(1, batch.size(0) - 1)
        logits = torch.ones((bsz, game.action_size), dtype=torch.float32)
        values = torch.zeros((bsz, 1), dtype=torch.float32)
        return logits, values

    mock_inference = MagicMock()
    mock_inference.infer.side_effect = bad_side_effect

    engine = MultiprocessEngine(game, params, mock_inference)
    print("\n[Test] Verifying mismatched batch size handling...")

    with pytest.raises(RuntimeError):
        engine.search(roots, add_noise=True)
