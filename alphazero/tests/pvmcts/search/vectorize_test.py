from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from gomoku.core.game_config import PLAYER_1, PLAYER_2, set_pos, xy_to_index
from gomoku.pvmcts.search.vectorize import VectorizeEngine
from gomoku.pvmcts.treenode import TreeNode


def test_vectorize_single_start_node_input(vectorize_engine: VectorizeEngine) -> None:
    """Ensure a single TreeNode input is accepted and processed correctly."""
    print("\n[Test] Starting single start-node input test...")

    target_searches = int(vectorize_engine.params.num_searches)
    print(f" -> Target searches set to: {target_searches}")

    state = vectorize_engine.game.get_initial_state()
    start_node = TreeNode(state=state)

    print(" -> Executing search...")
    vectorize_engine.search(start_node, add_noise=True)

    print(f" -> Start-node visit count: {start_node.visit_count}")
    assert start_node.visit_count == target_searches, (
        f"Expected {target_searches} visits, got {start_node.visit_count}"
    )

    print(f" -> Start-node children count: {len(start_node.children)}")
    assert len(start_node.children) > 0, "Start node should be expanded"
    print("[Test] Passed.")


def test_vectorize_empty_input(vectorize_engine: VectorizeEngine) -> None:
    """Ensure an empty list input returns immediately without computation."""
    print("\n[Test] Starting empty input test...")

    vectorize_engine.inference = MagicMock()
    print(" -> Inference mocked.")

    print(" -> Calling search with empty list...")
    vectorize_engine.search([], add_noise=True)

    print(" -> Verifying inference calls...")
    vectorize_engine.inference.infer.assert_not_called()
    vectorize_engine.inference.infer_batch.assert_not_called()
    print("[Test] Passed: No inference calls detected.")


def test_vectorize_start_node_dirichlet_only(vectorize_engine: VectorizeEngine) -> None:
    """Dirichlet noise should affect only the initial start-node expansion."""
    print("\n[Test] Starting start-node Dirichlet-only test...")
    np.random.seed(0)
    torch.manual_seed(0)

    start_node = TreeNode(state=vectorize_engine.game.get_initial_state())
    vectorize_engine.search(start_node, add_noise=True)

    logits = torch.ones(vectorize_engine.game.action_size, dtype=torch.float32)

    legal_mask_head = vectorize_engine._legal_mask_tensor(
        start_node.state, device=logits.device
    )
    pure_policy_head = (
        vectorize_engine._masked_policy_from_logits(
            logits, legal_mask_head, apply_dirichlet=False
        )
        .cpu()
        .numpy()
    )

    head_priors = np.zeros(vectorize_engine.game.action_size, dtype=np.float32)
    for child in start_node.children.values():
        if child.action_taken:
            idx = (
                child.action_taken[0]
                + child.action_taken[1] * vectorize_engine.game.col_count
            )
            head_priors[idx] = child.prior

    legal_mask_np = vectorize_engine._legal_mask_numpy(start_node.state)
    top_idx_root = np.argmax(head_priors)
    diff_root = np.abs(head_priors - pure_policy_head)
    diff_count = int((diff_root > 1e-4).sum())
    print(
        f" -> Start-node priors: sum={head_priors.sum():.3f}, "
        f"top_idx={top_idx_root}, prior={head_priors[top_idx_root]:.4f}, "
        f"pure={pure_policy_head[top_idx_root]:.4f}, "
        f"diff_count>{1e-4}: {diff_count}, max_diff={diff_root.max():.4f}"
    )
    assert not np.allclose(
        head_priors[legal_mask_np],
        pure_policy_head[legal_mask_np],
        rtol=1e-3,
        atol=1e-4,
    ), "Start-node priors should differ due to Dirichlet noise."

    expanded_child = next((c for c in start_node.children.values() if c.children), None)
    assert expanded_child is not None, "Expected at least one expanded child."

    legal_mask_child = vectorize_engine._legal_mask_tensor(
        expanded_child.state, device=logits.device
    )
    expected_child_policy = (
        vectorize_engine._masked_policy_from_logits(
            logits, legal_mask_child, apply_dirichlet=False
        )
        .cpu()
        .numpy()
    )

    child_priors = np.zeros(vectorize_engine.game.action_size, dtype=np.float32)
    for grandchild in expanded_child.children.values():
        if grandchild.action_taken:
            idx = (
                grandchild.action_taken[0]
                + grandchild.action_taken[1] * vectorize_engine.game.col_count
            )
            child_priors[idx] = grandchild.prior

    legal_mask_child_np = vectorize_engine._legal_mask_numpy(expanded_child.state)
    top_idx_child = np.argmax(child_priors)
    diff_child = np.abs(child_priors - expected_child_policy)
    print(
        f" -> Child priors: sum={child_priors.sum():.3f}, "
        f"top_idx={top_idx_child}, prior={child_priors[top_idx_child]:.4f}, "
        f"expected={expected_child_policy[top_idx_child]:.4f}, "
        f"max_diff={diff_child.max():.4f}"
    )
    assert np.allclose(
        child_priors[legal_mask_child_np],
        expected_child_policy[legal_mask_child_np],
        atol=1e-5,
    ), "Non-root expansions should match pure softmax (no Dirichlet)."


def test_vectorize_respects_num_searches(vectorize_engine: VectorizeEngine) -> None:
    """Each start node must strictly reach exactly the configured num_searches."""
    print("\n[Test] Verifying num_searches enforcement...")

    # [Fix 1] Frozen dataclass 수정 불가 -> replace로 새 설정 객체 생성 후 주입
    target_searches = 5
    new_params = vectorize_engine.params.model_copy(
        update={"num_searches": target_searches}
    )
    vectorize_engine.params = new_params
    print(f" -> Configured target_searches: {target_searches}")

    state_a = vectorize_engine.game.get_initial_state()
    state_b = vectorize_engine.game.get_initial_state()

    start_a = TreeNode(state=state_a)
    start_b = TreeNode(state=state_b)

    # Act
    vectorize_engine.search([start_a, start_b], add_noise=True)

    # Assert
    print(
        f" -> Post-search visits: start_a={start_a.visit_count}, start_b={start_b.visit_count}"
    )

    assert start_a.visit_count == target_searches, (
        f"start_a visits {start_a.visit_count} != target {target_searches}"
    )
    assert start_b.visit_count == target_searches, (
        f"start_b visits {start_b.visit_count} != target {target_searches}"
    )

    print("[Test] Passed: Both start nodes reached exact target count.")


def test_vectorize_active_set_drops_finished(vectorize_engine: VectorizeEngine) -> None:
    """Start nodes meeting the target count should be skipped immediately."""
    print("\n[Test] Verifying active set drop for pre-finished start nodes...")

    target_searches = 5
    new_params = vectorize_engine.params.model_copy(
        update={"num_searches": target_searches}
    )
    vectorize_engine.params = new_params

    finished_start = TreeNode(state=vectorize_engine.game.get_initial_state())
    finished_start.visit_count = target_searches

    active_start = TreeNode(state=vectorize_engine.game.get_initial_state())

    print(
        f" -> Pre-search visits: finished={finished_start.visit_count}, "
        f"active={active_start.visit_count}"
    )

    vectorize_engine.search([finished_start, active_start], add_noise=True)

    print(
        f" -> Post-search visits: finished={finished_start.visit_count}, "
        f"active={active_start.visit_count}"
    )

    assert finished_start.visit_count == target_searches, (
        "Finished start node was incorrectly visited again."
    )
    assert not finished_start.children, (
        "Finished start node should not have been expanded."
    )

    assert active_start.visit_count == target_searches, (
        "Active start node failed to reach target visits."
    )

    print("[Test] Passed: Finished start node skipped, active start node processed.")


def test_vectorize_terminal_short_circuit(vectorize_engine: VectorizeEngine) -> None:
    """Terminal start node should trigger immediate backup without inference."""
    print("\n[Test] Verifying terminal short-circuit...")

    target_searches = 5
    vectorize_engine.params = vectorize_engine.params.model_copy(
        update={"num_searches": target_searches}
    )

    # Build a terminal state (winning line)
    term_state = vectorize_engine.game.get_initial_state()
    for x in range(5):
        set_pos(term_state.board, x, 0, PLAYER_1)
    term_state.last_move_idx = xy_to_index(4, 0, vectorize_engine.game.col_count)
    term_state.empty_count -= 5
    term_state.legal_indices_cache = None

    class CountingInference:
        """Track calls to ensure inference is skipped."""

        def __init__(self, action_size: int):
            self.calls = 0
            self.logits = torch.ones((1, action_size), dtype=torch.float32)
            self.value = torch.zeros((1, 1), dtype=torch.float32)

        def infer(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            self.calls += 1
            return self.logits, self.value

        def infer_batch(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            self.calls += 1
            return self.logits, self.value

    counter = CountingInference(vectorize_engine.game.action_size)
    vectorize_engine.inference = counter

    start_node = TreeNode(state=term_state)
    vectorize_engine.search(start_node, add_noise=True)

    print(
        f" -> Calls={counter.calls}, visits={start_node.visit_count}, "
        f"children={len(start_node.children)}, value_sum={start_node.value_sum}"
    )
    assert counter.calls == 0, "Inference must be skipped for terminal nodes."
    assert start_node.visit_count == 1, "Terminal backup must still bump visits."
    assert not start_node.children, "Terminal node must not expand children."
    assert start_node.value_sum == pytest.approx(-1.0), (
        "Winning last move should back up as loss (-1.0) for current player."
    )


def test_vectorize_mixed_terminal_and_active(
    vectorize_engine: VectorizeEngine,
) -> None:
    """Terminal nodes should be skipped while active nodes are processed."""
    print("\n[Test] Verifying mixed terminal and active start nodes...")

    target_searches = 5
    vectorize_engine.params = vectorize_engine.params.model_copy(
        update={"num_searches": target_searches}
    )

    # Terminal node
    term_state = vectorize_engine.game.get_initial_state()
    for x in range(5):
        set_pos(term_state.board, x, 0, PLAYER_1)
    term_state.last_move_idx = xy_to_index(4, 0, vectorize_engine.game.col_count)
    term_state.empty_count -= 5
    term_state.legal_indices_cache = None
    terminal_node = TreeNode(state=term_state)

    # Active node
    active_node = TreeNode(state=vectorize_engine.game.get_initial_state())

    vectorize_engine.search([terminal_node, active_node], add_noise=True)

    print(
        f" -> Terminal visits={terminal_node.visit_count}, "
        f"children={len(terminal_node.children)}"
    )
    print(
        f" -> Active visits={active_node.visit_count}, "
        f"children={len(active_node.children)}"
    )

    assert terminal_node.visit_count == 1
    assert not terminal_node.children

    assert active_node.visit_count == target_searches
    assert active_node.children, "Active node should be expanded."
    print("[Test] Passed: Terminal node skipped, active node processed correctly.")


def test_vectorize_batch_inference_shapes(vectorize_engine: VectorizeEngine) -> None:
    """infer_batch is used once and preserves batch shapes."""
    print("\n[Test] Verifying batch inference shapes and mapping...")

    target_searches = 1
    vectorize_engine.params = vectorize_engine.params.model_copy(
        update={"num_searches": target_searches, "dirichlet_epsilon": 0.0}
    )

    class CaptureInference:
        """Capture batch inputs and return indexed logits/values."""

        def __init__(self, action_size: int):
            self.action_size = action_size
            self.calls: list[torch.Tensor] = []

        def infer_batch(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            self.calls.append(batch)
            bsz = batch.size(0)
            logits = torch.zeros((bsz, self.action_size), dtype=torch.float32)
            for i in range(bsz):
                logits[i, i % self.action_size] = 5.0
            values = torch.arange(bsz, dtype=torch.float32).view(bsz, 1)
            return logits, values

    capture = CaptureInference(vectorize_engine.game.action_size)
    vectorize_engine.inference = capture

    start_a = TreeNode(state=vectorize_engine.game.get_initial_state())
    start_b = TreeNode(state=vectorize_engine.game.get_initial_state())

    vectorize_engine.search([start_a, start_b], add_noise=True)

    assert len(capture.calls) == 1, "infer_batch should be called exactly once."
    batch = capture.calls[0]
    print(
        f" -> infer_batch calls={len(capture.calls)}, batch_shape={tuple(batch.shape)}"
    )
    print(
        f" -> start_a value_sum={start_a.value_sum:.1f}, "
        f"start_b value_sum={start_b.value_sum:.1f}"
    )

    assert batch.dim() == 4, f"Expected 4D batch input, got {batch.shape}"
    assert batch.size(0) == 2, "Batch size must match number of start nodes."
    assert start_a.value_sum == pytest.approx(0.0)
    assert start_b.value_sum == pytest.approx(1.0)


def test_vectorize_fallback_to_standard_infer(
    vectorize_engine: VectorizeEngine,
) -> None:
    """Infer is used when infer_batch is unavailable."""
    print("\n[Test] Verifying fallback to standard infer...")

    target_searches = 1
    vectorize_engine.params = vectorize_engine.params.model_copy(
        update={"num_searches": target_searches, "dirichlet_epsilon": 0.0}
    )

    class FallbackInference:
        """Expose only infer to trigger fallback path."""

        def __init__(self, action_size: int):
            self.action_size = action_size
            self.calls: list[torch.Tensor] = []

        def infer(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            self.calls.append(batch)
            bsz = batch.size(0)
            logits = torch.ones((bsz, self.action_size), dtype=torch.float32)
            values = torch.zeros((bsz, 1), dtype=torch.float32)
            return logits, values

    fallback = FallbackInference(vectorize_engine.game.action_size)
    vectorize_engine.inference = fallback

    start_a = TreeNode(state=vectorize_engine.game.get_initial_state())
    start_b = TreeNode(state=vectorize_engine.game.get_initial_state())

    vectorize_engine.search([start_a, start_b], add_noise=True)

    assert len(fallback.calls) == 1, "infer should be called exactly once."
    batch = fallback.calls[0]
    print(f" -> infer calls={len(fallback.calls)}, batch_shape={tuple(batch.shape)}")

    assert batch.dim() == 4, f"Expected 4D batch input, got {batch.shape}"
    assert batch.size(0) == 2, "Fallback must still stack inputs."


def test_vectorize_enforces_cpu_tensors(vectorize_engine: VectorizeEngine) -> None:
    """Inference inputs must be forced to CPU."""
    print("\n[Test] Verifying CPU enforcement on inference inputs...")

    vectorize_engine.params = vectorize_engine.params.model_copy(
        update={"num_searches": 1.0}
    )

    class DeviceCheckInference:
        """Record device types passed into inference."""

        def __init__(self, action_size: int):
            self.action_size = action_size
            self.devices: list[str] = []

        def infer(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            self.devices.append(batch.device.type)
            bsz = batch.size(0)
            logits = torch.ones((bsz, self.action_size), dtype=torch.float32)
            values = torch.zeros((bsz, 1), dtype=torch.float32)
            return logits, values

    recorder = DeviceCheckInference(vectorize_engine.game.action_size)
    vectorize_engine.inference = recorder

    start_node = TreeNode(state=vectorize_engine.game.get_initial_state())
    vectorize_engine.search(start_node, add_noise=True)

    print(f" -> Recorded devices: {recorder.devices}")
    assert recorder.devices, "Inference should be invoked at least once."
    assert all(d == "cpu" for d in recorder.devices), "Inputs must be on CPU."


def test_vectorize_no_zero_backup_bias(vectorize_engine: VectorizeEngine) -> None:
    """Revisited leaf (visits>0) must trigger inference, not 0.0 backup."""
    print("\n[Test] Verifying re-evaluation of visited leaves...")

    # Setup: Target 2 searches.
    # Start node has 1 visit already. So it needs 1 more simulation.
    target_searches = 2
    vectorize_engine.params = vectorize_engine.params.model_copy(
        update={"num_searches": target_searches, "dirichlet_epsilon": 0.0}
    )

    # 1. Prepare a visited leaf node
    start_node = TreeNode(state=vectorize_engine.game.get_initial_state())
    start_node.visit_count = 1
    start_node.value_sum = 0.5  # Previous average value = 0.5

    # 2. Mock Inference with a specific value (e.g., 0.9)
    # If the bug exists (0.0 backup), new value_sum will be 0.5 + 0.0 = 0.5
    # If fixed (inference called), new value_sum will be 0.5 + 0.9 = 1.4
    expected_new_value = 0.9

    mock_inference = MagicMock()
    dummy_logits = torch.zeros((1, vectorize_engine.game.action_size))
    dummy_value = torch.tensor([[expected_new_value]])  # (1, 1)

    # Handle both infer and infer_batch for robustness
    mock_inference.infer.return_value = (
        dummy_logits.squeeze(0),
        dummy_value.squeeze(0),
    )
    mock_inference.infer_batch.return_value = (dummy_logits, dummy_value)

    vectorize_engine.inference = mock_inference

    # Act
    vectorize_engine.search(start_node, add_noise=True)

    # Assert
    print(
        f" -> Calls (infer/batch)={mock_inference.infer.call_count}/{mock_inference.infer_batch.call_count}"
    )
    print(
        f" -> Visits: {start_node.visit_count} (Expected {target_searches}), "
        f"ValueSum: {start_node.value_sum:.2f} (Expected {0.5 + expected_new_value:.2f})"
    )

    # [Crucial] Inference MUST be called.
    # Validates that we did NOT skip the node or just backup 0.0 locally.
    assert mock_inference.infer.called or mock_inference.infer_batch.called, (
        "Inference should be triggered for visited leaf nodes."
    )

    # [Crucial] Value Check
    assert start_node.visit_count == target_searches
    assert start_node.value_sum == pytest.approx(0.5 + expected_new_value), (
        "Bias detected! It seems 0.0 was backed up instead of inference result."
    )


def test_vectorize_handles_no_legal_moves(vectorize_engine: VectorizeEngine) -> None:
    """Ensure a terminal start node (win/no moves) short-circuits without inference."""
    print("\n[Test] Verifying terminal start node without inference...")

    vectorize_engine.params = vectorize_engine.params.model_copy(
        update={"num_searches": 1.0, "dirichlet_epsilon": 0.0}
    )

    # Build a real winning position for PLAYER_1 along the top row.
    state = vectorize_engine.game.get_initial_state()
    for x in range(5):
        set_pos(state.board, x, 0, PLAYER_1)
    state.last_move_idx = xy_to_index(4, 0, vectorize_engine.game.col_count)
    state.empty_count -= 5
    state.legal_indices_cache = None
    state.next_player = PLAYER_2

    vectorize_engine.inference = MagicMock()
    start_node = TreeNode(state=state)

    vectorize_engine.search(start_node, add_noise=True)

    print(
        f" -> visits={start_node.visit_count}, children={len(start_node.children)}, "
        f"value_sum={start_node.value_sum:.2f}"
    )

    # Should stop immediately with one backup and no inference.
    assert start_node.visit_count == 1
    assert not start_node.children
    assert start_node.value_sum != 0.0
    vectorize_engine.inference.infer.assert_not_called()
    vectorize_engine.inference.infer_batch.assert_not_called()


def test_vectorize_duplicate_states_in_batch(vectorize_engine: VectorizeEngine) -> None:
    """Duplicate start nodes should keep batch mapping order."""
    print("\n[Test] Verifying batch mapping with duplicate states...")

    vectorize_engine.params = vectorize_engine.params.model_copy(
        update={"num_searches": 1.0, "dirichlet_epsilon": 0.0}
    )

    class MappingInference:
        """Return distinct values per batch index to verify mapping."""

        def __init__(self, action_size: int):
            self.action_size = action_size
            self.calls = 0

        def infer_batch(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            self.calls += 1
            bsz = batch.size(0)
            logits = torch.ones((bsz, self.action_size), dtype=torch.float32)
            # Return [0.0, 1.0] to identify which result goes where
            values = torch.arange(bsz, dtype=torch.float32).view(bsz, 1)
            return logits, values

    mapper = MappingInference(vectorize_engine.game.action_size)
    vectorize_engine.inference = mapper

    # Create two nodes with the IDENTICAL state object/content
    state = vectorize_engine.game.get_initial_state()
    start_a = TreeNode(state=state)
    start_b = TreeNode(state=state)

    # Act
    vectorize_engine.search([start_a, start_b], add_noise=True)

    # Assert
    print(
        f" -> infer_batch calls={mapper.calls}, "
        f"start_a value={start_a.value_sum:.1f}, "
        f"start_b value={start_b.value_sum:.1f}"
    )

    assert mapper.calls == 1
    assert start_a.value_sum == pytest.approx(0.0), (
        "First node should get 1st batch item"
    )
    assert start_b.value_sum == pytest.approx(1.0), (
        "Second node should get 2nd batch item"
    )
    assert start_a.children and start_b.children
