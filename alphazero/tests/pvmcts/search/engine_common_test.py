import numpy as np
import pytest
import torch

from gomoku.core.game_config import PLAYER_1, PLAYER_2, set_pos, xy_to_index
from gomoku.pvmcts.search.engine import _safe_dirichlet_noise
from gomoku.pvmcts.search.sequential import SequentialEngine
from gomoku.pvmcts.treenode import TreeNode


def test_safe_dirichlet_noise_mechanics() -> None:
    """
    Verify Dirichlet noise stays on legal moves and preserves a valid probability
    distribution.
    """
    np.random.seed(0)

    policy = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.0], dtype=torch.float32)
    legal_mask = torch.tensor([True, False, True, True, False], dtype=torch.bool)

    noisy = _safe_dirichlet_noise(
        policy,
        legal_mask,
        epsilon=0.25,
        alpha=0.3,
    )

    assert torch.all(noisy[~legal_mask] == 0.0), (
        "Illegal moves must remain zero probability."
    )
    assert noisy.sum().item() == pytest.approx(1.0), "Sum must stay 1.0."
    assert torch.all(noisy >= 0), "Probabilities must be non-negative."
    assert not torch.allclose(noisy, policy), (
        "Noise should modify the policy distribution."
    )


def test_safe_dirichlet_noise_edge_cases() -> None:
    """Verify that noise is skipped when epsilon is 0 or legal moves are sparse."""
    policy = torch.tensor([0.2, 0.8], dtype=torch.float32)

    legal_mask = torch.tensor([True, True], dtype=torch.bool)
    noisy_zero_eps = _safe_dirichlet_noise(policy, legal_mask, epsilon=0.0, alpha=0.3)

    assert torch.equal(noisy_zero_eps, policy), (
        "Original policy must be returned when epsilon is 0."
    )

    legal_mask_single = torch.tensor([True, False], dtype=torch.bool)
    noisy_single = _safe_dirichlet_noise(
        policy, legal_mask_single, epsilon=0.5, alpha=0.3
    )

    assert torch.equal(noisy_single, policy), (
        "Original policy must be returned when only one legal move exists."
    )

    legal_mask_none = torch.tensor([False, False], dtype=torch.bool)
    noisy_none = _safe_dirichlet_noise(policy, legal_mask_none, epsilon=0.5, alpha=0.3)

    assert torch.equal(noisy_none, policy), (
        "Original policy must be returned when no legal moves exist."
    )


def test_masked_policy_consistency(
    sequential_engine: SequentialEngine,
) -> None:
    """
    Masking should zero out illegal logits and produce a valid probability distribution.
    """
    logits = torch.tensor([5.0, 1000.0, -2.0, 3.0], dtype=torch.float32)
    legal_mask = torch.tensor([True, False, True, True], dtype=torch.bool)

    policy = sequential_engine._masked_policy_from_logits(logits, legal_mask)

    assert torch.all(policy[~legal_mask] == 0.0), "Illegal moves must be zeroed."
    assert policy.sum().item() == pytest.approx(1.0), "Policy must sum to 1.0."


def test_masked_policy_nan_prevention(
    sequential_engine: SequentialEngine,
) -> None:
    """
    Verify that masking handles extreme or all-illegal cases without producing NaNs.

    This test enforces numerical stability:
    1. If all moves are illegal, the engine should return a zero vector (safe failure)
       instead of NaNs, or handle it gracefully.
    2. If logits are extremely large (e.g., 1e9), Softmax should effectively be
       one-hot/argmax without overflowing to NaN.
    """
    # Case 1: All moves illegal
    # Note: Standard torch.softmax([-inf, -inf]) returns [nan, nan].
    # This assertion ensures the engine explicitly handles this edge case.
    logits = torch.tensor([1000.0, -1000.0], dtype=torch.float32)
    legal_mask_none = torch.tensor([False, False], dtype=torch.bool)

    policy_none = sequential_engine._masked_policy_from_logits(logits, legal_mask_none)

    # Expectation: No NaNs, and preferably a zero vector since no move is legal.
    assert not torch.isnan(policy_none).any(), (
        "Policy must not contain NaNs even if all moves are illegal."
    )
    assert torch.all(policy_none == 0.0), (
        "Policy should be a zero vector when no moves are legal."
    )

    # Case 2: Extreme logits with mixed legality
    # 1e9 is large enough to cause exp(x) overflow if not handled by stable softmax.
    logits_extreme = torch.tensor([1e9, -1e9, 0.0], dtype=torch.float32)
    legal_mask = torch.tensor([True, False, True], dtype=torch.bool)

    policy_extreme = sequential_engine._masked_policy_from_logits(
        logits_extreme, legal_mask
    )

    # 1. Illegal move (-1e9 logit, but masked) must be exactly 0.0
    assert policy_extreme[1] == 0.0, "Masked index must be zero."

    # 2. Extreme value (1e9) should dominate, making probability ~1.0
    #    The 0.0 logit (index 2) becomes negligible compared to 1e9.
    assert policy_extreme[0] == pytest.approx(1.0), "Extreme logit should dominate."
    assert policy_extreme[2] == pytest.approx(0.0), "Smaller logit should vanish."

    # 3. Stability check
    assert not torch.isnan(policy_extreme).any(), (
        "Extreme logits should not produce NaNs."
    )
    assert policy_extreme.sum().item() == pytest.approx(1.0), "Sum must be 1.0."


def test_legal_mask_tensor_mapping(
    sequential_engine: SequentialEngine,
) -> None:
    """Verify that board legal moves map correctly to a boolean tensor mask."""
    # 1. Setup: Construct a simple state with specific occupied cells
    state = sequential_engine.game.get_initial_state()

    # Manually occupy two cells via game transitions
    state = sequential_engine.game.get_next_state(state, (0, 0), state.next_player)
    state = sequential_engine.game.get_next_state(state, (1, 0), state.next_player)

    # CRITICAL: Clear cache to force _legal_indices to scan the modified board
    state.legal_indices_cache = None

    # 2. Action
    mask = sequential_engine._legal_mask_tensor(state, device=torch.device("cpu"))

    # 3. Assertions
    assert mask.dtype == torch.bool, "Mask must be boolean type."
    assert mask.shape[0] == sequential_engine.game.action_size

    # Check Illegal Moves (Indices 0 and 1)
    # Instead of looping, check the slice directly
    assert not torch.any(mask[0:2]), "Occupied cells must be marked illegal."

    # Check Legal Moves (Indices 2 to End)
    assert torch.all(mask[2:]), "Empty cells must be marked legal."


def test_evaluate_terminal_logic(
    sequential_engine: SequentialEngine,
) -> None:
    """
    Verify terminal evaluation logic for both root and child nodes with debug prints.

    Run with `pytest -s` to see the output.
    """
    game = sequential_engine.game
    print("\n\n[Test Start] Evaluate Terminal Logic")

    # -------------------------------------------------------------------------
    # Case 1: Root Node - Already Lost
    # -------------------------------------------------------------------------
    print("\n--- Case 1: Root Node (Already Lost) ---")
    print("Scenario: Opponent just connected 5 stones. It is now my turn.")

    root_state = game.get_initial_state()
    # Manually place 5 stones for PLAYER_1 (Black) in a row: (0,0) to (4,0)
    for x in range(5):
        set_pos(root_state.board, x, 0, PLAYER_1)

    # Set metadata to simulate "Player 1 just moved at (4,0)"
    root_state.last_move_idx = xy_to_index(4, 0, game.col_count)
    root_state.empty_count -= 5
    root_state.next_player = PLAYER_2  # Now it's Player 2's turn
    root_state.legal_indices_cache = None

    root_node = TreeNode(state=root_state)
    val_root, is_term_root = sequential_engine._evaluate_terminal(root_node)

    print(f"  -> Result: Value={val_root}, Terminal={is_term_root}")

    assert is_term_root, "Root node with a winning line must be terminal."
    assert val_root == -1.0, "Root value must be -1.0 (Loss) for the current player."
    print("  -> Assertion Passed: Root identified as Loss.")

    # -------------------------------------------------------------------------
    # Case 2: Root Node - Draw
    # -------------------------------------------------------------------------
    print("\n--- Case 2: Root Node (Draw) ---")
    print("Scenario: Board is full, no winner found.")

    draw_state = game.get_initial_state()
    draw_state.empty_count = 0  # Board full
    draw_state.last_move_idx = -1  # Skip win check for safety
    draw_state.legal_indices_cache = None

    draw_node = TreeNode(state=draw_state)
    val_draw, is_term_draw = sequential_engine._evaluate_terminal(draw_node)

    print(f"  -> Result: Value={val_draw}, Terminal={is_term_draw}")

    assert is_term_draw, "Full board state must be terminal."
    assert val_draw == 0.0, "Full board must be 0.0 (Draw)."
    print("  -> Assertion Passed: Root identified as Draw.")

    # -------------------------------------------------------------------------
    # Case 3: Child Node - Just Moved & Won
    # -------------------------------------------------------------------------
    print("\n--- Case 3: Child Node (Win by Move) ---")
    print("Scenario: Simulated move (4,0) completed a line.")

    child_state = game.get_initial_state()
    for x in range(5):
        set_pos(child_state.board, x, 0, PLAYER_1)

    child_state.last_move_idx = xy_to_index(4, 0, game.col_count)
    child_state.empty_count -= 5
    child_state.next_player = PLAYER_2
    child_state.legal_indices_cache = None

    # Child node represents the state AFTER taking action (4,0)
    child_node = TreeNode(state=child_state, action_taken=(4, 0))

    val_child, is_term_child = sequential_engine._evaluate_terminal(child_node)

    print(f"  -> Result: Value={val_child}, Terminal={is_term_child}")
    print("     (Note: Value is inverted to -1.0 for the next player)")

    assert is_term_child, "Child node completing a line must be terminal."
    assert val_child == -1.0, (
        "Terminal value should be negated (-1.0) indicating loss for the next player."
    )
    print("  -> Assertion Passed: Child identified as Win (Value Inverted).")
    print("\n[Test Complete] All cases verified successfully.")


def test_encode_state_shape_and_device(
    sequential_engine: SequentialEngine,
) -> None:
    """Verify that the encoded state tensor matches the expected shape, device, and dtype."""
    # 1. Setup
    state = sequential_engine.game.get_initial_state()

    # Ground Truth: Get the shape directly from the numpy-based game logic
    # get_encoded_state returns (Batch, C, H, W), so indexing [0] gives (C, H, W)
    expected_np = sequential_engine.game.get_encoded_state([state])[0]
    expected_shape = expected_np.shape

    # 2. Action
    # Use the actual method name '_encode_state' (not _encode_state_to_tensor)
    tensor = sequential_engine._encode_state_to_tensor(
        state, dtype=torch.float32, device=torch.device("cpu")
    )

    # 3. Assertions
    # Shape Check: e.g., (C, 15, 15)
    assert tensor.shape == expected_shape, (
        f"Tensor shape {tensor.shape} must match the numpy encoding shape {expected_shape}."
    )

    # Device Check
    assert tensor.device.type == "cpu", "Tensor must remain on CPU as requested."

    # Dtype Check
    assert tensor.dtype == torch.float32, "Tensor must be converted to float32."

    # Value Check (Optional but recommended sanity check)
    # Ensure the values are actually copied correctly
    assert torch.allclose(tensor, torch.from_numpy(expected_np).float()), (
        "Tensor values must match the numpy source data."
    )


def test_backup_value_propagation_and_sign_flip(
    sequential_engine: SequentialEngine,
) -> None:
    """
    Verify backup propagates visits and flips value signs at each depth.
    """
    print("\n[Backup] Verifying value propagation and sign flipping.")

    root = TreeNode(state=sequential_engine.game.get_initial_state())
    child_state = sequential_engine.game.get_initial_state()
    child = TreeNode(state=child_state, parent=root, action_taken=(0, 0), prior=0.5)
    leaf_state = sequential_engine.game.get_initial_state()
    leaf = TreeNode(state=leaf_state, parent=child, action_taken=(1, 0), prior=0.5)

    value = 0.8
    print(f"[Action] Backing up value={value} from leaf.")
    leaf.backup(value)

    print(
        f"[Check] Visits -> leaf:{leaf.visit_count}, "
        f"child:{child.visit_count}, root:{root.visit_count}"
    )
    assert leaf.visit_count == 1, "Leaf visit must increment by 1."
    assert child.visit_count == 1, "Child visit must increment by 1."
    assert root.visit_count == 1, "Root visit must increment by 1."

    print(
        f"[Check] Value sums -> leaf:{leaf.value_sum}, "
        f"child:{child.value_sum}, root:{root.value_sum}"
    )
    assert leaf.value_sum == pytest.approx(value), "Leaf stores original value."
    assert child.value_sum == pytest.approx(-value), (
        "Child must accumulate flipped sign (-value)."
    )
    assert root.value_sum == pytest.approx(value), (
        "Root accumulates flipped again (+value)."
    )
