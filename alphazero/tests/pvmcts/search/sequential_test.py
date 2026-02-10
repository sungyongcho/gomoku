import numpy as np
import pytest
import torch

from gomoku.core.gomoku import GameState
from gomoku.core.game_config import PLAYER_1, set_pos, xy_to_index
from gomoku.pvmcts.search.sequential import SequentialEngine
from gomoku.pvmcts.treenode import TreeNode


def test_start_node_expansion_and_dirichlet(
    sequential_engine: SequentialEngine,
) -> None:
    """
    Verify that Dirichlet noise is applied ONLY to the start node with debug prints.

    Test Steps:
    1. Run search from a fresh head.
    2. Check head expansion and visit counts.
    3. Verify head children's priors differ from raw softmax (due to noise).
    4. Verify non-head children (expanded during search) allow pure softmax (no noise).
    """
    print("\n\n[Test Start] Start-node Expansion & Dirichlet Noise Logic")

    # 1. Setup
    np.random.seed(0)
    torch.manual_seed(0)

    engine: SequentialEngine = sequential_engine
    state: GameState = engine.game.get_initial_state()
    head: TreeNode = TreeNode(state=state)

    print(
        f"  Configuration: Num Searches={engine.params.num_searches}, "
        f"Epsilon={engine.params.dirichlet_epsilon}"
    )

    # 2. Action: Run MCTS
    engine.search(head, add_noise=True)

    # 3. Assertion: Root Expansion
    print("\n  [Check 1] Start-node Expansion Stats")
    print(f"    - Start-node Visits: {head.visit_count}")
    print(f"    - Children Count: {len(head.children)}")

    assert head.children, "Start node must be expanded after search."
    assert head.visit_count == int(engine.params.num_searches), (
        "Start-node visit count must match the number of searches performed."
    )
    print("    -> PASS: Start node expanded and visit count correct.")

    # 4. Assertion: Root Dirichlet Noise
    print("\n  [Check 2] Start-node Dirichlet Noise Application")

    # Calculate 'Pure' Policy (Softmax without Noise)
    logits = engine.inference.logits.to(device=torch.device("cpu"))
    legal_mask = engine._legal_mask_tensor(state, device=logits.device)

    # Manual Masked Softmax computation
    masked_logits = torch.where(
        legal_mask,
        logits,
        torch.tensor(-float("inf"), device=logits.device, dtype=logits.dtype),
    )
    pure_policy = torch.softmax(masked_logits, dim=-1).numpy()

    # Extract priors from children mapping them to the correct index
    head_priors = np.zeros_like(pure_policy)
    for child in head.children.values():
        if child.action_taken:
            idx = child.action_taken[0] + child.action_taken[1] * engine.game.col_count
            head_priors[idx] = child.prior

    # Visual Comparison for Debugging
    # Index 0 is the dominant move (logit=100), Index 1 is a normal move (logit=1)
    idx_dom, idx_norm = 0, 1
    print(f"    - Comparison at Dominant Index {idx_dom}:")
    print(f"      Pure Softmax: {pure_policy[idx_dom]:.6f}")
    print(f"      Start Prior : {head_priors[idx_dom]:.6f} (Should differ)")
    print(f"    - Comparison at Normal Index {idx_norm}:")
    print(f"      Pure Softmax: {pure_policy[idx_norm]:.6f}")
    print(f"      Start Prior : {head_priors[idx_norm]:.6f} (Should differ)")

    legal_indices = np.where(legal_mask.numpy())[0]
    assert not np.allclose(head_priors[legal_indices], pure_policy[legal_indices]), (
        "Start-node policy must include Dirichlet noise (should differ from pure softmax)."
    )
    print("    -> PASS: Start-node priors contain noise (Values differ).")

    # 5. Assertion: Child Expansion (No Noise)
    print("\n  [Check 3] Non-Root Child Expansion (No Noise)")

    expanded_child = None
    for child in head.children.values():
        if not child.is_leaf:
            expanded_child = child
            break

    if expanded_child:
        print(f"    - Found Expanded Child Node: Action {expanded_child.action_taken}")

        child_state = expanded_child.state
        child_mask = engine._legal_mask_tensor(child_state, device=logits.device)

        # Calculate Expected Pure Policy for the child
        child_masked_logits = torch.where(
            child_mask,
            logits,
            torch.tensor(-float("inf"), device=logits.device, dtype=logits.dtype),
        )
        child_expected_policy = torch.softmax(child_masked_logits, dim=-1).numpy()

        # Extract priors from the expanded child's children
        child_priors = np.zeros_like(child_expected_policy)
        for grandchild in expanded_child.children.values():
            if grandchild.action_taken:
                idx = (
                    grandchild.action_taken[0]
                    + grandchild.action_taken[1] * engine.game.col_count
                )
                child_priors[idx] = grandchild.prior

        legal_indices_child = np.where(child_mask.numpy())[0]

        # Visual Comparison
        # Index 0 is likely legal and dominant
        print(f"    - Comparison at Dominant Index {idx_dom}:")
        print(f"      Pure Softmax: {child_expected_policy[idx_dom]:.6f}")
        print(f"      Child Prior : {child_priors[idx_dom]:.6f} (Should match)")

        assert np.allclose(
            child_priors[legal_indices_child],
            child_expected_policy[legal_indices_child],
            atol=1e-5,
        ), "Non-start node expansion must NOT include Dirichlet noise."
        print("    -> PASS: Child priors match pure softmax (No noise applied).")
    else:
        print("    [WARNING] No child node was expanded! Increase num_searches.")
        pytest.warns(
            UserWarning, match="No child node was expanded; check num_searches."
        )

    print("\n[Test Complete] All checks passed successfully.")


def test_simulation_loop_and_selection(
    sequential_engine: SequentialEngine,
) -> None:
    """
    Ensure the search loop runs num_searches times and drives UCB-based expansion.
    """
    print("\n[Simulation] Sequential search loop and selection trace")

    np.random.seed(0)
    torch.manual_seed(0)

    engine: SequentialEngine = sequential_engine
    head = TreeNode(state=engine.game.get_initial_state())

    print("[Action] Running engine.search on head.")
    engine.search(head, add_noise=True)

    expected_visits = int(engine.params.num_searches)
    print(f"[Check] head.visit_count={head.visit_count}, expected={expected_visits}")
    assert head.visit_count == expected_visits, (
        "Start-node visit count must match configured num_searches."
    )

    print(f"[Check] head children count={len(head.children)}")
    assert head.children, "Start node must be expanded during search."

    visited_children = sum(
        1 for child in head.children.values() if child.visit_count > 0
    )
    print(f"[Check] visited children={visited_children}")
    assert visited_children > 0, (
        "At least one child must be visited during simulations."
    )

    expanded_depth = sum(len(child.children) for child in head.children.values())
    print(f"[Check] grandchildren created={expanded_depth}")
    assert expanded_depth > 0, (
        "Selection/expansion must create deeper nodes beyond the head."
    )


def test_terminal_node_handling(sequential_engine: SequentialEngine) -> None:
    """
    Ensure terminal states are detected and backed up without extra inference.
    """
    print("\n[Terminal] Verifying immediate backup on terminal head.")

    engine: SequentialEngine = sequential_engine
    engine.params = engine.params.model_copy(update={"num_searches": 1})
    col_count = engine.game.col_count

    term_state = engine.game.get_initial_state()
    for x in range(5):
        set_pos(term_state.board, x, 0, PLAYER_1)
    term_state.last_move_idx = xy_to_index(4, 0, col_count)
    term_state.empty_count -= 5
    term_state.legal_indices_cache = None

    class CountingInference:
        """Count how many times infer is called."""

        def __init__(self, action_size: int):
            self.calls = 0
            self.logits = torch.ones(1, action_size, dtype=torch.float32)
            self.value = torch.tensor([[0.0]], dtype=torch.float32)

        def infer(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            self.calls += 1
            return self.logits, self.value

    counter = CountingInference(engine.game.action_size)
    engine.inference = counter

    head = TreeNode(state=term_state)
    engine.search(head, add_noise=True)

    print(
        f"[Check] infer calls={counter.calls}, "
        f"head visits={head.visit_count}, children={len(head.children)}"
    )
    assert counter.calls == 0, "Terminal detection must skip inference."
    assert head.visit_count == 1, "Terminal backup must still bump visit count."
    assert head.value_sum == pytest.approx(-1.0), (
        "Winning last move should be recorded as loss (-1.0) for current player."
    )
    assert not head.children, "Terminal nodes must not expand children."


def test_draw_handling_on_full_board(sequential_engine: SequentialEngine) -> None:
    """
    Verify behavior when the board is completely full (Draw).
    """
    print("\n[Draw] Simulating full board (Draw) scenario.")

    engine: SequentialEngine = sequential_engine
    engine.params = engine.params.model_copy(update={"num_searches": 1})

    state = engine.game.get_initial_state()
    state.empty_count = 0
    state.last_move_idx = -1  # Skip win check; full board with no winner.
    state.legal_indices_cache = None

    head = TreeNode(state=state)
    engine.search(head, add_noise=True)

    print(
        f"[Check] visits={head.visit_count}, children={len(head.children)}, "
        f"value_sum={head.value_sum}"
    )
    assert head.visit_count == 1, "Draw head must be visited once."
    assert not head.children, "Full board must no expand children."
    assert head.value_sum == pytest.approx(0.0), "Draw should backup 0.0."


def test_single_legal_move_processing(sequential_engine: SequentialEngine) -> None:
    """
    Ensure search proceeds correctly when only one legal move exists.
    """
    print("\n[Single Move] Verifying handling of single legal action.")

    engine: SequentialEngine = sequential_engine
    state = engine.game.get_initial_state()

    # Fill the board except one cell to leave a single legal move at (0, 0)
    for y in range(engine.game.row_count):
        for x in range(engine.game.col_count):
            if x == 0 and y == 0:
                continue
            set_pos(state.board, x, y, PLAYER_1)
    state.empty_count = 1
    state.last_move_idx = -1
    state.legal_indices_cache = None

    head = TreeNode(state=state)
    engine.search(head, add_noise=True)

    print(
        f"[Check] head visits={head.visit_count}, children={len(head.children)} "
        f"single child visits="
        f"{list(head.children.values())[0].visit_count if head.children else 'N/A'}"
    )
    assert len(head.children) == 1, "Exactly one child must be expanded."
    assert head.visit_count == int(engine.params.num_searches), (
        "Start-node visits must match configured num_searches."
    )
    only_child = next(iter(head.children.values()))
    assert only_child.action_taken == (0, 0), "Single legal move must be selected."
    assert only_child.visit_count > 0, "Single child must be visited."
    assert only_child.prior == pytest.approx(1.0), (
        "Single move must have prior 1.0 (no Dirichlet noise)."
    )


def test_batch_size_ignorance(sequential_engine: SequentialEngine) -> None:
    """
    Ensure SequentialEngine ignores batch_size and performs single inference calls.
    """
    print(
        "\n[Batch Ignore] Verifying sequential mode uses single infer, not infer_batch."
    )

    engine: SequentialEngine = sequential_engine
    engine.params = engine.params.model_copy(
        update={"num_searches": 1, "batch_infer_size": 8}  # ignored in sequential
    )

    class CountingInference:
        """Track infer vs infer_batch usage."""

        def __init__(self, action_size: int):
            self.action_size = action_size
            self.infer_calls = 0
            self.infer_batch_calls = 0

        def infer(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            self.infer_calls += 1
            # Mimic LocalInference: single-state output without batch dim.
            logits = torch.ones(self.action_size, dtype=torch.float32)
            value = torch.tensor([0.0], dtype=torch.float32)
            return logits, value

        def infer_batch(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            self.infer_batch_calls += 1
            raise AssertionError("infer_batch should not be called in sequential mode.")

    counter = CountingInference(engine.game.action_size)
    engine.inference = counter

    head = TreeNode(state=engine.game.get_initial_state())
    engine.search(head, add_noise=True)

    print(
        f"[Check] infer_calls={counter.infer_calls}, "
        f"infer_batch_calls={counter.infer_batch_calls}"
    )
    assert counter.infer_calls == 1, "Sequential engine should call infer once."
    assert counter.infer_batch_calls == 0, (
        "infer_batch must never be invoked in sequential mode."
    )


def test_reproducibility_with_fixed_seed(
    sequential_engine: SequentialEngine,
) -> None:
    """
    Verify that fixed seeds produce identical visit distributions.
    """
    print("\n[Determinism] Checking reproducibility with fixed seeds.")

    engine: SequentialEngine = sequential_engine
    col_count = engine.game.col_count

    def run_once() -> np.ndarray:
        np.random.seed(0)
        torch.manual_seed(0)
        head = TreeNode(state=engine.game.get_initial_state())
        engine.search(head, add_noise=True)
        visits_vec = np.zeros(engine.game.action_size, dtype=np.int64)
        for action, child in head.children.items():
            idx = xy_to_index(action[0], action[1], col_count)
            visits_vec[idx] = child.visit_count
        return visits_vec

    visits_1 = run_once()
    visits_2 = run_once()

    print("[Check] Comparing visit vectors from two seeded runs.")
    assert np.array_equal(visits_1, visits_2), (
        "Visit distribution must be identical when seeds are fixed."
    )


def test_search_result_policy_distribution(
    sequential_engine: SequentialEngine,
) -> None:
    """
    Verify that a biased policy from inference steers visit counts accordingly.

    Test Conditions:
    - Dirichlet noise disabled (epsilon=0).
    - Logits heavily biased towards index 0.
    - Expected: Index 0 receives the vast majority of visits.
    """
    print("\n[Policy] Checking visit distribution follows model logits.")

    engine: SequentialEngine = sequential_engine
    engine.params = engine.params.model_copy(
        update={"num_searches": 15, "dirichlet_epsilon": 0.0}
    )

    # Force biased logits toward index 0 for test self-containment.
    biased_logits = torch.zeros(engine.game.action_size, dtype=torch.float32)
    biased_logits[0] = 100.0
    engine.inference.logits = biased_logits

    np.random.seed(0)
    torch.manual_seed(0)

    head = TreeNode(state=engine.game.get_initial_state())
    engine.search(head, add_noise=True)

    visits_vec = np.zeros(engine.game.action_size, dtype=np.int64)
    for action, child in head.children.items():
        idx = xy_to_index(action[0], action[1], engine.game.col_count)
        visits_vec[idx] = child.visit_count

    top_idx = int(visits_vec.argmax())
    top_visits = int(visits_vec.max())
    if visits_vec.size > 1:
        flat_sorted = np.sort(visits_vec.flatten())
        second_visits = int(flat_sorted[-2])
    else:
        second_visits = 0

    print(
        f"[Check] top_idx={top_idx}, top_visits={top_visits}, "
        f"second_visits={second_visits}"
    )

    assert top_idx == 0, "Highest-logit action (0) must receive most visits."
    assert top_visits > 0, "Dominant action must be visited."
    assert top_visits > second_visits, (
        "Dominant action visits must strictly exceed runner-up."
    )
