import numpy as np
import torch

from gomoku.core.game_config import NUM_LINES
from gomoku.core.gomoku import Gomoku
from gomoku.inference.local import LocalInference
from gomoku.model.policy_value_net import PolicyValueNet
from gomoku.pvmcts.search.mp import MultiprocessEngine
from gomoku.pvmcts.search.sequential import SequentialEngine
from gomoku.pvmcts.search.vectorize import VectorizeEngine
from gomoku.pvmcts.treenode import TreeNode
from gomoku.utils.config.loader import BoardConfig, MctsConfig


def _make_game_and_params(num_searches: float = 3.0) -> tuple[Gomoku, MctsConfig]:
    board_config = BoardConfig(
        num_lines=NUM_LINES,
        gomoku_goal=5,
        capture_goal=5,
        enable_doublethree=True,
        enable_capture=True,
    )
    game = Gomoku(board_config)
    params = MctsConfig(
        C=1.4,
        num_searches=num_searches,
        exploration_turns=1,
        dirichlet_epsilon=0.0,  # deterministic
        dirichlet_alpha=0.3,
        batch_infer_size=2,
        max_batch_wait_ms=0,
        min_batch_size=1,
    )
    return game, params


def _make_deterministic_inference(game: Gomoku) -> LocalInference:
    # Tiny model that returns zeros (policy uniform, value zero)
    from gomoku.model.model_helpers import calc_num_planes
    from gomoku.utils.config.loader import ModelConfig

    model_cfg = ModelConfig(
        num_planes=calc_num_planes(game.history_length),
        num_hidden=16,
        num_resblocks=1,
        policy_channels=2,
        value_channels=1,
    )
    model = PolicyValueNet(game, model_cfg, device="cpu")
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    return LocalInference(model)


def _run_engine(engine, roots):
    engine.search(roots, add_noise=False)
    return [engine.get_search_result(r)[0] for r in roots]


def test_engines_produce_same_policy_for_same_roots() -> None:
    """Deterministic cross-engine consistency on identical roots."""
    game, params = _make_game_and_params(num_searches=2.0)
    inference = _make_deterministic_inference(game)

    roots = [TreeNode(game.get_initial_state()) for _ in range(2)]

    # Distributed engines (MP/Ray) expect IPC/actor-based inference clients;
    # this unit test sticks to in-process engines for determinism.
    engines = [
        SequentialEngine(game, params, inference),
        VectorizeEngine(game, params, inference),
        MultiprocessEngine(game, params, inference),
    ]

    policies = []
    for eng in engines:
        # Fresh copy of roots per engine to avoid reuse artifacts
        local_roots = [TreeNode(r.state) for r in roots]
        outputs = _run_engine(eng, local_roots)
        policies.append(outputs)

    # Compare all engine policy outputs pairwise
    base = policies[0]
    for other in policies[1:]:
        for p_base, p_other in zip(base, other, strict=True):
            assert np.allclose(p_base, p_other, atol=1e-6)


def test_vectorize_runner_skips_finished_slots() -> None:
    """Ensure Runner-level contract: finished slots are not re-sent to engine."""
    game, params = _make_game_and_params(num_searches=1.0)
    inference = _make_deterministic_inference(game)
    engine = VectorizeEngine(game, params, inference)

    finished_root = TreeNode(game.get_initial_state())
    finished_root.visit_count = int(params.num_searches)
    active_root = TreeNode(game.get_initial_state())

    engine.search([finished_root, active_root], add_noise=False)

    # Finished root should remain at visit_count set, active should reach target
    assert finished_root.visit_count == int(params.num_searches)
    assert active_root.visit_count == int(params.num_searches)
