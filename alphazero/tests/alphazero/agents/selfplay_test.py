import numpy as np
import pytest

from gomoku.alphazero.agent import AlphaZeroAgent
from gomoku.alphazero.runners.selfplay import SelfPlayRunner
from gomoku.alphazero.types import GameRecord, xy_to_action
from gomoku.core.gomoku import Gomoku
from gomoku.utils.config.loader import RootConfig


def test_selfplay_runs_one_game(
    sequential_components: tuple[RootConfig, Gomoku, AlphaZeroAgent, SelfPlayRunner],
) -> None:
    """Ensure one sequential self-play game finishes and record fields are consistent."""
    _, _, agent, selfplay = sequential_components

    record = selfplay.play_one_game(agent, temperature=1.0, add_noise=False)

    print(f"[selfplay] moves={len(record.moves)}, last_outcome={record.outcomes[-1]}")
    assert isinstance(record, GameRecord)
    assert len(record.moves) > 0

    n_moves = len(record.moves)
    assert len(record.policies) == n_moves
    assert len(record.outcomes) == n_moves
    assert len(record.players) == n_moves
    assert len(record.states_raw) == n_moves

    assert np.all(np.abs(record.outcomes) <= 1.0)
    last_outcome = record.outcomes[-1]
    if last_outcome != 0.0:
        assert last_outcome == 1.0


def test_policy_sum_and_nan_free(
    sequential_components: tuple[RootConfig, Gomoku, AlphaZeroAgent, SelfPlayRunner],
) -> None:
    """Verify policies are normalized (~1.0) and finite (no NaN/Inf)."""
    _, _, agent, selfplay = sequential_components

    record = selfplay.play_one_game(agent, add_noise=True)
    policies = record.policies

    print(f"[policy] sample sums={policies.sum(axis=1)[:3]}")
    assert np.isfinite(policies).all(), "Policy contains NaN or Inf"

    sums = policies.sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-3, err_msg="Policy sum is not 1.0")


def test_outcome_sign_by_player(
    sequential_components: tuple[RootConfig, Gomoku, AlphaZeroAgent, SelfPlayRunner],
) -> None:
    """Outcomes follow the turn player's perspective (+1 winner, -1 loser)."""
    _, _, agent, selfplay = sequential_components

    record = selfplay.play_one_game(agent)
    outcomes = record.outcomes
    players = record.players
    print(f"[outcome] sample={outcomes[:5]}, players={players[:5]}")

    if np.all(outcomes == 0):
        return

    winner_indices = np.where(outcomes > 0)[0]
    assert winner_indices.size > 0
    winner_player = players[winner_indices[0]]

    for o, p in zip(outcomes, players, strict=True):
        if o > 0:
            assert p == winner_player
        elif o < 0:
            assert p != winner_player
        assert abs(o) <= 1.0


def test_players_moves_length_match(
    sequential_components: tuple[RootConfig, Gomoku, AlphaZeroAgent, SelfPlayRunner],
) -> None:
    """Players, moves, outcomes, states lengths match."""
    _, _, agent, selfplay = sequential_components
    record = selfplay.play_one_game(agent)

    n = len(record.moves)
    print(
        f"[lengths] moves={n}, players={len(record.players)}, outcomes={len(record.outcomes)}"
    )
    assert len(record.players) == n
    assert len(record.outcomes) == n
    assert len(record.states_raw) == n


def test_noise_toggle_changes_pi(
    sequential_components: tuple[RootConfig, Gomoku, AlphaZeroAgent, SelfPlayRunner],
) -> None:
    """Dirichlet noise toggle should alter the policy distribution."""
    cfg, game, agent, _ = sequential_components

    if cfg.mcts.dirichlet_epsilon <= 0:
        pytest.skip("dirichlet_epsilon=0; noise disabled in config.")

    state = game.get_initial_state()

    np.random.seed(42)
    agent.reset()
    pi_no_noise = agent.get_action_probs(state, temperature=1.0, add_noise=False)

    np.random.seed(42)
    agent.reset()
    pi_with_noise = agent.get_action_probs(state, temperature=1.0, add_noise=True)

    print(f"[noise] l1 diff={(np.abs(pi_no_noise - pi_with_noise).sum()):.6f}")
    assert not np.allclose(pi_no_noise, pi_with_noise)


def test_agent_policy_masking_logic(
    sequential_components: tuple[RootConfig, Gomoku, AlphaZeroAgent, SelfPlayRunner],
) -> None:
    """Illegal actions must be masked to zero probability."""
    _, game, agent, _ = sequential_components

    state = game.get_initial_state()
    cx, cy = game.col_count // 2, game.row_count // 2
    state = game.get_next_state(state, (cx, cy), state.next_player)
    occupied_idx = xy_to_action(cx, cy, game.col_count)

    pi = agent.get_action_probs(state, temperature=1.0, add_noise=False)

    print(
        f"[mask] occupied_idx={occupied_idx}, prob={pi[occupied_idx]:.6f}, sum={pi.sum():.6f}"
    )
    assert pi[occupied_idx] == 0.0
    assert np.isclose(pi.sum(), 1.0, atol=1e-5)


def test_action_and_policy_shapes(
    sequential_components: tuple[RootConfig, Gomoku, AlphaZeroAgent, SelfPlayRunner],
) -> None:
    """Policies have shape (T, action_size) and moves within range."""
    _, game, agent, selfplay = sequential_components

    record = selfplay.play_one_game(agent)

    expected_shape = (len(record.moves), game.action_size)
    print(f"[shape] policies shape={record.policies.shape}, expected={expected_shape}")
    assert record.policies.shape == expected_shape
    assert np.all(record.moves >= 0)
    assert np.all(record.moves < game.action_size)


def test_temperature_near_zero_yields_one_hot(
    sequential_components: tuple[RootConfig, Gomoku, AlphaZeroAgent, SelfPlayRunner],
) -> None:
    """Very low temperature produces near one-hot policy."""
    _, game, agent, _ = sequential_components
    state = game.get_initial_state()

    pi = agent.get_action_probs(state, temperature=1e-3, add_noise=False)
    print(f"[temp≈0] max={pi.max():.6f}, sum={pi.sum():.6f}")
    assert np.isclose(pi.sum(), 1.0, atol=1e-3)
    assert pi.max() > 0.9


def test_temperature_one_preserves_distribution(
    sequential_components: tuple[RootConfig, Gomoku, AlphaZeroAgent, SelfPlayRunner],
) -> None:
    """Temperature=1.0 should keep distribution stable for deterministic runs."""
    _, game, agent, _ = sequential_components
    state = game.get_initial_state()

    pi1 = agent.get_action_probs(state, temperature=1.0, add_noise=False)
    pi2 = agent.get_action_probs(state, temperature=1.0, add_noise=False)

    diff = np.abs(pi1 - pi2).sum()
    print(f"[temp=1] sum1={pi1.sum():.6f}, sum2={pi2.sum():.6f}, l1 diff={diff:.6f}")
    np.testing.assert_allclose(pi1.sum(), 1.0, atol=1e-3)
    np.testing.assert_allclose(pi2.sum(), 1.0, atol=1e-3)
    assert np.allclose(pi1, pi2)


def test_deterministic_when_noise_off(
    sequential_components: tuple[RootConfig, Gomoku, AlphaZeroAgent, SelfPlayRunner],
) -> None:
    """With noise off and fixed seeds, policy should be deterministic."""
    _, game, agent, _ = sequential_components
    state = game.get_initial_state()

    np.random.seed(42)
    agent.reset()
    pi1 = agent.get_action_probs(state, temperature=1.0, add_noise=False)

    np.random.seed(42)
    agent.reset()
    pi2 = agent.get_action_probs(state, temperature=1.0, add_noise=False)

    print(f"[noise_off] l1 diff={np.abs(pi1 - pi2).sum():.6f}")
    np.testing.assert_array_equal(pi1, pi2)


def test_pi_differs_when_noise_on(
    sequential_components: tuple[RootConfig, Gomoku, AlphaZeroAgent, SelfPlayRunner],
) -> None:
    """With noise on, policies should differ."""
    cfg, game, agent, _ = sequential_components

    if cfg.mcts.dirichlet_epsilon <= 0:
        pytest.skip("dirichlet_epsilon=0; noise disabled in config.")

    state = game.get_initial_state()

    np.random.seed(42)
    agent.reset()
    pi_no_noise = agent.get_action_probs(state, temperature=1.0, add_noise=False)

    np.random.seed(42)
    agent.reset()
    pi_with_noise = agent.get_action_probs(state, temperature=1.0, add_noise=True)

    print(f"[noise_on] l1 diff={np.abs(pi_no_noise - pi_with_noise).sum():.6f}")
    assert not np.allclose(pi_no_noise, pi_with_noise)


def test_update_root_reuses_child_or_resets(
    sequential_components: tuple[RootConfig, Gomoku, AlphaZeroAgent, SelfPlayRunner],
) -> None:
    """Valid action moves root to child; invalid action resets root."""
    _, game, agent, _ = sequential_components
    state = game.get_initial_state()

    agent.get_action_probs(state, temperature=1.0, add_noise=False)
    root = agent.root
    assert root is not None
    assert root.children

    action_xy = next(iter(root.children.keys()))
    action_idx = xy_to_action(action_xy[0], action_xy[1], game.col_count)
    agent.update_root(action_idx)
    assert agent.root is root.children[action_xy]
    assert agent.root.parent is None

    invalid_action = game.action_size + 1
    agent.update_root(invalid_action)
    print(f"[update_root] valid_action={action_idx}, invalid_action={invalid_action}")
    assert agent.root is None


def test_root_resets_on_state_mismatch(
    sequential_components: tuple[RootConfig, Gomoku, AlphaZeroAgent, SelfPlayRunner],
) -> None:
    """Root should be refreshed when state does not match cached root."""
    _, game, agent, _ = sequential_components
    state = game.get_initial_state()

    agent.get_action_probs(state, temperature=1.0, add_noise=False)
    old_root = agent.root
    assert old_root is not None

    action = (0, 0)
    next_state = game.get_next_state(state, action, state.next_player)
    agent.get_action_probs(next_state, temperature=1.0, add_noise=False)

    print("[state_mismatch] root refreshed")
    assert agent.root is not old_root
    assert agent.root is not None
    assert agent.root.state is next_state
