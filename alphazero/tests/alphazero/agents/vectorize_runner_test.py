from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import MagicMock, patch

import numpy as np

from gomoku.alphazero.agent import AlphaZeroAgent
from gomoku.alphazero.runners.vectorize_runner import VectorizeRunner
from gomoku.core.gomoku import Gomoku
from gomoku.inference.local import LocalInference
from gomoku.utils.config.loader import RootConfig


def test_run_batch_games_returns_correct_count(
    vectorize_components: tuple[
        RootConfig, Gomoku, AlphaZeroAgent, VectorizeRunner, LocalInference
    ],
) -> None:
    """배치 크기와 target_games에 맞게 기록이 수집되는지 확인한다."""
    _, _, agent, runner, _ = vectorize_components

    records = runner.play_batch_games(agent, batch_size=4, add_noise=True)
    print(f"[vectorize] batch_size=4, returned_records={len(records)}")

    assert len(records) == 4


def test_noise_flag_only_on_turn_zero(
    vectorize_components: tuple[
        RootConfig, Gomoku, AlphaZeroAgent, VectorizeRunner, LocalInference
    ],
) -> None:
    """리필 이후에도 turn 0 슬롯만 노이즈가 켜지는지 검증한다."""
    _, game, agent, runner, _ = vectorize_components

    noise_history: list[list[bool]] = []

    def _spy(
        states_raw: Sequence[object],
        temperature: float | Sequence[float],
        add_noise_flags: bool | Sequence[bool],
        active_indices: list[int] | None = None,
    ):
        flags = (
            [bool(flag) for flag in add_noise_flags]
            if isinstance(add_noise_flags, (list, tuple))
            else [bool(add_noise_flags)] * len(states_raw)
        )
        noise_history.append(flags)
        policy = np.full(game.action_size, 1.0 / game.action_size, dtype=np.float32)
        return [policy.copy() for _ in states_raw]

    def _stub_update_root_batch(actions: Sequence[int]) -> None:
        return

    call_counter = {"n": 0}

    def _fake_get_value_and_terminated(state, move):
        call_counter["n"] += 1
        # 첫 호출(슬롯 0)에서만 종료로 보고 리필을 유도한다.
        is_terminal = call_counter["n"] % 2 == 1
        return (1.0, True) if is_terminal else (0.0, False)

    agent.get_action_probs_batch = MagicMock(side_effect=_spy)
    agent.update_root_batch = MagicMock(side_effect=_stub_update_root_batch)
    agent.reset_game = MagicMock(side_effect=lambda idx: None)
    runner.game.get_next_state = MagicMock(side_effect=lambda s, a, p: s)
    runner.game.get_value_and_terminated = MagicMock(
        side_effect=_fake_get_value_and_terminated
    )

    runner.play_batch_games(
        agent,
        batch_size=2,
        add_noise=True,
        target_games=3,
    )

    print(f"[noise_history] {noise_history}")
    assert len(noise_history) >= 2
    assert noise_history[0] == [True, True]
    assert [True, False] in noise_history


def test_finished_games_are_removed_correctly(
    vectorize_components: tuple[
        RootConfig, Gomoku, AlphaZeroAgent, VectorizeRunner, LocalInference
    ],
) -> None:
    """서로 다른 턴에 끝나는 게임이 누락 없이 수집되는지 검증한다."""
    _, game, agent, runner, _ = vectorize_components

    termination_calls = {"n": 0}
    reset_calls: list[int] = []

    def _stub_get_action_probs(
        states_raw: Sequence[object],
        temperature: float | Sequence[float],
        add_noise_flags: bool | Sequence[bool],
        active_indices: list[int] | None = None,
    ):
        policy = np.full(game.action_size, 1.0 / game.action_size, dtype=np.float32)
        return [policy.copy() for _ in states_raw]

    def _stub_update_root_batch(actions: Sequence[int]) -> None:
        return

    def _stub_reset_game(idx: int) -> None:
        reset_calls.append(idx)

    def _fake_get_value_and_terminated(state, move):
        termination_calls["n"] += 1
        # 5의 배수에서만 종료시켜 배치 크기(3)와 서로소 패턴을 만든다.
        is_terminal = termination_calls["n"] % 5 == 0
        return (1.0, True) if is_terminal else (0.0, False)

    agent.get_action_probs_batch = MagicMock(side_effect=_stub_get_action_probs)
    agent.update_root_batch = MagicMock(side_effect=_stub_update_root_batch)
    agent.reset_game = MagicMock(side_effect=_stub_reset_game)
    runner.game.get_next_state = MagicMock(side_effect=lambda s, a, p: s)
    runner.game.get_value_and_terminated = MagicMock(
        side_effect=_fake_get_value_and_terminated
    )

    target_games = 5
    records = runner.play_batch_games(
        agent,
        batch_size=3,
        add_noise=False,
        target_games=target_games,
    )

    move_lengths = [len(rec.moves) for rec in records]
    print(
        f"[finish] target={target_games}, records={len(records)}, "
        f"moves={move_lengths}, reset_calls={reset_calls}"
    )

    assert len(records) == target_games
    assert all(length > 0 for length in move_lengths)
    assert len(set(move_lengths)) > 1
    assert reset_calls, "reset_game should be called when slots are refilled."


def test_sampling_temperature_logic(
    vectorize_components: tuple[
        RootConfig, Gomoku, AlphaZeroAgent, VectorizeRunner, LocalInference
    ],
) -> None:
    """탐험/결정 구간에서 샘플링 방식이 달라지는지 검증한다."""
    _, game, agent, runner, _ = vectorize_components
    runner.mcts_cfg = runner.mcts_cfg.model_copy(update={"exploration_turns": 1})

    pi = np.zeros(game.action_size, dtype=np.float32)
    pi[0] = 0.1
    pi[1] = 0.9

    agent.get_action_probs_batch = MagicMock(return_value=[pi])
    agent.update_root_batch = MagicMock(side_effect=lambda actions: None)
    agent.reset_game = MagicMock(side_effect=lambda idx: None)

    call_counter = {"n": 0}

    def _fake_get_value_and_terminated(state, move):
        call_counter["n"] += 1
        is_terminal = call_counter["n"] >= 2
        return (1.0, True) if is_terminal else (0.0, False)

    runner.game.get_value_and_terminated = MagicMock(
        side_effect=_fake_get_value_and_terminated
    )
    runner.game.get_next_state = MagicMock(
        side_effect=lambda s, a, p: type(
            "State", (), {"next_player": 2 if p == 1 else 1}
        )()
    )

    with patch("numpy.random.choice", MagicMock(return_value=0)) as choice_mock:
        records = runner.play_batch_games(
            agent,
            batch_size=1,
            add_noise=False,
            target_games=1,
        )

    calls = [call.args[0] for call in agent.update_root_batch.mock_calls]
    print(f"[temp_logic] update_root_batch calls={calls}")
    print(f"[temp_logic] np.random.choice calls={choice_mock.call_count}")

    assert choice_mock.call_count == 1
    assert calls[0] == [0]
    assert calls[1] == [1]
    assert len(records) == 1
    assert list(records[0].moves) == [0, 1]


def test_update_root_batch_called_with_correct_actions(
    vectorize_components: tuple[
        RootConfig, Gomoku, AlphaZeroAgent, VectorizeRunner, LocalInference
    ],
) -> None:
    """매 턴 전달된 액션 리스트가 정확한지 검증한다."""
    _, game, agent, runner, _ = vectorize_components
    runner.mcts_cfg = runner.mcts_cfg.model_copy(update={"exploration_turns": 0})

    actions_seq = [2, 3]

    def _policy_side_effect(
        states_raw, temperature, add_noise_flags, active_indices=None
    ):
        action = actions_seq[min(len(agent.update_root_batch.mock_calls), 1)]
        policy = np.zeros(game.action_size, dtype=np.float32)
        policy[action] = 1.0
        return [policy]

    move_counter = {"n": 0}

    def _fake_get_value_and_terminated(state, move):
        move_counter["n"] += 1
        is_terminal = move_counter["n"] >= 2
        return (1.0, True) if is_terminal else (0.0, False)

    agent.get_action_probs_batch = MagicMock(side_effect=_policy_side_effect)
    agent.update_root_batch = MagicMock(side_effect=lambda actions: None)
    agent.reset_game = MagicMock(side_effect=lambda idx: None)
    runner.game.get_next_state = MagicMock(
        side_effect=lambda s, a, p: type(
            "State", (), {"next_player": 2 if p == 1 else 1}
        )()
    )
    runner.game.get_value_and_terminated = MagicMock(
        side_effect=_fake_get_value_and_terminated
    )

    runner.play_batch_games(agent, batch_size=1, add_noise=False, target_games=1)

    calls = [call.args[0] for call in agent.update_root_batch.mock_calls]
    print(f"[update_root] calls={calls}")
    assert calls == [[2], [3]]


def test_records_integrity(
    vectorize_components: tuple[
        RootConfig, Gomoku, AlphaZeroAgent, VectorizeRunner, LocalInference
    ],
) -> None:
    """GameRecord 필드 길이 및 승패 부호 정합성을 검증한다."""
    _, game, agent, runner, _ = vectorize_components
    runner.mcts_cfg = runner.mcts_cfg.model_copy(update={"exploration_turns": 0})

    policies = []
    for action in (0, 1):
        policy = np.zeros(game.action_size, dtype=np.float32)
        policy[action] = 1.0
        policies.append(policy)

    def _policy_side_effect(
        states_raw, temperature, add_noise_flags, active_indices=None
    ):
        idx = min(len(agent.update_root_batch.mock_calls), 1)
        return [policies[idx]]

    move_counter = {"n": 0}

    def _fake_get_value_and_terminated(state, move):
        move_counter["n"] += 1
        is_terminal = move_counter["n"] >= 2
        return (1.0, True) if is_terminal else (0.0, False)

    agent.get_action_probs_batch = MagicMock(side_effect=_policy_side_effect)
    agent.update_root_batch = MagicMock(side_effect=lambda actions: None)
    agent.reset_game = MagicMock(side_effect=lambda idx: None)
    runner.game.get_next_state = MagicMock(
        side_effect=lambda s, a, p: type(
            "State", (), {"next_player": 2 if p == 1 else 1}
        )()
    )
    runner.game.get_value_and_terminated = MagicMock(
        side_effect=_fake_get_value_and_terminated
    )

    records = runner.play_batch_games(
        agent, batch_size=1, add_noise=False, target_games=1
    )
    record = records[0]

    print(
        f"[integrity] moves={record.moves}, outcomes={record.outcomes}, "
        f"players={record.players}, policies_shape={record.policies.shape}"
    )

    assert (
        len(record.moves)
        == len(record.policies)
        == len(record.outcomes)
        == len(record.players)
    )
    assert list(record.moves) == [0, 1]
    assert record.outcomes.tolist() == [-1, 1]
    assert record.policies.shape[1] == game.action_size


def test_empty_slots_short_circuit(
    vectorize_components: tuple[
        RootConfig, Gomoku, AlphaZeroAgent, VectorizeRunner, LocalInference
    ],
) -> None:
    """batch_size가 0일 때 에이전트 호출 없이 빈 결과를 반환한다."""
    _, _, agent, runner, _ = vectorize_components

    agent.get_action_probs_batch = MagicMock(
        side_effect=RuntimeError("should not call")
    )

    records = runner.play_batch_games(
        agent, batch_size=0, add_noise=False, target_games=0
    )
    print(f"[short_circuit] records_len={len(records)}")

    assert records == []
    assert not agent.get_action_probs_batch.mock_calls


def test_policy_sum_and_nan_free_batch(
    vectorize_components: tuple[
        RootConfig, Gomoku, AlphaZeroAgent, VectorizeRunner, LocalInference
    ],
) -> None:
    """배치 정책이 NaN/Inf 없이 합≈1을 유지하는지 검증한다."""
    _, game, agent, runner, _ = vectorize_components
    runner.mcts_cfg = runner.mcts_cfg.model_copy(update={"exploration_turns": 0})

    policy = np.full(game.action_size, 1.0 / game.action_size, dtype=np.float32)

    agent.get_action_probs_batch = MagicMock(return_value=[policy, policy])
    agent.update_root_batch = MagicMock(side_effect=lambda actions: None)
    agent.reset_game = MagicMock(side_effect=lambda idx: None)
    runner.game.get_next_state = MagicMock(
        side_effect=lambda s, a, p: type(
            "State", (), {"next_player": 2 if p == 1 else 1}
        )()
    )
    runner.game.get_value_and_terminated = MagicMock(
        side_effect=lambda state, move: (1.0, True)
    )

    records = runner.play_batch_games(
        agent, batch_size=2, add_noise=False, target_games=2
    )
    sums = [rec.policies.sum(axis=1) for rec in records]
    finite = [np.isfinite(rec.policies).all() for rec in records]

    print(f"[policy_sum] sums={[s.tolist() for s in sums]}")
    assert all(finite)
    for s in sums:
        np.testing.assert_allclose(s, 1.0, atol=1e-3)


def test_illegal_moves_masked_batch(
    vectorize_components: tuple[
        RootConfig, Gomoku, AlphaZeroAgent, VectorizeRunner, LocalInference
    ],
) -> None:
    """비합법 위치 확률이 0이고 move가 범위 내인지 검증한다."""
    _, game, agent, runner, _ = vectorize_components
    runner.mcts_cfg = runner.mcts_cfg.model_copy(update={"exploration_turns": 0})

    illegal_idx = 0
    legal_idx = 1 if game.action_size > 1 else 0
    policy = np.zeros(game.action_size, dtype=np.float32)
    policy[legal_idx] = 1.0
    policy[illegal_idx] = 0.0

    agent.get_action_probs_batch = MagicMock(return_value=[policy])
    agent.update_root_batch = MagicMock(side_effect=lambda actions: None)
    agent.reset_game = MagicMock(side_effect=lambda idx: None)
    runner.game.get_next_state = MagicMock(
        side_effect=lambda s, a, p: type(
            "State", (), {"next_player": 2 if p == 1 else 1}
        )()
    )
    runner.game.get_value_and_terminated = MagicMock(
        side_effect=lambda state, move: (1.0, True)
    )

    records = runner.play_batch_games(
        agent, batch_size=1, add_noise=False, target_games=1
    )
    record = records[0]

    print(
        f"[illegal] move={record.moves}, policy_illegal={record.policies[0][illegal_idx]}"
    )

    assert record.policies[0][illegal_idx] == 0.0
    assert record.moves[0] != illegal_idx
    assert 0 <= record.moves[0] < game.action_size


def test_temperature_pass_through_batch(
    vectorize_components: tuple[
        RootConfig, Gomoku, AlphaZeroAgent, VectorizeRunner, LocalInference
    ],
) -> None:
    """Temperature 인자가 배치 호출로 그대로 전달되는지 확인한다."""
    _, game, agent, runner, _ = vectorize_components
    temp = 0.42

    policy = np.full(game.action_size, 1.0 / game.action_size, dtype=np.float32)
    agent.get_action_probs_batch = MagicMock(return_value=[policy])
    agent.update_root_batch = MagicMock(side_effect=lambda actions: None)
    agent.reset_game = MagicMock(side_effect=lambda idx: None)
    runner.game.get_next_state = MagicMock(
        side_effect=lambda s, a, p: type(
            "State", (), {"next_player": 2 if p == 1 else 1}
        )()
    )
    runner.game.get_value_and_terminated = MagicMock(
        side_effect=lambda state, move: (1.0, True)
    )

    runner.play_batch_games(
        agent, batch_size=1, temperature=temp, add_noise=False, target_games=1
    )

    assert agent.get_action_probs_batch.call_count >= 1
    call = agent.get_action_probs_batch.call_args
    passed_temp = call.kwargs.get(
        "temperature", call.args[1] if len(call.args) > 1 else None
    )
    print(f"[temp_passthrough] passed_temp={passed_temp}")
    assert passed_temp == temp


def test_noise_group_split_behavior(
    vectorize_components: tuple[
        RootConfig, Gomoku, AlphaZeroAgent, VectorizeRunner, LocalInference
    ],
) -> None:
    """노이즈 on/off 섞일 때 플래그 패턴이 기대대로 전달되는지 확인한다."""
    _, game, agent, runner, _ = vectorize_components
    runner.mcts_cfg = runner.mcts_cfg.model_copy(update={"exploration_turns": 0})

    policy = np.full(game.action_size, 1.0 / game.action_size, dtype=np.float32)
    flags_history: list[list[bool]] = []

    def _spy(states_raw, temperature, add_noise_flags, active_indices=None):
        flags = list(add_noise_flags)
        flags_history.append(flags)
        return [policy for _ in states_raw]

    call_counter = {"n": 0}

    def _fake_get_value_and_terminated(state, move):
        call_counter["n"] += 1
        # 슬롯 0만 첫 턴 종료 → 리필 후 turn 0, 슬롯 1은 계속 진행.
        is_terminal = call_counter["n"] % 2 == 1
        return (1.0, True) if is_terminal else (0.0, False)

    agent.get_action_probs_batch = MagicMock(side_effect=_spy)
    agent.update_root_batch = MagicMock(side_effect=lambda actions: None)
    agent.reset_game = MagicMock(side_effect=lambda idx: None)
    runner.game.get_next_state = MagicMock(
        side_effect=lambda s, a, p: type(
            "State", (), {"next_player": 2 if p == 1 else 1}
        )()
    )
    runner.game.get_value_and_terminated = MagicMock(
        side_effect=_fake_get_value_and_terminated
    )

    runner.play_batch_games(agent, batch_size=2, add_noise=True, target_games=2)

    print(f"[noise_split] flags_history={flags_history}")
    assert flags_history[0] == [True, True]
    assert any(flags in flags_history[1:] for flags in ([True, False], [False, True]))


def test_partial_games_continue(
    vectorize_components: tuple[
        RootConfig, Gomoku, AlphaZeroAgent, VectorizeRunner, LocalInference
    ],
) -> None:
    """일부 슬롯만 종료돼도 나머지 슬롯이 정상 진행되는지 확인한다."""
    _, game, agent, runner, _ = vectorize_components
    runner.mcts_cfg = runner.mcts_cfg.model_copy(update={"exploration_turns": 0})

    policy = np.full(game.action_size, 1.0 / game.action_size, dtype=np.float32)

    agent.get_action_probs_batch = MagicMock(return_value=[policy, policy])
    agent.update_root_batch = MagicMock(side_effect=lambda actions: None)
    agent.reset_game = MagicMock(side_effect=lambda idx: None)
    runner.game.get_next_state = MagicMock(
        side_effect=lambda s, a, p: type(
            "State", (), {"next_player": 2 if p == 1 else 1}
        )()
    )

    call_counter = {"n": 0}

    def _fake_get_value_and_terminated(state, move):
        call_counter["n"] += 1
        slot_idx = (call_counter["n"] - 1) % 2
        if slot_idx == 1 and call_counter["n"] <= 2:
            return 1.0, True
        if slot_idx == 0 and call_counter["n"] >= 3:
            return 1.0, True
        return 0.0, False

    runner.game.get_value_and_terminated = MagicMock(
        side_effect=_fake_get_value_and_terminated
    )

    records = runner.play_batch_games(
        agent, batch_size=2, add_noise=False, target_games=2
    )
    lengths = sorted([len(rec.moves) for rec in records])
    print(f"[partial] lengths={lengths}")
    assert lengths == [1, 2]


def test_outcome_sign_by_player_batch(
    vectorize_components: tuple[
        RootConfig, Gomoku, AlphaZeroAgent, VectorizeRunner, LocalInference
    ],
) -> None:
    """마지막 착수자 기준 outcomes 부호가 일관적인지 확인한다."""
    _, game, agent, runner, _ = vectorize_components
    runner.mcts_cfg = runner.mcts_cfg.model_copy(update={"exploration_turns": 0})

    policies = []
    for action in (0, 1):
        policy = np.zeros(game.action_size, dtype=np.float32)
        policy[action] = 1.0
        policies.append(policy)

    def _policy_side_effect(
        states_raw, temperature, add_noise_flags, active_indices=None
    ):
        idx = min(len(agent.update_root_batch.mock_calls), 1)
        return [policies[idx]]

    move_counter = {"n": 0}

    def _fake_get_value_and_terminated(state, move):
        move_counter["n"] += 1
        is_terminal = move_counter["n"] >= 2
        return (1.0, True) if is_terminal else (0.0, False)

    agent.get_action_probs_batch = MagicMock(side_effect=_policy_side_effect)
    agent.update_root_batch = MagicMock(side_effect=lambda actions: None)
    agent.reset_game = MagicMock(side_effect=lambda idx: None)
    runner.game.get_next_state = MagicMock(
        side_effect=lambda s, a, p: type(
            "State", (), {"next_player": 2 if p == 1 else 1}
        )()
    )
    runner.game.get_value_and_terminated = MagicMock(
        side_effect=_fake_get_value_and_terminated
    )

    records = runner.play_batch_games(
        agent, batch_size=1, add_noise=False, target_games=1
    )
    record = records[0]

    print(
        f"[outcome] players={record.players}, outcomes={record.outcomes}, moves={record.moves}"
    )

    assert record.outcomes.tolist() == [-1, 1]
    assert record.players.tolist() == [1, 2]
