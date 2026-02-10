from dataclasses import dataclass
from typing import Any

import numpy as np

from gomoku.alphazero.agent import AlphaZeroAgent
from gomoku.alphazero.runners.common import (
    SelfPlaySample,
    build_game_record,
    random_legal_action,
    sample_action,
)
from gomoku.alphazero.types import Action, GameRecord, action_to_xy
from gomoku.core.gomoku import GameState, Gomoku
from gomoku.utils.config.loader import MctsConfig, TrainingConfig
from gomoku.utils.progress import make_progress


@dataclass
class GameSlot:
    """Track a single game slot during vectorized self-play."""

    state: GameState
    turn: int
    memory: list[SelfPlaySample]


class VectorizeRunner:
    """Batch self-play runner that advances multiple games in parallel."""

    def __init__(
        self,
        game: Gomoku,
        mcts_cfg: MctsConfig,
        train_cfg: TrainingConfig,
    ):
        """Initialize runner configuration."""
        self.game = game
        self.mcts_cfg = mcts_cfg
        self.train_cfg = train_cfg

    def play_batch_games(
        self,
        agent: AlphaZeroAgent,
        batch_size: int,
        temperature: float | None = None,
        add_noise: bool = True,
        target_games: int | None = None,
        progress_desc: str | None = None,
        random_ratio: float = 0.0,
        random_opening_turns: int = 0,
        opponent: Any = None,
        agent_first: bool = True,
    ) -> list[GameRecord]:
        """
        Generate multiple self-play games concurrently using batch MCTS.

        Parameters
        ----------
        agent :
            Batch-capable MCTS adapter.
        batch_size :
            Number of games to run in parallel.
        temperature :
            Sampling temperature. Defaults to training config when None.
        add_noise :
            Whether to apply Dirichlet noise on the first move of each game.
        target_games :
            Number of completed games to return. Defaults to ``batch_size``.
        opponent :
            Optional opponent; when provided, agent only moves for its color.
            Opponent is expected to expose ``get_action_probs``.
        agent_first :
            Agent plays as player 1 when True, player 2 otherwise.

        Returns
        -------
        list[GameRecord]
            Completed game records.

        """
        if batch_size <= 0:
            return []

        target = target_games if target_games is not None else batch_size
        if target <= 0:
            return []

        agent.reset()
        if opponent is not None and hasattr(opponent, "reset"):
            opponent.reset()
        temp = (
            float(self.train_cfg.temperature)
            if temperature is None
            else float(temperature)
        )
        rnd_ratio = max(0.0, float(random_ratio))
        rnd_opening = max(0, int(random_opening_turns))
        agent_player = 1 if agent_first else 2

        finished_records: list[GameRecord] = []
        progress = make_progress(
            total=target, desc=progress_desc, unit="game", disable=progress_desc is None
        )

        slots: list[GameSlot] = []
        for _ in range(batch_size):
            slots.append(
                GameSlot(state=self.game.get_initial_state(), turn=0, memory=[])
            )

        while len(finished_records) < target:
            alive_indices = [
                idx for idx, slot in enumerate(slots) if slot.state is not None
            ]
            if not alive_indices:
                break

            alive_slots = [slots[i] for i in alive_indices]
            alive_states = [slot.state for slot in alive_slots]
            slot_to_root_idx = {slot_idx: i for i, slot_idx in enumerate(alive_indices)}

            # 1. Identify which indices belong to the agent vs opponent
            agent_indices = [
                i for i, slot in enumerate(alive_slots)
                if opponent is None or int(slot.state.next_player) == agent_player
            ]
            opp_indices = [
                i for i, slot in enumerate(alive_slots)
                if opponent is not None and int(slot.state.next_player) != agent_player
            ]

            # 2. Agent Batch Inference
            agent_noise = [(add_noise and slot.turn == 0) for slot in alive_slots]
            agent_policies = agent.get_action_probs_batch(
                states_raw=alive_states,
                temperature=temp,
                add_noise_flags=agent_noise,
                active_indices=agent_indices,
            )

            # 3. Opponent Batch Inference (if supported)
            opp_policies: list[np.ndarray | None] = [None] * len(alive_slots)
            if opp_indices:
                if hasattr(opponent, "get_action_probs_batch"):
                    opp_policies = opponent.get_action_probs_batch(
                        states_raw=alive_states,
                        temperature=1.0,
                        add_noise_flags=[False] * len(alive_slots),
                        active_indices=opp_indices,
                    )
                else:
                    # Fallback to sequential calls for simpler bots
                    for i in opp_indices:
                        opp_policies[i] = opponent.get_action_probs(
                            alive_states[i], temperature=1.0, add_noise=False
                        )

            actions_alive: list[Action] = []
            for i, slot in enumerate(alive_slots):
                if i in agent_indices:
                    pi = agent_policies[i]
                else:
                    pi = opp_policies[i]
                    if pi is None:
                        # Safety fallback
                        pi = np.zeros(self.game.action_size, dtype=np.float32)

                is_agent_turn = i in agent_indices
                if is_agent_turn:
                    if rnd_opening > 0 and slot.turn < rnd_opening:
                        random_idx = random_legal_action(self.game, slot.state)
                        if random_idx is not None:
                            action = random_idx
                            pi = np.zeros_like(pi)
                            pi[action] = 1.0
                        else:
                            action = sample_action(
                                pi, slot.turn, self.mcts_cfg.exploration_turns
                            )
                    elif rnd_ratio > 0.0 and np.random.rand() < rnd_ratio:
                        random_idx = random_legal_action(self.game, slot.state)
                        action = (
                            random_idx
                            if random_idx is not None
                            else sample_action(
                                pi, slot.turn, self.mcts_cfg.exploration_turns
                            )
                        )
                    else:
                        action = sample_action(
                            pi, slot.turn, self.mcts_cfg.exploration_turns
                        )
                else:
                    random_idx = None
                    if getattr(opponent, "is_uniform_random", False):
                        random_idx = random_legal_action(self.game, slot.state)
                    if random_idx is not None:
                        action = random_idx
                    else:
                        action = sample_action(
                            pi, slot.turn, self.mcts_cfg.exploration_turns
                        )
                actions_alive.append(action)

                player = int(slot.state.next_player)
                slot.memory.append(
                    SelfPlaySample(
                        state=slot.state,
                        policy_probs=pi,
                        move=action,
                        player=player,
                    )
                )

                x, y = action_to_xy(action, self.game.col_count)
                slot.state = self.game.get_next_state(slot.state, (x, y), player)
                slot.turn += 1

            agent.update_root_batch(actions_alive)
            if opponent is not None and hasattr(opponent, "update_root_batch"):
                opponent.update_root_batch(actions_alive)

            for slot_idx, slot in zip(alive_indices, alive_slots, strict=True):
                last_action = slot.memory[-1].move
                x, y = action_to_xy(last_action, self.game.col_count)
                value, is_terminal = self.game.get_value_and_terminated(
                    slot.state, (x, y)
                )
                if not is_terminal:
                    continue

                last_player = slot.memory[-1].player

                record = build_game_record(slot.memory, float(value), last_player)
                finished_records.append(record)
                progress.update(1)

                # Skip refill when target count is reached
                if len(finished_records) >= target:
                    # Disable the slot when it will no longer be reused
                    slot.state = None
                    continue

                # Refill: restart the slot from the initial state
                slot.state = self.game.get_initial_state()
                slot.turn = 0
                slot.memory = []
                agent.reset_game(slot_to_root_idx[slot_idx])

            if len(finished_records) >= target:
                break

        progress.close()
        return finished_records[:target]
