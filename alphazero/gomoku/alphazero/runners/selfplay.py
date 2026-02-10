from typing import Any

import numpy as np

from gomoku.alphazero.agent import AlphaZeroAgent
from gomoku.alphazero.runners.common import (
    SelfPlaySample,
    build_game_record,
    random_legal_action,
    sample_action,
)
from gomoku.alphazero.types import GameRecord, action_to_xy
from gomoku.core.gomoku import Gomoku
from gomoku.utils.config.loader import MctsConfig, TrainingConfig


class SelfPlayRunner:
    """
    Sequential self-play runner.

    Drives a single game from empty board to termination while delegating search and
    inference to the provided agent.
    """

    def __init__(
        self,
        game: Gomoku,
        mcts_cfg: MctsConfig,
        train_cfg: TrainingConfig,
    ):
        """Initialize the self-play runner."""
        self.game = game
        self.mcts_cfg = mcts_cfg
        self.train_cfg = train_cfg

    def play_one_game(
        self,
        agent: AlphaZeroAgent,
        temperature: float | None = None,
        add_noise: bool = True,
        random_ratio: float = 0.0,
        random_opening_turns: int = 0,
        opponent: Any = None,
        agent_first: bool = True,
    ) -> GameRecord:
        """
        Play a full self-play game and return its training record.

        Parameters
        ----------
        agent :
            Adapter that wraps MCTS and neural network inference.
        temperature :
            Sampling temperature. Falls back to training config when None.
        add_noise :
            Whether to apply Dirichlet noise to the root on the first move.
        random_opening_turns :
            Number of initial turns to play uniformly random moves (opening
            diversification). Unlike ``random_ratio`` which applies stochastically
            to all turns, this deterministically forces the first N moves to be
            uniform random.
        opponent :
            Optional opponent. When ``None``, the agent plays both sides.
            When provided, agent plays as the first player by default.
        agent_first :
            Whether the agent moves first when an opponent is provided.

        Returns
        -------
        GameRecord
            Game data encoded for training.

        Notes
        -----
        - Samples actions from pi and stores ``(state, pi, move, player)`` per turn.
        - Encodes outcomes from each turn player's perspective at termination.

        """
        # 1. Initialize game state and agent roots
        agent.reset()
        state = self.game.get_initial_state()
        memory: list[SelfPlaySample] = []
        turn = 0
        rnd_ratio = max(0.0, float(random_ratio))
        rnd_opening = max(0, int(random_opening_turns))
        opponent_bot = opponent
        agent_player = 1 if agent_first else 2
        if opponent_bot is not None and hasattr(opponent_bot, "reset"):
            opponent_bot.reset()

        # Load temperature setting
        temp = (
            float(self.train_cfg.temperature)
            if temperature is None
            else float(temperature)
        )

        while True:
            is_agent_turn = (
                opponent_bot is None or int(state.next_player) == agent_player
            )
            actor = agent if is_agent_turn else opponent_bot
            if actor is None:
                raise RuntimeError("Opponent not provided for non-agent turn.")

            # 2. Query actor policy
            if is_agent_turn:
                # 3. Choose action: random opening -> random ratio -> explore -> argmax
                if rnd_opening > 0 and turn < rnd_opening:
                    # Deterministic random opening: skip MCTS, play uniform random
                    random_idx = random_legal_action(self.game, state)
                    if random_idx is not None:
                        action = random_idx
                        board_size = self.game.col_count * self.game.row_count
                        pi = np.zeros(board_size, dtype=np.float32)
                        pi[action] = 1.0
                    else:
                        pi = agent.get_action_probs(state, temp, add_noise=add_noise)
                        action = sample_action(
                            pi, turn, self.mcts_cfg.exploration_turns
                        )
                else:
                    pi = agent.get_action_probs(state, temp, add_noise=add_noise)
                    if rnd_ratio > 0.0 and np.random.rand() < rnd_ratio:
                        random_idx = random_legal_action(self.game, state)
                        action = (
                            random_idx
                            if random_idx is not None
                            else sample_action(
                                pi, turn, self.mcts_cfg.exploration_turns
                            )
                        )
                    else:
                        action = sample_action(
                            pi, turn, self.mcts_cfg.exploration_turns
                        )
            else:
                pi = actor.get_action_probs(state, temperature=1.0, add_noise=False)
                random_idx = None
                if getattr(actor, "is_uniform_random", False):
                    random_idx = random_legal_action(self.game, state)
                if random_idx is not None:
                    action = random_idx
                else:
                    action = sample_action(pi, turn, self.mcts_cfg.exploration_turns)

            # Convert to coordinates and capture player id
            x, y = action_to_xy(action, self.game.col_count)
            current_player = int(state.next_player)

            # 4. Buffer the turn for later outcome annotation
            memory.append(
                SelfPlaySample(
                    state=state,
                    policy_probs=pi,
                    move=action,
                    player=current_player,
                )
            )

            # 5. Advance game state and move agent root
            state = self.game.get_next_state(state, (x, y), current_player)
            agent.update_root(action)
            if opponent_bot is not None and hasattr(opponent_bot, "update_root"):
                opponent_bot.update_root(action)
            turn += 1

            # 6. Check termination
            value, is_terminal = self.game.get_value_and_terminated(state, (x, y))

            # Adjudication: Check for early resignation based on value head
            if not is_terminal and self.mcts_cfg.resign_enabled and turn >= self.mcts_cfg.min_moves_before_resign:
                # Get value estimate from agent's last search
                if is_agent_turn and hasattr(agent, 'get_last_value'):
                    predicted_value = agent.get_last_value()
                    if abs(predicted_value) >= self.mcts_cfg.resign_threshold:
                        # Adjudicate: predict winner based on value sign
                        # value is from current_player's perspective
                        value = 1.0 if predicted_value > 0 else -1.0
                        is_terminal = True

            if is_terminal:
                return build_game_record(memory, float(value), current_player)

