from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from gomoku.core.game_config import (
    EMPTY_SPACE,
    PLAYER_1,
    PLAYER_2,
    bulk_index_to_xy,
    get_pos,
    index_to_xy,
    opponent_player,
    set_pos,
    xy_to_index,
)
from gomoku.core.rules.capture import detect_captured_stones
from gomoku.core.rules.doublethree import detect_doublethree
from gomoku.core.rules.terminate import check_local_gomoku
from gomoku.utils.config.loader import BoardConfig

try:
    from gomoku.cpp_ext import gomoku_cpp
except Exception:  # pragma: no cover - optional native dependency
    gomoku_cpp = None


@dataclass(slots=True)
class GameState:
    """Snapshot of the board and match metadata."""

    board: np.ndarray  # (19, 19) int8 board
    p1_pts: np.int16  # Capture score for PLAYER_1 (black)
    p2_pts: np.int16  # Capture score for PLAYER_2 (white)
    next_player: np.int8
    last_move_idx: np.int16  # -1 when no prior move
    empty_count: np.int16
    history: tuple[int, ...] = ()
    legal_indices_cache: np.ndarray | None = None
    native_state: Any | None = None

    def __getstate__(self):
        return {k: getattr(self, k) for k in self.__slots__ if k != "native_state"}

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)
        self.native_state = None


class Gomoku:
    """Engine for Gomoku with optional double-three and capture rules."""

    def __init__(self, cfg: BoardConfig, use_native: bool = False):
        """
        Initialize game parameters from a ``BoardConfig``.

        Parameters
        ----------
        cfg : BoardConfig
            Board/game rule configuration.
        use_native : bool, optional
            Whether to prefer C++ core (`gomoku_cpp`) if available.

        """
        self.row_count = cfg.num_lines
        self.col_count = cfg.num_lines
        self.gomoku_goal = cfg.gomoku_goal
        self.capture_goal = cfg.capture_goal
        self.last_captures = []
        self.action_size = self.row_count * self.col_count
        self.enable_doublethree = cfg.enable_doublethree
        self.enable_capture = cfg.enable_capture
        self.history_length = cfg.history_length
        self.use_native = bool(use_native and gomoku_cpp is not None)
        self._native_core = None
        if self.use_native and gomoku_cpp is not None:
            try:
                self._native_core = gomoku_cpp.GomokuCore(
                    self.row_count,
                    self.enable_doublethree,
                    self.enable_capture,
                    self.capture_goal,
                    self.gomoku_goal,
                    self.history_length,
                )
            except Exception:
                self._native_core = None
                self.use_native = False

        self._coord_cache = [
            f"{chr(ord('A') + x)}{y + 1}"
            for y in range(self.row_count)
            for x in range(self.col_count)
        ]

    def get_initial_state(self) -> GameState:
        """
        Create a fresh ``GameState`` with an empty board.

        Returns
        -------
        GameState
            New state with zeroed scores, no history, and all cells empty.

        """
        board = np.zeros((self.row_count, self.col_count), dtype=np.int8)
        native_state = None
        if self.use_native and self._native_core is not None:
            native_state = self._native_core.initial_state()
            board = np.array(native_state.board, dtype=np.int8).reshape(
                self.row_count, self.col_count
            )
        return GameState(
            board=board,
            p1_pts=np.int16(0),
            p2_pts=np.int16(0),
            next_player=np.int8(PLAYER_1),
            last_move_idx=np.int16(-1),
            empty_count=np.int16(self.action_size),
            history=tuple(),
            legal_indices_cache=None,
            native_state=native_state,
        )

    def get_next_state(
        self, state: GameState, action: tuple[int, int], player: int
    ) -> GameState:
        """
        Apply a move for ``player`` and return the resulting state.

        Parameters
        ----------
        state : GameState
            Current game snapshot.
        action : tuple[int, int]
            Coordinate of the placed stone as ``(x, y)``.
        player : int
            Player identifier placing the stone.

        Returns
        -------
        GameState
            Updated state with board, scores, empty count, and history adjusted.

        """
        if self.use_native and self._native_core is not None and state.native_state:
            native_next = self._native_core.apply_move(
                state.native_state, action[0], action[1], int(player)
            )
            return self._from_native_state(native_next)

        board = state.board.copy()
        p1_pts, p2_pts = np.int16(state.p1_pts), np.int16(state.p2_pts)
        empty_count = np.int16(state.empty_count - 1)

        x, y = action
        set_pos(board, x, y, player)
        last_move_idx = np.int16(xy_to_index(x, y, len(board)))

        if self.enable_capture:
            cap_indices = detect_captured_stones(board, x, y, player, self.row_count)
            if cap_indices.size:
                self.last_captures = cap_indices
                for idx in cap_indices:
                    cx, cy = index_to_xy(int(idx), len(board))
                    set_pos(board, cx, cy, EMPTY_SPACE)
                cap_pairs = cap_indices.shape[0] // 2
                if player == PLAYER_1:
                    p1_pts = np.int16(p1_pts + cap_pairs)
                else:
                    p2_pts = np.int16(p2_pts + cap_pairs)
                empty_count = np.int16(empty_count + cap_indices.shape[0])

        next_player_turn = np.int8(opponent_player(player))

        new_history = state.history
        if self.history_length > 0:
            prev = state.history
            new_history = (int(last_move_idx),) + prev[
                : max(self.history_length - 1, 0)
            ]

        return GameState(
            board=board,
            p1_pts=p1_pts,
            p2_pts=p2_pts,
            next_player=next_player_turn,
            last_move_idx=last_move_idx,
            empty_count=empty_count,
            history=new_history,
            legal_indices_cache=None,
        )

    def get_legal_moves(self, state: GameState) -> list[str]:
        """
        Compute playable moves for the current player.

        Returns
        -------
        list[str]
            Legal coordinates in ``"A1"`` style; cached per state for reuse.

        """
        if (
            self.use_native
            and self._native_core is not None
            and state.native_state is not None
            and state.legal_indices_cache is None
        ):
            native_moves = self._native_core.get_legal_moves(state.native_state)
            native_indices = np.array(native_moves, dtype=np.int16)
            state.legal_indices_cache = native_indices
            return [self._coord_cache[int(idx)] for idx in native_indices]

        if state.legal_indices_cache is not None:
            return [self._coord_cache[int(idx)] for idx in state.legal_indices_cache]

        board = state.board
        empty_indices = np.flatnonzero(board == EMPTY_SPACE)

        if empty_indices.size == 0:
            state.legal_indices_cache = np.array([], dtype=np.int16)
            return []

        player_stone_count = np.count_nonzero(board == state.next_player)

        if not self.enable_doublethree or player_stone_count < 4:
            indices = empty_indices.astype(np.int16, copy=False)
            state.legal_indices_cache = indices
            return [self._coord_cache[int(idx)] for idx in indices]

        x_coords, y_coords = bulk_index_to_xy(empty_indices, self.col_count)
        legal_buffer = np.empty_like(empty_indices, dtype=np.int16)
        legal_count = 0

        for idx, x, y in zip(empty_indices, x_coords, y_coords, strict=True):
            if detect_doublethree(
                board, int(x), int(y), state.next_player, self.row_count
            ):
                continue
            legal_buffer[legal_count] = np.int16(idx)
            legal_count += 1

        legal_indices = legal_buffer[:legal_count]
        state.legal_indices_cache = legal_indices
        return [self._coord_cache[int(idx)] for idx in legal_indices]

    def check_win(self, state: GameState, action: int | tuple[int, int] | None) -> bool:
        """
        Determine whether a given move produces a terminal win.

        Parameters
        ----------
        state : GameState
            Current game snapshot.
        action : int | tuple[int, int] | None
            Flat index (RL use), ``(x, y)`` tuple, or ``None`` when no move.

        Returns
        -------
        bool
            ``True`` if five-in-a-row or capture goal is achieved.

        """
        if action is None:
            return False

        if (
            self.use_native
            and self._native_core is not None
            and state.native_state is not None
        ):
            if isinstance(action, tuple):
                x, y = action
            else:
                x, y = index_to_xy(int(action), self.col_count)
            return bool(self._native_core.check_win(state.native_state, x, y))

        if isinstance(action, int):
            x, y = index_to_xy(action, self.col_count)
        elif isinstance(action, np.integer):
            x, y = index_to_xy(int(action), self.col_count)
        elif isinstance(action, tuple):
            x, y = action
        else:
            raise TypeError(f"Action must be int or tuple, got {type(action)}")
        player = get_pos(state.board, x, y)

        if check_local_gomoku(
            state.board, x, y, player, self.row_count, self.gomoku_goal
        ):
            return True
        if self.capture_goal > 0:
            if player == PLAYER_1 and state.p1_pts >= self.capture_goal:
                return True
            if player == PLAYER_2 and state.p2_pts >= self.capture_goal:
                return True

        return False

    def get_value_and_terminated(
        self, state: GameState, action: tuple[int, int]
    ) -> tuple[int, bool]:
        """
        Evaluate outcome after applying ``action`` for the current player.

        Returns
        -------
        tuple[int, bool]
            ``(value, terminated)`` where value is 1 for win, 0 otherwise.

        """
        if self.check_win(state, action):
            return 1, True
        if state.empty_count <= 0:
            return 0, True
        if self.enable_doublethree:
            legal = self.get_legal_moves(state)
            if not legal:
                return 0, True
        return 0, False

    def format_board(self, state: GameState) -> str:
        """
        Render the board state as human-readable text.

        Parameters
        ----------
        state : GameState
            Board and capture status to render.

        Returns
        -------
        str
            Multiline string with column letters, row numbers, and capture counts.

        """
        board = state.board
        column_labels = " ".join(chr(ord("A") + i) for i in range(self.col_count))
        lines = ["   " + column_labels]
        for i, row in enumerate(board):
            lines.append(f"{i + 1:>2} " + " ".join(str(int(c)) for c in row))
        lines.append(f"Captures  P1:{state.p1_pts}  P2:{state.p2_pts}")
        # Keep a trailing blank line to match legacy print_board output.
        return "\n".join(lines) + "\n\n"

    def print_board(self, state: GameState) -> None:
        """
        Print the board with column letters (A-T) and row numbers (1-19).

        Parameters
        ----------
        state : GameState
            Board to display.

        """
        print(self.format_board(state), end="")

    def get_encoded_state(self, states: Sequence[GameState] | GameState) -> np.ndarray:
        """
        Encode game state(s) into a tensor suitable for neural network input.

        The output tensor has the shape (Batch, Channels, Height, Width).
        The total number of channels is `8 + self.history_length`.

        The `self.history_length` is determined by the `BoardConfig` passed to the
        `Gomoku` constructor (default is typically 5).

        Channel Layout
        --------------
        0. Me (current player): binary plane where the current player has stones.
        1. Opponent: binary plane where the opponent has stones.
        2. Empty: binary plane where intersections are empty.
        3. Last Move: one-hot plane for the last move made.
        4. My Capture Score: constant plane of (score / capture_goal).
        5. Opponent Capture Score: constant plane of (score / capture_goal).
        6. Color Plane: 1.0 for Black (Player 1), -1.0 for White (Player 2).
        7. Forbidden Points: binary plane for current-player double-three forbiddens.
        8+. History: ``self.history_length`` planes for past moves (T-1, T-2, ...).
        """
        if isinstance(states, GameState):
            states = [states]

        batch_size = len(states)
        if self.use_native and self._native_core is not None:
            encoded_native: list[np.ndarray] = []
            for st in states:
                if st.native_state is None:
                    break
                encoded_native.append(self._encode_native(st))
            if len(encoded_native) == batch_size:
                return np.stack(encoded_native, axis=0)

        h, w = self.row_count, self.col_count

        base_channels = 8
        total_channels = base_channels + self.history_length
        features = np.zeros((batch_size, total_channels, h, w), dtype=np.float32)

        boards = np.stack([st.board for st in states])
        next_players = np.array([st.next_player for st in states])
        last_moves = np.array([st.last_move_idx for st in states], dtype=np.int16)

        next_players_exp = next_players[:, None, None]

        features[:, 0] = boards == next_players_exp
        opp_exp = (3 - next_players)[:, None, None]
        features[:, 1] = boards == opp_exp
        features[:, 2] = boards == EMPTY_SPACE

        valid_idx = np.where(last_moves >= 0)[0]
        if valid_idx.size:
            moves = last_moves[valid_idx]
            ys = moves // self.col_count
            xs = moves % self.col_count
            features[valid_idx, 3, ys, xs] = 1.0

        if self.enable_capture and self.capture_goal > 0:
            p1_pts = np.array([st.p1_pts for st in states], dtype=np.float32)
            p2_pts = np.array([st.p2_pts for st in states], dtype=np.float32)
            my_pts = np.where(next_players == PLAYER_1, p1_pts, p2_pts)
            opp_pts = np.where(next_players == PLAYER_1, p2_pts, p1_pts)
            my_ratio = np.clip(my_pts / self.capture_goal, 0.0, 1.0)
            opp_ratio = np.clip(opp_pts / self.capture_goal, 0.0, 1.0)
            features[:, 4] = my_ratio[:, None, None]
            features[:, 5] = opp_ratio[:, None, None]

        color_vals = np.where(next_players == PLAYER_1, 1.0, -1.0)
        features[:, 6] = color_vals[:, None, None]

        if self.enable_doublethree:
            for b_idx in range(batch_size):
                board = boards[b_idx]
                player = int(next_players[b_idx])
                empties = np.flatnonzero(board == EMPTY_SPACE)
                if not empties.size:
                    continue
                xs, ys = bulk_index_to_xy(empties, self.col_count)
                for x, y in zip(xs, ys, strict=False):
                    if detect_doublethree(
                        board, int(x), int(y), player, self.row_count
                    ):
                        features[b_idx, 7, int(y), int(x)] = 1.0

        if self.history_length > 0:
            hist_buffer = np.full((batch_size, self.history_length), -1, dtype=np.int32)
            for idx, st in enumerate(states):
                if st.history:
                    capped = st.history[: self.history_length]
                    hist_buffer[idx, : len(capped)] = capped

            start_ch = base_channels
            for k in range(self.history_length):
                moves = hist_buffer[:, k]
                valid = moves >= 0
                if not np.any(valid):
                    continue
                xs = moves[valid] % self.col_count
                ys = moves[valid] // self.col_count
                features[valid, start_ch + k, ys, xs] = 1.0

        return features

    def _from_native_state(self, native_state: Any) -> GameState:
        """Convert native GomokuState to Python GameState."""
        board = np.array(native_state.board, dtype=np.int8).reshape(
            self.row_count, self.col_count
        )
        return GameState(
            board=board,
            p1_pts=np.int16(native_state.p1_pts),
            p2_pts=np.int16(native_state.p2_pts),
            next_player=np.int8(native_state.next_player),
            last_move_idx=np.int16(native_state.last_move_idx),
            empty_count=np.int16(native_state.empty_count),
            history=tuple(int(x) for x in native_state.history),
            legal_indices_cache=None,
            native_state=native_state,
        )

    def _encode_native(self, state: GameState) -> np.ndarray:
        """Encode using native core; returns (C, H, W)."""
        if self._native_core is None or state.native_state is None:
            raise ValueError("Native core not available for encoding.")
        features = self._native_core.encode_state(state.native_state)
        total_channels = 8 + self.history_length
        arr = np.array(features, dtype=np.float32).reshape(
            total_channels, self.row_count, self.col_count
        )
        return arr
