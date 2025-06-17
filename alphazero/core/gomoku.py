from ai.ai_config import DRAW
from core.board import Board
from core.rules.terminate import (
    board_is_functionally_full,
    check_local_gomoku,
    is_won_by_score,
)


class Gomoku:
    def __init__(self, board: Board | None = None):
        self.board: Board = board or Board.empty()
        self.turn: int = 0
        self.history: list[tuple[int, int, int]] = []
        self.winner: int | None = None
        self.last_captures: list[dict] = []

    @property
    def current_player(self):
        return self.board.next_player

    def play_move(self, x: int, y: int) -> bool:
        player = self.board.next_player
        if not self.board.is_legal_move(x, y, player):
            return False
        self.board.apply_move(x, y, player)  # Board 쪽에서 next_player 토글 수행
        self.history.append((x, y, player))
        self._check_terminal(player)  # ← 플레이어 전달
        self.turn += 1
        return True

    def _check_terminal(self, player: int) -> None:
        # 5목·점수·가득참 체크 → self.winner 갱신
        if self.board.last_x is None or self.board.last_y is None:
            return

        if check_local_gomoku(
            self.board, self.board.last_x, self.board.last_y, player
        ) or is_won_by_score(self.board, player):
            self.winner = player

        elif board_is_functionally_full(self.board):
            self.winner = DRAW

    def set_board(self, data: dict):
        self.board.update_from_dict(data)
        self.turn = data.get("turn", 0)
        self.history.clear()
        self.winner = None
        self.last_captures = []

    # 디버그 전체 출력
    def debug_print(self):
        print(
            f"Turn {self.turn}  Player {self.board.next_player}"
            f"  Score B:{self.board.last_pts} W:{self.board.next_pts}"
            f"  Winner:{self.winner}"
        )
        self.board.print_board()
