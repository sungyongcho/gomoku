from dataclasses import dataclass

import numpy as np
import torch
from core.game_config import (
    CAPTURE_GOAL,
    EMPTY_DOT,
    EMPTY_SPACE,
    PLAYER_1,
    PLAYER_2,
    PLAYER_X,
    get_pos,
    set_pos,
)
from core.rules.capture import detect_captured_stones
from core.rules.doublethree import detect_doublethree


@dataclass
class Board:
    pos: np.ndarray
    last_x: int | None = None
    last_y: int | None = None
    last_player: int = PLAYER_2
    next_player: int = PLAYER_1
    last_pts: int = 0
    next_pts: int = 0
    enable_capture: bool = True
    enable_doublethree: bool = True
    goal: int = CAPTURE_GOAL

    @classmethod
    def empty(
        cls,
        size: int = 19,
        enable_capture: bool = True,
        enable_doublethree: bool = True,
        goal: int = 10,
    ) -> "Board":
        return cls(
            pos=np.full((size, size), EMPTY_SPACE, np.uint8),
            goal=goal,
            enable_capture=enable_capture,
            enable_doublethree=enable_doublethree,
        )

    def is_legal_move(self, x: int, y: int, player: int) -> bool:
        if get_pos(self.pos, x, y) != EMPTY_SPACE:
            return False
        if self.enable_doublethree and detect_doublethree(self.pos, x, y, player):
            return False
        return True

    def apply_move(self, x: int, y: int, player: int) -> list[dict]:
        """합법이라고 가정"""
        set_pos(self.pos, x, y, player)
        self.last_x, self.last_y = x, y
        captures: list[dict] = []

        # ─ 캡처 처리 ─
        if self.enable_capture:
            captures = detect_captured_stones(self.pos, x, y, player)
            for s in captures:
                set_pos(self.pos, s["x"], s["y"], EMPTY_SPACE)
            # 점수 가산
            if player == PLAYER_1:
                self.last_pts += len(captures) // 2
            else:
                self.next_pts += len(captures) // 2

        # ─ 턴 교대 ─
        self.last_player = player
        self.next_player = PLAYER_2 if player == PLAYER_1 else PLAYER_1
        # 점수 레이블(last/next)도 플레이어 교대에 맞춰 스왑
        # self.last_pts, self.next_pts = self.next_pts, self.last_pts

        return captures

    def legal_moves(self, player: int) -> list[tuple[int, int]]:
        ys, xs = np.where(self.pos == EMPTY_SPACE)
        return [(x, y) for x, y in zip(xs, ys) if self.is_legal_move(x, y, player)]

    def update_from_dict(self, board_data: dict) -> None:
        """
        FastAPI 웹소켓이 보내주는 board_data(JSON)를
        현재 Board 인스턴스에 반영한다.
        """
        # 1) 점수・플래그
        self.goal = board_data["goal"]
        self.enable_capture = board_data["enableCapture"]
        self.enable_doublethree = board_data["enableDoubleThreeRestriction"]

        # 2) 플레이어 정보
        self.last_player = (
            PLAYER_1 if board_data["lastPlay"]["stone"] == PLAYER_X else PLAYER_2
        )
        self.next_player = (
            PLAYER_1 if board_data["nextPlayer"] == PLAYER_X else PLAYER_2
        )

        # 3) 점수
        self.last_pts = next(
            s["score"]
            for s in board_data["scores"]
            if s["player"] == board_data["lastPlay"]["stone"]
        )
        self.next_pts = next(
            s["score"]
            for s in board_data["scores"]
            if s["player"] == board_data["nextPlayer"]
        )

        # 4) 바둑판 배열 ('.' → EMPTY_SPACE 등)
        self.pos = np.array(
            [
                [
                    EMPTY_SPACE
                    if cell == EMPTY_DOT
                    else (PLAYER_1 if cell == PLAYER_X else PLAYER_2)
                    for cell in row
                ]
                for row in board_data["board"]
            ],
            dtype=np.uint8,
        )

        # 5) 마지막 착수 좌표
        self.last_x = board_data["lastPlay"]["coordinate"].get("x")
        self.last_y = board_data["lastPlay"]["coordinate"].get("y")

    def print_board(self) -> None:
        """Prints the board with column letters (A-T) and row numbers (1-19)."""
        size = len(self.pos)
        column_labels = " ".join(chr(ord("A") + i) for i in range(size))
        print("   " + column_labels)
        for i, row in enumerate(self.pos):
            row_label = f"{i + 1:>2}"  # Right-align single-digit numbers
            row_str = " ".join(map(str, row))
            print(f"{row_label} {row_str}")

    def to_tensor(self) -> torch.Tensor:
        from ai.state_encoder import encode  # 지연 임포트

        # TODO: check encode function
        return encode(self).unsqueeze(0)


