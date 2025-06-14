from __future__ import annotations

import math
from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
import torch
from ai.ai_config import DRAW, LOSE, WIN
from core.board import Board
from core.game_config import NUM_LINES
from core.rules.terminate import is_terminal


class Node:
    def __init__(self, state: Board, parent: Node | None = None, prior: float = 0.0):
        self.state: Board = state
        self.parent: Node | None = parent
        self.children: dict[Tuple[int, int], Node] = {}  # move -> Node
        self.P: float = prior  # 네트워크 prior (policy)
        self.N: int = 0  # 방문 횟수
        self.W: float = 0.0  # 누적 가치
        self.Q: int = 0.0  # 평균 가치 (W / N)

    def is_leaf(self) -> bool:  # 자식 없는지
        return len(self.children) == 0

    def expand(self, move_probs: dict[tuple, float]):
        """
        move_probs : {(row, col): prior_P}
        Board.state 는 이미 현재 플레이어 차례여야 함.
        """
        for (x, y), p in move_probs.items():
            if (x, y) in self.children:  # 이미 있는 경우 skip
                continue
            child_state = deepcopy(self.state)
            child_state.apply_move(x, y, child_state.next_player)
            child_state.last_player, child_state.next_player = (
                child_state.next_player,
                child_state.last_player,
            )
            child_state.last_pts, child_state.next_pts = (
                child_state.next_pts,
                child_state.last_pts,
            )
            self.children[(x, y)] = Node(child_state, parent=self, prior=p)

    # def best_child(self, c_puct: float):
    #     """
    #     Q + c * P * sqrt(N_parent)/(1+N_child) 최대인 자식을 반환.
    #     c_puct : 탐색vs활용 계수.
    #     """
    #     parent_visits = math.sqrt(self.N)
    #     best_score, best_move, best_node = -float("inf"), None, None

    #     for move, child in self.children.items():
    #         ucb = child.Q + c_puct * child.P * parent_visits / (1 + child.N)
    #         if ucb > best_score:
    #             best_score, best_move, best_node = ucb, move, child
    #     return best_node

    def best_child(self, scorer, c_puct: float) -> "Node":
        """
        scorer : Callable[Node, sqrt(N_parent)] → float
        parent 가 전달한 sqrt(N_parent)를 이용해 최대 점수 자식 반환
        """
        sqrt_Np = math.sqrt(self.N)
        return max(self.children.values(), key=lambda ch: scorer(ch, sqrt_Np, c_puct))

    def backup(self, value: float):
        """
        리프에서 얻은 value(현재 노드의 플레이어 관점)
        → 부모 방향으로 -value, +value 번갈아 전파.
        부모로 올라갈 때마다 부호 반전하며 누적.
        """
        node, v = self, value
        while node is not None:
            node.N += 1
            node.W += v
            node.Q = node.W / node.N
            v = -v  # 플레이어 전환
            node = node.parent


class PVMCTS:
    """
    PV-MCTS with pluggable child-selection formula
    ---------------------------------------------
    policy : "ucb1"  – Q + c * P * sqrt(Np)/(1+Nc)          (기본)
             "arc"   – Q + c * P / (1+Nc) + γ * sqrt(Np)/(1+Nc)
                       (예시 ‘ARC’ 변형)
    """

    def __init__(
        self,
        model,
        sims: int = 800,
        c_puct: float = 1.4,
        gamma_arc: float = 0.2,  # ARC only
        policy: str = "ucb1",  # "ucb1" | "arc"
        device: str = "cpu",
    ):
        self.model = model.eval()
        self.sims = sims
        self.c_puct = c_puct
        self.gamma_arc = gamma_arc
        self.policy = policy.lower()
        assert self.policy in {"ucb1", "arc"}
        self.device = device

    # ────────────────────────────────────────────
    # Selection helper
    # ────────────────────────────────────────────
    def _score(self, child: Node, parent_sqrt: float, c: float) -> float:
        if self.policy == "ucb1":
            return child.Q + c * child.P * parent_sqrt / (1 + child.N)
        # ARC :  Q + c·P/(1+N) + γ·√N_p /(1+N)
        return (
            child.Q
            + c * child.P / (1 + child.N)
            + self.gamma_arc * parent_sqrt / (1 + child.N)
        )

    # ────────────────────────────────────────────
    # Search (Selection → Expansion/Eval → Backup)
    # ────────────────────────────────────────────
    def search(self, root_state: Board) -> Node:
        root = Node(deepcopy(root_state))
        self._expand_eval(root)

        for _ in range(self.sims):
            node = root
            # Selection
            while not node.is_leaf():
                node = node.best_child(self._score, self.c_puct)
            # Expansion + Evaluation
            self._expand_eval(node)
        return root

    def _expand_eval(self, node: Node):
        # 1) 게임 종료면 바로 백업
        winner = is_terminal(node.state)
        if winner is not None:
            if winner == node.state.last_player:
                v = WIN
            elif winner == node.state.next_player:
                v = LOSE
            else:
                v = DRAW
            node.backup(v)
            return

        # 2) 신경망 추론
        with torch.no_grad():
            planes = node.state.to_tensor().to(self.device)
            log_pi, v_raw = self.model(planes)
        policy = (
            torch.exp(log_pi.squeeze(0))  # (N²,) 확률
            .view(NUM_LINES, NUM_LINES)  # → (N, N)
            .cpu()
            .numpy()
        )
        value = float(v_raw.item())  # value-head 는 이미 tanh(−1 ~ 1) 범위임

        move_probs: Dict[Tuple[int, int], float] = {}
        total_p = 0.0
        for x, y in node.state.legal_moves(node.state.next_player):
            p = float(policy[y, x])
            move_probs[(x, y)] = p
            total_p += p
        if total_p > 0:
            for k in move_probs:
                move_probs[k] /= total_p
        else:  # degenerate → 균등
            uniform = 1.0 / len(move_probs)
            for k in move_probs:
                move_probs[k] = uniform

        # 4) 자식 노드 확장
        node.expand(move_probs)

        # 5) 가치 백업
        node.backup(value)

    def get_move_and_pi(self, root: Node):
        """
        Returns
        -------
        best_move : (x, y) - 방문수 최다
        pi        : (N, N) - 방문수 정규화 확률
        """
        visits = np.zeros((NUM_LINES, NUM_LINES), dtype=np.float32)
        for (x, y), child in root.children.items():
            visits[y, x] = child.N

        if visits.sum() == 0:
            visits += 1.0

        pi = visits / visits.sum()
        best_y, best_x = divmod(visits.argmax(), NUM_LINES)  # unravel
        return (best_x, best_y), pi

    @staticmethod
    def apply_dirichlet_noise(
        root: Node,
        alpha: float = 0.3,  # 19×19 기준 추천값
        epsilon: float = 0.25,
    ) -> None:
        """
        AlphaZero 방식의 root exploration noise.

        P' = (1-ε)·P  +  ε·Dir(α)
        """
        if not root.children:  # leaf safety
            return
        moves = list(root.children.keys())
        noise = np.random.dirichlet([alpha] * len(moves))
        for move, n in zip(moves, noise):
            child = root.children[move]
            child.P = child.P * (1 - epsilon) + n * epsilon

    @staticmethod
    def sample_with_temperature(
        pi: np.ndarray,
        temperature: float = 1.0,
    ) -> tuple[int, int]:
        """
        볼츠만 분포로 (x, y) 선택.
        temperature → 0  이면 argmax, 1 은 그대로,  >1 은 더 균등.
        """
        flat = pi.flatten()
        if temperature != 1.0:
            flat = np.power(flat, 1.0 / temperature)
            flat /= flat.sum()
        choice = np.random.choice(len(flat), p=flat)
        y, x = divmod(choice, NUM_LINES)
        return int(x), int(y)
