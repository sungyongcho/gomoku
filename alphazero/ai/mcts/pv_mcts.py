import math
from copy import deepcopy

import numpy as np
import torch
from ai.state_encoder import encode
from core.board import Board
from core.game_config import DRAW, LOSE, NUM_LINES, WIN
from core.rules.terminate import is_terminal


class Node:
    def __init__(self, state, parent=None, prior=0.0):
        self.state: Board = state
        self.parent: Node | None = parent
        self.children: dict[Node] = {}  # move -> Node
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
        for move, p in move_probs.items():
            if move in self.children:  # 이미 있는 경우 skip
                continue
            new_state = deepcopy(self.state)
            r, c = move
            new_state.set_value(c, r, new_state.current_player)  # (col, row) 순서 주의
            self.children[move] = Node(new_state, parent=self, prior=p)

    def best_child(self, c_puct: float):
        """
        Q + c * P * sqrt(N_parent)/(1+N_child) 최대인 자식을 반환.
        c_puct : 탐색vs활용 계수.
        """
        parent_visits = math.sqrt(self.N)
        best_score, best_move, best_node = -float("inf"), None, None

        for move, child in self.children.items():
            ucb = child.Q + c_puct * child.P * parent_visits / (1 + child.N)
            if ucb > best_score:
                best_score, best_move, best_node = ucb, move, child
        return best_node

    def backup(self, value: float):
        """
        리프에서 얻은 value(현재 노드의 플레이어 관점)
        → 부모 방향으로 -value, +value 번갈아 전파.
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
        gamma_arc: float = 0.2,
        policy: str = "ucb1",
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
    def _score(self, child: Node, parent_visits: float) -> float:
        if self.policy == "ucb1":
            return child.Q + self.c_puct * child.P * parent_visits / (1 + child.N)

        # ARC 예시: priors와 방문수를 분리 가중
        # score = Q + c*P/(1+N) + γ*sqrt(N_parent)/(1+N)
        return (
            child.Q
            + self.c_puct * child.P / (1 + child.N)
            + self.gamma_arc * parent_visits / (1 + child.N)
        )

    # ────────────────────────────────────────────
    # Search (Selection → Expansion/Eval → Backup)
    # ────────────────────────────────────────────
    def search(self, root_state):
        root = Node(deepcopy(root_state))
        self._expand_eval(root)

        for _ in range(self.sims):
            node = root
            # Selection
            while not node.is_leaf():
                parent_vis = math.sqrt(node.N)
                node = max(
                    node.children.values(),
                    key=lambda ch: self._score(ch, parent_vis),
                )
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
        x = encode(node.state).unsqueeze(0).to(self.device)  # (1,C,N,N)
        with torch.no_grad():
            p_tensor, v_tensor = self.model(x)  # p: (1,N,N), v: (1,1)

        # 2-a. policy (확률 평면 → numpy 2-D)
        policy_plane = p_tensor.squeeze(0).cpu().numpy()  # (N,N) 확률

        # 2-b. value (0~1 → -1~1로 매핑)
        value = 2.0 * v_tensor.item() - 1.0  # [-1,1]

        # 3) 합법 수 prior 추출 + 정규화
        move_probs, prob_sum = {}, 0.0
        for r, c in node.state.get_legal_moves():  # (row,col)
            p = policy_plane[r, c]
            move_probs[(r, c)] = p
            prob_sum += p

        if prob_sum > 0:
            for m in move_probs:
                move_probs[m] /= prob_sum
        else:  # 모든 prior가 0 → 균등
            uniform = 1.0 / len(move_probs)
            for m in move_probs:
                move_probs[m] = uniform

        # 4) 자식 노드 확장
        node.expand(move_probs)

        # 5) 가치 백업
        node.backup(value)

    def get_move_and_pi(self, root: Node):
        """
        반환
        -------
        best_move : (row, col)  – 방문수가 가장 많은 착수
        pi        : (N, N) np.ndarray  – 방문수 분포를 확률로 정규화
        """
        visits = np.zeros((NUM_LINES, NUM_LINES), dtype=np.float32)

        for (r, c), child in root.children.items():
            visits[r, c] = child.N

        if visits.sum() == 0:  # 안전 장치
            visits += 1.0

        pi = visits / visits.sum()
        best_idx = np.unravel_index(np.argmax(visits), visits.shape)
        return best_idx, pi
