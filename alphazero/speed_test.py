from core.gomoku import Gomoku
from ai.pv_mcts import PVMCTS
from ai.policy_value_net import PolicyValueNet

# 흑(PLAYER_1)이 바로 승리하는 판
g = Gomoku()
for x in range(5):
    g.play_move(x, 0)          # 흑
    if x < 4:
        g.play_move(x, 1)      # 백

mcts = PVMCTS(PolicyValueNet().eval(), sims=1)
root = mcts.search(g.board)    # 리프 평가/백업 실행

print("root.Q (흑 시점):", root.Q)  # +1.0 근처 → 부호 정상
