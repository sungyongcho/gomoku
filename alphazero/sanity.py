# sanity.py
from pathlib import Path
import torch
from ai.policy_value_net import PolicyValueNet
from ai.self_play import SelfPlayConfig, play_one_game
from core.game_config import  PLAYER_1, PLAYER_2
from core.gomoku import Gomoku

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ① 최신 체크포인트 로드 (없으면 랜덤 초기화)
model = PolicyValueNet().to(DEVICE).eval()
ckpts = sorted(Path("checkpoints").glob("*.pkl"))
if ckpts:
    print("loading", ckpts[-1])
    model.load_state_dict(torch.load(ckpts[-1], map_location=DEVICE)["model_state_dict"])

# ② self-play 한 판 돌리기 (MCTS sims=600)
cfg = SelfPlayConfig(device=DEVICE, sims=600)
samples = play_one_game(model, cfg)        # [(state, π, z), …]
state, pi, z = samples[0]                  # 첫 턴 샘플만 조사

print("\nπ 분포 통계 (첫 턴)")
print("   max :", float(pi.max()), "min :", float(pi.min()))
print("   전체 합 :", pi.sum())

# ③ 임의 종료국 value 테스트
game = Gomoku()
# 흑(1) 5목 만들기
for x in range(5):
    game.play_move(x, 0)         # 흑
    if x < 4:
        game.play_move(x, 1)     # 백 인터리브
print("winner =", game.winner)

with torch.no_grad():
    planes = game.board.to_tensor().to(DEVICE)
    _, v_pred = model(planes)
v = float(v_pred.item())
print("value head 출력 =", v)
print("  (흑 승자면 + 값이어야 정상)")

