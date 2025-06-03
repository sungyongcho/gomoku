# ai/train_loop.py
# ─────────────────────────────────────────────────────────────
# ‘셀프플레이 → 버퍼추가 → 학습’ 을 반복하는 간단한 학습 루프
# ─────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F
from ai.policy_value_net import PolicyValueNet
from ai.pv_mcts import PVMCTS
from ai.replay_buffer import ReplayBuffer
from ai.self_play import play_one_game

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 하이퍼파라미터
N_GAMES_PER_ITER = 25  # 셀프플레이 판수
BATCH_SIZE = 128
EPOCHS_PER_ITER = 5
TOTAL_ITERS = 1000
LR = 1e-3

# 객체 준비
model = PolicyValueNet().to(DEVICE)
mcts = PVMCTS(model, sims=400, device=DEVICE)
buffer = ReplayBuffer(max_len=100_000)
optim = torch.optim.Adam(model.parameters(), lr=LR)

for it in range(TOTAL_ITERS):
    # 1) ───── 셀프플레이로 경험 수집 ─────
    for _ in range(N_GAMES_PER_ITER):
        play_one_game(mcts, buffer)  # 버퍼 크기가 점점 증가

    # 버퍼에 샘플이 충분히 쌓일 때까지 대기
    if len(buffer) < BATCH_SIZE:
        print(f"iter {it}: buffer {len(buffer)}  (skip training)")
        continue

    # 2) ───── 파라미터 갱신 파트 ─────
    model.train()
    for _ in range(EPOCHS_PER_ITER):
        s, pi, z = buffer.sample(BATCH_SIZE)  # torch.Tensor
        s, pi, z = s.to(DEVICE), pi.to(DEVICE), z.to(DEVICE)

        pred_pi, pred_v = model(s)  # pred_pi : (B,N,N)
        # policy loss :  CE(π, π̂) ≈ KLDiv
        loss_p = F.kl_div(torch.log(pred_pi + 1e-8), pi, reduction="batchmean")
        # value loss :  MSE(z, v̂)
        loss_v = F.mse_loss(pred_v, z)
        loss = loss_p + loss_v

        optim.zero_grad()
        loss.backward()
        optim.step()

    model.eval()  # 탐색용은 항상 eval 모드
    print(f"iter {it:03d} | buffer {len(buffer):6d} | loss {loss.item():.4f}")

    # 3) ───── (선택) 체크포인트 저장 ─────
    if it % 50 == 0:
        torch.save(model.state_dict(), f"models/iter_{it:03d}.pth")
