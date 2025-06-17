import time

from ai.policy_value_net import PolicyValueNet
from ai.self_play import SelfPlayConfig, play_one_game

cfg = SelfPlayConfig(
    device="cpu", sims=200, temperature=0, resign_threshold=-1.1
)  # 현재 sims 확인
m = PolicyValueNet().eval()

t0 = time.time()
samples = play_one_game(m, cfg)
dt = time.time() - t0
print(f"걸린 시간 {dt:.2f}s, 턴 수 {len(samples)}")
