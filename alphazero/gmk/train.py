import json

import torch

from alphazero import AlphaZero
from game_config import NUM_LINES
from gomoku import Gomoku
from policy_value_net import PolicyValueNet


def calc_num_hidden(num_lines: int, min_ch: int = 32, max_ch: int = 128) -> int:
    """
    보드 한 변(N)을 받아 적절한 num_hidden(2의 거듭제곱)을 리턴.
    규칙: N ≤ 6 → 32, 7~12 → 64, 13↑ → 128
    """
    if num_lines <= 6:
        val = 32
    elif num_lines <= 12:
        val = 64
    else:
        val = 128
    # 혹시 모르는 사용자가 min/max를 바꿔도 안전하게
    val = max(min_ch, min(max_ch, val))
    return val


def calc_num_resblocks(num_lines: int) -> int:
    # 3~6 → 2, 7~12 → 4, 13↑ → 6
    if num_lines <= 6:
        return 2
    elif num_lines <= 12:
        return 4
    else:
        return 6


args = {
    # -------------------- MCTS-related --------------------
    "C": 2,  # UCB exploration constant
    #   larger → explores unseen / low-visit moves more
    #   smaller → relies on high-Q, well-visited moves more
    "num_searches": 60,  # Monte-Carlo tree-search rollouts per move
    #   larger → better statistics, slower per turn
    #   smaller → faster, less accurate
    # -------- self-play volume per outer iteration --------
    "num_iterations": 3,  # how many [self-play → train] outer cycles
    #   larger → longer overall training, stronger model
    "num_selfPlay_iterations": 500,  # self-play games generated in ONE iteration
    #   larger → more fresh data, runtime ↑
    #   smaller → risk of over-fit / noisy gradients
    # ---------------- training loop ----------------------
    "num_epochs": 4,  # passes over the SAME memory set in one iteration
    #   larger → better convergence, risk of over-fit
    #   smaller → may under-fit
    "batch_size": 64,  # mini-batch size used by the optimiser
    #   larger → steadier gradients & better GPU utilisation
    #   smaller → fits in low VRAM, noisier gradients
    "learning_rate": 0.002,
    "weight_decay": 0.0001,
    # --------------- exploration modifiers ---------------
    "temperature": 1.25,  # **Boltzmann temperature** applied to π before sampling
    #   >1.0 → flatter distribution  (more exploration)
    #   <1.0 → sharper distribution (more exploitation)
    #   0.0  → argmax (greedy)      (typically evaluation)
    # Dirichlet noise is mixed into root prior to force
    # at-least-once exploration of every legal child
    "dirichlet_epsilon": 0.25,  # how much of the root prior is replaced by noise
    #   0.25 → 75 % model policy + 25 % noise
    #   0.0  → no noise
    "dirichlet_alpha": 0.3,  # concentration parameter of Dirichlet(α,…,α)
    #   smaller α  → “spiky” noise (one child gets a big boost)
    #   larger α   → almost uniform noise
    "exploration_turns": 15,  # Number of initial moves where temperature is applied for exploration.
    #   For the first `exploration_turns` moves, the AI samples an action from the MCTS policy distribution.
    #   After this, it switches to a greedy approach, always selecting the move with the highest visit count.
    #   This ensures a balance between exploring diverse opening strategies and exploiting strong lines later in the game.
    # ---------------Neural Network Architecture --------------------
    "num_planes": 6,
    "num_resblocks": calc_num_resblocks(NUM_LINES),
    "num_hidden": calc_num_hidden(NUM_LINES),
    # ---game config---
    "enable_doublethree": False,
    "enable_capture": False,
    # --- arena----
    "num_eval_games": 20,
    "eval_win_rate": 0.55,
}


gomoku = Gomoku(
    enable_doublethree=args["enable_doublethree"],
    enable_capture=args["enable_capture"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = PolicyValueNet(
    gomoku, args["num_planes"], args["num_resblocks"], args["num_hidden"], device
)

optimizer = torch.optim.Adam(
    model.parameters(), lr=args["learning_rate"], weight_decay=args["weight_decay"]
)

alphaZero = AlphaZero(model, optimizer, gomoku, args)

alphaZero.learn()


# [추가] 모델 구조와 관련된 설정만 따로 저장합니다.
model_config = {
    "num_planes": args["num_planes"],
    "num_resblocks": args["num_resblocks"],
    "num_hidden": args["num_hidden"],
}
with open("model_config.json", "w") as f:
    json.dump(model_config, f)

print("Model config saved to model_config.json")
