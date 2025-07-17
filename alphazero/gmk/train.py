import torch

from alphazero import AlphaZero
from game_config import NUM_HIDDEN_LAYERS, NUM_RESBLOCKS
from gomoku import Gomoku
from policy_value_net import PolicyValueNet

gomoku = Gomoku()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PolicyValueNet(gomoku, NUM_RESBLOCKS, NUM_HIDDEN_LAYERS, device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=0.0001)

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
}


alphaZero = AlphaZero(model, optimizer, gomoku, args)

alphaZero.learn()
