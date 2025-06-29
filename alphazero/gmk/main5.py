import torch

from alphazero import AlphaZero
from gomoku import Gomoku
from policy_value_net import PolicyValueNet

gomoku = Gomoku()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PolicyValueNet(gomoku, 4, 64, device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=0.0001)

args = {
    "C": 1.4,
    "num_searches": 300,
    "num_iterations": 10,
    "num_selfPlay_iterations": 120,
    "num_epochs": 4,
    "batch_size": 64,
    "temperature": 1.25,
    "dirichlet_epsilon": 0.25,
    "dirichlet_alpha": 0.3,
}

alphaZero = AlphaZero(model, optimizer, gomoku, args)

alphaZero.learn()
