import torch

from alphazero import AlphaZero
from gomoku import Gomoku
from policy_value_net import PolicyValueNet

gomoku = Gomoku()

model = PolicyValueNet(gomoku, 4, 64)

optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

args = {
    "C": 1.4,
    "num_searches": 300,
    "num_iterations": 10,
    "num_selfPlay_iterations": 120,
    "num_epochs": 4,
    "batch_size": 64,
}

alphaZero = AlphaZero(model, optimizer, gomoku, args)

alphaZero.learn()
