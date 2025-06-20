import torch
from polictvaluenet import ResNet
from selfplay import AlphaZero
from tictactoe import TicTacToe

tictactoe = TicTacToe()

model = ResNet(tictactoe, 4, 64)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

args = {
    "C": 2,
    "num_searches": 60,
    "num_iterations": 3,
    "num_selfPlay_iterations": 500,
    "num_epochs": 4,
    "batch_size": 64,
}

alphaZero = AlphaZero(model, optimizer, tictactoe, args)
alphaZero.learn()
