import matplotlib.pyplot as plt
import torch

from game_config import PLAYER_1, PLAYER_2
from gomoku import Gomoku
from policy_value_net import PolicyValueNet

torch.manual_seed(0)

gomoku = Gomoku()
player = PLAYER_1

args = {"C": 1.41, "num_searches": 1000}

model = PolicyValueNet(gomoku, 4, 64)
model.eval()
state = gomoku.get_initial_state()
state = gomoku.get_next_state(state, (2, 0), PLAYER_1)
state = gomoku.get_next_state(state, (1, 2), PLAYER_2)

gomoku.print_board(state)

encoded_state = gomoku.get_encoded_state(state)

print(encoded_state)

tensor_state = torch.tensor(encoded_state).unsqueeze(0)

model = PolicyValueNet(gomoku, 4, 64)

policy, value = model(tensor_state)

value = value.item()

policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()

print(value, policy)

plt.bar(range(gomoku.action_size), policy)
plt.show()
