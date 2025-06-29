import matplotlib.pyplot as plt
import torch

from game_config import PLAYER_1, PLAYER_2
from gomoku import Gomoku
from policy_value_net import PolicyValueNet

gomoku = Gomoku()
state = gomoku.get_initial_state()
# state = gomoku.get_next_state(state, (2, 0), PLAYER_2)
# state = gomoku.get_next_state(state, (1, 1), PLAYER_2)
# state = gomoku.get_next_state(state, (0, 2), PLAYER_1)
# state = gomoku.get_next_state(state, (2, 2), PLAYER_1)

state = gomoku.get_next_state(state, (2, 0), PLAYER_1)
state = gomoku.get_next_state(state, (1, 2), PLAYER_2)

encoded_state = gomoku.get_encoded_state(state)

tensor_state = torch.tensor(encoded_state).unsqueeze(0)

model = PolicyValueNet(gomoku, 4, 64)
model.load_state_dict(torch.load("model_7.pt"))
model.eval()


policy, value = model(tensor_state)
value = value.item()
policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()

gomoku.print_board(state)
print(value, policy)
print(tensor_state)

plt.bar(range(gomoku.action_size), policy)
plt.show()
