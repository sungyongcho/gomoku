import matplotlib.pyplot as plt
import torch
from polictvaluenet import ResNet
from tictactoe import TicTacToe

tictactoe = TicTacToe()

state = tictactoe.get_initial_state()
state = tictactoe.get_next_state(state, 2, -1)
state = tictactoe.get_next_state(state, 4, -1)

state = tictactoe.get_next_state(state, 6, 1)
state = tictactoe.get_next_state(state, 8, 1)


encoded_state = tictactoe.get_encoded_state(state)

tensor_state = torch.tensor(encoded_state).unsqueeze(0)


model = ResNet(tictactoe, 4, 64)
model.load_state_dict(torch.load("model_2.pt"))
model.eval()

policy, value = model(tensor_state)

value = value.item()
policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()

print(value)
print(state)
print(tensor_state)

plt.bar(range(tictactoe.action_size), policy)
plt.show()
