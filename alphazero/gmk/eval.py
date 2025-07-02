import numpy as np
import torch

from game_config import NUM_HIDDEN_LAYERS, NUM_LINES, NUM_RESBLOCKS, PLAYER_1
from gomoku import GameState, Gomoku, convert_coordinates_to_index
from policy_value_net import PolicyValueNet
from pvmcts import PVMCTS

game = Gomoku()

args = {
    "C": 2,
    "num_searches": 100,
    "dirichlet_epsilon": 0.0,
    "dirichlet_alpha": 0.3,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("bbb")
model = PolicyValueNet(game, NUM_RESBLOCKS, NUM_HIDDEN_LAYERS, device)
model.load_state_dict(torch.load("model_9.pt", map_location=device))
model.eval()
print("aaa")
mcts = PVMCTS(game, args, model)
state: GameState = game.get_initial_state()

while True:
    game.print_board(state)

    if state.next_player == PLAYER_1:
        valid_moves = game.get_legal_moves(state)
        print("valid_moves: ", valid_moves)
        action_str = input(f"{state.next_player} (e.g. A1): ").strip().upper()
        if action_str not in valid_moves:
            print("Action not valid -- invalid string")
            continue
        action = convert_coordinates_to_index(action_str)
        if action is None:
            print("Action not valid -- 2 ")
            continue
    else:
        print(state)
        mcts_probs = mcts.search(state)
        flat_idx = np.argmax(mcts_probs)
        # (y, x) = np.unravel_index(flat_idx, (NUM_LINES, NUM_LINES))
        # action = (x, y)
        action = (flat_idx % NUM_LINES, flat_idx // NUM_LINES)  # (x, y)

        print("action:", action[0], action[1])

    det_player = state.next_player
    state = game.get_next_state(state, action, state.next_player)

    value, is_terminal = game.get_value_and_terminated(state, action)

    if is_terminal:
        game.print_board(state)
        if value == 1:
            print("Player", det_player, "won")
        else:
            print("draw")
        break

    # player = opponent_player(player)
