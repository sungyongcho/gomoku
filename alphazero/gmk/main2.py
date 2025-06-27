import numpy as np

from game_config import NUM_LINES, PLAYER_1, opponent_player
from gomoku import Gomoku, convert_coordinates_to_index
from mcts import MCTS

tictactoe = Gomoku()
player = PLAYER_1

args = {"C": 1.41, "num_searches": 1000}

mcts = MCTS(tictactoe, args)

state = tictactoe.get_initial_state()


while True:
    tictactoe.print_board(state)

    if player == 1:
        valid_moves = tictactoe.get_legal_moves(state)
        print("valid_moves: ", valid_moves)
        action_str = input(f"{player} (e.g. A1): ").strip().upper()
        if action_str not in valid_moves:
            print("Action not valid -- invalid string")
            continue
        action = convert_coordinates_to_index(action_str)
        if action is None:
            print("Action not valid -- 2 ")
            continue

    else:
        # neutral_state = tictactoe.change_perspective(state, player)
        print(state)
        mcts_probs = mcts.search(state)
        flat_idx = np.argmax(mcts_probs)
        # (y, x) = np.unravel_index(flat_idx, (NUM_LINES, NUM_LINES))
        # action = (x, y)
        action = (flat_idx % NUM_LINES, flat_idx // NUM_LINES)  # (x, y)

        print("action:", action[0], action[1])

    state = tictactoe.get_next_state(state, action, player)

    value, is_terminal = tictactoe.get_value_and_terminated(state, action)

    if is_terminal:
        tictactoe.print_board(state)
        if value == 1:
            print(player, "won")
        else:
            print("draw")
        break

    player = opponent_player(player)
