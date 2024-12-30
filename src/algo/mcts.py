import numpy as np
from config import *


class TreeNode:
    def __init__(self, board, parent, prior_probs=0.0):
        self.board = board
        # init is node terminal flag
        if board.is_win_board() or board.is_draw():
            self.is_terminal = True
        else:
            self.is_terminal = False

        # init is fully expanded flag
        self.is_fully_expanded = self.is_terminal
        self.parent = parent
        # the number of node visitis
        self.visits = 0
        # the action value of the node
        self.total_value = 0
        self.prior_probs = prior_probs
        # current node's children
        self.children = {}


class MCTS:
    def __init__(self, model):
        self.model = model
        self.game_state = [np.zeros((NUM_LINES, NUM_LINES)) for _ in range(17)]

    # search for the best move in the current position
    def search(self, initial_state):
        # create root node
        self.root = TreeNode(initial_state, None)

        for iteration in range(10):
            # select a node (selection phase)
            node = self.select(self.root)

            move_probs, value = self.evaluate_board(node.board)

            # backpropagate results
            self.backpropagate(node, value)

        # pick up the best move in the current position
        action = self.select_action(self.root)
        # print("action in mcts search:", action)
        return initial_state.make_move(action[0], action[1])
        # return action

    # select most promising node
    def select(self, node):
        # print("Selection...")
        while not node.is_terminal:
            if node.is_fully_expanded:
                # print("node is fully expanded", node.children)
                node = node.children[self.select_action(node)]
            else:
                return self.expand(node)
        return node

    def convert_game_state(self):
        game_state = np.array(self.game_state)
        game_state = np.transpose(game_state, (1, 2, 0))
        # game_state = game_state.reshape(-1, 9, 9, 17)
        # game_state.shape becomes (1, 9, 9, 17)
        game_state = np.expand_dims(game_state, axis=0)
        # print("game_state.shape:", game_state.shape, type(game_state))
        return game_state

    # expand node
    def expand(self, node):
        # print("Expansion...")
        # generate legal states for the given node
        state_lst = node.board.generate_states()
        # print(f"generated states: {state_lst}")
        for board, action in state_lst:
            # print("state:", state)
            # make sure the current state is not present in child nodes
            if action not in node.children:
                # Get the prior probability for this state from the policy head of the neural network.
                self.preprocess_board(board, board.turn)
                policy_probs, _ = self.model.predict(self.convert_game_state())
                action_index = self.action_to_index(action)
                prior_prob = policy_probs[0][action_index]
                """
                print(
                    f"policy probs: {policy_probs}\naction index: {action_index}\nprior_prob: {prior_prob}"
                )
                """

                # create a new node
                new_node = TreeNode(board, node, prior_prob)

                # add child node to parent's node children list (dict)
                node.children[action] = new_node

                # case when node is fully expanded
                if len(state_lst) == len(node.children):
                    node.is_fully_expanded = True

                # return newly created node
                return new_node

    def evaluate_board(self, board):
        # self.preprocess_board(board, board.turn)

        # predict the move probabilities(p) and the value(v) of the board state.
        move_probs, value = self.model.predict(self.convert_game_state())

        return move_probs, value

    def preprocess_board(self, board, player_turn) -> None:
        """
        Convert the board state to a suitable format for the model.

            self.game_state = [X_t, Y_t, X_t-1, Y_t-1, ... X_t-7, Y_t-7, C]
            X: 8 feature planes of black stone
            Y: 8 feature planes of white stone
            C: the color of player stone; 1 for black, 0 for white
        """
        # get each board state of black and white from the current board position
        player_1_state = board.create_board_state(PLAYER_1)
        player_2_state = board.create_board_state(PLAYER_2)

        # insert each board state of black and white to the beginning of the game_state list
        self.game_state.insert(0, player_1_state)
        self.game_state.insert(1, player_2_state)

        # remove the second and third board states to the last
        del self.game_state[-3:-1]

        # last element of the game_state standing for the color of the player stone; 1 for black, 0 for white
        if player_turn == PLAYER_1:
            self.game_state[-1] = np.zeros((NUM_LINES, NUM_LINES))
        else:
            self.game_state[-1] = np.ones((NUM_LINES, NUM_LINES))

    # backpropagate the number of visits and total_value up to the root node
    def backpropagate(self, node, action_value):
        # update nodes's up to root node
        while node is not None:
            # update node's visits
            node.visits += 1

            # update node's total_value
            node.total_value += action_value
            action_value -= action_value

            # set node to parent
            node = node.parent

    def select_action(self, node):
        best_score = float("-inf")
        best_action = None

        for action, child_node in node.children.items():
            Q = child_node.total_value / child_node.visits  # average value
            U = child_node.prior_probs / (1 + child_node.visits)  # exploration term
            score = Q + U
            """
            print("score in select_action():", score, type(score))
            print("best_score in select_action():", best_score, type(best_score))
            """
            max_score = np.argmax(score)

            if max_score > best_score:
                # if score > best_score:
                best_score = max_score
                best_action = action

        return best_action

    def action_to_index(self, action):
        # Convert an action to an index into the policy_probs array.
        # Since your action is likely a 2D tuple (row, col) and policy_probs is a 1D array,
        # you need to flatten the action to get the correct index.
        """
        print("action: ", action)
        print("action index: ", np.ravel_multi_index(action, (NUM_LINES, NUM_LINES)))
        """
        return np.ravel_multi_index(action, (NUM_LINES, NUM_LINES))
