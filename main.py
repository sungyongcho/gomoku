from src.interface.controller.game_controller import GameController
from src.algo.conv import create_CNN_model, create_mini_CNN_model
from src.algo.mcts import MCTS
from config import *
from src.game.board import Board
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gomoku game with optional modes.")

    parser.add_argument("--selfplay", action="store_true", help="Enable selfplay mode.")
    parser.add_argument("--train", action="store_true", help="Enable train mode.")
    args = parser.parse_args()

    if NUM_LINES == 19:
        model = create_CNN_model()
    elif NUM_LINES == 9:
        model = create_mini_CNN_model()
    else:
        raise ValueError(f"Unsupported board size: {NUM_LINES}, choose either 19 or 9.")

    if args.train == True:
        print("coming soon")

    elif args.selfplay == True:
        game = GameController(SCREEN_WIDTH, SCREEN_HEIGHT, model)
        game.init_selfplay()
        while game.running == True:
            game.run()

    else:
        game = GameController(SCREEN_WIDTH, SCREEN_HEIGHT, model)
        game.init_game()
        while game.running == True:
            game.run()
