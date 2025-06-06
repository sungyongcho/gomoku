# ai/arena.py
from __future__ import annotations

import argparse, itertools, torch
from ai.policy_value_net import PolicyValueNet
from ai.pv_mcts          import PVMCTS
from core.gomoku         import Gomoku
from core.game_config    import PLAYER_1, PLAYER_2, DRAW

def load_model(path: str, device: str):
    net = PolicyValueNet().to(device).eval()
    net.load_state_dict(torch.load(path, map_location=device)["model_state_dict"])
    return net

def play_one(game: Gomoku, mcts_a, mcts_b, turn_cap=400):
    while game.winner is None and len(game.history) < turn_cap:
        mcts = mcts_a if game.current_player == PLAYER_1 else mcts_b
        root = mcts.search(game.board)
        move, _ = mcts.get_move_and_pi(root)
        game.play_move(*move)
    return game.winner

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-a", required=True)
    p.add_argument("--model-b", required=True)
    p.add_argument("--games",   type=int, default=200)
    p.add_argument("--sims",    type=int, default=400)
    p.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    net_a = load_model(args.model_a, args.device)
    net_b = load_model(args.model_b, args.device)

    mcts_a = PVMCTS(net_a, sims=args.sims, device=args.device)
    mcts_b = PVMCTS(net_b, sims=args.sims, device=args.device)

    score_a = score_b = draws = 0
    for g in range(1, args.games + 1):
        game = Gomoku()
        # 흑/백 번갈아 가며 동일 판수
        if g % 2 == 0:
            mcts_first, mcts_second = mcts_b, mcts_a
        else:
            mcts_first, mcts_second = mcts_a, mcts_b

        winner = play_one(game, mcts_first, mcts_second, turn_cap=400)
        if winner == PLAYER_1:
            score = (1, 0) if g % 2 == 1 else (0, 1)  # 흑이 model-a?
        elif winner == PLAYER_2:
            score = (0, 1) if g % 2 == 1 else (1, 0)
        else:
            score = (0, 0); draws += 1
        score_a += score[0]; score_b += score[1]

        if g % 20 == 0:
            print(f"[{g}/{args.games}]  A:{score_a}  B:{score_b}  D:{draws}")

    print("\n=== Final ===")
    print(f"Model-A wins : {score_a}")
    print(f"Model-B wins : {score_b}")
    print(f"Draws        : {draws}")
    winrate = 100 * score_a / max(1, score_a + score_b)
    print(f"Model-A win-rate: {winrate:.2f} %")

if __name__ == "__main__":
    main()
