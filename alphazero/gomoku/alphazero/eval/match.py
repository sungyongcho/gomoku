import numpy as np

from gomoku.alphazero.types import action_to_xy
from gomoku.core.gomoku import Gomoku
from gomoku.pvmcts.pvmcts import PVMCTS


def _calculate_q_drop(
    stats: dict, best_q: float, game: Gomoku, action_idx: int
) -> float:
    """Compute Q-value drop for the chosen move (blunder detection).

    Parameters
    ----------
    stats : dict
        Search statistics dictionary.
    best_q : float
        Best Q value among considered moves.
    game : Gomoku
        Game instance for coordinate conversion.
    action_idx : int
        Flat index of the chosen action.

    Returns
    -------
    float
        Non-negative Q-drop value.
    """
    if "q_selected" in stats:
        chosen_q = float(stats["q_selected"])
    else:
        # Fallback: read chosen child Q directly from the tree
        chosen_q = best_q
        if "root" in stats:
            root = stats["root"]
            x, y = action_to_xy(action_idx, game.col_count)
            child = root.children.get((x, y))
            if child:
                chosen_q = float(getattr(child, "q_value", 0.0))

    return max(0.0, best_q - chosen_q)


def play_match(
    game: Gomoku,
    p1: PVMCTS,
    p2: PVMCTS,
    *,
    opening_turns: int,
    temperature: float,
    blunder_threshold: float,
) -> tuple[int, int, int]:
    """Play a single game and return ``(winner, blunder_count, move_count)``.

    Parameters
    ----------
    game : Gomoku
        Game engine instance.
    p1 : PVMCTS
        Player 1 MCTS.
    p2 : PVMCTS
        Player 2 MCTS.
    opening_turns : int
        Number of turns to sample before switching to argmax.
    temperature : float
        Sampling temperature for the opening phase.
    blunder_threshold : float
        Q-drop threshold to count a blunder.

    Returns
    -------
    tuple[int, int, int]
        Winner (1 for p1, -1 for p2, 0 draw), total blunder count, and move count.
    """
    players = {1: p1, 2: p2}
    state = game.get_initial_state()
    turn = 0
    blunders = 0
    move_count = 0

    for _ in range(game.row_count * game.col_count * 2):  # safety guard
        player = players[state.next_player]
        root = player.create_root(state)
        if hasattr(player, "run_search_on_root"):
            policy, stats = player.run_search_on_root(root)
        else:
            policy, stats = player.run_search([root], add_noise=False)[0]
        stats = dict(stats)
        stats.setdefault("root", root)

        # Action selection
        if turn < opening_turns:
            probs = np.nan_to_num(policy, nan=0.0, posinf=0.0, neginf=0.0)
            probs = np.maximum(probs, 0.0)
            total = probs.sum()
            if total > 1e-9:
                probs = probs / total
                action_idx = int(np.random.choice(game.action_size, p=probs))
            else:
                # If the sum is zero, fall back to uniform over legal moves only
                legal_indices = None
                root_node = stats.get("root")
                if root_node is not None and hasattr(root_node, "children"):
                    legal_indices = [
                        x + y * game.col_count for (x, y) in root_node.children
                    ]
                if legal_indices:
                    action_idx = int(np.random.choice(legal_indices))
                else:
                    action_idx = int(np.argmax(policy))
        else:
            action_idx = int(np.argmax(policy))

        # Blunder accounting
        best_q = float(stats.get("q_max", 0.0))
        q_drop = _calculate_q_drop(stats, best_q, game, action_idx)
        if q_drop >= blunder_threshold:
            blunders += 1

        # Apply move
        x, y = action_to_xy(action_idx, game.col_count)
        state = game.get_next_state(state, (x, y), state.next_player)
        turn += 1
        move_count += 1

        # Check termination
        value, is_terminal = game.get_value_and_terminated(state, (x, y))
        if is_terminal:
            if value > 0:
                winner = 1 if player is p1 else -1
            elif value < 0:
                winner = -1 if player is p1 else 1
            else:
                winner = 0
            return winner, blunders, move_count
    raise RuntimeError("Game loop exceeded max turns")
