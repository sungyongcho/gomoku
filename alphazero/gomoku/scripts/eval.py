import argparse
import os
from typing import Any

import fsspec
import numpy as np
import torch

from gomoku.alphazero.agent import AlphaZeroAgent
from gomoku.alphazero.types import action_to_xy
from gomoku.core.game_config import (
    PLAYER_1,
    convert_coordinates_to_index,
    convert_index_to_coordinates,
)
from gomoku.core.gomoku import Gomoku
from gomoku.inference.local import LocalInference
from gomoku.model.model_helpers import (
    POLICY_CHANNELS,
    VALUE_CHANNELS,
    calc_num_planes,
)
from gomoku.model.policy_value_net import PolicyValueNet
from gomoku.utils.config.loader import (
    BoardConfig,
    MctsConfig,
    ModelConfig,
    load_and_parse_config,
)
from gomoku.utils.manifest import ensure_manifest
from gomoku.utils.paths import format_path, load_state_dict_from_fs
from gomoku.utils.state_dict_utils import align_state_dict_to_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play/evaluate a trained Gomoku AlphaZero model against a human."
    )
    parser.add_argument(
        "--config",
        default="configs/config_alphazero_test.yaml",
        help="Path to RootConfig YAML.",
    )
    parser.add_argument(
        "--run-id",
        help="Run ID to evaluate (overrides config run_id).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu"],
        help="Device selection (LocalInference is CPU-only).",
    )
    parser.add_argument("--visits", type=int, default=1000, help="MCTS num_searches.")
    parser.add_argument("--C", type=float, default=2.0, help="PUCT constant.")
    parser.add_argument(
        "--exploration-turns",
        type=int,
        default=0,
        help="Exploration turns (recommended 0).",
    )
    parser.add_argument(
        "--dirichlet-epsilon", type=float, default=0.0, help="Dirichlet epsilon."
    )
    parser.add_argument(
        "--dirichlet-alpha", type=float, default=0.3, help="Dirichlet alpha."
    )
    parser.add_argument(
        "--strict-opening",
        action="store_true",
        help="If set, disallow moves outside trained candidate mask (default allows any empty cell).",
    )
    parser.add_argument(
        "--debug-policy",
        action="store_true",
        help="Print top policy moves on the initial board for debugging.",
    )
    parser.add_argument(
        "--debug-mcts",
        action="store_true",
        help="Print top policy moves at each AI turn.",
    )
    parser.add_argument(
        "--human-power",
        action="store_true",
        help="Force aggressive MCTS settings for human play (higher visits, no noise, batch size 1).",
    )
    parser.add_argument(
        "--ai-first",
        action="store_true",
        help="Let AI play first (as Player 1). By default, human plays first.",
    )
    return parser.parse_args()


def _choose_fs(paths_cfg: Any) -> fsspec.AbstractFileSystem:
    protocol = "file"
    if getattr(paths_cfg, "use_gcs", False):
        protocol = "gcs"
    return fsspec.filesystem(protocol)


def _resolve_weights_path(
    manifest: dict[str, Any], paths_cfg, fs: fsspec.AbstractFileSystem
) -> str:
    candidates: list[str] = []
    champ = manifest.get("champion_model_path")
    if champ:
        candidates.append(champ)
        base = os.path.basename(champ)
        if base:
            ckpt_dir = format_path(paths_cfg, paths_cfg.ckpt_dir)
            candidates.append(os.path.join(ckpt_dir, base))
    ckpt_dir = format_path(paths_cfg, paths_cfg.ckpt_dir)
    candidates.append(os.path.join(ckpt_dir, "champion.pt"))

    tried = set()
    for cand in candidates:
        local = os.path.abspath(cand)
        if local in tried:
            continue
        tried.add(local)
        if fs.exists(local):
            return local
    raise FileNotFoundError(
        f"Champion weights not found. Tried: {', '.join(sorted(tried))}"
    )


def load_model_and_game(
    paths_cfg,
    fs: fsspec.AbstractFileSystem,
    device_str: str,
    use_native: bool = False,
) -> tuple[PolicyValueNet, Gomoku]:
    manifest = ensure_manifest(paths_cfg, fs)

    board_cfg = BoardConfig.model_validate(manifest["board"])
    model_cfg_raw = manifest["model"]
    model_cfg = ModelConfig.model_validate(model_cfg_raw)
    updates: dict[str, Any] = {}
    if model_cfg.num_planes is None:
        updates["num_planes"] = calc_num_planes(board_cfg.history_length)
    if model_cfg.policy_channels is None:
        updates["policy_channels"] = POLICY_CHANNELS
    if model_cfg.value_channels is None:
        updates["value_channels"] = VALUE_CHANNELS
    if updates:
        model_cfg = model_cfg.model_copy(update=updates)
    game = Gomoku(board_cfg, use_native=use_native)

    weights_file = _resolve_weights_path(manifest, paths_cfg, fs)
    model = PolicyValueNet(game, model_cfg, device=device_str)
    state_dict = load_state_dict_from_fs(fs, weights_file, torch.device(device_str))
    resized, missing, dropped = align_state_dict_to_model(
        state_dict, model.state_dict()
    )
    if resized:
        print(
            "[StateDict] Resized tensors:",
            ", ".join(resized[:10]) + ("..." if len(resized) > 10 else ""),
        )
    if missing:
        print(f"[StateDict] Missing keys in checkpoint: {len(missing)}")
    if dropped:
        print(f"[StateDict] Dropped unmatched keys from checkpoint: {len(dropped)}")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, game


def play_vs_human(
    agent: AlphaZeroAgent,
    game: Gomoku,
    mcts_cfg: MctsConfig,
    human_is: int = PLAYER_1,
    allow_full_open: bool = True,
    debug_mcts: bool = False,
) -> None:
    state = game.get_initial_state()
    while True:
        game.print_board(state)
        if state.next_player == human_is:
            legal_moves = game.get_legal_moves(state)
            if allow_full_open:
                legal_moves = [mv for mv in legal_moves if mv in legal_moves]
            move_str = (
                input(f"Player {state.next_player} (human), enter move (e.g. A1): ")
                .strip()
                .upper()
            )
            action = convert_coordinates_to_index(move_str, game.col_count)
            if move_str not in legal_moves or action is None:
                print("Invalid move, please try again.")
                continue
        else:
            policy_debug = None
            if debug_mcts:
                # temperature=1.0로 방문 기반 분포를 그대로 보기 위한 추가 호출
                policy_debug = agent.get_action_probs(
                    state, temperature=1.0, add_noise=False
                )
            policy = agent.get_action_probs(state, temperature=0.0, add_noise=False)
            idx = int(np.argmax(policy))
            x, y = action_to_xy(idx, game.col_count)
            action = (x, y)
            move_str = convert_index_to_coordinates(x, y, game.col_count)
            if debug_mcts:
                root = agent.root
                if root is not None:
                    visited_children = sum(
                        1 for c in root.children.values() if c.visit_count > 0
                    )
                    print(
                        f"[Debug] root_visits={root.visit_count} "
                        f"children={len(root.children)} visited_children={visited_children}"
                    )
                dist = policy_debug if policy_debug is not None else policy
                topk = np.argsort(dist)[::-1][:10]
                print("[Debug] AI policy top-k (after MCTS visits):")
                for i in topk:
                    px, py = action_to_xy(int(i), game.col_count)
                    mv = convert_index_to_coordinates(px, py, game.col_count)
                    print(f"  p={float(dist[i]):.4f} move={mv} ({px},{py})")
            print(f"AI chooses: {move_str} -> ({x}, {y})")

        current_player = state.next_player
        state = game.get_next_state(state, action, current_player)
        value, terminal = game.get_value_and_terminated(state, action)
        if terminal:
            game.print_board(state)
            if value == 1:
                winner = "Human" if current_player == human_is else "AI"
                print(f"{winner} wins!")
            else:
                print("Draw.")
            break


def main() -> None:
    args = parse_args()
    cfg = load_and_parse_config(args.config)
    paths_cfg = cfg.paths
    if args.run_id:
        paths_cfg = paths_cfg.model_copy(update={"run_id": args.run_id})

    fs = _choose_fs(paths_cfg)
    device_str = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device_str}")

    model, game = load_model_and_game(
        paths_cfg, fs, device_str, use_native=cfg.mcts.use_native
    )
    if args.debug_policy:
        state0 = game.get_initial_state()
        encoded = torch.as_tensor(
            game.get_encoded_state([state0])[0], device=device_str
        ).unsqueeze(0)
        with torch.no_grad():
            logits, value = model(encoded)
            policy = torch.softmax(logits[0], dim=-1)
        topk = torch.topk(policy, k=min(10, policy.numel()))
        print(f"[Debug] value={float(value.squeeze().item()):.4f}")
        for p, idx in zip(topk.values, topk.indices):
            x, y = action_to_xy(int(idx), game.col_count)
            mv = convert_index_to_coordinates(x, y, game.col_count)
            print(f"[Debug] p={float(p):.4f} move={mv} ({x},{y})")
    inference = LocalInference(model)
    base_mcts_cfg = cfg.mcts
    visits_override = int(args.visits)
    if args.human_power:
        print(">>> [Human Mode] Forcing high-strength MCTS (no noise, batch size 1).")
        visits_override = max(visits_override, 800)
        base_mcts_cfg = base_mcts_cfg.model_copy(
            update={
                "dirichlet_epsilon": 0.0,
                "exploration_turns": 0,
                "batch_infer_size": 1,
                "min_batch_size": 1,
                "max_batch_wait_ms": 0,
            }
        )
    eval_mcts_cfg = base_mcts_cfg.model_copy(
        update={
            "C": float(args.C),
            "num_searches": visits_override,
            "exploration_turns": 0 if args.human_power else int(args.exploration_turns),
            "dirichlet_epsilon": 0.0
            if args.human_power
            else float(args.dirichlet_epsilon),
            "dirichlet_alpha": float(args.dirichlet_alpha),
        }
    )
    agent = AlphaZeroAgent(
        game=game,
        mcts_cfg=eval_mcts_cfg,
        inference_client=inference,
        engine_type="sequential",
    )

    allow_full_open = not args.strict_opening
    human_is = 2 if args.ai_first else PLAYER_1  # AI first = human is Player 2
    play_vs_human(
        agent,
        game,
        eval_mcts_cfg,
        human_is=human_is,
        allow_full_open=allow_full_open,
        debug_mcts=args.debug_mcts,
    )


if __name__ == "__main__":
    main()
