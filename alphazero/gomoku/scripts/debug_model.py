import argparse
from typing import Any

import numpy as np
import torch

from gomoku.alphazero.types import action_to_xy
from gomoku.core.game_config import PLAYER_1, PLAYER_2, convert_index_to_coordinates
from gomoku.core.gomoku import Gomoku
from gomoku.model.model_helpers import (
    POLICY_CHANNELS,
    VALUE_CHANNELS,
    calc_num_planes,
)
from gomoku.model.policy_value_net import PolicyValueNet
from gomoku.utils.config.loader import (
    BoardConfig,
    ModelConfig,
    RootConfig,
    load_and_parse_config,
)
from gomoku.utils.manifest import ensure_manifest
from gomoku.utils.paths import format_path, load_state_dict_from_fs
from gomoku.utils.state_dict_utils import align_state_dict_to_model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug model policy/value on a hand-crafted board situation."
    )
    parser.add_argument(
        "--config",
        default="configs/9x9_local_hardcore.yaml",
        help="RootConfig YAML 경로.",
    )
    parser.add_argument(
        "--run-id",
        help="manifest를 덮어쓸 run_id (없으면 config의 run_id 사용).",
    )
    parser.add_argument(
        "--checkpoint",
        help="직접 지정할 ckpt 경로(없으면 manifest의 champion 우선 사용).",
    )
    parser.add_argument("--topk", type=int, default=10, help="출력할 상위 수 개수.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu"],
        help="디바이스 선택(cpu-only 모델).",
    )
    return parser.parse_args()


def _resolve_weights(
    cfg: RootConfig, fs: Any, checkpoint_override: str | None
) -> tuple[str, dict[str, Any]]:
    paths_cfg = cfg.paths
    if checkpoint_override:
        return checkpoint_override, {}

    manifest = ensure_manifest(paths_cfg, fs)
    candidates: list[str] = []
    champ = manifest.get("champion_model_path")
    if champ:
        candidates.append(champ)
        base = champ.split("/")[-1]
        if base:
            candidates.append(f"{format_path(paths_cfg, paths_cfg.ckpt_dir)}/{base}")
    candidates.append(f"{format_path(paths_cfg, paths_cfg.ckpt_dir)}/champion.pt")

    tried = []
    for cand in candidates:
        tried.append(cand)
        if fs.exists(cand):
            return cand, manifest
    raise FileNotFoundError(f"Champion weights not found. Tried: {', '.join(tried)}")


def _choose_fs(paths_cfg) -> Any:
    import fsspec

    protocol = "file"
    if getattr(paths_cfg, "use_gcs", False):
        protocol = "gcs"
    return fsspec.filesystem(protocol)


def _load_model(
    fs: Any,
    board_cfg: BoardConfig,
    model_cfg_raw: Any,
    ckpt_path: str,
    device: str,
) -> PolicyValueNet:
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

    game = Gomoku(board_cfg)
    model = PolicyValueNet(game, model_cfg, device=device)
    state_dict = load_state_dict_from_fs(fs, ckpt_path, torch.device(device))
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
    return model


def _build_block_state(game: Gomoku) -> Any:
    """
    흑(PLAYER_1)이 (0,0),(0,1),(0,2)에 돌을 두는 상황을 만든다.
    순서를 맞추기 위해 백(PLAYER_2)도 의미 없는 곳에 착수하여
    History Plane이 [상대, 나, 상대, 나, ...] 순서를 유지하도록 한다.

    Returns
    -------
    GameState
        수순이 올바르게 맞춰진 게임 상태.
    """
    state = game.get_initial_state()

    # 시나리오:
    # 1. 흑(0,0) -> 2. 백(8,8) -> 3. 흑(0,1) -> 4. 백(8,7) -> 5. 흑(0,2)
    # 현재 차례: 백 (다음 수로 0,3을 막아야 함)
    moves = [
        ((0, 0), PLAYER_1),
        ((8, 8), PLAYER_2),  # 백의 덤 수 (구석)
        ((0, 1), PLAYER_1),
        ((8, 7), PLAYER_2),  # 백의 덤 수 (구석)
        ((0, 2), PLAYER_1),
    ]

    for action, player in moves:
        # get_next_state는 player 인자를 받지만, 내부적으로
        # state.next_player가 맞지 않으면 로직이 꼬일 수 있으므로
        # 순차적으로 올바르게 호출하는 것이 안전함.
        if state.next_player != player:
            raise ValueError(
                f"Turn mismatch! Expected {state.next_player}, got {player}"
            )
        state = game.get_next_state(state, action, player)

    return state


def _mask_policy_to_legal(
    game: Gomoku, state: Any, policy: torch.Tensor
) -> torch.Tensor:
    mask = np.zeros(game.action_size, dtype=np.float32)
    legal_indices = state.legal_indices_cache
    if legal_indices is None:
        game.get_legal_moves(state)
        legal_indices = state.legal_indices_cache
    if legal_indices is not None:
        mask[legal_indices.astype(int)] = 1.0
    masked = policy * torch.as_tensor(mask, device=policy.device, dtype=policy.dtype)
    total = masked.sum()
    if total > 1e-6:
        masked = masked / total
    return masked


def _print_topk(game: Gomoku, policy: torch.Tensor, topk: int) -> None:
    values, indices = torch.topk(policy, k=min(topk, policy.numel()))
    print(f"[Top-{topk}]")
    for p, idx in zip(values, indices, strict=True):
        x, y = action_to_xy(int(idx), game.col_count)
        mv = convert_index_to_coordinates(x, y, game.col_count)
        print(f"  p={float(p):.4f} move={mv} ({x},{y})")


def main() -> None:
    args = _parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"

    cfg = load_and_parse_config(args.config)
    if args.run_id:
        cfg = cfg.model_copy(
            update={"paths": cfg.paths.model_copy(update={"run_id": args.run_id})}
        )

    fs = _choose_fs(cfg.paths)
    ckpt_path, manifest = _resolve_weights(cfg, fs, args.checkpoint)
    board_cfg = BoardConfig.model_validate(manifest["board"]) if manifest else cfg.board
    model_cfg_raw = manifest["model"] if manifest else cfg.model

    game = Gomoku(board_cfg)
    model = _load_model(fs, board_cfg, model_cfg_raw, ckpt_path, device)

    state = _build_block_state(game)
    print("[Scenario] 흑이 (A1,A2,A3) 3목. 백이 막아야 함 (차례=백).")
    game.print_board(state)

    encoded = torch.as_tensor(
        game.get_encoded_state([state])[0], device=device
    ).unsqueeze(0)
    with torch.no_grad():
        logits, value = model(encoded)
        policy = torch.softmax(logits[0], dim=-1)
    masked_policy = _mask_policy_to_legal(game, state, policy)

    print(f"[Value] {float(value.squeeze().item()):.4f}")
    _print_topk(game, masked_policy, args.topk)

    best_idx = int(torch.argmax(masked_policy).item())
    bx, by = action_to_xy(best_idx, game.col_count)
    best_coord = convert_index_to_coordinates(bx, by, game.col_count)
    print(
        f"[Best] move={best_coord} ({bx},{by}) prob={float(masked_policy[best_idx]):.4f}"
    )
    if (bx, by) == (0, 3):
        print("✅ 3목 차단 수를 찾았습니다.")
    else:
        print("⚠️ 3목을 막지 못했습니다. 인코딩/학습 품질을 재검토하세요.")


if __name__ == "__main__":
    main()
