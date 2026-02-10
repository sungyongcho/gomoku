from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from gomoku.alphazero.types import action_to_xy
from gomoku.core.gomoku import GameState, Gomoku
from gomoku.inference.local import LocalInference
from gomoku.model.policy_value_net import PolicyValueNet
from gomoku.pvmcts.pvmcts import PVMCTS
from gomoku.utils.config.loader import MctsConfig, load_and_parse_config
from gomoku.utils.state_dict_utils import align_state_dict_to_model


class AlphaZeroEngine:
    def __init__(self, config_path: str, checkpoint_path: str, device: str = "cpu"):
        config = load_and_parse_config(config_path)
        self.device = self._resolve_device(device)
        self.game = Gomoku(config.board, use_native=False)
        self.model = PolicyValueNet(self.game, config.model, self.device)
        self._load_checkpoint(checkpoint_path)
        self.model.eval()
        self.inference_client = LocalInference(self.model, self.device)

        # Serving receives Python GameState objects built from frontend payloads,
        # so keep MCTS on the Python path (no native_state required).
        self.mcts_config: MctsConfig = config.mcts.model_copy(update={"use_native": False})

    def get_best_move(self, state: GameState, num_searches: int | None = None) -> int:
        """Run MCTS and return a flat action index."""
        cfg = self.mcts_config
        if num_searches is not None:
            cfg = cfg.model_copy(update={"num_searches": float(num_searches)})

        pvmcts = PVMCTS(
            game=self.game,
            mcts_params=cfg,
            inference_client=self.inference_client,
            mode="sequential",
        )
        root = pvmcts.create_root(state)
        [(policy, _)] = pvmcts.run_search([root], add_noise=False)
        return int(np.argmax(policy))

    def apply_move(self, state: GameState, action: int) -> tuple[GameState, list[int]]:
        """Apply a flat action index and return the next state + captured flat indices."""
        x, y = action_to_xy(action, self.game.col_count)
        player = int(state.next_player)

        # Gomoku.get_next_state only updates last_captures when capture occurs.
        self.game.last_captures = []
        new_state = self.game.get_next_state(state, (x, y), player)
        captures = [int(idx) for idx in self.game.last_captures]
        return new_state, captures

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        ckpt = Path(checkpoint_path)
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. "
                "Set ALPHAZERO_CHECKPOINT to a valid .pt file."
            )

        raw = torch.load(str(ckpt), map_location=self.device)
        state_dict = self._extract_state_dict(raw)
        align_state_dict_to_model(state_dict, self.model.state_dict())
        self.model.load_state_dict(state_dict, strict=False)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return requested

    @staticmethod
    def _extract_state_dict(raw_checkpoint: Any) -> dict[str, torch.Tensor]:
        if isinstance(raw_checkpoint, dict):
            if raw_checkpoint and all(
                isinstance(v, torch.Tensor) for v in raw_checkpoint.values()
            ):
                return raw_checkpoint

            for key in ("state_dict", "model_state_dict", "model"):
                candidate = raw_checkpoint.get(key)
                if isinstance(candidate, dict) and candidate and all(
                    isinstance(v, torch.Tensor) for v in candidate.values()
                ):
                    return candidate

        raise TypeError(
            "Unsupported checkpoint format: expected a state-dict mapping tensor keys "
            "or a dict containing 'state_dict'/'model_state_dict'."
        )
