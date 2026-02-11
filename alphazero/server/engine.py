from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gomoku.alphazero.types import action_to_xy
from gomoku.core.gomoku import GameState, Gomoku
from gomoku.inference.local import LocalInference
try:
    from gomoku.inference.onnx_inference import OnnxInference
except ImportError:
    OnnxInference = None
from gomoku.model.policy_value_net import PolicyValueNet
from gomoku.pvmcts.pvmcts import PVMCTS
from gomoku.utils.config.loader import MctsConfig, load_and_parse_config
from gomoku.utils.state_dict_utils import align_state_dict_to_model

logger = logging.getLogger(__name__)


class AlphaZeroEngine:
    def __init__(self, config_path: str, checkpoint_path: str, device: str = "cpu"):
        config = load_and_parse_config(config_path)
        self.device = self._resolve_device(device)
        self._configure_torch_threads()
        requested_native = bool(getattr(config.mcts, "use_native", False))
        self.game = Gomoku(config.board, use_native=requested_native)
        self.model = PolicyValueNet(self.game, config.model, self.device)
        self._load_checkpoint(checkpoint_path)
        self.model.eval()
        self.model.eval()

        infer_backend = os.getenv("ALPHAZERO_INFER_BACKEND", "local")
        if infer_backend.startswith("onnx"):
            if OnnxInference is None:
                logger.warning("ONNX Runtime not available, falling back to LocalInference")
                self.inference_client = LocalInference(self.model, self.device)
            else:
                logger.info("Initializing ONNX Runtime inference backend...")
                quantize = "int8" in infer_backend
                # Cache directory for ONNX models
                onnx_cache = os.getenv("ONNX_CACHE_DIR", "/tmp/onnx_cache")
                self.inference_client = OnnxInference(
                    model=self.model,
                    num_planes=config.model.dim,  # Assuming dim is correct
                    board_h=config.board.size,
                    board_w=config.board.size,
                    quantize=quantize,
                    onnx_cache_dir=onnx_cache,
                )
        else:
            self.inference_client = LocalInference(self.model, self.device)

        native_enabled = bool(
            requested_native
            and self.game.use_native
            and getattr(self.game, "_native_core", None) is not None
        )
        if requested_native and not native_enabled:
            logger.warning(
                "Native MCTS requested but unavailable; falling back to Python MCTS."
            )
        self.mcts_config: MctsConfig = config.mcts.model_copy(
            update={"use_native": native_enabled}
        )

    def get_best_move(self, state: GameState, num_searches: int | None = None) -> int:
        """Run MCTS and return a flat action index."""
        cfg = self.mcts_config
        if num_searches is not None:
            cfg = cfg.model_copy(update={"num_searches": float(num_searches)})

        if bool(getattr(cfg, "use_native", False)):
            state = self._ensure_native_state(state)

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

        source_state = state
        if self.game.use_native:
            source_state = self._ensure_native_state(state)

        # Gomoku.get_next_state only updates last_captures in the Python path.
        self.game.last_captures = []
        new_state = self.game.get_next_state(source_state, (x, y), player)
        if self.game.use_native:
            captures = [
                int(idx)
                for idx in np.flatnonzero(
                    (source_state.board != 0).reshape(-1) & (new_state.board == 0).reshape(-1)
                )
            ]
        else:
            captures = [int(idx) for idx in self.game.last_captures]
        return new_state, captures

    def _ensure_native_state(self, state: GameState) -> GameState:
        if state.native_state is not None:
            return state

        native_core = getattr(self.game, "_native_core", None)
        if native_core is None:
            return state

        native_state = native_core.initial_state()
        native_state.board = state.board.astype(np.int8, copy=False).reshape(-1).tolist()
        native_state.p1_pts = int(state.p1_pts)
        native_state.p2_pts = int(state.p2_pts)
        native_state.next_player = int(state.next_player)
        native_state.last_move_idx = int(state.last_move_idx)
        native_state.empty_count = int(state.empty_count)
        native_state.history = [int(x) for x in state.history]

        return GameState(
            board=state.board,
            p1_pts=state.p1_pts,
            p2_pts=state.p2_pts,
            next_player=state.next_player,
            last_move_idx=state.last_move_idx,
            empty_count=state.empty_count,
            history=state.history,
            legal_indices_cache=state.legal_indices_cache,
            native_state=native_state,
        )

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

    def _configure_torch_threads(self) -> None:
        """Apply server-side CPU thread settings so env overrides are honored reliably."""
        if self.device.type != "cpu":
            return

        available_cpus = self._detect_available_cpu_count()
        raw_threads = os.getenv("TORCH_NUM_THREADS")
        target_threads: int | None = None
        if raw_threads:
            try:
                parsed = int(raw_threads)
                if parsed > 0:
                    target_threads = parsed
            except ValueError:
                logger.warning("Ignoring invalid TORCH_NUM_THREADS=%r", raw_threads)
        if target_threads is None:
            # Maximize throughput by default up to the CPU count available to this container.
            target_threads = available_cpus
        target_threads = max(1, min(target_threads, available_cpus))
        torch.set_num_threads(target_threads)

        raw_interop = os.getenv("TORCH_NUM_INTEROP_THREADS")
        if raw_interop:
            try:
                interop_threads = max(1, int(raw_interop))
            except ValueError:
                logger.warning(
                    "Ignoring invalid TORCH_NUM_INTEROP_THREADS=%r", raw_interop
                )
                interop_threads = None
            if interop_threads is not None:
                try:
                    torch.set_num_interop_threads(interop_threads)
                except RuntimeError:
                    # PyTorch allows this only before any parallel work starts.
                    pass

        logger.info(
            "Torch threads configured: num_threads=%s interop_threads=%s available_cpus=%s",
            torch.get_num_threads(),
            torch.get_num_interop_threads(),
            available_cpus,
        )

    @staticmethod
    def _read_int_file(path: str) -> int | None:
        try:
            raw = Path(path).read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    @classmethod
    def _detect_cpu_quota_limit(cls) -> int | None:
        # cgroup v2
        try:
            cpu_max = Path("/sys/fs/cgroup/cpu.max").read_text(encoding="utf-8").strip()
            parts = cpu_max.split()
            if len(parts) >= 2 and parts[0] != "max":
                quota_us = int(parts[0])
                period_us = int(parts[1])
                if quota_us > 0 and period_us > 0:
                    return max(1, int(math.ceil(quota_us / period_us)))
        except OSError:
            pass
        except ValueError:
            pass

        # cgroup v1
        quota_us = cls._read_int_file("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        period_us = cls._read_int_file("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if quota_us is not None and period_us is not None and quota_us > 0 and period_us > 0:
            return max(1, int(math.ceil(quota_us / period_us)))
        return None

    @classmethod
    def _detect_available_cpu_count(cls) -> int:
        if hasattr(os, "sched_getaffinity"):
            try:
                affinity_count = len(os.sched_getaffinity(0))
                if affinity_count > 0:
                    host_visible = affinity_count
                else:
                    host_visible = max(1, int(os.cpu_count() or 1))
            except OSError:
                host_visible = max(1, int(os.cpu_count() or 1))
        else:
            host_visible = max(1, int(os.cpu_count() or 1))

        quota_limit = cls._detect_cpu_quota_limit()
        if quota_limit is None:
            return host_visible
        return max(1, min(host_visible, quota_limit))

    @staticmethod
    def _extract_state_dict(raw_checkpoint: Any) -> dict[str, torch.Tensor]:
        if isinstance(raw_checkpoint, dict):
            if raw_checkpoint and all(
                isinstance(v, torch.Tensor) for v in raw_checkpoint.values()
            ):
                return raw_checkpoint

            for key in ("state_dict", "model_state_dict", "model"):
                candidate = raw_checkpoint.get(key)
                if (
                    isinstance(candidate, dict)
                    and candidate
                    and all(isinstance(v, torch.Tensor) for v in candidate.values())
                ):
                    return candidate

        raise TypeError(
            "Unsupported checkpoint format: expected a state-dict mapping tensor keys "
            "or a dict containing 'state_dict'/'model_state_dict'."
        )
