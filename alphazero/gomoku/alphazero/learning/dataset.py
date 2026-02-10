from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
import json
import os
from typing import NamedTuple
import uuid

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from gomoku.alphazero.types import GameRecord
from gomoku.core.game_config import EMPTY_SPACE
from gomoku.core.gomoku import GameState, Gomoku
from gomoku.utils import io_helpers


def decode_board(board_field: object, game: Gomoku) -> np.ndarray:
    """Restore a board field into an int8 ndarray.

    Parameters
    ----------
    board_field : object
        Buffer or array-like containing serialized board data.
    game : Gomoku
        Game instance providing board dimensions.

    Returns
    -------
    np.ndarray
        Reshaped board array of dtype int8.
    """
    if isinstance(board_field, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(board_field, dtype=np.int8)
    else:
        arr = np.asarray(board_field, dtype=np.int8)
    return arr.reshape(game.row_count, game.col_count)


def decode_policy(policy_field: object) -> np.ndarray:
    """Restore a policy field into a float32 ndarray.

    Parameters
    ----------
    policy_field : object
        Buffer or array-like containing serialized policy data.

    Returns
    -------
    np.ndarray
        Policy array of dtype float32.
    """
    if isinstance(policy_field, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(policy_field, dtype=np.float16)
    else:
        arr = np.asarray(policy_field, dtype=np.float16)
    return arr.astype(np.float32)


def _apply_symmetry(
    encoded_state: np.ndarray, policy: np.ndarray, game: Gomoku
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a random D4 symmetry to the encoded state and policy."""
    h, w = game.row_count, game.col_count
    if policy.size != h * w:
        return encoded_state, policy

    k = int(np.random.randint(0, 8))
    obs = encoded_state
    pi_2d = policy.reshape(h, w)

    if k >= 4:
        obs = np.flip(obs, axis=2)  # horizontal flip on width axis
        pi_2d = np.flip(pi_2d, axis=1)
        k -= 4

    if k > 0:
        obs = np.rot90(obs, k=k, axes=(1, 2))
        pi_2d = np.rot90(pi_2d, k=k)

    obs = np.ascontiguousarray(obs)
    pi_flat = np.ascontiguousarray(pi_2d.reshape(-1))
    return obs, pi_flat


class GameSample(NamedTuple):
    """Flattened training sample at a single turn."""

    state: GameState
    policy_probs: np.ndarray
    value: float
    priority: float = 1.0


@dataclass(slots=True)
class ReplayDataset(Dataset):
    """Convert GameRecords into tensors ready for training."""

    samples: Sequence[GameSample]
    game: Gomoku
    include_priority: bool = True
    return_index: bool = False
    priorities: list[float] = field(init=False)

    def __post_init__(self) -> None:
        """Initialize cached priorities for all samples."""
        self.priorities = [
            _safe_priority(float(getattr(s, "priority", 1.0))) for s in self.samples
        ]

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        """Return training tensors for the given index."""
        sample = self.samples[idx]
        encoded_state = self.game.get_encoded_state([sample.state])[0]
        policy_np = np.asarray(sample.policy_probs, dtype=np.float32)
        aug_state, aug_policy = _apply_symmetry(encoded_state, policy_np, self.game)
        state_tensor = torch.tensor(aug_state, dtype=torch.float32)
        policy_tensor = torch.tensor(aug_policy, dtype=torch.float32)
        value_tensor = torch.tensor(sample.value, dtype=torch.float32).unsqueeze(0)
        if not self.include_priority:
            if self.return_index:
                return idx, state_tensor, policy_tensor, value_tensor
            return state_tensor, policy_tensor, value_tensor
        priority_tensor = torch.tensor(
            _safe_priority(float(getattr(sample, "priority", 1.0))), dtype=torch.float32
        )
        if self.return_index:
            return idx, state_tensor, policy_tensor, value_tensor, priority_tensor
        return state_tensor, policy_tensor, value_tensor, priority_tensor


def flatten_game_records(
    records: Iterable[GameRecord], game: Gomoku
) -> list[GameSample]:
    """Flatten GameRecords into per-turn GameSamples.

    Parameters
    ----------
    records : Iterable[GameRecord]
        Game records to flatten.
    game : Gomoku
        Game instance used for encoding (kept for signature parity).

    Returns
    -------
    list[GameSample]
        Flattened per-turn samples.
    """
    flat: list[GameSample] = []
    for rec in records:
        priority = float(getattr(rec, "priority", 1.0))
        for state, pi, z in zip(
            rec.states_raw,
            rec.policies,
            rec.outcomes,
            strict=True,
        ):
            flat.append(
                GameSample(
                    state=state,
                    policy_probs=pi,
                    value=float(z),
                    priority=priority,
                )
            )
    return flat


def reconstruct_state(sample: dict, game: Gomoku) -> GameState:
    """Rebuild a GameState from a serialized sample dictionary.

    Parameters
    ----------
    sample :
        Serialized sample dictionary.
    game :
        Game instance carrying board dimensions.

    Returns
    -------
    GameState
        Reconstructed game state.

    Notes
    -----
    - Used when reading serialized dicts from Parquet/JSON.
    - Not used in the in-memory training path.
    """
    board_array = decode_board(sample["board"], game)
    last_move_idx = int(sample.get("last_move_idx", -1))
    empty_count = int(np.count_nonzero(board_array == EMPTY_SPACE))
    return GameState(
        board=board_array,
        p1_pts=sample["p1_pts"],
        p2_pts=sample["p2_pts"],
        next_player=sample["next_player"],
        last_move_idx=last_move_idx,
        empty_count=empty_count,
        history=tuple(sample.get("history", ())),
        legal_indices_cache=None,
    )


def _state_to_row(
    state: GameState,
    policy: np.ndarray,
    value: float,
    priority: float = 1.0,
    config_snapshot: dict | None = None,
) -> dict:
    """Serialize a GameState and targets into a Parquet row."""
    board_bytes = np.asarray(state.board, dtype=np.int8).tobytes()
    policy_bytes = np.asarray(policy, dtype=np.float16).tobytes()
    last_move_idx = int(state.last_move_idx)
    return {
        "board": board_bytes,
        "p1_pts": int(state.p1_pts),
        "p2_pts": int(state.p2_pts),
        "next_player": int(state.next_player),
        "last_move": last_move_idx,  # compatibility with io_helpers.validation
        "last_move_idx": last_move_idx,
        "history": list(state.history) if state.history else [],
        "policy_probs": policy_bytes,
        "value": float(value),
        "priority": float(priority),
        "config_snapshot": config_snapshot or {},
    }


def _safe_priority(raw: float) -> float:
    """Clamp priority to a finite positive value."""
    if not np.isfinite(raw) or raw <= 0.0:
        return 1e-6
    return float(raw)


def game_records_to_rows(records: Iterable[GameRecord]) -> list[dict]:
    """Convert GameRecords into rows suitable for Parquet storage."""
    rows: list[dict] = []
    for rec in records:
        priority = _safe_priority(getattr(rec, "priority", 1.0))
        cfg_snap = getattr(rec, "config_snapshot", None)
        for state, pi, outcome in zip(
            rec.states_raw, rec.policies, rec.outcomes, strict=True
        ):
            rows.append(_state_to_row(state, pi, float(outcome), priority, cfg_snap))
    return rows


def save_records_to_parquet_shard(
    records: Iterable[GameRecord],
    shard_path: str,
    *,
    fs: fsspec.AbstractFileSystem | None = None,
    compression: str = "snappy",
) -> str:
    """Persist GameRecords into a single Parquet shard."""
    filesystem = fs or fsspec.filesystem("file")
    rows = game_records_to_rows(records)
    if not rows:
        return shard_path

    table = pa.Table.from_pylist(rows)
    metadata = table.schema.metadata or {}
    first_snapshot = getattr(next(iter(records), None), "config_snapshot", None)
    if first_snapshot:
        metadata = dict(metadata)
        metadata[b"config_snapshot"] = json.dumps(first_snapshot).encode("utf-8")
        table = table.replace_schema_metadata(metadata)

    tmp_path = f"{shard_path}.tmp.{uuid.uuid4().hex}"
    parent_dir = os.path.dirname(shard_path)
    if not filesystem.exists(parent_dir):
        filesystem.makedirs(parent_dir, exist_ok=True)

    with filesystem.open(tmp_path, "wb") as f:
        pq.write_table(table, where=f, compression=compression, version="2.6")
    filesystem.mv(tmp_path, shard_path, atomic=True)
    return shard_path


class ShardDataset(Dataset):
    """Dataset that loads samples from Parquet shards."""

    _REQUIRED_KEYS = {
        "board",
        "p1_pts",
        "p2_pts",
        "next_player",
        "last_move",
        "policy_probs",
        "value",
    }
    _OPTIONAL_KEYS = {"last_move_idx", "history", "priority", "config_snapshot"}
    _ALLOWED_KEYS = _REQUIRED_KEYS | _OPTIONAL_KEYS

    def __init__(
        self,
        shard_paths: Sequence[str],
        game: Gomoku,
        *,
        fs: fsspec.AbstractFileSystem | None = None,
        include_priority: bool = True,
    ):
        """Construct a dataset backed by one or more Parquet shards."""
        self.game = game
        self.fs = fs or fsspec.filesystem("file")
        self.shards = list(shard_paths)
        self._index: list[tuple[str, int]] = []
        self.priorities: list[float] = []
        self.include_priority = include_priority
        self._cached_path: str | None = None
        self._cached_rows: list[dict] | None = None
        self._build_index()
        if self._index:
            self._validate_all_rows()

    def _build_index(self) -> None:
        """Scan shard paths and build an index of row locations and priorities."""
        offset = 0
        for path in self.shards:
            rows = io_helpers.read_parquet_shard(
                self.fs, path, columns=["board", "priority"]
            )
            length = len(rows)
            for i, row in enumerate(rows):
                self._index.append((path, i))
                self.priorities.append(_safe_priority(float(row.get("priority", 1.0))))
            offset += length

    def __len__(self) -> int:
        """Return the total number of indexed samples across shards."""
        return len(self._index)

    def _load_row(self, idx: int) -> dict:
        """Load and validate a single row from the indexed shard."""
        path, row_idx = self._index[idx]
        if path != self._cached_path:
            self._cached_rows = io_helpers.read_parquet_shard(
                self.fs, path, columns=None
            )
            self._cached_path = path
        assert self._cached_rows is not None
        row = self._cached_rows[row_idx]
        self._validate_row(row, path, row_idx)
        return row

    def _validate_row(self, row: dict, path: str, row_idx: int) -> None:
        """Validate a row against required and allowed keys."""
        missing = [k for k in self._REQUIRED_KEYS if k not in row]
        if missing:
            raise ValueError(
                f"[ReplayError] Missing required keys at {path}[{row_idx}]: "
                f"{', '.join(sorted(missing))}"
            )
        unknown = set(row.keys()) - self._ALLOWED_KEYS
        if unknown:
            raise ValueError(
                f"[ReplayError] Unknown keys at {path}[{row_idx}]: "
                f"{', '.join(sorted(unknown))}"
            )

    def _validate_all_rows(self) -> None:
        """Load every shard row once to validate schema consistency."""
        for idx in range(len(self._index)):
            self._load_row(idx)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        """Return tensors for a single indexed sample."""
        row = self._load_row(idx)
        state = reconstruct_state(
            {
                "board": row["board"],
                "p1_pts": row["p1_pts"],
                "p2_pts": row["p2_pts"],
                "next_player": row["next_player"],
                "last_move_idx": row.get("last_move_idx", row.get("last_move", -1)),
                "history": row.get("history", ()),
            },
            self.game,
        )
        encoded_state = self.game.get_encoded_state([state])[0]
        state_tensor = torch.tensor(encoded_state, dtype=torch.float32)

        policy = decode_policy(row["policy_probs"])
        policy_tensor = torch.tensor(policy, dtype=torch.float32)

        value_tensor = torch.tensor(float(row["value"]), dtype=torch.float32).unsqueeze(
            0
        )
        if not self.include_priority:
            return state_tensor, policy_tensor, value_tensor
        priority_tensor = torch.tensor(
            _safe_priority(float(row.get("priority", 1.0))), dtype=torch.float32
        )
        return state_tensor, policy_tensor, value_tensor, priority_tensor
