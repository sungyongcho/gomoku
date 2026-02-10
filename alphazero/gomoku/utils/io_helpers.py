# gmk/utils/io_helpers.py


from collections.abc import Sequence
import json
import os
import shutil
from typing import Any
import uuid

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq


def _validate_rows(rows: Sequence[dict[str, Any]]) -> None:
    """Performs minimal validation on the format of the data to be saved."""
    if not rows:
        raise ValueError("Cannot save an empty shard because 'rows' is empty.")

    required_keys = (
        "board",
        "p1_pts",
        "p2_pts",
        "next_player",
        "last_move",
        "policy_probs",
        "value",
    )
    optional_keys = (
        "last_move_idx",
        "history",
        "priority",
        "config_snapshot",
    )
    allowed_keys = set(required_keys) | set(optional_keys)

    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            raise TypeError(f"Item at rows[{i}] is not a dictionary.")

        if not all(k in r for k in required_keys):
            missing_keys = sorted([k for k in required_keys if k not in r])
            raise KeyError(f"Row {i} is missing required keys. Missing: {missing_keys}")
        unknown = set(r.keys()) - allowed_keys
        if unknown:
            raise ValueError(
                f"Row {i} contains unknown keys: {', '.join(sorted(unknown))}"
            )


def save_as_parquet_shard(
    fs: fsspec.AbstractFileSystem,
    shard_path: str,
    rows: Sequence[dict[str, Any]],
    *,
    compression: str = "snappy",
) -> str:
    """리플레이 샤드를 Parquet 형식으로 원자적으로 저장합니다."""
    _validate_rows(rows)
    # 원본 precision 그대로 저장
    table = pa.Table.from_pylist(rows)
    # UUID를 포함한 임시 경로 생성
    tmp_path = f"{shard_path}.tmp.{uuid.uuid4().hex}"

    parent_dir = os.path.dirname(shard_path)
    if not fs.exists(parent_dir):
        fs.makedirs(parent_dir, exist_ok=True)

    with fs.open(tmp_path, "wb") as f:
        pq.write_table(table, where=f, compression=compression, version="2.6")
    fs.mv(tmp_path, shard_path, atomic=True)  # fsspec의 원자적 이동 기능 사용
    return shard_path


def read_parquet_shard(
    fs: Any,
    shard_path: str,
    columns: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Parquet 샤드를 읽어 딕셔너리 리스트로 반환합니다."""
    with fs.open(shard_path, "rb") as f:
        table = pq.read_table(f, columns=columns)
    return table.to_pylist()


def list_replay_shards(
    fs: Any,
    shards_dir: str,
    suffix: str = ".parquet",
) -> list[str]:
    """디렉토리 내의 샤드 파일 목록을 반환합니다."""
    pat = shards_dir.rstrip("/") + f"/*{suffix}"
    return sorted(fs.glob(pat))


def atomic_write_json(fs: fsspec.AbstractFileSystem, path: str, data: dict):
    """JSON 파일을 원자적으로 저장합니다."""
    # UUID를 포함한 임시 경로 생성
    tmp_path = f"{path}.tmp.{uuid.uuid4().hex}"

    parent_dir = os.path.dirname(path)
    if not fs.exists(parent_dir):
        fs.makedirs(parent_dir, exist_ok=True)

    with fs.open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    fs.mv(tmp_path, path, atomic=True)  # fsspec의 원자적 이동 기능 사용


def atomic_append_jsonl(
    fs: fsspec.AbstractFileSystem, path: str, entry: dict[str, Any]
) -> None:
    """JSONL 파일에 항목을 안전하게 추가합니다.

    일부 원격 파일 시스템은 append 모드를 지원하지 않으므로
    기존 파일을 복사한 뒤에 새로운 엔트리를 추가하여 원자적으로 교체합니다.
    """
    tmp_path = f"{path}.tmp.{uuid.uuid4().hex}"

    parent_dir = os.path.dirname(path)
    if not fs.exists(parent_dir):
        fs.makedirs(parent_dir, exist_ok=True)

    with fs.open(tmp_path, "wb") as dst:
        if fs.exists(path):
            with fs.open(path, "rb") as src:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
        dst.write((json.dumps(entry) + "\n").encode("utf-8"))

    fs.mv(tmp_path, path, atomic=True)
