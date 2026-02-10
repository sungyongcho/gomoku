from datetime import UTC, datetime
import json
from typing import Any

import fsspec

from gomoku.utils.config.loader import PathsConfig, RootConfig
from gomoku.utils.paths import manifest_path


def _read_json_file(
    fs: "fsspec.AbstractFileSystem", path: str
) -> dict[str, Any] | None:
    """Fsspec 파일 시스템을 사용하여 JSON 파일을 읽습니다."""
    if not fs.exists(path):
        return None
    try:
        with fs.open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None


def load_manifest(
    paths_cfg: PathsConfig, fs: "fsspec.AbstractFileSystem"
) -> dict[str, Any] | None:
    """Fs 객체를 전달받아 manifest를 로드합니다."""
    return _read_json_file(fs, manifest_path(paths_cfg))


def create_new_manifest(
    run_id: str,
    cfg: RootConfig,
    device_type: str,
    config_name: str | None = None,
    hardware_info: dict | None = None,
) -> dict[str, Any]:
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "total_elapsed_sec": 0,
        "status": "running",
        "hardware": hardware_info or {},
        "board": cfg.board.model_dump(),
        "model": cfg.model.model_dump(
            exclude={"num_planes", "policy_channels", "value_channels"}
        ),
        "device_type": device_type,
        "progress": {"completed_iterations": 0},
        "elo": {
            "champion": {"iteration": 0, "elo": 1500.0},
            "history": [{"iteration": 0, "elo": 1500.0}],
        },
        "promotions": [],
    }

    if config_name:
        manifest["configs"] = [
            {"revision": 0, "name": config_name, "begin_iteration": 1}
        ]

    return manifest


def ensure_elo_structure(
    manifest: dict[str, Any], *, default_iteration: int = 0
) -> dict[str, Any]:
    """Manifest 내 elo 섹션을 iteration/elo 딕셔너리 형태로 정규화합니다."""
    elo_section = manifest.setdefault("elo", {})

    champion_entry = elo_section.get("champion")
    default_elo = 1500.0

    if isinstance(champion_entry, dict):
        champion_iteration = int(champion_entry.get("iteration", default_iteration))
        champion_elo = float(champion_entry.get("elo", default_elo))
    elif isinstance(champion_entry, (int, float)):
        champion_iteration = int(default_iteration)
        champion_elo = float(champion_entry)
    else:
        champion_iteration = int(default_iteration)
        champion_elo = default_elo

    normalized_champion = {"iteration": champion_iteration, "elo": champion_elo}

    history: list[Any] = elo_section.get("history", [])
    if not isinstance(history, list):
        history = [history]

    normalized_history: list[dict[str, Any]] = []
    for idx, item in enumerate(history):
        if isinstance(item, dict):
            normalized_history.append(
                {
                    "iteration": int(item.get("iteration", champion_iteration + idx)),
                    "elo": float(item.get("elo", champion_elo)),
                }
            )
        elif isinstance(item, (int, float)):
            normalized_history.append(
                {
                    "iteration": int(champion_iteration + idx),
                    "elo": float(item),
                }
            )

    if not normalized_history:
        normalized_history.append(normalized_champion.copy())

    elo_section["champion"] = normalized_champion
    elo_section["history"] = normalized_history

    return elo_section


def ensure_manifest(
    paths_cfg: PathsConfig, fs: "fsspec.AbstractFileSystem"
) -> dict[str, Any]:
    """Fs 객체를 전달받아 manifest를 확인하고 반환합니다."""
    mpath = manifest_path(paths_cfg)

    if not fs.exists(mpath):
        raise RuntimeError(f"manifest.json not found for run_id='{paths_cfg.run_id}'.")

    try:
        with fs.open(mpath, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        raise RuntimeError(f"Failed to parse manifest.json at {mpath}: {e}") from e

    if not isinstance(manifest, dict):
        raise RuntimeError(f"Invalid manifest: expected a JSON object at {mpath}.")

    if not isinstance(manifest.get("board"), dict):
        raise RuntimeError(
            f"Invalid manifest: missing or malformed 'board' section at {mpath}."
        )
    if not isinstance(manifest.get("model"), dict):
        raise RuntimeError(
            f"Invalid manifest: missing or malformed 'model' section at {mpath}."
        )

    if "run_id" in manifest and manifest["run_id"] != paths_cfg.run_id:
        print(
            f"Warning: manifest run_id '{manifest['run_id']}' differs from requested run_id '{paths_cfg.run_id}'. Using file value."
        )
    else:
        manifest.setdefault("run_id", paths_cfg.run_id)

    return manifest
