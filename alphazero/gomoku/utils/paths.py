# gmk/utils/paths.py
import io
import os
import socket
import time
import uuid

import fsspec
import torch

from gomoku.utils.config.loader import PathsConfig, RunnerParams


def get_paths_cfg(params: RunnerParams) -> PathsConfig:
    return params.paths


def format_path(paths: PathsConfig, template: str) -> str:
    """
    경로 템플릿을 포맷팅합니다. GCS 경로도 올바르게 처리합니다.
    """
    base_path = "gs://" if getattr(paths, "use_gcs", False) else ""
    run_path = template.format(run_prefix=paths.run_prefix, run_id=paths.run_id)

    # GCS 경로와 로컬 경로를 올바르게 결합합니다.
    return os.path.join(base_path, run_path) if base_path else str(run_path)


def ensure_run_dirs(params: RunnerParams, fs: "fsspec.AbstractFileSystem"):
    """최상위 실행 디렉토리 및 주요 하위 디렉토리를 생성합니다."""
    paths = get_paths_cfg(params)
    # format_path를 사용하여 GCS/로컬 경로를 올바르게 생성합니다.
    fs.makedirs(format_path(paths, "{run_prefix}/{run_id}"), exist_ok=True)
    fs.makedirs(format_path(paths, paths.replay_dir), exist_ok=True)
    fs.makedirs(format_path(paths, paths.ckpt_dir), exist_ok=True)
    fs.makedirs(format_path(paths, paths.evaluation_logs_dir), exist_ok=True)


def iteration_ckpt_path(paths_cfg: PathsConfig, iteration: int) -> str:
    """이터레이션 번호가 포함된 모델 체크포인트 경로를 생성합니다."""
    ckpt_dir = format_path(paths_cfg, paths_cfg.ckpt_dir)
    return f"{ckpt_dir}/iteration_{iteration:04d}.pt"


def heartbeat_ckpt_path(paths_cfg: PathsConfig) -> str:
    """하트비트 체크포인트 경로를 생성합니다."""
    ckpt_dir = format_path(paths_cfg, paths_cfg.ckpt_dir)
    return f"{ckpt_dir}/heartbeat.pt"


def manifest_path(paths_cfg: PathsConfig) -> str:
    return format_path(paths_cfg, paths_cfg.manifest)


def new_replay_shard_path(paths: PathsConfig, iteration: int) -> str:
    ts = int(time.time())
    host = socket.gethostname()
    pid = os.getpid()
    uid = uuid.uuid4().hex
    replay_dir = format_path(paths, paths.replay_dir)
    return f"{replay_dir}/shard-iter{iteration:04d}-{ts}-{host}-{pid}-{uid}.parquet"


def evaluation_log_path(paths_cfg: PathsConfig) -> str:
    """
    고정된 평가 로그 파일 경로를 반환합니다.
    """
    log_dir = format_path(paths_cfg, paths_cfg.evaluation_logs_dir)
    return f"{log_dir}/evaluation_log.jsonl"


def save_state(fs: "fsspec.AbstractFileSystem", model_state: dict, path: str):
    """모델 가중치를 원자적으로 저장하며, 필요 시 디렉토리를 생성합니다."""
    # UUID를 포함한 임시 경로 생성
    tmp_path = f"{path}.tmp.{uuid.uuid4().hex}"

    parent_dir = os.path.dirname(path)
    if not fs.exists(parent_dir):
        fs.makedirs(parent_dir, exist_ok=True)

    with fs.open(tmp_path, "wb") as f:
        buffer = io.BytesIO()
        torch.save(model_state, buffer)
        buffer.seek(0)
        f.write(buffer.getvalue())
    fs.mv(tmp_path, path, atomic=True)  # fsspec의 원자적 이동 기능 사용


def load_state_dict_from_fs(
    fs: "fsspec.AbstractFileSystem", path: str, device: torch.device
) -> dict:
    """Fsspec 파일 시스템을 통해 모델 가중치를 로드합니다."""
    with fs.open(path, "rb") as f:
        buffer = io.BytesIO(f.read())
        buffer.seek(0)
        return torch.load(buffer, map_location=device)
