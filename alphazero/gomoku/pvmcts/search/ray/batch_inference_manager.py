from collections import deque
from dataclasses import dataclass
import time
from typing import NamedTuple

import ray
import torch

from gomoku.inference.ray_client import RayInferenceClient
from gomoku.pvmcts.treenode import TreeNode


class PendingNodeInfo(NamedTuple):
    """비동기 추론 요청과 매핑될 트리 노드 메타데이터."""

    node: TreeNode
    is_start_node: bool


@dataclass
class BatchResult:
    """비동기 추론 결과 묶음."""

    mapping: list[PendingNodeInfo]
    policy_logits: torch.Tensor
    values: torch.Tensor


class _QueueItem(NamedTuple):
    mapping: PendingNodeInfo
    tensor: torch.Tensor
    native_state: object | None


class BatchInferenceManager:
    """Ray 기반 비동기 추론 배치 관리자."""

    def __init__(
        self,
        client: RayInferenceClient,
        batch_size: int,
        max_wait_ms: float | None = None,
        max_inflight_batches: int | None = None,
    ):
        self.client = client
        self.batch_size = max(1, int(batch_size))
        self.max_wait_ns = (
            int(max_wait_ms * 1_000_000) if max_wait_ms is not None else 0
        )
        self.max_inflight_batches = max_inflight_batches

        self._queue: deque[_QueueItem] = deque()
        self._queue_start_ns: int | None = None
        self._inflight_refs: dict[ray.ObjectRef, list[PendingNodeInfo]] = {}

    def enqueue(
        self,
        mapping: PendingNodeInfo,
        tensor: torch.Tensor,
        native_state: object | None = None,
    ) -> None:
        """요청을 큐에 적재하고 배치 조건이 되면 즉시 발송한다."""
        self._queue.append(_QueueItem(mapping=mapping, tensor=tensor, native_state=native_state))
        if self._queue_start_ns is None:
            self._queue_start_ns = time.monotonic_ns()

        # 배치가 가득 찼으면 바로 발송
        if len(self._queue) >= self.batch_size:
            # inflight 제한에 걸리면 발송을 미루고 큐에 쌓아둔다.
            if (
                self.max_inflight_batches is not None
                and len(self._inflight_refs) >= self.max_inflight_batches
            ):
                return
            self._dispatch_batch()

    def dispatch_ready(self, *, force: bool = False) -> bool:
        """사이즈/타임아웃 조건을 만족하면 배치를 발송한다."""
        if not self._queue:
            return False

        if (
            self.max_inflight_batches is not None
            and len(self._inflight_refs) >= self.max_inflight_batches
        ):
            return False

        if not force and not self._should_dispatch():
            return False

        return self._dispatch_batch()

    def drain_ready(self, timeout_s: float = 0.0) -> list[BatchResult]:
        """도착한 배치를 회수한다. 대기 중 타임아웃이 지나면 강제 발송한다."""
        self.check_and_flush()

        if not self._inflight_refs:
            return []

        inflight_keys = list(self._inflight_refs.keys())
        # num_returns=1: return as soon as ANY result is ready instead of
        # waiting for all results or timeout expiry.  This reduces pipeline
        # stall when some batches complete earlier than others.
        ready_refs, _ = ray.wait(
            inflight_keys,
            num_returns=1,
            timeout=timeout_s,
            fetch_local=True,
        )

        results: list[BatchResult] = []
        for ref in ready_refs:
            mapping = self._inflight_refs.pop(ref, [])
            try:
                policy_logits, values = ray.get(ref)
                results.append(
                    BatchResult(
                        mapping=mapping,
                        policy_logits=policy_logits.cpu(),
                        values=values.cpu(),
                    )
                )
            except Exception as exc:  # pragma: no cover - 보호용
                print(f"[Error] Batch processing failed: {exc}")
            finally:
                _notify = getattr(self.client, "notify_result", None)
                if _notify is not None:
                    _notify(ref)

        # inflight 슬롯이 비었으니 대기열이 있다면 즉시 발송을 시도한다.
        if ready_refs:
            self.dispatch_ready(force=False)

        return results

    def pending_count(self) -> int:
        """큐 + 처리중 요청 수를 반환한다."""
        inflight = sum(len(v) for v in self._inflight_refs.values())
        return len(self._queue) + inflight

    def cleanup(self) -> None:
        """남은 작업을 취소하고 상태를 초기화한다."""
        for ref in list(self._inflight_refs.keys()):
            try:
                ray.cancel(ref, force=True)
            except Exception:
                pass
        self._inflight_refs.clear()
        self._queue.clear()
        self._queue_start_ns = None


    # 내부 헬퍼
    def _should_dispatch(self) -> bool:
        """배치를 발송할 조건을 검사한다."""
        if len(self._queue) >= self.batch_size:
            return True
        if self.max_wait_ns <= 0 or self._queue_start_ns is None:
            return False
        elapsed = time.monotonic_ns() - self._queue_start_ns
        return elapsed >= self.max_wait_ns

    def check_and_flush(self) -> bool:
        """타임아웃이 지났으면 강제로 배치를 발송한다."""
        if not self._queue:
            return False
        if self.max_wait_ns <= 0 or self._queue_start_ns is None:
            return False
        elapsed = time.monotonic_ns() - self._queue_start_ns
        if elapsed < self.max_wait_ns:
            return False
        return self._dispatch_batch()

    def _dispatch_batch(self) -> bool:
        """큐에서 배치를 꺼내 Ray에 전송하고 in-flight로 기록한다."""
        if not self._queue:
            return False

        count = min(len(self._queue), self.batch_size)
        batch_items = [self._queue.popleft() for _ in range(count)]
        self._queue_start_ns = time.monotonic_ns() if self._queue else None

        batch_tensor = torch.stack([item.tensor for item in batch_items])
        native_payload = [item.native_state for item in batch_items]
        batch_ref = self.client.infer_async(batch_tensor, native_payload=native_payload)
        self._inflight_refs[batch_ref] = [item.mapping for item in batch_items]
        return True
