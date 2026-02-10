from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import ray
import torch

from gomoku.inference.base import InferenceClient


class _InferenceRequest:
    __slots__ = ("batch", "native_payload", "user_future", "model_slot")

    def __init__(
        self,
        batch: np.ndarray,
        native_payload: list[object] | None,
        model_slot: str | None = None,
    ):
        self.batch = batch
        self.native_payload = native_payload
        self.model_slot = model_slot
        self.user_future: asyncio.Future | None = None


@ray.remote(num_cpus=1)
class RayInferenceActor:
    """Ray Actor that performs model inference with server-side batching using AsyncIO."""

    def __init__(
        self,
        model_builder: Callable[[], Any],
        batch_size: int = 32,
        min_batch_size: int = 1,
        max_wait_ms: int = 10,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_builder = model_builder
        self.model = model_builder()
        self.model.to(self.device)
        self.model.eval()

        self.model_slots: dict[str, Any] = {"current": self.model}
        self.default_slot = "current"

        self.batch_size = max(1, batch_size)
        self.min_batch_size = max(1, min_batch_size)
        self.max_wait_s = max(0.0, max_wait_ms / 1000.0)

        self._queue: asyncio.Queue[_InferenceRequest] = asyncio.Queue()
        self._stop_event = asyncio.Event()

        # Start the batch consumer task
        # print(f"[RayInferenceActor] Initialized. BatchSize={batch_size}, Min={min_batch_size}, MaxWait={max_wait_ms}ms")
        asyncio.create_task(self._consumer_loop())

    async def infer(
        self,
        inputs: np.ndarray,
        native_payload: list[object] | None = None,
        model_slot: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Async inference entrypoint."""
        # print("DEBUG: infer called")
        loop = asyncio.get_running_loop()
        req = _InferenceRequest(inputs, native_payload, model_slot)

        # Attach future
        fut = loop.create_future()
        req.user_future = fut

        await self._queue.put(req)
        return await fut

    async def _consumer_loop(self) -> None:
        pending: list[_InferenceRequest] = []
        loop = asyncio.get_running_loop()

        while not self._stop_event.is_set():
            try:
                # 1. Wait efficiently for the first item
                if not pending:
                    # print("DEBUG: Waiting for queue item...")
                    req = await self._queue.get()
                    # print("DEBUG: Got first queue item")
                    pending.append(req)

                # 2. Smart Batching: Fetch remaining items aggressively
                # Set a deadline for batch completion
                deadline = loop.time() + self.max_wait_s

                current_samples = 0
                if pending:
                    current_samples = sum(req.batch.shape[0] for req in pending)

                while current_samples < self.batch_size:
                    # If queue has items, grab them immediately (Zero latency for high throughput)
                    if not self._queue.empty():
                        try:
                            req = self._queue.get_nowait()
                            pending.append(req)
                            current_samples += req.batch.shape[0]
                            continue
                        except asyncio.QueueEmpty:
                            pass

                    # Optimization: If min_batch_size met and queue empty, fire immediately
                    if current_samples >= self.min_batch_size:
                        break

                    # Wait for next item until deadline
                    timeout = deadline - loop.time()
                    if timeout <= 0:
                        break

                    try:
                        req = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                        pending.append(req)
                        current_samples += req.batch.shape[0]
                        # print(f"DEBUG: Pending samples: {current_samples}/{self.batch_size}")
                    except TimeoutError:
                        # print(f"DEBUG: Batch timeout reached. Processing {current_samples} samples.")
                        break

                # Process batch
                # print(f"[RayInferenceActor] Processing batch of {current_samples} samples")
                self._process_batch(pending)

            except Exception:
                import traceback

                traceback.print_exc()
            finally:
                pending.clear()

    def _process_batch(self, requests: list[_InferenceRequest]) -> None:
        if not requests:
            return

        # Group by slot
        by_slot: dict[str, list[_InferenceRequest]] = {}
        for r in requests:
            slot = r.model_slot or self.default_slot
            if slot not in by_slot:
                by_slot[slot] = []
            by_slot[slot].append(r)

        for slot, slot_reqs in by_slot.items():
            self._process_slot_batch(slot, slot_reqs)

    def _process_slot_batch(self, slot: str, requests: list[_InferenceRequest]) -> None:
        model = self.model_slots.get(slot)
        if model is None:
            # Slot not found, fallback or error? Error is safer.
            err = ValueError(f"Model slot '{slot}' not found")
            for r in requests:
                if hasattr(r, "user_future") and not r.user_future.done():
                    r.user_future.set_exception(err)
            return

        try:
            tensors = []
            counts = []
            payloads = []
            has_payload = False
            for r in requests:
                arr = r.batch
                if arr.ndim == 3:
                    arr = arr[np.newaxis, ...]
                tensors.append(torch.from_numpy(arr))
                counts.append(arr.shape[0])
                if r.native_payload:
                    payloads.extend(r.native_payload)
                    has_payload = True
                else:
                    payloads.extend([None] * arr.shape[0])

            inputs_t = torch.cat(tensors, dim=0).to(self.device, dtype=torch.float32)

            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type="cuda" if self.device.type == "cuda" else "cpu",
                    dtype=torch.float16
                    if self.device.type == "cuda"
                    else torch.bfloat16,
                ),
            ):
                # Only pass native_payload if the model signature likely supports it
                # or if we explicitly want to enable this for hybrid models.
                if has_payload:
                    try:
                        policy_logits, values = model(inputs_t, native_payload=payloads)
                    except TypeError:
                        # Fallback for models not supporting native_payload
                        policy_logits, values = model(inputs_t)
                else:
                    policy_logits, values = model(inputs_t)

            policy_logits = policy_logits.cpu()
            values = values.cpu()

            p_splits = policy_logits.split(counts, dim=0)
            v_splits = values.split(counts, dim=0)

            for r, p, v in zip(requests, p_splits, v_splits, strict=True):
                if hasattr(r, "user_future") and not r.user_future.done():
                    r.user_future.set_result((p, v))

        except Exception as e:
            for r in requests:
                if hasattr(r, "user_future") and not r.user_future.done():
                    r.user_future.set_exception(e)

    def set_weights(self, state_dict: Any, slot: str = "current") -> None:
        # Create slot if missing
        if slot not in self.model_slots:
            new_model = self.model_builder()
            new_model.to(self.device)
            new_model.eval()
            self.model_slots[slot] = new_model

        target_model = self.model_slots[slot]
        if hasattr(target_model, "load_state_dict"):
            target_model.load_state_dict(state_dict)
            target_model.eval()


class RayInferenceClient(InferenceClient):
    """Inference client that batches requests to Ray actors asynchronously."""

    def __init__(
        self,
        actors: Iterable[ray.actor.ActorHandle],
        max_batch_size: int = 8,
        # Note: min_batch_size is configured on the actor side during creation,
        # but we keep the signature clean here.
    ):
        self.actors: list[ray.actor.ActorHandle] = list(actors)
        self.max_batch_size = max_batch_size
        self._rr: int = 0
        self._pending_counts: list[int] = [0] * len(self.actors)
        self._ref_to_actor: dict[ray.ObjectRef, int] = {}
        self._device = torch.device("cpu")

    def notify_result(self, ref: ray.ObjectRef) -> None:
        """Decrement pending count when an inflight result is consumed."""
        actor_idx = self._ref_to_actor.pop(ref, None)
        if actor_idx is not None:
            self._pending_counts[actor_idx] = max(0, self._pending_counts[actor_idx] - 1)

    @property
    def device(self) -> torch.device:
        """Ray inference는 CPU 텐서로 직렬화하므로 CPU로 고정."""
        return self._device

    def infer(
        self,
        states: torch.Tensor,
        native_payload: list[object] | None = None,
        model_slot: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Synchronously run inference on a single or batched input."""
        ref = self.infer_async(
            states, native_payload=native_payload, model_slot=model_slot
        )
        result = ray.get(ref)
        # Ray가 (ObjectRef, ObjectRef) 또는 단일 ObjectRef를 반환할 가능성에 방어적으로 대응한다.
        if isinstance(result, ray.ObjectRef):
            result = ray.get(result)
        if isinstance(result, tuple) and len(result) == 2:
            policy_t, value_t = result
            if isinstance(policy_t, ray.ObjectRef):
                policy_t = ray.get(policy_t)
            if isinstance(value_t, ray.ObjectRef):
                value_t = ray.get(value_t)
        else:
            raise TypeError(f"Unexpected ray result type: {type(result)}")
        # actor가 torch.Tensor를 반환하므로 그대로 전달, 혹시 numpy여도 방어
        if not isinstance(policy_t, torch.Tensor):
            policy_t = torch.as_tensor(policy_t)
        if not isinstance(value_t, torch.Tensor):
            value_t = torch.as_tensor(value_t)
        return policy_t, value_t

    def infer_async(
        self,
        states: torch.Tensor,
        native_payload: list[object] | None = None,
        model_slot: str | None = None,
    ) -> ray.ObjectRef:
        """비동기 추론: 즉시 ObjectRef를 반환한다."""
        if isinstance(states, np.ndarray):
            batch_np = states
        elif states.dim() in (3, 4):
            batch_np = states.detach().cpu().numpy()
        else:
            raise ValueError(f"Unsupported state shape: {states.shape}")

        # Least-loaded actor selection: pick the actor with fewest pending
        # requests to avoid stalling behind a slow actor.
        min_idx = 0
        min_pending = self._pending_counts[0]
        for i in range(1, len(self.actors)):
            if self._pending_counts[i] < min_pending:
                min_pending = self._pending_counts[i]
                min_idx = i
        self._pending_counts[min_idx] += 1
        actor = self.actors[min_idx]
        ref = actor.infer.remote(batch_np, native_payload, model_slot=model_slot)
        self._ref_to_actor[ref] = min_idx
        return ref


class SlotInferenceClient:
    """Wrapper around RayInferenceClient that binds calls to a specific model slot."""

    def __init__(self, client: RayInferenceClient, slot: str):
        self.client = client
        self.slot = slot

    @property
    def device(self) -> torch.device:
        return self.client.device

    @property
    def async_inflight_limit(self):
        return getattr(self.client, "async_inflight_limit", None)

    def notify_result(self, ref: ray.ObjectRef) -> None:
        self.client.notify_result(ref)

    def infer(
        self, states: torch.Tensor, native_payload: list[object] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.client.infer(states, native_payload, model_slot=self.slot)

    def infer_async(
        self, states: torch.Tensor, native_payload: list[object] | None = None
    ) -> ray.ObjectRef:
        return self.client.infer_async(states, native_payload, model_slot=self.slot)
