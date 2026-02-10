import itertools
from multiprocessing import Queue
import queue

import torch

from gomoku.inference.base import InferenceClient


class MPInferenceClient(InferenceClient):
    """Inference client that forwards batch requests via multiprocessing queues."""

    def __init__(self, worker_id: int, in_q: Queue, out_q: Queue):
        self.worker_id = worker_id
        self.in_q = in_q
        self.out_q = out_q
        self._id_gen = itertools.count()
        self._device = torch.device("cpu")

    @property
    def device(self) -> torch.device:
        """Return the device used for IPC tensors."""
        return self._device

    def infer(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward encoded states (single or batch) to the worker.

        Parameters
        ----------
        states : torch.Tensor
            Encoded state tensor. Shape can be (C, H, W) or (B, C, H, W).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Policy logits and value tensors (batch-aligned).
        """
        if states.dim() == 3:
            batch = [states.cpu()]
        elif states.dim() == 4:
            # Split along batch dimension to keep server batching simple.
            batch = [states[i].cpu() for i in range(states.size(0))]
        else:
            raise ValueError(f"Unsupported state shape for MP inference: {tuple(states.shape)}")

        responses: list[tuple[torch.Tensor, torch.Tensor, int]] = []
        expected_ids: list[int] = []
        for _ in batch:
            req_id = next(self._id_gen)
            expected_ids.append(req_id)
            self.in_q.put((self.worker_id, req_id, _))

        seen_ids: set[int] = set()
        for _ in batch:
            try:
                response_id, policy, value, error_msg = self.out_q.get(timeout=30.0)
            except queue.Empty as exc:
                raise TimeoutError("Inference response timed out") from exc

            if error_msg:
                raise RuntimeError(f"Inference error: {error_msg}")
            if response_id not in expected_ids or response_id in seen_ids:
                print(f"[MPInferenceClient] ID mismatch: expected one of {expected_ids}, got {response_id}")
                raise RuntimeError("ID mismatch in MP inference response")
            seen_ids.add(response_id)
            # No strict ordering guarantee from queue; collect and sort later.
            responses.append(
                (
                    torch.as_tensor(policy),
                    torch.as_tensor(value),
                    response_id,
                )
            )

        if len(seen_ids) != len(expected_ids):
            print(f"[MPInferenceClient] ID mismatch: missing responses for {set(expected_ids) - seen_ids}")
            raise RuntimeError("ID mismatch in MP inference response")

        # Reorder by request id to align with inputs
        responses.sort(key=lambda x: x[2])
        policies = torch.stack([p for p, _, _ in responses])
        values = torch.stack([v.view(-1)[0] for _, v, _ in responses]).view(-1, 1)
        return policies, values
