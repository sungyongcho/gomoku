import numpy as np
import pytest
import torch
import ray

from gomoku.inference.ray_client import RayInferenceActor, RayInferenceClient


def test_ray_native_payload_roundtrip() -> None:
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=1)

    class DummyModel:
        def __init__(self, action_size: int = 4):
            self.action_size = action_size

        def __call__(
            self, inputs: torch.Tensor, native_payload: list[np.ndarray] | None = None
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # inputs shape: (B, C, H, W) or (B, dim)
            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(0)

            # Use native_payload in the result to verify roundtrip
            payload_sum = 0.0
            if native_payload:
                print(f"DEBUG: native_payload length={len(native_payload)}")
                for i, p in enumerate(native_payload):
                    if p is not None:
                        s = float(p.sum())
                        print(f"DEBUG: payload[{i}] sum={s}")
                        payload_sum += s
                    else:
                        print(f"DEBUG: payload[{i}] is None")
            else:
                 print("DEBUG: native_payload is None or empty")

            summed = inputs.view(inputs.size(0), -1).sum(dim=1) + payload_sum
            policy = summed.unsqueeze(1).repeat(1, self.action_size)
            value = summed
            return policy, value

        def to(self, device):
            return self

        def eval(self):
            return self

    # Payload sum is 8.0 (2*2*2 ones)
    payload = np.ones((2, 2, 2), dtype=np.float32)

    # Create actor with a factory that returns DummyModel
    # Note: RayInferenceActor expects a model_fn
    actor = RayInferenceActor.remote(lambda: DummyModel(action_size=4))
    client = RayInferenceClient([actor], max_batch_size=4)

    try:
        # Input sum is 0.0
        states = torch.zeros((2, 2, 2), dtype=torch.float32)

        # We pass a list of 1 payload because infer handles a batch of inputs?
        # Actually infer takes a single state or batch?
        # RayInferenceClient.infer signature: (state: Tensor, native_payload: Any = None)
        # It puts (state, payload) into a queue.

        policy_t, value_t = client.infer(states, native_payload=[payload])

        expected_total = 8.0
        print(f"ray native payload policy[0]: {policy_t[0]}")
        print(f"ray native payload value: {value_t}")
        print(f"Expected: {expected_total}, Got: {float(value_t.view(-1)[0])}")

        assert policy_t.shape[1] == 4
        assert float(value_t.view(-1)[0]) == pytest.approx(expected_total, rel=0, abs=1e-5)
    finally:
        ray.shutdown()
