import torch
import torch.nn as nn

from .base import InferenceClient

__all__ = ["LocalInference"]


class LocalInference(InferenceClient):
    """Local inference wrapper that respects the provided device.

    ``prefers_sync`` tells the native search to call ``infer`` directly: the
    async path would round-trip every batch through the Ray object store.

    With ``dtype="bfloat16"`` the forward pass runs under bf16 autocast on the
    model's device: about twice as fast as float32 on CPUs with native bf16, and
    on tensor cores on the GPU. Outputs are returned as float32 so callers can
    convert them to numpy.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | str | None = None,
        dtype: str = "float32",
    ):
        if dtype not in ("float32", "bfloat16"):
            raise ValueError(f"Unsupported local inference dtype: {dtype}")
        target_device: torch.device
        if device is not None:
            target_device = torch.device(device)
        elif hasattr(model, "device"):
            target_device = torch.device(model.device)
        else:
            try:
                target_device = next(model.parameters()).device  # type: ignore[call-overload]
            except StopIteration:
                target_device = torch.device("cpu")

        self._model = model.to(target_device)
        self._device = target_device
        self._model.eval()
        self._bf16 = dtype == "bfloat16"

    prefers_sync = True

    @property
    def device(self) -> torch.device:
        return self._device

    def infer(
        self, inputs: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run inference on a single state or batch.

        Parameters
        ----------
        inputs : torch.Tensor
            State tensor of shape (C, H, W) or (B, C, H, W). A 3D tensor is
            implicitly treated as a batch of size 1.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Policy logits and value tensors.
        """
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)
            squeeze_output = True
        elif inputs.dim() == 4:
            squeeze_output = False
        else:
            raise ValueError(
                f"Expected input shape (C, H, W) or (B, C, H, W), got {inputs.shape}"
            )

        inputs = inputs.to(self._device)

        with torch.inference_mode():
            if self._bf16:
                with torch.autocast(device_type=self._device.type, dtype=torch.bfloat16):
                    policy, value = self._model(inputs)
                policy, value = policy.float(), value.float()
            else:
                policy, value = self._model(inputs)

        if squeeze_output:
            policy = policy.squeeze(0)
            value = value.squeeze(0)

        return policy, value

    def infer_batch(
        self, batch_inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run batch inference. Requires input shape (B, C, H, W).
        """
        if batch_inputs.dim() != 4:
            raise ValueError(
                f"Expected 4D tensor (B, C, H, W) for batch inference, got {batch_inputs.shape}"
            )
        return self.infer(batch_inputs)

    def infer_async(
        self,
        inputs: torch.Tensor,
        native_payload: list[object] | None = None,
        model_slot: str | None = None,
    ):
        """
        Mimic async inference by wrapping result in ray.ObjectRef.
        Required for compatibility with RayAsyncEngine.
        """
        import ray

        res = self.infer(inputs)
        return ray.put(res)
