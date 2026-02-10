from abc import ABC, abstractmethod

import torch


class InferenceClient(ABC):
    """
    Abstract base class for inference backends (Local, Ray, TensorRT, etc.).

    This defines the contract required by SearchEngines.
    """

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Return the torch device where the model resides."""
        pass

    @abstractmethod
    def infer(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run inference on a single state or a batch of states.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (C, H, W) or (B, C, H, W).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (policy_logits, value)
            - policy_logits: (Action_Dim,) or (B, Action_Dim)
            - value: (1,) or (B, 1)
        """
        pass

    def infer_batch(
        self, batch_inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run inference on a batch of states.

        Default implementation simply delegates to infer(), but subclasses
        can override this for batch-specific optimizations (e.g., dynamic batching).
        """
        return self.infer(batch_inputs)
