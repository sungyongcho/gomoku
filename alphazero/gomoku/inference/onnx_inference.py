import logging
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from gomoku.inference.base import InferenceClient

logger = logging.getLogger(__name__)


def _export_to_onnx(
    model: nn.Module, num_planes: int, board_h: int, board_w: int, output_path: str
) -> None:
    """Export PyTorch model to ONNX."""
    dummy_input = torch.randn(1, num_planes, board_h, board_w, device="cpu")

    # Ensure model is in eval mode
    model.eval()

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["policy", "value"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "policy": {0: "batch_size"},
            "value": {0: "batch_size"},
        },
    )
    logger.info("Model exported to ONNX: %s", output_path)


def _quantize_dynamic(model_input: str, model_output: str) -> None:
    """Apply dynamic INT8 quantization to ONNX model."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input,
        model_output,
        weight_type=QuantType.QUInt8,
    )
    logger.info("Model quantized to INT8: %s", model_output)


class OnnxInference(InferenceClient):
    """ONNX Runtime inference backend with optional INT8 quantization.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to convert to ONNX.
    num_planes : int
        Number of input planes (channels).
    board_h : int
        Board height.
    board_w : int
        Board width.
    quantize : bool
        If True, apply INT8 dynamic quantization for faster CPU inference.
    onnx_cache_dir : str | None
        Directory to cache ONNX files. If None, uses a temp directory.
    """

    def __init__(
        self,
        model: nn.Module,
        num_planes: int,
        board_h: int,
        board_w: int,
        quantize: bool = True,
        onnx_cache_dir: str | None = None,
    ):
        import onnxruntime as ort

        self._device = torch.device("cpu")

        # Determine cache directory
        if onnx_cache_dir:
            cache_dir = Path(onnx_cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._tmp_dir = tempfile.TemporaryDirectory()
            cache_dir = Path(self._tmp_dir.name)

        onnx_path = str(cache_dir / "model.onnx")
        quantized_path = str(cache_dir / "model_int8.onnx")

        # Export model to ONNX
        # Always re-export to ensure sync with current checkpoint?
        # Ideally check hash, but for now re-exporting is safer and fast enough.
        _export_to_onnx(model, num_planes, board_h, board_w, onnx_path)

        # Optionally quantize
        final_path = onnx_path
        if quantize:
            try:
                _quantize_dynamic(onnx_path, quantized_path)
                final_path = quantized_path
            except Exception:
                logger.warning(
                    "INT8 quantization failed, falling back to FP32 ONNX",
                    exc_info=True,
                )

        # Create ONNX Runtime session with optimizations
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        # Use all available CPU threads
        sess_options.intra_op_num_threads = 0  # 0 = auto (all cores)
        sess_options.inter_op_num_threads = 0

        self._session = ort.InferenceSession(
            final_path,
            sess_options,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

        model_type = "INT8" if final_path == quantized_path else "FP32"
        logger.info(
            "ONNX Runtime session created (%s), input=%s",
            model_type,
            self._input_name,
        )

    @property
    def device(self) -> torch.device:
        return self._device

    def infer(
        self, inputs: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run inference via ONNX Runtime."""
        squeeze_output = False
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)
            squeeze_output = True
        elif inputs.dim() != 4:
            raise ValueError(
                f"Expected input shape (C, H, W) or (B, C, H, W), got {inputs.shape}"
            )

        # Convert to numpy for ONNX Runtime
        input_np = inputs.cpu().numpy().astype(np.float32)

        # Run inference
        policy_np, value_np = self._session.run(
            None, {self._input_name: input_np}
        )

        # Convert back to torch tensors
        policy = torch.from_numpy(policy_np)
        value = torch.from_numpy(value_np)

        if squeeze_output:
            policy = policy.squeeze(0)
            value = value.squeeze(0)

        return policy, value

    def infer_batch(
        self, batch_inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run batch inference via ONNX Runtime."""
        if batch_inputs.dim() != 4:
            raise ValueError(
                f"Expected 4D tensor (B, C, H, W), got {batch_inputs.shape}"
            )
        return self.infer(batch_inputs)
