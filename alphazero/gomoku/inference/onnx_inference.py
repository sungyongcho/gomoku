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
        # Torch 2.9 dynamo exporter produces ONNX shape metadata that currently
        # breaks ORT dynamic quantization (ShapeInferenceError: 722 vs 361).
        dynamo=False,
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


class _CalibrationReader:
    """Feed calibration samples to ONNX Runtime static quantization."""

    def __init__(self, input_name: str, samples: np.ndarray) -> None:
        self._items = iter({input_name: sample[None]} for sample in samples)

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self._items, None)


def _quantize_static(
    model_input: str, model_output: str, samples: np.ndarray
) -> None:
    """Apply static QDQ INT8 quantization calibrated on real positions."""
    import onnxruntime as ort
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    prepared = f"{model_output}.prepared.onnx"
    quant_pre_process(model_input, prepared, skip_symbolic_shape=True)
    input_name = ort.InferenceSession(
        model_input, providers=["CPUExecutionProvider"]
    ).get_inputs()[0].name
    quantize_static(
        prepared,
        model_output,
        _CalibrationReader(input_name, samples),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    logger.info("Model statically quantized to INT8: %s", model_output)


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
        calibration_samples: np.ndarray | None = None,
        num_threads: int = 0,
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
                if calibration_samples is None:
                    _quantize_dynamic(onnx_path, quantized_path)
                else:
                    _quantize_static(onnx_path, quantized_path, calibration_samples)
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
        sess_options.intra_op_num_threads = num_threads  # 0 = auto (all cores)
        sess_options.inter_op_num_threads = 1 if num_threads else 0

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
