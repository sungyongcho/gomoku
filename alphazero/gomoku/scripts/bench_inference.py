"""Compare ONNX inference speed and accuracy across quantization modes and model sizes.

Pin the process to the target core count, for example:
    taskset -c 0,1 python -m gomoku.scripts.bench_inference --checkpoint models/champion.pt
"""

import argparse
import os
import random
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
import torch

from gomoku.core.gomoku import Gomoku
from gomoku.inference.onnx_inference import _export_to_onnx
from gomoku.model.policy_value_net import PolicyValueNet
from gomoku.utils.config.loader import load_and_parse_config
from gomoku.utils.state_dict_utils import align_state_dict_to_model


class _Reader(CalibrationDataReader):
    def __init__(self, name: str, samples: np.ndarray) -> None:
        self._items = iter({name: sample[None]} for sample in samples)

    def get_next(self) -> dict | None:
        return next(self._items, None)


def _session(path: Path, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(path), options, providers=["CPUExecutionProvider"]
    )


def _collect_positions(
    game: Gomoku, session: ort.InferenceSession, count: int, seed: int
) -> np.ndarray:
    """Walk games through the top network moves so samples resemble real play."""
    rng = random.Random(seed)
    name = session.get_inputs()[0].name
    samples: list[np.ndarray] = []
    while len(samples) < count:
        state = game.get_initial_state()
        for _ in range(rng.randint(4, 60)):
            encoded = game.get_encoded_state(state)
            encoded = np.asarray(encoded[0] if encoded.ndim == 4 else encoded, np.float32)
            samples.append(encoded)
            logits = session.run(None, {name: encoded[None]})[0].reshape(-1)
            logits[state.board.reshape(-1) != 0] = -np.inf
            action = int(rng.choice(np.argsort(logits)[-3:].tolist()))
            state = game.get_next_state(
                state, (action % game.col_count, action // game.col_count), state.next_player
            )
            if game.get_value_and_terminated(state, (action % game.col_count, action // game.col_count))[1]:
                break
    rng.shuffle(samples)
    return np.stack(samples[:count])


def _latency_ms(session: ort.InferenceSession, samples: np.ndarray) -> float:
    name = session.get_inputs()[0].name
    timings = []
    for sample in samples:
        started = time.perf_counter()
        session.run(None, {name: sample[None]})
        timings.append(time.perf_counter() - started)
    return 1000 * statistics.median(timings)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    probs = np.exp(shifted)
    return probs / probs.sum()


def _accuracy(
    reference: ort.InferenceSession, other: ort.InferenceSession, samples: np.ndarray
) -> tuple[float, float, float]:
    name = reference.get_inputs()[0].name
    agree, distance, value_error = [], [], []
    for sample in samples:
        ref_policy, ref_value = reference.run(None, {name: sample[None]})
        new_policy, new_value = other.run(None, {name: sample[None]})
        p, q = _softmax(ref_policy.reshape(-1)), _softmax(new_policy.reshape(-1))
        agree.append(p.argmax() == q.argmax())
        distance.append(0.5 * np.abs(p - q).sum())
        value_error.append(abs(float(ref_value.reshape(-1)[0]) - float(new_value.reshape(-1)[0])))
    return float(np.mean(agree)), float(np.mean(distance)), float(np.mean(value_error))


def _build(config, game: Gomoku, blocks: int, channels: int, checkpoint: str | None):
    model_cfg = config.model.model_copy(update={"num_resblocks": blocks, "num_hidden": channels})
    model = PolicyValueNet(game, model_cfg, torch.device("cpu"))
    if checkpoint:
        raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = raw.get("model_state_dict", raw.get("state_dict", raw))
        align_state_dict_to_model(state, model.state_dict())
        model.load_state_dict(state)
    return model.eval(), model_cfg


def _variants(model, model_cfg, config, game, work: Path, calibration_count: int, seed: int):
    fp32 = work / "fp32.onnx"
    _export_to_onnx(model, model_cfg.num_planes, config.board.num_lines, config.board.num_lines, str(fp32))
    prepared = work / "prepared.onnx"
    quant_pre_process(str(fp32), str(prepared), skip_symbolic_shape=True)
    dynamic = work / "dynamic.onnx"
    quantize_dynamic(str(fp32), str(dynamic), weight_type=QuantType.QUInt8)
    base = _session(fp32, os.cpu_count() or 1)
    calibration = _collect_positions(game, base, calibration_count, seed)
    static = work / "static.onnx"
    quantize_static(
        str(prepared),
        str(static),
        _Reader(base.get_inputs()[0].name, calibration),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    return {"fp32": fp32, "dynamic": dynamic, "static": static}, base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/deploy.yaml")
    parser.add_argument("--checkpoint", default="models/champion.pt")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--budget-s", type=float, default=2.5)
    parser.add_argument("--sizes", nargs="+", default=["12x128", "8x96", "6x96", "6x64", "4x64"])
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--calibration", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    config = load_and_parse_config(args.config)
    game = Gomoku(config.board, use_native=False)
    full_size = f"{config.model.num_resblocks}x{config.model.num_hidden}"
    print(f"threads={args.threads} affinity={len(os.sched_getaffinity(0))} budget={args.budget_s}s")
    print("size    mode     ms    sims_in_budget  top1_agree  tv_dist  value_mae")
    for size in args.sizes:
        blocks, channels = (int(part) for part in size.split("x"))
        # Only the served size can load the real weights; others measure speed only.
        checkpoint = args.checkpoint if size == full_size else None
        model, model_cfg = _build(config, game, blocks, channels, checkpoint)
        with tempfile.TemporaryDirectory() as tmp:
            paths, base = _variants(model, model_cfg, config, game, Path(tmp), args.calibration, args.seed)
            samples = _collect_positions(game, base, args.samples, args.seed + 1)
            reference = _session(paths["fp32"], args.threads)
            for mode, path in paths.items():
                session = _session(path, args.threads)
                latency = _latency_ms(session, samples)
                sims = int(args.budget_s * 1000 / latency)
                quality = ""
                if checkpoint and mode != "fp32":
                    agree, distance, value_error = _accuracy(reference, session, samples)
                    quality = f"{agree:10.3f}  {distance:7.4f}  {value_error:9.4f}"
                print(f"{size:7s} {mode:7s} {latency:5.1f}  {sims:14d}  {quality}", flush=True)


if __name__ == "__main__":
    main()
