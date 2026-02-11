# AlphaZero Deployment Performance Optimization

## Context

The AlphaZero deployment on GCP takes ~30 seconds for just 20 MCTS searches. The expected time is ~1-2 seconds. This document identifies the root causes and proposes fixes.

## Root Cause Analysis

### The Real Bottleneck: `renju_cpp.detect_doublethree()` is O(n^2) per call

The **dominant bottleneck** is NOT the neural network inference -- it's the double-three detection in the Python MCTS path.

**File**: `alphazero/cpp/src/bindings.cpp:11-87`

Every single call to `renju_cpp.detect_doublethree(board, x, y, player, board_size)` does:

1. **pybind11 forcecast**: Board is `np.int8` but binding expects `int` (int32) -- pybind11 creates a **full copy** of the 19x19 board cast to int32 on every call
2. **New `CForbiddenPointFinder(board_size)`**: Allocates a new finder object
3. **361 `SetStone()` calls**: Loops over the entire board to copy it into the finder's internal representation (with color swapping for white player)
4. **`IsDoubleThree(x, y)`**: Finally checks the single point
5. **Destroy everything**

This happens in two places per MCTS simulation:

**Place 1 -- `get_legal_moves()` (`gomoku.py:238-244`)**:
```python
for idx, x, y in zip(empty_indices, x_coords, y_coords):
    if detect_doublethree(board, x, y, player, board_size):  # Called ~150 times!
        continue
```

**Place 2 -- `get_encoded_state()` (`gomoku.py:441-445`)**:
```python
for x, y in zip(xs, ys):
    if detect_doublethree(board, x, y, player, board_size):  # Called ~150 times!
        features[b_idx, 7, y, x] = 1.0
```

**Per MCTS simulation (Python path)**: ~300 calls to `detect_doublethree`, each creating a new finder + copying the full board twice (pybind11 forcecast + SetStone loop).

**For 20 simulations**: ~6,000 finder constructions + ~2,160,000 `SetStone` calls + 6,000 `IsDoubleThree` checks.

Additionally, `get_value_and_terminated()` (`gomoku.py:319-320`) also calls `get_legal_moves()` when double-three is enabled, adding more overhead to every terminal check.

### The Native C++ Path Avoids This Entirely

When `use_native: true` is working, the `GomokuCore` C++ class:
- Maintains internal `CForbiddenPointFinder` state (no per-call reconstruction)
- `get_legal_moves()` routes through C++: `self._native_core.get_legal_moves(state.native_state)` -- fast
- State encoding uses `_native_core.write_state_features()` -- fast
- C++ MCTS (`CppSearchStrategy`) handles tree search internally, only calling Python for NN forward pass

**If native is working**: 20 searches ≈ 20 × NN forward pass ≈ **4-10 seconds** (FP32 on shared CPU)
**If native is broken**: 20 searches ≈ 6000 detect_doublethree calls + 20 × NN forward pass ≈ **20-30 seconds**

The 30-second observation strongly suggests **native C++ MCTS is not working in production**, falling back to the Python path silently.

### Secondary: NN Forward Pass is Slow on Shared CPU

Even with native C++ working, 20 FP32 forward passes on an e2-standard-2 shared core ≈ 4-10 seconds.

Model compute: ~1.3 GFLOPS per forward pass (128ch, 12 ResBlocks, 19x19 input).
e2-standard-2 shared core effective throughput: ~5-15 GFLOPS.
Per inference: ~100-250ms. For 200 searches: ~20-50 seconds.

ONNX Runtime + INT8 quantization would give ~3-5x speedup here.

### Bug: NUM_SEARCHES Hardcoded

`websocket.py:32`: `NUM_SEARCHES: int | None = 100` -- ignores both env var and config file.

### Model Cannot Be Downsized

The `champion.pt` checkpoint weights are shaped for 128 channels / 12 blocks. Changing `deploy.yaml` would cause a shape mismatch. Requires retraining.

## Cost Breakdown (per 20 searches, Python MCTS path)

| Operation | Calls | Cost/call | Total | % of 30s |
|-----------|-------|-----------|-------|----------|
| `detect_doublethree` (get_legal_moves) | ~3,000 | ~2-5ms | **6-15s** | **40-50%** |
| `detect_doublethree` (get_encoded_state) | ~3,000 | ~2-5ms | **6-15s** | **40-50%** |
| NN forward pass (FP32, 2 shared cores) | 20 | ~200-500ms | **4-10s** | **15-30%** |
| `get_next_state` (board copies, expand) | ~200 | ~0.05ms | ~10ms | <1% |
| Everything else | - | - | ~100ms | <1% |

## Diagnostic: Verify Native Status

**Before implementing any fix, check production logs for:**
```
"Native MCTS requested but unavailable; falling back to Python MCTS."
"Torch threads configured: num_threads=X interop_threads=Y available_cpus=Z"
```

If the native warning appears, the C++ extensions (`gomoku_cpp`) failed to load in the Docker image. This means the entire MCTS falls back to the Python path with the O(n^2) double-three disaster.

**Quick local check:**
```bash
cd alphazero
python -c "from gomoku.cpp_ext import renju_cpp, gomoku_cpp; print('C++ extensions OK')"
```

## Implementation Plan

### Priority 1: Ensure Native C++ MCTS Works (fixes 30s → ~5-10s)

1. **Verify C++ extensions build in Docker** -- check `Dockerfile.prod` build logs for gomoku_cpp compilation errors
2. **Add explicit startup logging** -- log whether native MCTS or Python fallback is used
3. **Fix if broken** -- ensure `gomoku_cpp.so` is built and importable in the production image

### Priority 2: Fix NUM_SEARCHES Bug

**File**: `alphazero/server/websocket.py` lines 27-32

```python
# Before:
_raw = os.environ.get("ALPHAZERO_MCTS_NUM_SEARCHS")
NUM_SEARCHES: int | None = 100  # HARDCODED

# After:
_raw = os.environ.get("ALPHAZERO_MCTS_NUM_SEARCHS")
NUM_SEARCHES: int | None = int(_raw) if _raw and _raw.strip().isdigit() else None
```

### Priority 3: ONNX Runtime + INT8 (fixes ~5-10s → ~1-3s for 200 searches)

1. **New file**: `alphazero/gomoku/scripts/export_onnx.py` -- export model to ONNX + INT8 quantization
2. **New file**: `alphazero/gomoku/inference/onnx_runtime.py` -- `OnnxInference` implementing `InferenceClient` interface
3. **Modify**: `alphazero/server/engine.py` -- select backend via `ALPHAZERO_INFER_BACKEND` env var
4. **Modify**: `alphazero/pyproject.toml` -- add `onnxruntime>=1.19.0` to serve extras
5. **Modify**: `deploy/02_deploy.sh` -- add ONNX export step + env var

### Priority 4: Config Updates

- `alphazero/configs/deploy.yaml`: `num_searches: 200`
- `.env`: `ALPHAZERO_MCTS_NUM_SEARCHS=200`, `ALPHAZERO_INFER_BACKEND=onnx-int8`
- `docker-compose.yml`: pass new env vars

## Expected Performance After Fixes

| Scenario | Per-inference | 200 searches |
|----------|-------------|-------------|
| Current (Python MCTS + FP32, broken native) | N/A (30s for 20) | hours |
| After P1: Native C++ MCTS + FP32 | ~200-500ms | ~40-100s |
| After P1+P3: Native C++ MCTS + ONNX INT8 | ~50-150ms | **~10-30s** |
| After P1+P3 on c2-standard-4 (dedicated cores) | ~25-75ms | **~5-15s** |

## GCP Instance Options

| Instance | vCPUs | Type | $/day (eu-west1) |
|----------|-------|------|-----------------|
| e2-standard-2 (current) | 2 shared | General | $1.77 |
| c2-standard-4 | 4 dedicated Cascade Lake | Compute-optimized | $5.51 |

## Files to Modify/Create

| File | Action |
|------|--------|
| `alphazero/server/websocket.py` | Fix NUM_SEARCHES bug |
| `alphazero/server/engine.py` | Add ONNX backend selection + startup diagnostics |
| `alphazero/gomoku/inference/onnx_runtime.py` | **New** -- ONNX inference client |
| `alphazero/gomoku/scripts/export_onnx.py` | **New** -- ONNX export + quantization |
| `alphazero/pyproject.toml` | Add onnxruntime to serve extras |
| `alphazero/configs/deploy.yaml` | Update num_searches to 200 |
| `deploy/02_deploy.sh` | Add ONNX export step + env var |
| `docker-compose.yml` | Pass new env vars |
| `.env` | Set inference backend and search count |

## Key Reusable Code

- `InferenceClient` interface: `alphazero/gomoku/inference/base.py`
- `LocalInference` pattern: `alphazero/gomoku/inference/local.py`
- `align_state_dict_to_model`: `alphazero/gomoku/utils/state_dict_utils.py`
- `AlphaZeroEngine._extract_state_dict`: `alphazero/server/engine.py:238-258`
- `AlphaZeroEngine._detect_available_cpu_count`: `alphazero/server/engine.py:219-236`
- `calc_num_planes`: `alphazero/gomoku/model/model_helpers.py`

## Sources

- [INT8 Quantization for x86 CPU in PyTorch](https://pytorch.org/blog/int8-quantization/)
- [ONNX vs PyTorch Speed Comparison](https://dev-kit.io/blog/machine-learning/onnx-vs-pytorch-speed-comparison)
- [GCP c2-standard-4 pricing](https://gcloud-compute.com/c2-standard-4.html)
- [GCP e2-standard-2 pricing](https://gcloud-compute.com/e2-standard-2.html)
