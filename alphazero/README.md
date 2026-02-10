# gmk-alphazero

Refactored Gomoku AlphaZero codebase; packaging and build follow `pyproject.toml`.

If you are new to virtual environments, create and activate one before installation:
```bash
python -m venv .venv  # pick your own env name if you prefer
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

## Installation

- Base deps (without torch):
  ```bash
  pip install -e .
  ```

- Torch CPU:
  ```bash
  pip install -e ".[torch-cpu]" --extra-index-url https://download.pytorch.org/whl/cpu
  ```

- Torch CUDA 12.1:
  ```bash
  pip install -e ".[torch-cu121]" --extra-index-url https://download.pytorch.org/whl/cu121
  ```

- Ray + CPU torch:
  ```bash
  pip install -e ".[ray,torch-cpu]" --extra-index-url https://download.pytorch.org/whl/cpu
  ```

- Ray + CUDA 12.1 torch:
  ```bash
  pip install -e ".[ray,torch-cu121]" --extra-index-url https://download.pytorch.org/whl/cu121
  ```

- Dev tools (pytest/coverage/ruff/flake8/mypy/pre-commit):
  ```bash
  pip install -e ".[dev]"
  ```
- Combine extras as needed, for example Ray + CPU torch + dev tools:
  ```bash
  pip install -e ".[ray,torch-cpu,dev]" --extra-index-url https://download.pytorch.org/whl/cpu
  ```

> Choose the torch extra that matches your environment (CPU vs CUDA). After install, verify with:
> `python -c "import gomoku; from gomoku.cpp_ext import renju_cpp; print('ok')"`

### Notes on requirements files

- `pip install .` / `pip install -e .` uses only `pyproject.toml`.
- `requirements.txt` / `requirements.313.txt` are optional freeze files; use them only if you need exact pins for a specific environment, and keep them consistent with `pyproject.toml`.

### Replay data format

- No explicit schema version is used. Replay shards are validated for required keys and unknown fields raise errors on load. Keep writer/reader in sync with the current codebase.

## C++ extension build

- The pybind11 module is built via `scikit-build-core` and the CMake project under `cpp/`.
- Editable or wheel install will run CMake automatically; ensure build deps are present: `cmake`, `ninja`, `pybind11`, `scikit-build-core` (declared in `pyproject.toml`).
- You can force a clean rebuild by reinstalling:
  ```bash
  pip install -e . -v
  ```
- To force-reinstall and rebuild the extension (overwrite existing .so):
  ```bash
  pip install -e . -v --force-reinstall --no-cache-dir
  ```
- If you installed with extras (e.g., `.[ray,torch-cpu,dev]`), repeat the same extras on reinstall:
  ```bash
  pip install -e ".[ray,torch-cpu,dev]" -v --force-reinstall --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu
  ```
- To rebuild only the C++ extension after code changes (keeping the same extras), you can rerun the install with the same extras and `--no-deps` to avoid reinstalling Python deps:
  ```bash
  pip install -e ".[ray,torch-cpu,dev]" -v --force-reinstall --no-cache-dir --no-deps --extra-index-url https://download.pytorch.org/whl/cpu
  ```
- Post-install check:
  ```bash
  python -c "from gomoku.cpp_ext import renju_cpp; print('renju_cpp ok')"
  ```
- Post-install check (gomoku_cpp):
  ```bash
  python - <<'PY'
  from gomoku.cpp_ext import renju_cpp, gomoku_cpp
  print("renju_cpp ok:", hasattr(renju_cpp, "detect_doublethree"))
  core_g = gomoku_cpp.GomokuCore(9, True, True, 5, 5, 5)
  st = core_g.apply_move(core_g.initial_state(), 0, 0, 1)
  print("gomoku_cpp ok, action_size:", core_g.action_size())
  PY
  ```

### Manual CMake build (optional)

- If you want to build the extension manually (for debugging), from repo root:
  ```bash
  cmake -S cpp -B cpp/build -DPYTHON_EXECUTABLE=$(which python) -DCMAKE_BUILD_TYPE=Release
  cmake --build cpp/build
  ```
- The first `cmake -S ...` configures the build directory; run it again only if you clean `cpp/build` or change options/Python path. After that, `cmake --build cpp/build` alone rebuilds after source changes.
- To build specific targets manually:
  ```bash
  cmake --build cpp/build --target gomoku_cpp
  cmake --build cpp/build --target renju_cpp
  ```
- Build type is fixed to `Release` for now; Debug configuration is not provided.
- The built module is placed under `gomoku/cpp_ext/` (as configured in `cpp/CMakeLists.txt`).
