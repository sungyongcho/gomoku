# CLAUDE.md

This file provides guidance to Claude Code when working with the gomoku repository.

## Project Overview

Full-stack Gomoku game with a Nuxt 3 frontend, C++ minimax AI backend, and Python AlphaZero AI backend. The game uses custom rules on a 19x19 board: 5-in-a-row wins, double-three restriction (forbidden move creating 2+ open threes), and two-stone capture (sandwich pattern). Capturing 10 total stones (5 pairs) also counts as a win.

## Architecture

```
gomoku/
├── front/       # Nuxt 3 frontend (Vue 3 + TypeScript + Pinia + PrimeVue + Tailwind)
├── minimax/     # C++ minimax AI backend (libwebsockets + RapidJSON)
├── alphazero/   # Python AlphaZero AI backend (PyTorch + MCTS + FastAPI)
│   ├── gomoku/          # Core Python package
│   │   ├── core/        # Game engine & rules
│   │   ├── model/       # ResNet policy/value network
│   │   ├── pvmcts/      # MCTS facade & search engines
│   │   ├── alphazero/   # Agent, trainer, dataset, evaluation, self-play runners
│   │   ├── inference/   # Inference abstraction (local, multiprocess, Ray)
│   │   ├── scripts/     # Training & eval CLI entry points
│   │   └── utils/       # Config loader, paths, serialization
│   ├── cpp/             # C++ pybind11 extensions (rules + native MCTS)
│   ├── server/          # FastAPI serving layer (WebSocket)
│   ├── configs/         # YAML training & serving configs
│   ├── tests/           # Test suite
│   ├── infra/           # Cloud cluster scripts (Ray on GCP)
│   └── models/          # Trained checkpoints
```

All three services communicate via WebSocket. The frontend connects to either minimax (`ws://localhost:8005/ws`) or alphazero (`ws://localhost:8080/ws`) based on the user's AI selection in game settings.

## Running the Project

```bash
# Copy env file
cp .env.example .env

# Source Docker aliases
source alias.sh

# Start all services
docker compose up front minimax alphazero

# Or use alias
dev-up
```

- Frontend: http://localhost:3000
- Minimax WS: ws://localhost:8005/ws
- AlphaZero WS: ws://localhost:8080/ws

### Aliases (alias.sh)
```bash
dev-up          # Start front + minimax + alphazero
dev-down        # Stop all services
dev-up-valgrind # Start with valgrind minimax
dev-up-debug    # Start with gdb minimax

# AlphaZero Ray cluster management
alphazero-ray-attach        # Attach to Ray cluster
alphazero-restart-cluster   # Restart cluster
alphazero-reserve-gpu       # Reserve GPU instance
alphazero-purge-gcp         # Purge GCP resources
```

## Frontend (front/)

- **Framework**: Nuxt 3.15, Vue 3.5, TypeScript
- **State**: Pinia stores (`game.store.ts` is the main one)
- **UI**: PrimeVue 4 components + Tailwind CSS
- **Pages**: index (home), game (play), debug (rule testing), test (evaluation), docs/[uid] (docs)
- **Key composables**: `useEnv.ts` (socket URLs), `useGameLogic.ts` (game rules), `games/` (modular rule logic)
- **Docs**: Markdown in `front/content/docs/`, rendered via Nuxt Content

### Frontend commands
```bash
cd front && npm install && npm run dev    # Development
cd front && npm run build                 # Production build
```

## Minimax Backend (minimax/)

- **Language**: C++98 with g++
- **WebSocket**: libwebsockets on port 8005
- **Search**: Alpha-beta pruning with PVS, iterative deepening, transposition table, killer moves
- **Difficulty**: easy (depth 5), medium (depth 10 + 0.4s limit), hard (depth 10 + hard eval)

### Minimax commands
```bash
cd minimax && make all       # Release build
cd minimax && make debug     # Debug build
cd minimax && make re        # Clean rebuild
```

## AlphaZero Backend (alphazero/)

Full AlphaZero reinforcement learning system with training pipeline and FastAPI serving layer. Python 3.13+, hybrid Python-C++ architecture.

- **ML Framework**: PyTorch 2.9 with ResNet policy-value network (128 channels, 12 residual blocks)
- **Search**: PUCT-based MCTS with multiple backends (sequential, vectorize, multiprocess, Ray)
- **Serving**: FastAPI + WebSocket on port 8080, loads a trained checkpoint
- **C++ Extensions**: pybind11 modules for rules (`renju_cpp`) and native MCTS (`gomoku_cpp`)
- **Config**: YAML configs loaded into Pydantic models, supports scheduled parameters
- **Build**: scikit-build-core + CMake for C++ extensions (C++14, pybind11 3.0.1)

### AlphaZero commands
```bash
cd alphazero && pip install -e ".[serve,torch-cpu,dev]"  # Install for serving
cd alphazero && pip install -e ".[ray,torch-cpu,dev]"    # Install for training
cd alphazero && pytest                                    # Run tests
cd alphazero && ruff check . && ruff format .             # Lint & format

# Training (modes: sequential, vectorize, mp, ray)
python -m gomoku.scripts.train --config configs/config_alphazero_test.yaml --mode sequential

# C++ extension rebuild
cd alphazero && pip install -e . -v --force-reinstall --no-cache-dir --no-deps
# Verify: python -c "from gomoku.cpp_ext import renju_cpp, gomoku_cpp; print('ok')"
```

### Python-C++ hybrid architecture
- **Python** (`gomoku/`): Game logic, MCTS tree management, training loops, inference
- **C++** (`cpp/`): Performance-critical rule validation (double-three, captures) and optional native MCTS engine
- Two pybind11 modules: `renju_cpp` (low-level rules) and `gomoku_cpp` (high-level game state + MCTS)
- C++ is optional; pure Python fallback is the default. Enable with `use_native: true` in config.

### Search engine modes (strategy pattern)
Four interchangeable MCTS backends in `gomoku/pvmcts/search/`, all implementing the `SearchEngine` interface:
- **sequential**: Single-threaded, for dev/debugging
- **vectorize**: Interleaved batched search, GPU-friendly
- **mp**: Multiprocess via process pool
- **ray**: Distributed across Ray cluster

### Configuration system
YAML configs are loaded into Pydantic `BaseConfig` subclasses defined in `gomoku/utils/config/loader.py`. Supports **scheduled parameters** (values that change by training iteration):
```yaml
learning_rate:
  - { until: 30, value: 0.002 }
  - { until: 137, value: 0.0008 }
```

Key config classes: `BoardConfig`, `ModelConfig`, `TrainingConfig`, `MctsConfig`, `EvaluationConfig`, `ParallelConfig`.

### Training pipeline
`gomoku/scripts/train.py` dispatches to pipeline runners in `gomoku/scripts/pipelines/` based on `--mode`. The pipeline orchestrates: self-play game generation -> replay buffer (Parquet shards with 8x symmetry augmentation) -> trainer (AMP-enabled backprop, optional PER) -> evaluation (arena/SPRT) -> model promotion. Each run is tracked via `manifest.json` in `runs/{run_id}/`.

### AlphaZero key modules
- `gomoku/core/`: Game engine (`Gomoku` class, `GameState` dataclass), rules (double-three, capture, termination)
- `gomoku/model/`: ResNet policy/value network (`policy_value_net.py`) — 128 channels, 12 residual blocks
- `gomoku/pvmcts/`: MCTS facade (`pvmcts.py`), tree nodes, PUCT selection, search engines
- `gomoku/alphazero/`: Agent, trainer, dataset (replay + PER), evaluation (arena, SPRT, Elo), self-play runners
- `gomoku/inference/`: Inference abstraction (local, multiprocess server, Ray batch client)
- `gomoku/utils/config/`: Pydantic config models with scheduled parameter support
- `gomoku/scripts/pipelines/`: Pipeline orchestrators per mode (sequential, vectorize, mp, ray)
- `server/`: FastAPI serving layer (WebSocket endpoints, protocol conversion, engine wrapper)
- `cpp/`: C++ sources — `ForbiddenPointFinder` (double-three), `GomokuCore` (game state), `MctsEngine` (native MCTS)
- `configs/`: YAML configs for training (`elo1800-v*.yaml`, test configs) and serving (`local_play.yaml`)
- `infra/`: GCP cluster management scripts (Ray cluster bootstrap, GPU reservation)

### AlphaZero code style
- Python 3.13+, line length 88 (Black-compatible)
- Ruff for linting and formatting (rules: E, W, F, B, I, UP, N, D, ANN)
- Type hints throughout, PEP 604 union syntax (`X | None`)
- Dataclasses with `slots=True` for state objects (`GameState`, `TreeNode`)
- `known-first-party = ["gomoku"]` for import sorting

## WebSocket Protocol

Both minimax and alphazero backends use the same WebSocket JSON protocol.

### Request (frontend -> backend)
```json
{
  "type": "move",
  "difficulty": "easy|medium|hard",
  "nextPlayer": "X|O",
  "goal": 5,
  "enableCapture": true,
  "enableDoubleThreeRestriction": true,
  "lastPlay": {"coordinate": {"x": 9, "y": 10}, "stone": "X"},
  "board": [[".", "X", "O", ...], ...],
  "scores": [{"player": "X", "score": 3}, {"player": "O", "score": 1}]
}
```

### Response (backend -> frontend)
```json
{
  "type": "move",
  "status": "success",
  "lastPlay": {"coordinate": {"x": 9, "y": 10}, "stone": "O"},
  "board": [[".", "X", "O", ...], ...],
  "capturedStones": [{"x": 5, "y": 8, "stone": "X"}],
  "scores": [{"player": "X", "score": 3}, {"player": "O", "score": 1}],
  "evalScores": [{"player": "X", "evalScores": 500, "percentage": 45}, ...],
  "executionTime": {"s": 1, "ms": 234, "ns": 0}
}
```

## Docker Services

Defined in `docker-compose.yml`:
- **front**: Nuxt dev server (port 3000), depends on minimax + alphazero
- **minimax**: C++ WebSocket server (port 8005)
- **alphazero**: Python FastAPI server (port 8080), mounts `runs/elo1800-gcp-v4/ckpt/champion.pt` as model checkpoint, runs uvicorn with hot-reload on `server/`, `gomoku/`, `configs/`
- **minimax_valgrind**: Memory profiling variant
- **minimax_debug**: GDB debugging variant (port 8006)

## Environment Variables

See `.env` / `.env.example`:
- `LOCAL_FRONT=3000` - Frontend port
- `LOCAL_FRONT_NUXT_CONTENT_WS=4000` - Nuxt Content WebSocket port
- `LOCAL_MINIMAX=8005` - Minimax port
- `LOCAL_MINIMAX_GDB=8006` - Minimax GDB port
- `LOCAL_ALPHAZERO=8080` - AlphaZero port
- `FRONT_WHERE=local` - "local" or "prod"
- `ALPHAZERO_CONFIG=configs/local_play.yaml` - Serving config path
- `ALPHAZERO_CHECKPOINT=models/champion.pt` - Path to model checkpoint (.pt file)
- `ALPHAZERO_DEVICE=cpu` - "cpu" or "cuda"
- `ALPHAZERO_MCTS_NUM_SEARCHS=200` - MCTS simulations per move

## Game Rules

- **Board**: 19x19 intersections
- **Win conditions**: 5+ stones in a row OR capture 5 pairs (10 stones)
- **Capture**: When placing a stone creates pattern `XOO_` -> `X__X`, the two middle opponent stones are removed
- **Double-three**: Forbidden to place a stone that simultaneously creates two or more "open threes" (three consecutive stones with open ends on both sides)
- **Players**: X (Black, Player 1) goes first, O (White, Player 2) second

## Key Type Definitions

Frontend types are in `front/types/game.ts`:
- `Stone`: `"O" | "X" | "."`
- `Settings`: game config (capture, doublethree, difficulty, ai engine selection)
- `SocketMoveRequest` / `SocketMoveResponse`: WebSocket message shapes

AlphaZero types (in `alphazero/gomoku/`):
- `GameState`: dataclass (`slots=True`) with numpy board, capture scores, next player, move history, optional native state
- `BoardConfig`: Pydantic model for game rules (board size, goals, double-three, capture)
- `PLAYER_1=1` (X/Black), `PLAYER_2=2` (O/White), `EMPTY_SPACE=0`
- `PolicyValueNet`: ResNet CNN — policy head outputs 361 logits, value head outputs scalar in [-1, 1]
