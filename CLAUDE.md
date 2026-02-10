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

Python AlphaZero reinforcement learning system integrated from the gmk-refactor repository. Contains the full training pipeline and a FastAPI serving layer.

- **ML Framework**: PyTorch 2.9 with ResNet policy-value network (128 channels, 12 residual blocks)
- **Search**: PUCT-based MCTS with multiple backends (sequential, vectorize, multiprocess, Ray)
- **Serving**: FastAPI + WebSocket on port 8080, loads a trained checkpoint
- **C++ Extensions**: pybind11 modules for rules (`renju_cpp`) and native MCTS (`gomoku_cpp`)
- **Config**: YAML configs loaded into Pydantic models, supports scheduled parameters

### AlphaZero commands
```bash
cd alphazero && pip install -e ".[serve,torch-cpu,dev]"  # Install for serving
cd alphazero && pip install -e ".[ray,torch-cpu,dev]"    # Install for training
cd alphazero && pytest                                    # Run tests
cd alphazero && ruff check . && ruff format .             # Lint & format

# Training
python -m gomoku.scripts.train --config configs/config_alphazero_test.yaml --mode sequential

# C++ extension rebuild
cd alphazero && pip install -e . -v --force-reinstall --no-cache-dir --no-deps
```

### AlphaZero key modules
- `gomoku/core/`: Game engine, rules (double-three, capture, termination)
- `gomoku/model/`: ResNet policy/value network (`policy_value_net.py`)
- `gomoku/pvmcts/`: MCTS facade, tree nodes, PUCT selection, search engines
- `gomoku/alphazero/`: Agent, trainer, dataset, evaluation (arena, SPRT, Elo), self-play
- `gomoku/inference/`: Inference abstraction (local, multiprocess, Ray)
- `gomoku/utils/config/`: Pydantic config models with scheduled parameter support
- `server/`: FastAPI serving layer (WebSocket endpoints, protocol conversion, engine wrapper)

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
- **front**: Nuxt dev server (port 3000)
- **minimax**: C++ WebSocket server (port 8005)
- **alphazero**: Python FastAPI server (port 8080)
- **minimax_valgrind**: Memory profiling variant
- **minimax_debug**: GDB debugging variant (port 8006)

## Environment Variables

See `.env` / `.env.example`:
- `LOCAL_FRONT=3000` - Frontend port
- `LOCAL_MINIMAX=8005` - Minimax port
- `LOCAL_ALPHAZERO=8080` - AlphaZero port
- `FRONT_WHERE=local` - "local" or "prod"
- `ALPHAZERO_CHECKPOINT` - Path to model checkpoint (.pt file)
- `ALPHAZERO_DEVICE` - "cpu" or "cuda"
- `ALPHAZERO_NUM_SEARCHES` - MCTS simulations per move

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

AlphaZero types:
- `GameState`: dataclass with numpy board, capture scores, next player, move history
- `BoardConfig`: Pydantic model for game rules configuration
- `PLAYER_1=1` (X/Black), `PLAYER_2=2` (O/White), `EMPTY_SPACE=0`
