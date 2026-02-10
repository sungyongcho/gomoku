# Integration Plan: gmk-refactor into gomoku/alphazero

## Overview

Integrate the `gmk-refactor` AlphaZero training/inference system into `gomoku/alphazero/` so that:
1. All gmk-refactor source files live under `gomoku/alphazero/`
2. A trained model can be loaded and served via WebSocket to the Nuxt frontend
3. The alphazero backend speaks the same WebSocket protocol as the minimax backend
4. Training can still be run from within `gomoku/alphazero/`

---

## Current State

### Phase 1 (File Migration) — COMPLETE

All gmk-refactor source files have been copied into `gomoku/alphazero/`:

gomoku/alphazero/
├── gomoku/                    # gmk-refactor Python package (copied)
│   ├── core/                  # Game engine, rules, coordinate system
│   ├── model/                 # PolicyValueNet (ResNet, 128ch x 12 blocks)
│   ├── pvmcts/                # MCTS facade, tree nodes, search engines
│   ├── alphazero/             # Agent, trainer, dataset, eval, self-play runners
│   ├── inference/             # Inference abstraction (local, mp, ray)
│   ├── utils/                 # Config loader, manifest, paths, serialization
│   ├── scripts/               # train.py, eval.py, pipelines/
│   └── cpp_ext/               # Compiled .so files (built from cpp/)
│
├── server/                    # OLD code — needs full rewrite (Phase 2)
│   ├── websocket.py           # Imports from deleted `core.board` — broken
│   ├── engine.py              # Placeholder — needs implementation
│   └── protocol.py            # Placeholder — needs implementation
│   (missing: __init__.py)
│
├── cpp/                       # C++ extension source (pybind11)
├── configs/                   # Training YAML configs + local_play.yaml
├── tests/                     # Test suite
├── documents/                 # Reference docs
├── runs/                      # Training checkpoints (gitignored)
│   └── elo1800-gcp-v4/ckpt/
│       └── champion.pt        # Best trained model
│
├── pyproject.toml             # Python package config (needs `serve` extra)
├── requirements.txt           # OLD — to be deleted (pyproject.toml is source of truth)
├── requirements.313.txt       # OLD — to be deleted
├── Dockerfile                 # OLD — references deleted `adapters.api_fastapi`
└── pytest.toml                # Test config

### What still references deleted code

| File | Problem |
|------|---------|
| `server/websocket.py` | `from core.board import Board` / `from core.gomoku import Gomoku` — old `core/` was removed |
| `Dockerfile` | `CMD` runs `adapters.api_fastapi:create_app` — `adapters/` was removed |
| `requirements.txt` | Stale deps from the old Python server (numpy 2.2, anytree, pillow, etc.) |

---

## Verified API Signatures & Key Facts

These were verified by reading the actual gmk-refactor source code.

### Config loading

from gomoku.utils.config import load_and_parse_config

config: RootConfig = load_and_parse_config("configs/local_play.yaml")
# config.board  -> BoardConfig (num_lines=19, enable_doublethree=True, enable_capture=True, ...)
# config.model  -> ModelConfig (num_hidden=128, num_resblocks=12)
# config.mcts   -> MCTSConfig  (num_searches=2400, use_native=True, ...)

`local_play.yaml` is the serving config: 19x19 board, doublethree + capture enabled, 2400 searches, `use_native=true`.

### Model loading

from gomoku.model.policy_value_net import PolicyValueNet
from gomoku.utils.model import align_state_dict_to_model

game = Gomoku(board_config)
model = PolicyValueNet(game, model_config, device)
state_dict = torch.load("runs/elo1800-gcp-v4/ckpt/champion.pt", map_location=device)
aligned = align_state_dict_to_model(state_dict, model.state_dict())
model.load_state_dict(aligned, strict=False)
model.eval()

### MCTS inference

from gomoku.pvmcts import PVMCTS

pvmcts = PVMCTS(game, model, mcts_config, inference_client, device)
root = pvmcts.create_root(game_state)
results = pvmcts.run_search([root], add_noise=False)
# results: list[(policy_vector, stats_dict)]
# policy_vector: np.ndarray shape (361,) — probability for each board position
# Best move: action = int(np.argmax(policy_vector))
# Convert to (x, y): x = action % 19, y = action // 19

### Score semantics — NO conversion needed

Both the frontend and gmk-refactor use the same unit for capture scores:
- Frontend `scores[].score`: **pair count** (each capture event that removes 2 stones counts as 1)
- gmk-refactor `p1_pts` / `p2_pts`: also **pair count** (NOT individual stones)
- Win condition: 5 pairs = 10 stones captured

### Capture detection

After calling `game.get_next_state(state, action)`:
- `game.last_captures`: list of **flat indices** of captured stones from the *previous* board
- Flat index → coordinates: `x = idx % 19`, `y = idx // 19`
- Empty after a move with no captures

### GameState dataclass

@dataclass
class GameState:
    board: np.ndarray      # shape (19,19), dtype int8: 0=empty, 1=P1(X), 2=P2(O)
    p1_pts: np.int16       # P1 capture pair count
    p2_pts: np.int16       # P2 capture pair count
    next_player: np.int8   # 1 or 2
    last_move_idx: np.int16  # flat index (x + y*19), or -1 if no last move
    empty_count: np.int16
    history: tuple[int, ...]

---

## Remaining Phases

### Phase 2: Rewrite the Serving Layer (`server/`)

The existing `server/` files are broken (import deleted modules). All three files need a full rewrite, plus a new `__init__.py`.

#### 2.1 `server/__init__.py` — FastAPI App Factory

from fastapi import FastAPI

def create_app() -> FastAPI:
    app = FastAPI(title="Gomoku AlphaZero")

    from server.websocket import router
    app.include_router(router)

    from server.engine import AlphaZeroEngine
    import os

    @app.on_event("startup")
    async def load_model():
        app.state.engine = AlphaZeroEngine(
            config_path=os.getenv("ALPHAZERO_CONFIG", "configs/local_play.yaml"),
            checkpoint_path=os.getenv("ALPHAZERO_CHECKPOINT", "runs/elo1800-gcp-v4/ckpt/champion.pt"),
            device=os.getenv("ALPHAZERO_DEVICE", "cpu"),
        )

    @app.get("/")
    async def health():
        return {"status": "ok", "model": "alphazero"}

    return app

#### 2.2 `server/engine.py` — Model Loading & MCTS Wrapper

class AlphaZeroEngine:
    def __init__(self, config_path: str, checkpoint_path: str, device: str = "cpu"):
        config = load_and_parse_config(config_path)
        self.game = Gomoku(config.board)
        self.model = PolicyValueNet(self.game, config.model, device)
        # Load checkpoint with align_state_dict_to_model
        self.model.eval()
        self.mcts_config = config.mcts
        self.device = device

    def get_best_move(self, state: GameState, num_searches: int | None = None) -> int:
        """Run MCTS and return flat action index."""
        cfg = self.mcts_config
        if num_searches:
            cfg = cfg.model_copy(update={"num_searches": num_searches})
        pvmcts = PVMCTS(self.game, self.model, cfg, inference_client, self.device)
        root = pvmcts.create_root(state)
        [(policy, _)] = pvmcts.run_search([root], add_noise=False)
        return int(np.argmax(policy))

    def apply_move(self, state: GameState, action: int) -> tuple[GameState, list[int]]:
        """Apply action, return (new_state, captured_flat_indices)."""
        new_state = self.game.get_next_state(state, action)
        return new_state, list(self.game.last_captures)

Configuration via environment variables:
- `ALPHAZERO_CONFIG`: path to YAML config (default: `configs/local_play.yaml`)
- `ALPHAZERO_CHECKPOINT`: path to `.pt` checkpoint file (default: `runs/elo1800-gcp-v4/ckpt/champion.pt`)
- `ALPHAZERO_DEVICE`: `cpu` or `cuda` (default: `cpu`)
- `ALPHAZERO_MCTS_NUM_SEARCHS`: fixed MCTS search count for AlphaZero serving (`none` means use `config.mcts.num_searches`)

AlphaZero serving policy:
- Frontend difficulty mapping is removed.
- Search count is controlled only by `ALPHAZERO_MCTS_NUM_SEARCHS`.

#### 2.3 `server/protocol.py` — Format Conversion

Converts between frontend WebSocket JSON and gmk-refactor's internal types.

**Frontend → GameState:**

def frontend_to_gamestate(data: dict) -> GameState:
    """
    Frontend JSON → GameState.

    Mappings:
      board[y][x] "."/"X"/"O"  →  np.int8 0/1/2
      scores[player="X"].score  →  p1_pts (same units, no conversion)
      scores[player="O"].score  →  p2_pts
      nextPlayer "X"/"O"       →  next_player 1/2
      lastPlay.coordinate       →  last_move_idx = x + y * 19
    """

**GameState → Frontend response:**

def build_move_response(
    action: int,
    stone: str,
    new_state: GameState,
    captured_indices: list[int],
    execution_time_ns: int,
) -> dict:
    """
    Build SocketMoveResponse matching minimax format:
    {
        type: "move",
        status: "success",
        lastPlay: { coordinate: {x, y}, stone },
        board: Stone[][],
        capturedStones: [{x, y, stone}, ...],
        scores: [{player, score}, ...],
        evalScores: [{player, evalScores, percentage}, ...],
        executionTime: {s, ms, ns}
    }
    """

Key type mappings:

| Frontend | gmk-refactor | Notes |
|----------|-------------|-------|
| `"X"` (Black) | `PLAYER_1 = 1` | Goes first |
| `"O"` (White) | `PLAYER_2 = 2` | Goes second |
| `"."` (empty) | `EMPTY_SPACE = 0` | |
| `board[y][x]` | `board[y, x]` | Both row-major |
| `lastPlay.coordinate.{x,y}` | `last_move_idx = x + y * 19` | Flat index encoding |
| `scores[].score` | `p1_pts` / `p2_pts` | Same pair-count unit |

#### 2.4 `server/websocket.py` — WebSocket Endpoints

Full rewrite. The current file imports from deleted `core.board` / `core.gomoku`.

@router.websocket("/ws")
async def game_endpoint(websocket: WebSocket):
    """Main game endpoint — handles AI moves."""
    await websocket.accept()
    engine: AlphaZeroEngine = websocket.app.state.engine
    try:
        while True:
            data = await websocket.receive_json()
            if data["type"] == "move":
                # 1. Convert frontend board → GameState
                state = frontend_to_gamestate(data)
                # 2. Use fixed search budget from ALPHAZERO_MCTS_NUM_SEARCHS
                num_searches = NUM_SEARCHES
                # 3. Run MCTS to find AI's best move
                action = engine.get_best_move(state, num_searches)
                # 4. Apply AI's move, get captures
                new_state, captures = engine.apply_move(state, action)
                # 5. Build response in minimax-compatible format
                response = build_move_response(action, ...)
                await websocket.send_json(response)
    except WebSocketDisconnect:
        pass

@router.websocket("/ws/debug")
async def debug_endpoint(websocket: WebSocket):
    """Debug endpoint — rule validation without AI."""
    # Minimal: validate moves using Gomoku game engine directly

---

### Phase 3: Dependencies & Build Configuration

#### 3.1 Delete `requirements.txt` and `requirements.313.txt`

`pyproject.toml` is the single source of truth. Both requirements files are stale and redundant.

rm requirements.txt requirements.313.txt

#### 3.2 Update `pyproject.toml`

Add a `serve` extra for FastAPI/uvicorn/websockets. Separate heavy training deps (`ray`, `pyarrow`, `google-api-python-client`) from core deps. The existing Dockerfile pattern `pip install ".[ray]"` already proves extra-based installs work.

Changes needed:

[project]
dependencies = [
  "numpy==2.3.3",
  "pyyaml==6.0.2",
  "tqdm==4.67.1",
  "fsspec==2025.9.0",
  "pydantic==2.12.5",
]

[project.optional-dependencies]
serve = [
  "fastapi==0.115.6",
  "uvicorn==0.34.0",
  "websockets==14.1",
]
ray = [
  "ray==2.49.2",
  "pyarrow==21.0.0",
  "google-api-python-client",
]
# keep existing: dev, torch-cpu, torch-cu121

This means:
- `pip install ".[serve,torch-cpu]"` — serving with CPU inference
- `pip install ".[ray,torch-cpu]"` — training
- `pip install ".[serve,torch-cu121]"` — GPU serving

#### 3.3 Update `Dockerfile`

Replace the current Dockerfile which references deleted `adapters.api_fastapi`.

FROM python:3.13-slim

# Build deps for C++ extensions (pybind11)
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake g++ ninja-build && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install Python deps first (cache layer)
COPY pyproject.toml .
RUN pip install --no-cache-dir ".[serve,torch-cpu]"

# Copy source and build C++ extensions
COPY . .
RUN pip install -e . --no-deps

ENV PYTHONPATH=/app
ENV ALPHAZERO_CONFIG=configs/local_play.yaml
ENV ALPHAZERO_CHECKPOINT=runs/elo1800-gcp-v4/ckpt/champion.pt
ENV ALPHAZERO_DEVICE=cpu

EXPOSE 8080
CMD ["uvicorn", "server:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]

---

### Phase 4: Docker Compose & Environment Updates

#### 4.1 Update `docker-compose.yml`

The alphazero service needs updated build and volume config:

alphazero:
  build:
    context: ./alphazero
    dockerfile: Dockerfile
  volumes:
    - ./alphazero:/app
    - ./alphazero/runs:/app/runs  # model checkpoints
  ports:
    - "${LOCAL_ALPHAZERO}:8080"
  env_file:
    - .env
  environment:
    - ALPHAZERO_CONFIG=configs/local_play.yaml
    - ALPHAZERO_CHECKPOINT=runs/elo1800-gcp-v4/ckpt/champion.pt
    - ALPHAZERO_DEVICE=cpu

#### 4.2 Update `.env`

ALPHAZERO_CONFIG=configs/local_play.yaml
ALPHAZERO_CHECKPOINT=runs/elo1800-gcp-v4/ckpt/champion.pt
ALPHAZERO_DEVICE=cpu

---

### Phase 5: Model Checkpoint

The trained champion model already exists at `runs/elo1800-gcp-v4/ckpt/champion.pt`.

**Strategy**: Keep `runs/` gitignored. For deployment:
- **Development**: checkpoint is already present from the gmk-refactor copy
- **Docker**: volume-mount `./alphazero/runs:/app/runs`
- **Production**: download from GCS or bundle in image at build time

No additional provisioning work needed for local dev since the checkpoint was copied with the repo.

---

### Phase 6: Cloud Infrastructure (Optional)

Cloud/training infra files are now copied into `alphazero/infra`:

| File | Purpose |
|------|---------|
| `alphazero/infra/cluster_elo1800.yaml` | Ray cluster config for GCP training |
| `alphazero/infra/restart_cluster.sh` | Script to rebuild/push images and restart Ray cluster |
| `alphazero/infra/reserver_vm.sh` | GCP VM reservation script |
| `alphazero/infra/buildkitd.toml` | BuildKit config for Docker builds |
| `alphazero/infra/Dockerfile.py313-cpu` | Training container (CPU) |
| `alphazero/infra/Dockerfile.py313-cu124` | Training container (CUDA 12.4) |

These are only relevant for running distributed training on GCP, not for serving.

Operational notes:
- `restart_cluster.sh` now resolves paths relative to its own location and works from any current directory.
- `restart_cluster.sh` renders a runtime cluster YAML (`.cluster_elo1800.resolved.yaml`) using env overrides for project/region/zone/user/image tags.
- `reserver_vm.sh` supports optional `PROJECT` override and applies it to gcloud commands.

---

### Phase 7: Frontend Compatibility Verification

#### 7.1 WebSocket URL routing

`front/composables/useEnv.ts` routes to `ws://localhost:8080/ws` when `ai === "alphazero"` — already correct. The `/ws` endpoint is what Phase 2 creates.

#### 7.2 Response format parity

The alphazero `/ws` endpoint MUST return the exact same shape as minimax. Key fields the frontend reads (`front/pages/game.vue` + `front/stores/game.store.ts`):

| Field | Used by | Required |
|-------|---------|----------|
| `type` | Message routing | Yes — `"move"`, `"evaluate"`, or `"error"` |
| `status` | Error handling | Yes — `"success"` or `"doublethree"` |
| `lastPlay.coordinate.{x,y}` | `addStoneToBoardData()` | Yes |
| `lastPlay.stone` | Board rendering | Yes |
| `board` | Full board sync | Yes |
| `capturedStones` | Capture animation | Yes (can be `[]`) |
| `scores` | Score display | Yes |
| `evalScores` | Eval bar display | Optional (placeholder OK) |
| `executionTime` | Timing display | Optional |

**Note**: The frontend also runs its own capture/doublethree detection client-side via `useGameLogic.ts`. The `board` in the response should reflect the state *after* the AI's move with captures already applied.

#### 7.3 Difficulty settings

AlphaZero backend does not use frontend difficulty settings.
MCTS search count is controlled by one variable only:
- `ALPHAZERO_MCTS_NUM_SEARCHS`
- `none` means fallback to `configs/local_play.yaml` (`config.mcts.num_searches`)

---

### Phase 8: Testing & Validation

#### 8.1 Existing tests

Ensure gmk-refactor's test suite passes in the new location:

cd alphazero && pytest

#### 8.2 New tests for server/

- `server/protocol.py`: round-trip conversion (frontend → GameState → frontend)
- `server/engine.py`: model loads without error, produces valid action indices
- `server/websocket.py`: WebSocket round-trip with test client

#### 8.3 End-to-end smoke test

docker compose up front minimax alphazero
# Open http://localhost:3000
# Select AlphaZero in settings
# Play a game — verify: stones placed, captures work, game ends correctly

---

## Checklist

- [x] **Phase 1**: File migration — gmk-refactor copied into `gomoku/alphazero/`
- [x] **Phase 2**: Rewrite `server/` — `__init__.py`, `engine.py`, `protocol.py`, `websocket.py`
- [x] **Phase 3**: Delete requirements*.txt, add `serve` extra to pyproject.toml, update Dockerfile
- [x] **Phase 4**: Update docker-compose.yml and .env
- [x] **Phase 5**: Verify checkpoint is accessible (already present)
- [x] **Phase 6**: Copy cloud infra files (optional, only if training on GCP)
- [x] **Phase 7**: Verify frontend compatibility
- [ ] **Phase 8**: Run tests, end-to-end smoke test

---

## Risks & Notes

1. **C++ extension build**: pybind11 extensions need cmake + g++ in Docker. Build step is cached by Docker layers but adds ~2 min on first build.

2. **Inference latency**: MCTS search count directly affects response time on CPU. Mitigations:
   - Tune `ALPHAZERO_MCTS_NUM_SEARCHS` down for faster responses
   - Use `ALPHAZERO_MCTS_NUM_SEARCHS=none` only when you intentionally want config-level search count
   - `use_native: true` in config enables the C++ MCTS engine (much faster than Python)
   - GPU inference via `ALPHAZERO_DEVICE=cuda` if available

3. **PyTorch image size**: PyTorch CPU adds ~800 MB to the Docker image. Consider multi-stage build to keep the final image lean.

4. **Package namespace**: `gomoku/alphazero/gomoku/` nesting is intentional (same pattern as `django/django/`). Inside Docker at `/app`, imports resolve as `from gomoku.core...` via `PYTHONPATH=/app`. No namespace conflict with the outer `gomoku/` project directory.

5. **Training still works**: After integration, training runs unchanged:
   cd alphazero
   pip install -e ".[ray,torch-cpu]"
   python -m gomoku.scripts.train --config configs/local_play.yaml --mode sequential
