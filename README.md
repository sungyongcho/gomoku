# Gomoku

A full-stack Gomoku (five-in-a-row) game with two AI engines, built as a two-person École 42 project: a **C++ Minimax** engine (alpha-beta with iterative deepening, a transposition table and a PVS hard mode) and a **Python AlphaZero** engine (self-play reinforcement learning with MCTS + ResNet).

**[Play live](https://sungyongcho.com/gomoku)** | **[Documentation](https://sungyongcho.com/gomoku/docs/about-gomoku/intro)**

<img src="front/public/images/screenshot-game.gif" alt="Gomoku gameplay" width="720" />

## Stack

| Layer | Tech |
|---|---|
| Frontend | Nuxt 3, Vue 3, TypeScript |
| Minimax engine | C++, libwebsockets, Zobrist hashing, PVS with transposition table |
| AlphaZero engine | Python, PyTorch, MCTS (C++ native extension), ONNX inference |
| Training infra | Ray cluster on GCP, self-play + arena evaluation pipeline |
| Deployment | Docker containers: AlphaZero on a Google Cloud VM, Minimax on an Oracle Cloud Arm VM (since Sep 2026); Cloudflare Workers routing |

## Team

A two-person 42 project by [Sungyong Cho](https://sungyongcho.com) and [Woolim Park](https://woolimi.github.io).

- **Sungyong Cho**: the AlphaZero training pipeline, the C++ Minimax search, the C++ (libwebsockets) and FastAPI WebSocket servers, the deployment, the documentation pages and four interactive diagram components. Designed the hard-mode evaluation (pattern set, precomputed pattern-score lookup tables, incremental re-evaluation and move ordering), most of which Woolim implemented.
- **Woolim Park**: the game UI and most of the hard-mode evaluation implementation.
- Together: the WebSocket connection handling, since client and server had to change together.

## Local Development

Requires Docker and Docker Compose v2+.

```bash
git clone git@github.com:sungyongcho/gomoku.git
cd gomoku
cp .env.example .env
source alias.sh

# full stack (frontend + minimax + alphazero)
dev-up

# frontend + minimax only (no AlphaZero)
docker compose -f docker-compose.yml up front minimax
```

Open `http://localhost:${LOCAL_FRONT}/gomoku` (default port: `3000`).

Local documentation: `http://localhost:${LOCAL_FRONT}/gomoku/docs/about-gomoku/intro`

See `.env.example` for all configurable ports and variables.

## Project Structure

- **`alias.sh`** — Docker Compose shortcuts (`dev-up`, `dev-down`, `dev-up-debug`, etc.)
- **`Makefile`** — Original École 42 project submission entry point (front + minimax only)
- **`front/`** — Nuxt 3 frontend with interactive documentation and diagrams
- **`minimax/`** — C++ minimax engine (WebSocket server)
- **`alphazero/`** — AlphaZero training pipeline, inference server, and C++ MCTS extension

## Third-Party Code

- Double-three detection uses logic derived from the [Renju open source reference implementation](https://www.renju.se/renlib/opensrc/). Refer to the upstream page for license details and attribution.
- The AlphaZero engine started in May 2025 from the AlphaZeroFromScratch tutorial and continued in private repositories (Aug 2025 - Feb 2026) before the rewritten version was imported here.
