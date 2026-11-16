# Gomoku

A full-stack Gomoku (five-in-a-row) game with two AI engines, built by two people: a **C++ Minimax** engine (alpha-beta with iterative deepening, a transposition table and a PVS hard mode) and a **Python AlphaZero** engine (self-play reinforcement learning with MCTS + ResNet). The École 42 project covers the game and the Minimax engine; the Python AlphaZero engine goes beyond the school project.

**[Play live](https://sungyongcho.com/gomoku)** | **[Documentation](https://sungyongcho.com/gomoku/docs/about-gomoku/intro)**

<img src="front/public/images/screenshot-game.gif" alt="Gomoku gameplay" width="720" />

## Stack

| Layer | Tech |
|---|---|
| Frontend | Nuxt 4, Vue 3, TypeScript, PrimeVue, Tailwind CSS, Nuxt Content (docs site) |
| Minimax engine | C++, libwebsockets, Zobrist hashing, PVS with transposition table |
| AlphaZero engine | Python, PyTorch, MCTS (C++ native extension), static int8 ONNX Runtime inference, FastAPI WebSocket server |
| Training infra | Ray cluster on Google Cloud (L4 GPU head + CPU workers), self-play + arena evaluation pipeline |
| Deployment | Frontend on Netlify; Minimax (since Sep 2026) and AlphaZero (since Nov 2026) as Docker containers on Oracle Cloud Always Free Arm VMs; Cloudflare Worker routing |

## Team

By [Sungyong Cho](https://sungyongcho.com) and [Woolim Park](https://woolimi.github.io).

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
- **`front/`** — Nuxt 4 frontend with interactive documentation and diagrams
- **`minimax/`** — C++ minimax engine (WebSocket server)
- **`alphazero/`** — AlphaZero training pipeline, inference server, and C++ MCTS extension

## Third-Party Code

- Double-three detection uses logic derived from the [Renju open source reference implementation](https://www.renju.se/renlib/opensrc/) (`CForbiddenPointFinder`). It was brought in on 2026-02-03 (`f9ddbc6`), after the École 42 project and during the AlphaZero work: it replaced the Minimax engine's hand-written double-three check in `minimax/src/gomoku/Rules.cpp`, which is what the 42 submission used, and the AlphaZero engine's `renju_cpp` module uses the same logic. Refer to the upstream page for license details and attribution.

## References

### Papers

- D. Silver et al., "Mastering the game of Go with deep neural networks and tree search," *Nature* 529, 484–489 (2016). [Nature](https://www.nature.com/articles/nature16961) · [PDF (DeepMind)](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf) — **AlphaGo**
- D. Silver et al., "Mastering the game of Go without human knowledge," *Nature* 550, 354–359 (2017). [Nature](https://www.nature.com/articles/nature24270) — **AlphaGo Zero**
- D. Silver et al., "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play," *Science* 362, 1140–1144 (2018). [Science](https://www.science.org/doi/10.1126/science.aar6404) · [arXiv preprint](https://arxiv.org/abs/1712.01815) — **AlphaZero**

### Tutorials and lectures

- R. Förster, [AlphaZeroFromScratch](https://www.youtube.com/watch?v=wuSQpLinRB4) (video tutorial, [repository](https://github.com/foersterrobert/AlphaZeroFromScratch)). The AlphaZero engine in this repository grew out of this tutorial, starting in May 2025.
- Pang-Yo Lab, [쉽게 읽는 강화학습 논문 (Reinforcement Learning Papers Made Easy)](https://www.youtube.com/watch?v=SRVx2DFu_tY&list=PLpRS2w0xWHTfnWmr95LtIu4v4HbVxqTlM). Korean-language video series reviewing reinforcement learning papers, starting with AlphaGo.
