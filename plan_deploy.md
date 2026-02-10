# Deployment Plan: Gomoku WebSocket Backends to GCP

## Context

Deploy the minimax (C++ WebSocket) and alphazero (Python/PyTorch WebSocket) backends to Google Cloud as two separate VMs, accessible via path-based routing through Cloudflare (`sungyongcho.com/minimax/ws`, `sungyongcho.com/alphazero/ws`). The deployment must be fully scriptable (setup + teardown) following the same patterns as the existing training infra scripts in `alphazero/infra/`.

**Budget**: ~0.59 EUR/day (e2-micro + e2-small), well under 5 EUR/day limit.

---

## Architecture

```
Client (browser)
  │ wss://sungyongcho.com/alphazero/ws
  │ wss://sungyongcho.com/minimax/ws
  ▼
Cloudflare (TLS termination + path routing via Worker)
  │  /alphazero/* ──► GCP e2-small VM (alphazero, port 8080)
  │  /minimax/*  ──► GCP e2-micro VM (minimax, port 8005)
  │  /gomoku/*   ──► TBD (frontend, handled separately)
  │  /*          ──► GitHub Pages (existing personal site)
  ▼
GCP Compute Engine (Container-Optimized OS)
  ├── gomoku-minimax   (e2-micro, 1GB)  → minimax Docker container
  └── gomoku-alphazero (e2-small, 2GB)  → alphazero Docker container
```

---

## Files to Create

### 1. `minimax/Dockerfile.prod` — Production minimax image
Multi-stage build (no entr, no gdb, no git). Build stage compiles binary, runtime stage only has libwebsockets runtime lib + the binary.
- **Base reference**: `minimax/Dockerfile` (existing dev Dockerfile)
- Stage 1 (builder): debian:bookworm-slim + build deps → `make re`
- Stage 2 (runtime): debian:bookworm-slim + `libwebsockets19` only → `CMD ["./minimax"]`
- `ENV MINIMAX_PORT=8005`, `EXPOSE 8005`

### 2. `alphazero/infra/deploy/Dockerfile` — Production alphazero serving image
Based on existing `alphazero/Dockerfile` with production optimizations:
- Bake model checkpoint into image (deploy script copies `champion.pt` before build)
- Thread count optimization: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `TORCH_NUM_THREADS=1`
- `HEALTHCHECK` via `curl http://localhost:8080/`
- Uses existing `ALPHAZERO_DEVICE` and `ALPHAZERO_MCTS_NUM_SEARCHS` env vars (no separate deploy vars)
- No dev/ray/training deps — only `.[serve,torch-cpu]`

### 3. Unified `.env.example` — Single env file (no separate `.env.deploy` or `.env.train`)
All env vars consolidated into one `.env` / `.env.example` with sections:
```
# Local Development
LOCAL_FRONT, LOCAL_MINIMAX, LOCAL_ALPHAZERO, ...

# Frontend
FRONT_WHERE

# AlphaZero
ALPHAZERO_CONFIG, ALPHAZERO_CHECKPOINT, ALPHAZERO_DEVICE, ALPHAZERO_MCTS_NUM_SEARCHS

# AlphaZero Training (Ray on GCP)
GCP_PROJECT, GCP_REGION, GCP_ZONE, GCP_REPO, GCP_CLUSTER_NAME, ...

# Deployment (GCP VMs + Cloudflare)
DEPLOY_GCP_PROJECT, DEPLOY_GCP_REGION, DEPLOY_GCP_ZONE, DEPLOY_GCP_REPO,
DEPLOY_MINIMAX_VM, DEPLOY_ALPHAZERO_VM, DEPLOY_MINIMAX_MACHINE, DEPLOY_ALPHAZERO_MACHINE,
DEPLOY_DOMAIN, DEPLOY_SA_NAME, DEPLOY_USER_EMAIL,
CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN
```

AlphaZero serving vars (`ALPHAZERO_*`) are shared between local docker-compose and deployed containers — no duplication. Deploy section only has infra-specific vars (`DEPLOY_*`, `CLOUDFLARE_*`).

### 4. `infra/deploy/env_config.sh` — Load and validate `.env`
Follow pattern from `alphazero/infra/bootstrap/env_config.sh`:
- Resolve `DOTENV_PATH` → `.env` (repo root)
- `set -a; source; set +a`
- Validate required `DEPLOY_*` vars
- Export derived vars: `ARTIFACT_REGISTRY`, `SA_EMAIL`
- Print loaded config summary

### 5. `infra/deploy/setup.sh` — One-time GCP provisioning
Follow pattern from `alphazero/infra/cluster/restart_cluster.sh` (source env, log, gcloud commands):
1. Enable APIs (`compute`, `artifactregistry`)
2. Create Artifact Registry repo
3. Create service account with minimal roles
4. Create firewall rule (`gomoku-deploy` tag) allowing TCP 8005, 8080 from Cloudflare IP ranges + SSH from IAP
5. Reserve two static external IPs
6. Create minimax VM (`create-with-container`, Container-Optimized OS, e2-micro)
7. Create alphazero VM (`create-with-container`, Container-Optimized OS, e2-small)
8. Print external IPs for Cloudflare DNS setup

### 6. `infra/deploy/deploy.sh` — Build, push, and update containers
Follow pattern from `restart_cluster.sh` (DO_BUILD, DO_DELETE flags):
1. Source `env_config.sh`
2. Authenticate Docker with Artifact Registry
3. Copy model checkpoint: `cp -L alphazero/runs/elo1800-gcp-v4/ckpt/champion.pt alphazero/models/champion.pt`
4. Build minimax image: `docker buildx build -f minimax/Dockerfile.prod minimax/`
5. Build alphazero image: `docker buildx build -f alphazero/infra/deploy/Dockerfile alphazero/`
6. Push both images to Artifact Registry
7. Update containers on VMs via `gcloud compute instances update-container`
8. Verify container status via SSH

### 7. `infra/deploy/teardown.sh` — Remove all deploy resources
Follow pattern from `alphazero/infra/cluster/purge_gcp.sh` (confirm prompts, `--yes` flag):
1. Delete VMs
2. Release static IPs
3. Delete firewall rules
4. Delete Artifact Registry images
5. Optionally delete service account

### 8. `infra/deploy/cloudflare-worker.js` — Path-based routing Worker
Routes requests based on path prefix to the appropriate GCP VM origin:
- `/alphazero/*` → strip prefix, forward to `http://ALPHAZERO_IP:8080/*` (WebSocket-aware via `fetch()`)
- `/minimax/*` → strip prefix, forward to `http://MINIMAX_IP:8005/*`
- `/gomoku/*` → pass through (reserved for frontend)
- `/*` → pass through to GitHub Pages origin

**No changes to `sungyongcho.github.io` repo needed.** The Worker intercepts matching paths at the Cloudflare edge *before* requests reach any origin. GitHub Pages continues serving `sungyongcho.com/` as usual — it never sees `/minimax` or `/alphazero` requests. The Worker + `wrangler.toml` live in `gomoku/infra/deploy/` and are deployed via `npx wrangler deploy`.

### 9. `infra/deploy/wrangler.toml` — Cloudflare Worker config
Defines Worker name, routes (`sungyongcho.com/alphazero/*`, `sungyongcho.com/minimax/*`), and env vars (`ALPHAZERO_ORIGIN`, `MINIMAX_ORIGIN` with VM IPs).

**Prerequisites** (Cloudflare dashboard, one-time manual):
- `sungyongcho.com` must be in Proxied mode (orange cloud) — already set up per user
- No DNS record changes needed — existing A records to GitHub Pages stay as-is
- Worker routes are created automatically by `wrangler deploy` using the `wrangler.toml` config

---

## Files to Modify

### 1. `front/composables/useEnv.ts` — Switch to path-based URLs
Change production WebSocket URLs:
- `wss://minimax.sungyongcho.com/ws` → `wss://sungyongcho.com/minimax/ws`
- `wss://alphazero.sungyongcho.com/ws` → `wss://sungyongcho.com/alphazero/ws`
- Same for debug endpoints (`/ws/debug`)

### 2. `alphazero/server/websocket.py:28` — Make NUM_SEARCHES configurable
Change `NUM_SEARCHES: int | None = 200` to read from env:
```python
NUM_SEARCHES: int | None = int(os.getenv("ALPHAZERO_NUM_SEARCHES", "200"))
```

### 3. `alias.sh` — Add deployment aliases
```bash
## deployment
alias deploy-setup="bash infra/deploy/setup.sh"
alias deploy="bash infra/deploy/deploy.sh"
alias deploy-teardown="bash infra/deploy/teardown.sh"
alias deploy-ssh-minimax="..."
alias deploy-ssh-alphazero="..."
alias deploy-logs-minimax="..."
alias deploy-logs-alphazero="..."
```

### 4. `CLAUDE.md` — Add deployment section
Document the new infra/deploy structure, env vars, and workflow.

---

## Deployment Workflow (after implementation)

```bash
# One-time setup
cp .env.example .env                  # Fill in DEPLOY_* and CLOUDFLARE_* values
source alias.sh
deploy-setup                          # Provisions VMs, IPs, firewall

# Deploy (repeatable)
deploy                                # Build images, push, update containers

# Configure Cloudflare (manual or via wrangler)
cd infra/deploy && npx wrangler deploy  # Deploy the Worker

# Add Cloudflare DNS (manual, one-time):
#   A record: sungyongcho.com → GitHub Pages IPs (already exists)
#   Worker route: sungyongcho.com/alphazero/* and sungyongcho.com/minimax/*

# Teardown
deploy-teardown                       # Removes all GCP resources
```

---

## Performance Notes

- **AlphaZero on e2-small** (0.5 shared vCPU, 2GB): With 100 MCTS searches, expect ~2-5s per move. If too slow, upgrade to e2-medium (1 vCPU, 4GB, ~0.74 EUR/day) — still well within budget.
- **Minimax on e2-micro** (0.25 shared vCPU, 1GB): C++ is efficient; all difficulty levels should respond in <1s.
- Thread pinning (`OMP_NUM_THREADS=1`) prevents PyTorch from over-subscribing the shared vCPU.

---

## Verification

1. **Local Docker test**: Build both prod images, run locally, test with `wscat -c ws://localhost:8005/ws` and `ws://localhost:8080/ws`
2. **Direct VM test**: After `deploy`, test `wscat -c ws://<VM_IP>:8005/ws` and `ws://<VM_IP>:8080/ws`
3. **Cloudflare test**: After Worker deployment, test `wscat -c wss://sungyongcho.com/minimax/ws` and `wss://sungyongcho.com/alphazero/ws`
4. **Health check**: `curl https://sungyongcho.com/alphazero/` should return `{"status":"ok","model":"alphazero"}`
5. **End-to-end**: Run frontend with `FRONT_WHERE=prod`, play a game against each AI
