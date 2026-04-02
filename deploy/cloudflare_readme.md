# Cloudflare Deployment Notes

This document summarizes Cloudflare setup for the current deployment flow in
`deploy/03_deploy_cloudflare.sh`, focused on console actions and operational checks.

No real domain/account values are included here. Use placeholders like
`<YOUR_DOMAIN>`, `<YOUR_ACCOUNT_ID>`, and `<YOUR_TOKEN>`.

## What The Script Does

`deploy/03_deploy_cloudflare.sh` does three things:

1. Loads deployment values from `.env` via `deploy/deploy_env_config.sh`
2. Upserts DNS A records for backend origins:
   - `minimax-api.<YOUR_DOMAIN>` (DNS only / grey cloud)
   - `alphazero-api.<YOUR_DOMAIN>` (DNS only / grey cloud)
3. Deploys the Worker with:
   - `ALPHAZERO_ORIGIN=http://alphazero-api.<YOUR_DOMAIN>:8080`
   - `MINIMAX_ORIGIN=http://minimax-api.<YOUR_DOMAIN>:8080`

## Worker Behavior (Current Code)

`deploy/cloudflare-worker.js` routes:

- `/alphazero/*` -> `ALPHAZERO_ORIGIN`
- `/minimax/*` -> `MINIMAX_ORIGIN`
- `/gomoku` and `/gomoku/*` -> proxied to `https://omoku.netlify.app/*`

For `/gomoku`, it proxies (not browser redirect), so URL stays on your domain.

## Cloudflare Console Setup

## 1) API Token

Create a token with at least these permissions:

- `Zone:Zone:Read`
- `Zone:DNS:Edit`
- `Zone:Workers Routes:Edit`
- `Account:Workers Scripts:Edit`

Scope it only to the required account and zone.

## 2) Worker Routes

In Worker -> Domains & Routes, ensure these routes exist:

- `<YOUR_DOMAIN>/alphazero/*`
- `<YOUR_DOMAIN>/minimax/*`
- `<YOUR_DOMAIN>/gomoku`
- `<YOUR_DOMAIN>/gomoku/*`

If route failure mode is available:

- Recommended: `Fail open (proceed)` for public-facing availability
- Use `Fail closed (block)` only if strict fail-safe behavior is required

## 3) Worker Variables

In Worker -> Variables and Secrets (plaintext vars):

- `ALPHAZERO_ORIGIN` = `http://alphazero-api.<YOUR_DOMAIN>:8080`
- `MINIMAX_ORIGIN` = `http://minimax-api.<YOUR_DOMAIN>:8080`

No variable is needed for the gomoku target in current code
(`https://omoku.netlify.app` is hardcoded in Worker code).

## 4) DNS Records

In DNS (or via script), ensure:

- `A minimax-api.<YOUR_DOMAIN> -> <MINIMAX_VM_PUBLIC_IP>`
- `A alphazero-api.<YOUR_DOMAIN> -> <ALPHAZERO_VM_PUBLIC_IP>`
- Proxy status: `DNS only` (grey cloud)

## Deploy Command

From repo root:

```bash
bash deploy/03_deploy_cloudflare.sh
```

## Verification

`deploy/verify_connection.sh` is called automatically by the deploy script.
It checks:

- `https://<YOUR_DOMAIN>/alphazero/` health response
- WebSocket connectivity:
  - `wss://<YOUR_DOMAIN>/minimax/ws`
  - `wss://<YOUR_DOMAIN>/alphazero/ws`
- Gomoku proxy behavior:
  - `https://<YOUR_DOMAIN>/gomoku` returns 2xx/3xx
  - effective URL remains on `https://<YOUR_DOMAIN>/gomoku...`

Manual quick checks:

```bash
curl -I https://<YOUR_DOMAIN>/gomoku
curl -sSL -o /dev/null -w '%{url_effective}\n' https://<YOUR_DOMAIN>/gomoku
```

## Troubleshooting

- `Could not resolve host: api.cloudflare.com`
  - Local DNS/egress issue on the machine running deploy
- Worker route not applied
  - Confirm all 4 route patterns exist in Cloudflare console
- 403/404 on `/gomoku`
  - Check Worker routes and that Worker deploy succeeded
- WebSocket failure
  - Re-check backend DNS A records, VM service health, and firewall/open ports
