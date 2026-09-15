# Oracle Cloud deployment (minimax + docreview-rag)

Runs the minimax engine and [docreview-rag](https://github.com/sungyongcho/docreview-rag)
on one Oracle Cloud Always Free instance (`VM.Standard.A1.Flex`, 2 OCPU / 12 GB,
aarch64). AlphaZero stays on GCP (`deploy/02_deploy.sh`); the existing GCP scripts
in `deploy/` are untouched. Public routing keeps working through the same Cloudflare
Worker: only origin IPs change.

| Service | Port on the instance | Public path | Deployed from |
| --- | --- | --- | --- |
| minimax | 8080 (WebSocket) | `/minimax/*` via `minimax-api.<domain>` | this directory |
| docreview-rag | 8880 (Caddy → FastAPI) | `/docreview-rag/api/*` | `docreview-rag/deploy/oracle` |
| alphazero | GCP VM, unchanged | `/alphazero/*` | `deploy/02_deploy.sh` |

## 0. Instance (Oracle console, once)

1. Compute → Create instance: shape `VM.Standard.A1.Flex`, 2 OCPU, 12 GB, image
   **Canonical Ubuntu 24.04 (aarch64)**, boot volume 100 GB, upload your SSH public key.
   Retry later if the region reports "Out of host capacity".
2. Networking → the instance's VCN → default security list → ingress rules:
   - `tcp/22` from your own IP.
   - `tcp/8080` and `tcp/8880` from each Cloudflare IPv4 range only
     (`curl -fsSL https://www.cloudflare.com/ips-v4`, 15 CIDRs).
   Nothing else. The Worker is the only client that should reach the origin.
   With the OCI CLI configured (`oci setup config`) and
   `DEPLOY_ORACLE_SECURITY_LIST_OCID` in `.env`, `bash deploy/oracle/00_security_list.sh`
   appends exactly these rules and leaves existing ones alone.
3. Note the public IPv4; it is `DEPLOY_ORACLE_HOST` below. Do not stop/start the
   instance casually: the ephemeral IP changes and step 4 must be repeated.

## 1. Configure

Add to the repo `.env` (see `.env.example`):

```sh
DEPLOY_ORACLE_HOST=<public IPv4>
DEPLOY_ORACLE_SSH_USER=ubuntu
DEPLOY_ORACLE_SSH_KEY=~/.ssh/<key>   # optional if the key is in your agent
```

## 2. Prepare the host

```sh
bash deploy/oracle/01_host_setup.sh
```

Installs Docker, compose, rsync, a 2 GB swap file, opens `8080`/`8880` in the
host iptables (Oracle's Ubuntu image rejects everything but SSH by default) and
creates `/opt/docreview`, `/var/lib/docreview`, `~/build`. Idempotent.

## 3. Deploy minimax

```sh
bash deploy/oracle/02_deploy_minimax.sh
```

Syncs `minimax/` sources (no local build outputs), builds `Dockerfile.prod` on the
instance for aarch64 and starts the container with `restart: unless-stopped`.
Re-run for updates; `NO_CACHE=true` forces a clean build, `DO_BUILD=false` only restarts.

## 4. Deploy docreview-rag

From the docreview-rag checkout, follow `deploy/oracle/README.md` there
(`deploy_backend.sh first-install`). It publishes Caddy on `8880`.

## 5. Cut over the Cloudflare routing

```sh
bash deploy/oracle/03_print_origin.sh
```

Copy the printed `DEPLOY_MINIMAX_IP`, `DEPLOY_DOCREVIEW_IP` and
`DEPLOY_DOCREVIEW_ORIGIN` into `.env` (leave `DEPLOY_ALPHAZERO_IP` on GCP), then:

```sh
bash deploy/03_deploy_cloudflare.sh
```

This updates the `minimax-api` and `docreview-api` DNS records and redeploys the
Worker with the new DocReview origin; `verify_connection.sh` runs at the end.

## 6. Retire the GCP VMs

After a few days of clean logs, delete `gomoku-minimax` and the docreview VM plus
their static/ephemeral IPs in the GCP console. Keep `gomoku-alphazero`.

## Operations

```sh
# status / logs
ssh ubuntu@$DEPLOY_ORACLE_HOST 'sudo docker ps; sudo docker logs --tail 50 gomoku-minimax'
# restart without rebuilding
DO_BUILD=false bash deploy/oracle/02_deploy_minimax.sh
```

Always Free reclamation looks for seven idle days on CPU, network **and** memory;
the docreview Postgres and reranker keep memory well above the 20 % threshold.
