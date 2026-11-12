# AlphaZero deployment (Oracle Cloud)

Runs the AlphaZero engine on an Oracle Cloud Always Free instance (`VM.Standard.A1.Flex`,
2 OCPU / 12 GB, aarch64) on port 8080. The instance runs nothing else. Public routing goes
through the Cloudflare Worker in `deploy/cloudflare/`. `deploy/alphazero/` is the older GCP
deployment of the same engine.

## Why the model choice depends on this machine

The engine stops searching 2.5 seconds into a move, whatever the network size, so every model
answers within the 3-second target. What changes is how many searches fit. Measured on two
Neoverse-N1 cores (the A1 CPU) with `gomoku.scripts.bench_search`:

| Network | Backend | Threads | Searches in 2.5 s |
| --- | --- | --- | --- |
| 6x96 (v5) | static int8 ONNX | 2 | 634 |
| 10x128 (v6b) | static int8 ONNX | 2 | 272 |
| 6x96 (v5) | static int8 ONNX | 1 | 392 |
| 6x96 (v5) | float32 ONNX | 2 | 196 |

Static int8 on both cores is therefore the default. `03_bench_search.sh` prints the number
for the deployed model on the instance itself: on the A1 instance v5 iteration 144 completed
613 searches as trained and 671 with its unused channels removed
(`configs/deploy-v5-pruned.yaml`, `gomoku/scripts/prune_channels.py`).

## 0. Instance (Oracle console or CLI, once)

1. Create the instance: shape `VM.Standard.A1.Flex`, 2 OCPU, 12 GB, image
   **Canonical Ubuntu 24.04 (aarch64)**, in a public subnet with a public IPv4 address, with
   your SSH public key. Every tenancy gets 1,500 OCPU hours and 9,000 GB hours of A1 a month
   free, which is one such instance running all month and nothing more; a second A1 instance
   in the same tenancy is billed on a Pay As You Go account.
2. The subnet's security list needs:
   - `tcp/22` from your own IP.
   - `tcp/8080` from each Cloudflare IPv4 range only
     (`curl -fsSL https://www.cloudflare.com/ips-v4`, 15 CIDRs).
3. Note the public IPv4; it is `DEPLOY_ALPHAZERO_A1_HOST` below.

## 1. Configure

Add to the repo `.env` (see `.env.example`):

```sh
DEPLOY_ALPHAZERO_A1_HOST=<public IPv4>
DEPLOY_ALPHAZERO_A1_SSH_USER=ubuntu
DEPLOY_ALPHAZERO_A1_SSH_KEY=~/.ssh/<key>
DEPLOY_ALPHAZERO_A1_CONFIG=configs/deploy-v5.yaml       # under alphazero/
DEPLOY_ALPHAZERO_A1_CHECKPOINT=/path/to/iteration_0144.pt
```

The config names the network size and search settings; the checkpoint must match it
(`configs/deploy-v5.yaml` for 6x96, `configs/deploy-v6b.yaml` for 10x128).

## 2. Prepare the host

```sh
bash deploy/alphazero-oracle/01_host_setup.sh
```

Installs Docker, compose, rsync, a 2 GB swap file, opens the port in the host iptables
(Oracle's Ubuntu image rejects everything but SSH by default) and creates `~/build`.
Idempotent.

## 3. Deploy

```sh
bash deploy/alphazero-oracle/02_deploy_alphazero.sh
```

Syncs `alphazero/` (no training runs, virtualenv or native builds) and the checkpoint,
builds `infra/image/Dockerfile.prod` on the instance for aarch64 and starts the container
with `restart: unless-stopped`. Re-run for updates; `NO_CACHE=true` forces a clean build,
`DO_BUILD=false` only restarts.

## 4. Measure the search budget

```sh
STOP_ENGINE=true bash deploy/alphazero-oracle/03_bench_search.sh
```

## 5. Play a full game against it

The port is not public, so go through an SSH tunnel. The client makes the engine play both
sides and checks every answer against the repository's rules and the 3-second limit:

```sh
ssh -N -L 18080:127.0.0.1:8080 ubuntu@<host> &
cd alphazero && .venv/bin/python ../deploy/alphazero-oracle/smoke_game.py --url ws://127.0.0.1:18080/ws
```

## 6. Cut over the Cloudflare routing

`alphazero-api.<domain>` follows `DEPLOY_ALPHAZERO_IP`. Set it to the instance's address and
run:

```sh
bash deploy/cloudflare/deploy.sh
```

## Tests

```sh
node --test deploy/alphazero-oracle/tests/alphazero-oracle-deploy.test.mjs
```
