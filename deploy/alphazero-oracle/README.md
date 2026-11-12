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

### Searches kept between moves and searched while the opponent thinks

`configs/deploy-v5-pruned.yaml` turns on two switches of the native search, both off unless a
config sets them:

- `mcts.reuse_tree`: each WebSocket connection keeps its search tree. After answering, the
  engine keeps the subtree of the move it played; when the next request is that position or
  the position one move below it (the whole state is compared: board, capture points, side to
  move, last move, empty count and the last five moves), the search goes on from that subtree.
  Any other position, a `reset` message or a new connection starts a new tree.
- `mcts.ponder`: after answering, the engine searches the position it played into until the
  next move request arrives on any connection, the connection closes, `ponder_time_limit_s`
  (120 s) passes or all kept trees together reach `ponder_max_nodes` (50,000, 4 to 5 KB each).
  One background search runs at a time. A move request stops it before its own search starts
  and the time that takes comes out of the 2.5-second search limit, not out of the answer.

The answer's `search` field reports the searches carried over and the new ones. To measure a
game as an opponent sees it, give each colour its own connection and the opponent a thinking
time (one connection asked for both colours keeps the whole subtree of the engine's own move,
which no game against an opponent does):

```sh
.venv/bin/python ../deploy/alphazero-oracle/smoke_game.py --url ws://127.0.0.1:18080/ws \
  --connections 2 --think-s 5
```

Measured on the A1 instance (v5 iteration 144 pruned, static int8, 2 threads), one full game
per row, medians over the moves after each connection's first. A new tree gets 668 searches in
2.5 s (`03_bench_search.sh`):

| Setting | Searches behind a move | carried over | new | Answer mean / max | Container memory peak |
| --- | --- | --- | --- | --- | --- |
| no kept tree | 668 (the benchmark's median) | 0 | 668 | 2.52 s / 2.59 s | 262 MB |
| `reuse_tree` | 1,468 | 646 | 812 | 2.42 s / 2.53 s | 262 MB |
| `reuse_tree` + `ponder`, opponent answers at once | 1,468 | 651 | 799 | 2.42 s / 2.56 s | 263 MB |
| `reuse_tree` + `ponder`, opponent thinks 5 s | 6,363 | 5,303 | 1,044 | 2.51 s / 2.53 s | 307 MB |
| `reuse_tree` + `ponder`, opponent thinks 15 s | 13,859 | 12,484 | 977 | 2.51 s / 2.51 s | 415 MB |

Reading the table:

- The opponent in these games is the same engine, so its replies are the ones the kept tree
  expects far more often than a person's would be, and the carried-over numbers are an upper
  range for real games. An unexpected reply still keeps whatever was searched below it.
- The new searches are not fewer with `ponder` on (early moves: 704, 683, 684, 707 with it,
  721, 709, 681, 701 without), because the background search is stopped before the timed one
  starts.
- Late in a game most searches end at finished positions without evaluating the network, so
  the counts there run into the tens of thousands; that is why the table uses medians.
- With the opponent thinking for 130 s the background search ran 120 s on both cores (33,623
  searches), then stopped with the connection still open; the second connection's search
  stopped after about 55 s at the shared node cap. The container peaked at 461 MB. Memory freed by a
  closed connection is reused for later trees but not returned to the system.

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
