# Deployment

Each service deploys on its own. Nothing here depends on another service's scripts.

| Directory | Deploys | Where |
| --- | --- | --- |
| [`alphazero/`](alphazero) | AlphaZero engine, earlier deployment (replaced in Nov 2026; its GCP project is closed) | GCP VM |
| [`alphazero-oracle/`](alphazero-oracle/README.md) | AlphaZero engine | Oracle Cloud A1 |
| [`minimax/`](minimax/README.md) | Minimax engine | Oracle Cloud A1 |
| [`cloudflare/`](cloudflare/README.md) | Worker routing and backend DNS records | Cloudflare |

Training infrastructure is separate: see `alphazero/infra/training/`.

## AlphaZero

The live AlphaZero engine runs on the Oracle Cloud A1 instance; its steps are in [`alphazero-oracle/README.md`](alphazero-oracle/README.md). `deploy/alphazero/` keeps the scripts of the earlier GCP deployment (`01_setup.sh`, `02_deploy.sh`, `teardown.sh`); running them again needs a new GCP project.

`deploy/purge_local_gcloud.sh` wipes this machine's gcloud login state. It is not a deploy step.
