# Deployment

Each service deploys on its own. Nothing here depends on another service's scripts.

| Directory | Deploys | Where |
| --- | --- | --- |
| [`alphazero/`](alphazero) | AlphaZero engine | GCP VM |
| [`minimax/`](minimax/README.md) | Minimax engine | Oracle Cloud A1 |
| [`cloudflare/`](cloudflare/README.md) | Worker routing and backend DNS records | Cloudflare |

Training infrastructure is separate: see `alphazero/infra/training/`.

## AlphaZero on a new GCP project

```sh
bash deploy/alphazero/01_setup.sh    # prints DEPLOY_ALPHAZERO_IP; copy it into .env
bash deploy/alphazero/02_deploy.sh
bash deploy/cloudflare/deploy.sh     # points alphazero-api at the new IP
```

`deploy/alphazero/teardown.sh` removes the AlphaZero resources.
`deploy/purge_local_gcloud.sh` wipes this machine's gcloud login state. It is not a deploy step.
