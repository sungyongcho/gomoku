#!/bin/sh

### gomoku
### Docker compose commands

## for development environment
alias dev="docker compose -f docker-compose.yml"
alias dev-up="docker compose -f docker-compose.yml up front minimax alphazero"
alias dev-up-front="docker compose -f docker-compose.yml up front"
alias dev-up-valgrind="docker compose -f docker-compose.yml up front minimax_valgrind"
alias dev-up-debug="docker compose -f docker-compose.yml up front minimax_debug"
alias dev-down="docker compose -f docker-compose.yml down"

## alphazero cluster helper
alias alphazero-ray-attach=". alphazero/.venv/bin/activate && bash alphazero/infra/cluster/ray_attach.sh"
alias alphazero-restart-cluster=". alphazero/.venv/bin/activate && bash alphazero/infra/cluster/restart_cluster.sh"
alias alphazero-purge-gcp=". alphazero/.venv/bin/activate && bash alphazero/infra/cluster/purge_gcp.sh"

## deployment logs
alias deploy-logs-minimax='[ -f .env ] && set -a && source .env && set +a && gcloud compute ssh ${DEPLOY_MINIMAX_VM} --zone=${DEPLOY_GCP_ZONE} --command "sudo docker logs -f ${DEPLOY_MINIMAX_VM}"'
alias deploy-logs-alphazero='[ -f .env ] && set -a && source .env && set +a && gcloud compute ssh ${DEPLOY_ALPHAZERO_VM} --zone=${DEPLOY_GCP_ZONE} --command "sudo docker logs -f ${DEPLOY_ALPHAZERO_VM}"'
alias deploy-logs-worker='npx wrangler tail --config deploy/wrangler.toml'
