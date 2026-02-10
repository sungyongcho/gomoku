#!/bin/sh

### gomoku
### Docker compose commands

## for development environment
alias dev="docker compose -f docker-compose.yml"
alias dev-up="docker compose -f docker-compose.yml up front minimax alphazero"
alias dev-up-valgrind="docker compose -f docker-compose.yml up front minimax_valgrind"
alias dev-up-debug="docker compose -f docker-compose.yml up front minimax_debug"
alias dev-down="docker compose -f docker-compose.yml down"

## alphazero cluster helper
alias alphazero-ray-attach=". alphazero/.venv/bin/activate && bash alphazero/infra/cluster/ray_attach.sh"
alias alphazero-restart-cluster=". alphazero/.venv/bin/activate && bash alphazero/infra/cluster/restart_cluster.sh"
alias alphazero-reserve-gpu=". alphazero/.venv/bin/activate && bash alphazero/infra/cluster/reserve_gpu.sh"
alias alphazero-purge-gcp=". alphazero/.venv/bin/activate && bash alphazero/infra/cluster/purge_gcp.sh"
