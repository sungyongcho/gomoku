#!/bin/sh

### gomoku
### Docker compose commands

## for development environment
alias dev="docker compose -f docker-compose.yml"
alias dev-up="docker compose -f docker-compose.yml up back front minimax"
alias dev-up-valgrind="docker compose -f docker-compose.yml up back front minimax_valgrind"
alias dev-up-debug="docker compose -f docker-compose.yml up back front minimax_debug"
alias dev-down="docker compose -f docker-compose.yml down"

