#!/bin/sh

### gomoku
### Docker compose commands

## for development environment
alias dev="docker compose -f docker-compose.yml"
alias dev-up="docker compose -f docker-compose.yml up back front cpp_server"
alias dev-up-valgrind="docker compose -f docker-compose.yml up back front cpp_server_valgrind"
alias dev-up-debug="docker compose -f docker-compose.yml up back front cpp_server_debug"
# alias dev-up-valgrind-build="docker compose -f docker-compose.yml up back front cpp_server_valgrind"
# alias dev-up-build="docker compose -f docker-compose.yml up back front cpp_server --build"
alias dev-down="docker compose -f docker-compose.yml down"

