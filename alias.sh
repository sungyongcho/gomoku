#!/bin/sh

### gomoku
### Docker compose commands

## for development environment
alias dev="docker compose -f docker-compose.yml"
alias dev-up="docker compose -f docker-compose.yml up"
alias dev-up-build="docker compose -f docker-compose.yml up --build"
alias dev-down="docker compose -f docker-compose.yml down"
