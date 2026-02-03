### Gomoku

Gomoku is a classic board game where two players place stones on a 19x19 board. This project includes a Nuxt.js frontend and a C++ Minimax backend.

TBA: Documentation is still being updated and may be incomplete or outdated in some areas.

**Local Development (Docker, alpha-zero excluded)**
Docker is the recommended way to run the app locally.

1. Create `.env` from the example.
```bash
cp .env.example .env
```

2. Start the Minimax backend.
```bash
docker compose -f docker-compose.yml up minimax
```

3. In a separate terminal, start the frontend only (without dependencies).
```bash
docker compose -f docker-compose.yml up front --no-deps
```

**Run Frontend + Minimax Together (single command)**
If you want both services in one command (still excluding alpha-zero), use:
```bash
docker compose -f docker-compose.yml up front minimax
```
This runs both services in the same terminal and streams combined logs.

4. Open the app.
- Frontend: `http://localhost:3000`
- Minimax WebSocket: `ws://localhost:8005/ws`

5. Stop services.
```bash
docker compose -f docker-compose.yml down
```

**Environment Variables**
The default ports are defined in `.env.example`. These are the expected values for local usage:
```
LOCAL_FRONT=3000
LOCAL_FRONT_NUXT_CONTENT_WS=4000
LOCAL_MINIMAX=8005
LOCAL_MINIMAX_GDB=8006

FRONT_WHERE=local
```

**Notes**
- `alias.sh` contains Docker shortcuts, but the default aliases bring up additional services. Use the commands above to run only `front` and `minimax`.
- If you see missing include errors for Minimax, ensure system dependencies are installed (see `minimax/dependencies.txt`).

**Third-Party Code**
- Double-three detection uses logic derived from the Renju open source reference implementation:
  `https://www.renju.se/renlib/opensrc/`
  Please refer to the upstream page for license details and attribution requirements.
