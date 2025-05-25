# gomoku/adapters/api_fastapi/__init__.py
from fastapi import FastAPI

from .websocket import router as ws_router  # ← same router you already wrote


def create_app() -> FastAPI:
    app = FastAPI()
    app.include_router(ws_router)  # /ws/debug lives inside the router

    # keep .get("/") or any other REST endpoints here
    @app.get("/")
    async def read_root():
        return {"message": "Hello world!"}

    return app
