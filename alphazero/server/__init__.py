from fastapi import FastAPI


def create_app() -> FastAPI:
    app = FastAPI(title="Gomoku AlphaZero")

    from server.websocket import router

    app.include_router(router)

    import os

    from server.engine import AlphaZeroEngine

    @app.on_event("startup")
    async def load_model():
        app.state.engine = AlphaZeroEngine(
            config_path=os.getenv("ALPHAZERO_CONFIG", "configs/local_play.yaml"),
            checkpoint_path=os.getenv("ALPHAZERO_CHECKPOINT", "models/champion.pt"),
            device=os.getenv("ALPHAZERO_DEVICE", "cpu"),
        )

    @app.get("/")
    async def health():
        return {"status": "ok", "model": "alphazero"}

    return app
