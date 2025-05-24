from fastapi import FastAPI
from routers import websocket

app = FastAPI()

# Include Routers
app.include_router(websocket.router)


@app.get("/")
def read_root():
    return {"message": "Hello world!"}
