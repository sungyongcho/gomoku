from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from back.services.gomoku import (
    convert_board_for_print,
    get_board,
    play_next,
    reset_board,
    update_board,
)

router = APIRouter()


# Manage active connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()


@router.websocket("/ws/gomoku")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            if data["type"] == "move":
                x, y, player = (
                    data["new_stone"]["x"],
                    data["new_stone"]["y"],
                    data["new_stone"]["player"],
                )
                print(x, y, player)
                success = update_board(x, y, player)
                if success:
                    play_next()
                    board_to_print = convert_board_for_print()
                    print(board_to_print)
                    await websocket.send_json(
                        {"type": "move", "status": "success", "board": get_board()}
                    )
                else:
                    await websocket.send_json(
                        {"type": "error", "error": "Invalid move"}
                    )
            elif data["type"] == "reset":
                reset_board()
                # await websocket.send_json({"type": "reset", "board": get_board()})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
