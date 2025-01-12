from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from services.gomoku import Gomoku

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
        game = Gomoku()
        while True:
            data = await websocket.receive_json()
            if data["type"] == "move":
                x, y, player = (
                    data["new_stone"]["x"],
                    data["new_stone"]["y"],
                    data["new_stone"]["player"],
                )
                print(x, y, player)
                success = game.update_board(x, y, player)
                if success:
                    # play_next()
                    board_to_print = game.print_board()
                    print(board_to_print)
                    game.print_history()
                    await websocket.send_json(
                        {
                            "type": "move",
                            "status": "success",
                            "board": game.print_board(),
                        }
                    )
                else:
                    await websocket.send_json(
                        {"type": "error", "error": "Invalid move"}
                    )
            elif data["type"] == "reset":
                game.reset_board()
                await websocket.send_json({"type": "reset", "board": get_board()})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
