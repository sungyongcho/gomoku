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
                x, y, last_player, next_player, board = (
                    data["lastPlay"]["coordinate"]["x"],
                    data["lastPlay"]["coordinate"]["y"],
                    data["lastPlay"]["stone"],
                    data["nextPlayer"],
                    data["board"],
                )
                print(x, y, last_player, next_player, board, flush=True)
                game.set_game(board, last_player, next_player)
                # success = game.update_board(x, y, player)
            #     if success:
            #         # play_next()
            #         board_to_print = game.print_board()
            #         print(board_to_print)
            #         game.print_history()
            #         await websocket.send_json(
            #             {
            #                 "type": "move",
            #                 "status": "success",
            #                 "board": game.print_board(),
            #             }
            #         )
            #     else:
            #         await websocket.send_json(
            #             {"type": "error", "error": "Invalid move"}
            #         )
            # elif data["type"] == "reset":
            #     game.reset_board()
            #     await websocket.send_json(
            #         {"type": "reset", "board": game.print_board()}
            #     )
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@router.websocket("/ws/debug")
async def debug_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        game = Gomoku()
        while True:
            data = await websocket.receive_json()
            print(data, flush=True)
            if data["type"] == "move":
                # ai first
                if "lastPlay" not in data:
                    print("ai first", flush=True)
                    await websocket.send_json(
                        {
                            "type": "error",
                            "status": "tba",
                        }
                    )
                    continue
                # human player first
                x, y, last_player, next_player, goal, board, scores = (
                    data["lastPlay"]["coordinate"]["x"],
                    data["lastPlay"]["coordinate"]["y"],
                    data["lastPlay"]["stone"],
                    data["nextPlayer"],
                    data["goal"],
                    data["board"],
                    data["scores"],
                )

                # print(x, y, last_player, next_player, goal, board, scores, flush=True)
                game.set_game(board, last_player, next_player, scores, goal)
                print("before\n", game.print_board(), flush=True)
                success = game.update_board(x, y, last_player)
                print("after\n", game.print_board(), flush=True)
                if success:
                    # TODO: when play_next() executes, the "next player" changes
                    lastPlay = game.play_next()

                    print(lastPlay, flush=True)
                    await websocket.send_json(
                        {
                            "type": "move",
                            "status": "success",
                            "board": game.get_board(),
                            "scores": game.get_scores(),
                            "lastPlay": lastPlay,
                            "capturedStones": game.get_captured_stones(),
                        }
                    )
                else:
                    print("ssup")
                    await websocket.send_json({"type": "error", "error": "doublethree"})
            # elif data["type"] == "reset":
            #     game.reset_board()
            #     await websocket.send_json(
            #         {"type": "reset", "board": game.print_board()}
            #     )
    except WebSocketDisconnect:
        manager.disconnect(websocket)
