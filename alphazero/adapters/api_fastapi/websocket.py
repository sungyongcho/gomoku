from core.board import Board
from core.gomoku import Gomoku
from core.rules.capture import detect_captured_stones
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

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


@router.websocket("/ws/debug")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        game = Gomoku()
        while True:
            data = await websocket.receive_json()
            if data["type"] == "move":
                # print(data)
                (
                    last_x,
                    last_y,
                    last_player,
                    next_player,
                    board,
                    goal,
                    enable_capture,
                    enable_doublethree,
                ) = (
                    data["lastPlay"]["coordinate"]["x"],
                    data["lastPlay"]["coordinate"]["y"],
                    data["lastPlay"]["stone"],
                    data["nextPlayer"],
                    data["board"],
                    data["goal"],
                    data["enableCapture"],
                    data["enableDoubleThreeRestriction"],
                )
                # print(
                #     last_x,
                #     last_y,
                #     last_player,
                #     next_player,
                #     board,
                #     goal,
                #     enable_capture,
                #     enable_doublethree,
                # )
                game.set_game(data)
                game.print_board()
                print(
                    Board.convert_index_to_coordinates(
                        game.board.last_x, game.board.last_y
                    )
                )
                captured_test = detect_captured_stones(
                    game.board,
                    game.board.last_x,
                    game.board.last_y,
                    game.board.last_player,
                )

                if captured_test:
                    print("capture occured:", captured_test)
                    await websocket.send_json(
                        {"type": "error", "error": "capture test"}
                    )
                else:
                    await websocket.send_json({"type": "error", "error": "doublethree"})

                # game.set_game(board, last_player, next_player)
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


@router.websocket("/ws/debugbb")
async def debug_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        game = Gomoku()
        while True:
            data = await websocket.receive_json()
            print(data)
            if data["type"] == "move":
                # ai first
                if "lastPlay" not in data:
                    print("ai first")
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
                game.set_game(data)
                print("before\n", game.print_board())
                success = game.update_board(x, y, last_player)
                print("after\n", game.print_board())
                if success:
                    # TODO: when play_next() executes, the "next player" changes
                    lastPlay = game.play_next_minmax()

                    print(lastPlay)
                    print(
                        game.get_scores(),
                        game.board.last_player_score,
                        game.board.next_player_score,
                        game.get_captured_stones(),
                    )
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
                    game.captured_stones.clear()
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
