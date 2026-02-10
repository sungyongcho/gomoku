from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import torch

from gomoku.core.game_config import PLAYER_1, PLAYER_2
from server.engine import AlphaZeroEngine
from server.protocol import (
    build_error_response,
    build_evaluate_response,
    build_move_response,
    frontend_to_gamestate,
)

router = APIRouter()
# Reuse uvicorn's configured logger so app logs always appear in container stdout.
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)


def _parse_num_searches_env(name: str, default: int | None) -> int | None:
    raw = os.getenv(name)
    if raw is None:
        return default

    value = raw.strip().lower()
    if value in {"", "none", "default"}:
        return None

    try:
        parsed = int(value)
        if parsed <= 0:
            raise ValueError
        return parsed
    except ValueError:
        logger.warning(
            "Invalid %s=%r; using %s.",
            name,
            raw,
            "default config" if default is None else str(default),
        )
        return default


DIFFICULTY_MAP: dict[str, int | None] = {
    "easy": _parse_num_searches_env("ALPHAZERO_SEARCH_EASY", 100),
    "medium": _parse_num_searches_env("ALPHAZERO_SEARCH_MEDIUM", 400),
    "hard": _parse_num_searches_env("ALPHAZERO_SEARCH_HARD", 600),
}

DEFAULT_DIFFICULTY = os.getenv("ALPHAZERO_DEFAULT_DIFFICULTY", "hard").strip().lower()
if DEFAULT_DIFFICULTY not in DIFFICULTY_MAP:
    logger.warning(
        "Invalid ALPHAZERO_DEFAULT_DIFFICULTY=%r; falling back to 'hard'.",
        DEFAULT_DIFFICULTY,
    )
    DEFAULT_DIFFICULTY = "hard"


def _stone_for_player(player: int) -> str:
    if player == PLAYER_1:
        return "X"
    if player == PLAYER_2:
        return "O"
    raise ValueError(f"Invalid player value: {player}")


def _clamp_percentage(value: float) -> float:
    return max(0.0, min(100.0, value))


def _resolve_difficulty(raw_value: Any) -> str:
    difficulty = (
        str(raw_value).strip().lower() if raw_value is not None else DEFAULT_DIFFICULTY
    )
    if difficulty not in DIFFICULTY_MAP:
        raise ValueError("Game Difficulty Error")
    return difficulty


async def _safe_send_json(websocket: WebSocket, payload: dict[str, Any]) -> bool:
    """
    Send JSON unless the websocket is already closed.

    Returns
    -------
    bool
        True when send succeeded, False when the client is already disconnected.

    """
    try:
        await websocket.send_json(payload)
        return True
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected while sending payload.")
        return False
    except RuntimeError as exc:
        msg = str(exc)
        if "Unexpected ASGI message 'websocket.send'" in msg:
            logger.info("WebSocket already closed before payload send.")
            return False
        raise


def _opponent_stone(stone: str) -> str:
    return "O" if stone == "X" else "X"


def _format_request_board(board: Any) -> list[str]:
    if not isinstance(board, list):
        return []
    lines: list[str] = []
    for row in board:
        if not isinstance(row, list):
            continue
        rendered = []
        for cell in row:
            if cell == ".":
                rendered.append(".")
            elif cell == "X":
                rendered.append("1")
            elif cell == "O":
                rendered.append("2")
            else:
                rendered.append(str(cell))
        lines.append(" ".join(rendered) + " ")
    return lines


def _log_move_request(data: dict[str, Any], resolved_difficulty: str) -> None:
    last_play = data.get("lastPlay")
    next_player = str(data.get("nextPlayer", ""))
    if isinstance(last_play, dict):
        coordinate = last_play.get("coordinate", {})
        last_x = int(coordinate.get("x", -1))
        last_y = int(coordinate.get("y", -1))
        last_player = str(last_play.get("stone", ""))
    else:
        last_x = -1
        last_y = -1
        last_player = _opponent_stone(next_player) if next_player in {"X", "O"} else ""

    logger.info("Move received:")
    logger.info("  Last Play: (%s, %s) by %s", last_x, last_y, last_player)
    logger.info("  Next Player: %s", next_player)
    logger.info("  Goal: %s", data.get("goal"))
    logger.info("  Difficulty: %s", resolved_difficulty)
    logger.info("  Enable Capture: %s", int(bool(data.get("enableCapture"))))
    logger.info(
        "  Enable Double Three Restriction: %s",
        int(bool(data.get("enableDoubleThreeRestriction"))),
    )
    for line in _format_request_board(data.get("board")):
        logger.info(line)


async def _handle_move(
    websocket: WebSocket,
    engine: AlphaZeroEngine,
    data: dict[str, Any],
) -> bool:
    difficulty = _resolve_difficulty(data.get("difficulty"))

    state = frontend_to_gamestate(data)
    num_searches = DIFFICULTY_MAP[difficulty]
    stone = _stone_for_player(int(state.next_player))

    start_ns = time.perf_counter_ns()
    action = engine.get_best_move(state, num_searches)
    new_state, captures = engine.apply_move(state, action)
    elapsed_ns = time.perf_counter_ns() - start_ns
    elapsed_s = elapsed_ns / 1_000_000_000.0
    elapsed_ms = elapsed_ns / 1_000_000.0
    ai_x = int(action % engine.game.col_count)
    ai_y = int(action // engine.game.col_count)

    response = build_move_response(action, stone, new_state, captures, elapsed_ns)
    logger.info("AI played: (%s, %s) by %s", ai_x, ai_y, stone)
    sent = await _safe_send_json(websocket, response)
    logger.info(
        "Execution time: %.6f s, %.3f ms, %.5e ns",
        elapsed_s,
        elapsed_ms,
        float(elapsed_ns),
    )
    return sent


async def _handle_evaluate(
    websocket: WebSocket,
    engine: AlphaZeroEngine,
    data: dict[str, Any],
) -> bool:
    state = frontend_to_gamestate(data)
    encoded = engine.game.get_encoded_state(state)
    model_input = torch.from_numpy(encoded).to(
        device=engine.device, dtype=torch.float32
    )

    with torch.inference_mode():
        _, value_tensor = engine.inference_client.infer(model_input)

    value = float(value_tensor.squeeze().item())
    current_player_pct = _clamp_percentage((value + 1.0) * 50.0)

    if int(state.next_player) == PLAYER_1:
        x_pct = current_player_pct
        o_pct = 100.0 - current_player_pct
        x_eval = value
        o_eval = -value
    else:
        o_pct = current_player_pct
        x_pct = 100.0 - current_player_pct
        o_eval = value
        x_eval = -value

    response = build_evaluate_response(
        x_eval=x_eval,
        o_eval=o_eval,
        x_percentage=x_pct,
        o_percentage=o_pct,
    )
    return await _safe_send_json(websocket, response)


async def _handle_payload(
    websocket: WebSocket,
    engine: AlphaZeroEngine,
    data: dict[str, Any],
) -> bool:
    message_type = data.get("type")
    if message_type in {"move", "test"}:
        return await _handle_move(websocket, engine, data)

    if message_type == "evaluate":
        return await _handle_evaluate(websocket, engine, data)

    if message_type == "reset":
        # Stateless on server side for now.
        return True

    return await _safe_send_json(websocket, build_error_response("Unknown type"))


async def _serve(websocket: WebSocket) -> None:
    await websocket.accept()
    engine: AlphaZeroEngine = websocket.app.state.engine
    path = websocket.url.path
    logger.info("WebSocket `%s` connected!", path)
    prev_difficulty: str | None = None
    while True:
        try:
            data = await websocket.receive_json()
            if not isinstance(data, dict):
                sent = await _safe_send_json(
                    websocket,
                    build_error_response("Invalid payload: expected JSON object"),
                )
                if not sent:
                    break
                continue
            logger.info("Received: %s", json.dumps(data, separators=(",", ":")))
            message_type = data.get("type")
            if message_type in {"move", "test"}:
                difficulty = _resolve_difficulty(data.get("difficulty"))
                _log_move_request(data, difficulty)
                if prev_difficulty != difficulty:
                    if prev_difficulty is None:
                        logger.info("initial difficulty: %s", difficulty)
                    else:
                        logger.info(
                            "difficulty changed from %s to %s",
                            prev_difficulty,
                            difficulty,
                        )
                    prev_difficulty = difficulty
            if message_type == "reset":
                prev_difficulty = None
            sent = await _handle_payload(websocket, engine, data)
            if not sent:
                break
        except WebSocketDisconnect:
            break
        except ValueError as exc:
            logger.warning("Value error while handling websocket payload: %s", exc)
            sent = await _safe_send_json(websocket, build_error_response(str(exc)))
            if not sent:
                break
        except KeyError as exc:
            logger.warning("Missing field in websocket payload: %s", exc)
            sent = await _safe_send_json(
                websocket, build_error_response(f"Missing field: {exc}")
            )
            if not sent:
                break
        except Exception as exc:  # pragma: no cover - runtime safety net
            logger.exception("Unhandled websocket error")
            sent = await _safe_send_json(
                websocket, build_error_response(f"Internal server error: {exc}")
            )
            if not sent:
                break
    logger.info("WebSocket connection closed.")


@router.websocket("/ws")
async def game_endpoint(websocket: WebSocket) -> None:
    """Main game endpoint."""
    await _serve(websocket)


@router.websocket("/ws/debug")
async def debug_endpoint(websocket: WebSocket) -> None:
    """Debug endpoint."""
    await _serve(websocket)
