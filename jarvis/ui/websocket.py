from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from jarvis.orchestrator.events import State
from jarvis.telemetry.logging import get_logger


class FloatingUIBridge:
    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._router = APIRouter()
        self._router.add_api_websocket_route("/ws/state", self._websocket_handler)
        self._lock = asyncio.Lock()
        self._logger = get_logger(__name__)

    @property
    def router(self) -> APIRouter:
        return self._router

    async def _websocket_handler(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._clients.add(websocket)
        self._logger.info("ui.client.connected", count=len(self._clients))
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            async with self._lock:
                self._clients.discard(websocket)
            self._logger.info("ui.client.disconnected", count=len(self._clients))

    async def publish_state(self, state: State, payload: dict[str, Any] | None = None) -> None:
        message = {"state": state, "payload": payload or {}}
        async with self._lock:
            send_tasks = [client.send_json(message) for client in self._clients]
        if send_tasks:
            await asyncio.gather(*send_tasks, return_exceptions=True)

