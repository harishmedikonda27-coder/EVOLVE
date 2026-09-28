from fastapi import WebSocket
from typing import List, Dict, Any
from backend.events import Event


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, run_id: str, websocket: WebSocket):
        await websocket.accept()
        if run_id not in self.active_connections:
            self.active_connections[run_id] = []
        self.active_connections[run_id].append(websocket)

    def disconnect(self, run_id: str, websocket: WebSocket):
        if run_id in self.active_connections:
            if websocket in self.active_connections[run_id]:
                self.active_connections[run_id].remove(websocket)

    async def broadcast_event(self, run_id: str, event_data: Dict[str, Any]):
        if run_id in self.active_connections:
            disconnected = []
            for connection in self.active_connections[run_id]:
                try:
                    await connection.send_json(event_data)
                except Exception:
                    disconnected.append(connection)
            for ws in disconnected:
                self.disconnect(run_id, ws)
