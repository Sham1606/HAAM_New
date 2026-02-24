
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import logging
import time
from .audio_stream import AudioStreamManager
from .sprint_layer import SprintLayer
from .marathon_layer import MarathonLayer
from .utils import get_logger

logger = get_logger("HAAM_Live_API")

app = FastAPI(title="HAAM Live Analysis")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
class Pipeline:
    def __init__(self):
        self.sprint = SprintLayer()
        self.marathon = MarathonLayer()
        self.audio_manager = None
        self.active_websockets = [] # List of active connections

    def on_turn_detected(self, audio_data, start_time):
        """Callback from AudioStreamManager"""
        # 1. Run Sprint Layer (Latency critical)
        sprint_result = self.sprint.process_turn(audio_data, start_time)
        
        # 2. Run Marathon Layer (Aggregation)
        marathon_stats = self.marathon.update(sprint_result)
        
        # 3. Broadcast
        message = {
            "type": "turn_result",
            "sprint": sprint_result,
            "marathon": marathon_stats
        }
        asyncio.run_coroutine_threadsafe(self.broadcast(message), loop)

    async def broadcast(self, message):
        disconnected = []
        for ws in self.active_websockets:
            try:
                await ws.send_json(message)
            except Exception as e:
                logger.warning(f"WebSocket send failed: {e}")
                disconnected.append(ws)
        
        for ws in disconnected:
            self.active_websockets.remove(ws)

    def start_listening(self):
        if not self.audio_manager:
            self.audio_manager = AudioStreamManager(self.on_turn_detected)
            self.audio_manager.start()

    def stop_listening(self):
        if self.audio_manager:
            self.audio_manager.stop()
            self.audio_manager = None


pipeline = Pipeline()
loop = None

@app.on_event("startup")
async def startup_event():
    global loop
    loop = asyncio.get_running_loop()
    logger.info("HAAM Live Backend Started")
    # Auto-start listening? 
    # Better to start on first connection or explicit commands.
    pipeline.start_listening()

@app.on_event("shutdown")
def shutdown_event():
    pipeline.stop_listening()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    pipeline.active_websockets.append(websocket)
    logger.info("Client connected")
    try:
        while True:
            data = await websocket.receive_text()
            # Handle client commands if any
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        logger.info("Client disconnected")
        pipeline.active_websockets.remove(websocket)

@app.get("/")
def read_root():
    return {"status": "running", "service": "HAAM Live Analysis"}

@app.get("/stats")
def get_stats():
    return pipeline.marathon.get_session_stats()
