"""
WebSocket manager for real-time agent status streaming.

Admin dashboard connects to ws://localhost:8000/ws/agents and receives
live updates whenever any agent's status, emotion, or feedback changes.
"""

import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger("HAAM_WS")


class AgentStatusManager:
    """
    Manages live agent status and broadcasts updates to admin dashboards.
    Supports 10+ concurrent WebSocket connections.
    """

    def __init__(self):
        # Connected admin WebSocket clients
        self.active_connections: List[WebSocket] = []
        # Current status for each agent: {agent_id: {...status dict...}}
        self.agent_states: Dict[str, dict] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Admin WS connected. Total: {len(self.active_connections)}")
        # Send current state snapshot immediately
        try:
            await websocket.send_json({
                "type": "snapshot",
                "agents": self.agent_states,
            })
        except Exception:
            pass

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Admin WS disconnected. Remaining: {len(self.active_connections)}")

    async def update_agent(self, agent_id: str, data: dict):
        """
        Update an agent's live state and broadcast to all connected admins.
        data can include: status, live_emotion, confidence, feedback, last_ping, etc.
        """
        async with self._lock:
            if agent_id not in self.agent_states:
                self.agent_states[agent_id] = {}
            self.agent_states[agent_id].update(data)
            self.agent_states[agent_id]["last_update"] = datetime.utcnow().isoformat()

        await self.broadcast({
            "type": "agent_update",
            "agent_id": agent_id,
            "data": self.agent_states[agent_id],
        })

    async def broadcast(self, message: dict):
        """Send a message to all connected admin WebSockets."""
        disconnected = []
        for ws in self.active_connections:
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(ws)

        for ws in disconnected:
            self.disconnect(ws)

    async def remove_agent(self, agent_id: str):
        """Mark agent as offline and broadcast."""
        async with self._lock:
            if agent_id in self.agent_states:
                self.agent_states[agent_id]["status"] = "offline"
                self.agent_states[agent_id]["last_update"] = datetime.utcnow().isoformat()

        await self.broadcast({
            "type": "agent_update",
            "agent_id": agent_id,
            "data": self.agent_states.get(agent_id, {"status": "offline"}),
        })

    def get_all_statuses(self) -> dict:
        return dict(self.agent_states)


# ── Feedback Generator ────────────────────────────────────────────────────────

def generate_feedback(emotion: str, stress_score: float, confidence: float = 0.5) -> str:
    """
    Sprint-layer → natural language feedback.
    Input: emotion prediction + stress score
    Output: Actionable coaching text for agent or supervisor
    """
    if stress_score >= 0.8:
        if emotion == "anger":
            return "⚠️ Critical stress with anger detected — immediate supervisor intervention recommended. Consider a 10-minute break."
        elif emotion == "fear":
            return "⚠️ High anxiety detected — supportive check-in needed. Escalation protocol may be warranted."
        return "⚠️ High stress detected — take a 5-minute break and practice deep breathing."

    if stress_score >= 0.5:
        if emotion == "anger":
            return "🔶 Elevated frustration — try slowing speech pace. A brief pause between calls may help."
        elif emotion == "sadness":
            return "🔶 Emotional fatigue noted — consider rotating to lower-intensity tasks."
        elif emotion == "fear":
            return "🔶 Nervousness detected — you're doing well, stay with your training script."
        return "🔶 Moderate stress — maintain current pace, check in after 3 more calls."

    if emotion == "neutral":
        return "✅ Calm and professional tone — great job maintaining composure."
    if emotion == "anger":
        return "💡 Slight frustration detected — take a deep breath before the next response."
    if emotion == "sadness":
        return "💡 Low energy detected — try a brief stretch or standing break."
    if emotion == "disgust":
        return "💡 Mild negative reaction — stay objective and empathetic."

    return "✅ Good performance — keep up the steady pace."


# ── Global Instance ────────────────────────────────────────────────────────────
status_manager = AgentStatusManager()
