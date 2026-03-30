
import asyncio
import logging
import numpy as np
from collections import deque
from src.inference.live_predictor import LivePredictor
from src.utils.audio_stream import AudioStreamManager

logger = logging.getLogger(__name__)

class SprintLayer:
    """
    Handles real-time perception (Turn-by-turn).
    Input: Audio chunk (Turn)
    Output: Emotion, Confidence, Transcript, Attention
    """
    def __init__(self):
        self._predictor = None

    @property
    def predictor(self):
        """Lazy-load LivePredictor on first use to avoid blocking server startup."""
        if self._predictor is None:
            logger.info("Lazy-loading LivePredictor for Sprint Layer...")
            self._predictor = LivePredictor()
            logger.info("✅ LivePredictor loaded.")
        return self._predictor

    def process_turn(self, audio_data, start_time):
        """
        Process a single turn of audio.
        """
        logger.info(f"⚡ SPRINT: Processing turn starting at {start_time}")
        
        result = self.predictor.predict(audio_data)
        
        return {
            "timestamp": start_time,
            "transcript": result["transcript"],
            "emotion": result["emotion"],
            "confidence": result["confidence"],
            "attention": result["attention"]
        }

class MarathonLayer:
    """
    Handles long-term context (Session Aggregation).
    Input: Sprint Output (Turn Stats)
    Output: Session Stats, Trends, Risk Score
    """
    def __init__(self, history_size=50):
        self.history = deque(maxlen=history_size)
        self.emotion_counts = {}
        self.total_turns = 0
        
    def update(self, sprint_result):
        """
        Ingest a new turn result and update session state.
        """
        self.history.append(sprint_result)
        self.total_turns += 1
        
        # Update emotion counts
        emo = sprint_result["emotion"]
        self.emotion_counts[emo] = self.emotion_counts.get(emo, 0) + 1
        
        return self.get_session_stats()

    def get_session_stats(self):
        if not self.history:
            return {}
            
        # 1. Dominant Emotion
        dominant_emotion = max(self.emotion_counts, key=self.emotion_counts.get)
        
        # 2. Attention Trend (Last 10 turns)
        recent_turns = list(self.history)[-10:]
        avg_audio_attn = np.mean([t["attention"]["audio"] for t in recent_turns])
        avg_text_attn = np.mean([t["attention"]["text"] for t in recent_turns])
        
        # 3. Sentiment/Emotion Trend
        # Heuristic: convert to score (Anger/Fear/Sadness = -1, Neutral = 0, Joy = 1)
        score_map = {"anger": -1, "disgust": -1, "fear": -1, "sadness": 1, "neutral": 0, "joy": 1} # Adjusted sadness to 1? No, usually negative.
        # Wait, sadness is negative valence.
        score_map["sadness"] = -1
        
        scores = [score_map.get(t["emotion"], 0) for t in recent_turns]
        trend_slope = 0
        if len(scores) > 1:
            try:
                trend_slope = np.polyfit(range(len(scores)), scores, 1)[0]
            except:
                trend_slope = 0
            
        trend_desc = "Stable"
        if trend_slope > 0.1: trend_desc = "Inproving"
        elif trend_slope < -0.1: trend_desc = "Worsening"

        return {
            "total_turns": self.total_turns,
            "dominant_emotion": dominant_emotion,
            "avg_audio_attention": round(avg_audio_attn, 2),
            "avg_text_attention": round(avg_text_attn, 2),
            "trend": trend_desc,
            "risk_score": self._calculate_risk()
        }

    def _calculate_risk(self):
        # Simple risk calculation based on negative emotion frequency
        negatives = sum(self.emotion_counts.get(e, 0) for e in ["anger", "fear", "disgust"])
        if self.total_turns == 0: return 0.0
        return round(negatives / self.total_turns, 2)

class LivePipeline:
    def __init__(self):
        self._sprint = None
        self.marathon = MarathonLayer()
        self.audio_manager = None
        self.active_websockets = [] # List of active connections
        self.loop = None

    @property
    def sprint(self):
        """Lazy-load SprintLayer (and its LivePredictor) on first use."""
        if self._sprint is None:
            self._sprint = SprintLayer()
        return self._sprint

    def set_loop(self, loop):
        self.loop = loop

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
        
        if self.loop:
            asyncio.run_coroutine_threadsafe(self.broadcast(message), self.loop)

    async def broadcast(self, message):
        disconnected = []
        for ws in self.active_websockets:
            try:
                await ws.send_json(message)
            except Exception as e:
                logger.warning(f"WebSocket send failed: {e}")
                disconnected.append(ws)
        
        for ws in disconnected:
            if ws in self.active_websockets:
                self.active_websockets.remove(ws)

    def start_listening(self):
        if not self.audio_manager:
            self.audio_manager = AudioStreamManager(self.on_turn_detected)
            self.audio_manager.start()

    def stop_listening(self):
        if self.audio_manager:
            self.audio_manager.stop()
            self.audio_manager = None
