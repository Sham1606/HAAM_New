
from collections import deque
import numpy as np
from .utils import get_logger

logger = get_logger("MarathonLayer")

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
        score_map = {"anger": -1, "disgust": -1, "fear": -1, "sadness": -1, "neutral": 0, "joy": 1}
        scores = [score_map.get(t["emotion"], 0) for t in recent_turns]
        trend_slope = 0
        if len(scores) > 1:
            trend_slope = np.polyfit(range(len(scores)), scores, 1)[0]
            
        trend_desc = "Stable"
        if trend_slope > 0.1: trend_desc = "Improving"
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
