
from .predictor import HaamPredictor
from .utils import get_logger

logger = get_logger("SprintLayer")

class SprintLayer:
    """
    Handles real-time perception (Turn-by-turn).
    Input: Audio chunk (Turn)
    Output: Emotion, Confidence, Transcript, Attention
    """
    def __init__(self):
        self.predictor = HaamPredictor()

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
