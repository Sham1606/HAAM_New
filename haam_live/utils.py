import logging
import sys
import os

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def get_logger(name):
    return logging.getLogger(name)

# Audio Configuration
AUDIO_config = {
    "SAMPLE_RATE": 16000,
    "BLOCK_SIZE": 4096,        # Processing chunk size
    "VAD_THRESHOLD_DB": -35,   # Energy threshold for speech
    "SILENCE_DURATION": 2.0,   # Seconds of silence to end a turn
    "MIN_TURN_DURATION": 0.5,  # Minimum speech duration to process
    "MAX_TURN_DURATION": 15.0  # Max duration to force a cut
}

# Emotion Mapping (Sprint Model)
EMOTION_MAP = {
    0: "neutral",
    1: "anger",
    2: "disgust",
    3: "fear",
    4: "sadness",
    5: "joy",      # Check model specific mapping carefully!
    6: "surprise"
}
