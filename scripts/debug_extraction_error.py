
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.features.fixed_acoustic_extractor import RobustAcousticExtractor
from src.features.emotion_text import EmotionTextExtractor
import whisper
import numpy as np
import os

def test_single():
    print("Testing single file extraction...")
    # Get first file from metadata
    import pandas as pd
    df = pd.read_csv('data/real_metadata_with_split.csv')
    row = df.iloc[0]
    audio_path = row['filepath']
    print(f"File: {audio_path}")
    print(f"Exists: {os.path.exists(audio_path)}")

    try:
        print("1. Testing Acoustic...")
        acoustic_extractor = RobustAcousticExtractor(sr=16000)
        acoustic_features = acoustic_extractor.extract_array(audio_path)
        print(f"   Acoustic success! Shape: {acoustic_features.shape}")
        print(f"   Features: {acoustic_features}")

        print("2. Testing Whisper...")
        whisper_model = whisper.load_model("base")
        transcription = whisper_model.transcribe(audio_path)
        print(f"   Whisper success! Text: {transcription['text']}")

        print("3. Testing Text Emotion...")
        text_extractor = EmotionTextExtractor()
        text_res = text_extractor.extract(transcription['text'])
        print(f"   Text success! Dominant: {text_res['dominant_emotion']}")

        print("All steps succeeded for this file.")
    except Exception as e:
        print(f"\nFAILED during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_single()
