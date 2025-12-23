import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add project
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.features.improved_acoustic import ImprovedAcousticExtractor
from src.features.emotion_text import EmotionTextExtractor

def test_features():
    print("="*80)
    print("TESTING IMPROVED FEATURE EXTRACTION")
    print("="*80)
    
    # Init Extractors
    print("Initializing Extractors...")
    acoustic_ext = ImprovedAcousticExtractor()
    text_ext = EmotionTextExtractor()
    
    # Test Data
    crema_path = Path(r"D:\haam_framework\crema-d-mirror-main\AudioWAV\1001_DFA_ANG_XX.wav")
    
    if not crema_path.exists():
        print(f"Test file not found: {crema_path}")
        return

    print(f"\nProcessing File: {crema_path.name}")
    
    # 1. Acoustic Test
    print("\n--- Acoustic Features (12 core) ---")
    ac_feats = acoustic_ext.extract(str(crema_path))
    for k, v in ac_feats.items():
        print(f"{k:<20}: {v:.4f}")
        
    vec = acoustic_ext.extract_vector(str(crema_path))
    print(f"Vector Shape: {vec.shape} (Should be (12,))")
    
    # 2. Text Test
    print("\n--- Text Features (Emotion-Specific) ---")
    
    samples = [
        "I am so angry right now!",
        "This fits perfectly.",
        "I am terrified of heights."
    ]
    
    for txt in samples:
        print(f"\nInput: '{txt}'")
        res = text_ext.extract(txt)
        probs = res['emotion_probs']
        print(f"Dominant: {res['dominant_emotion'].upper()} ({res['confidence']:.2f})")
        print(f"Vector: {res['embedding'].shape} (Should be (768,))")
        print(f"Dist: {probs}")

    print("\nTest Complete.")

if __name__ == "__main__":
    test_features()
