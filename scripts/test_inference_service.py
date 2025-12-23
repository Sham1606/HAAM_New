import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.services.inference import HybridInference

def test():
    print("Initializing HybridInference v2...")
    try:
        engine = HybridInference()
    except Exception as e:
        print(f"FAILED to initialize: {e}")
        return

    # Use a sample file from CREMA-D 
    # 1001_DFA_ANG_XX.wav is a good candidate
    sample_path = r"d:\haam_framework\crema-d-mirror-main\AudioWAV\1001_DFA_ANG_XX.wav"
    
    if not os.path.exists(sample_path):
        print(f"Sample file not found: {sample_path}")
        return

    print(f"Testing prediction on: {sample_path}")
    try:
        result = engine.predict(sample_path)
        print("\n--- Prediction Results ---")
        print(f"Predicted: {result['predicted_emotion']} (Conf: {result['confidence']:.2f})")
        print(f"Transcript: {result['transcript']}")
        print(f"Acoustic: Pitch={result['acoustic_summary']['pitch_mean']}, RMS={result['acoustic_summary']['rms_mean']}")
        print(f"Fusion Weights: Ac={result['fusion_weights']['acoustic']}, Tx={result['fusion_weights']['text']}")
        print(f"Top 3: {result['top_3_predictions']}")
        print(f"Inference Time: {result['inference_time_ms']}ms")
    except Exception as e:
        print(f"Prediction FAILED: {e}")

if __name__ == "__main__":
    test()
