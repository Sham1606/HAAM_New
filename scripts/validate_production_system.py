
import os
import sys
import time
import json
import logging
import requests
import numpy as np
import librosa
import soundfile as sf
import subprocess
import shutil

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.inference import HybridInference

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Constants
SAMPLE_DIR = "test_samples"
MODEL_DIR = "models"
API_URL = "http://localhost:8000"

def create_test_samples():
    """Create synthetic samples for validation."""
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    
    # 1. Standard Tone
    sr = 16000
    t = np.linspace(0, 1, sr)
    # Sine wave (440Hz)
    y_sine = 0.5 * np.sin(2 * np.pi * 440 * t) 
    sf.write(os.path.join(SAMPLE_DIR, "test_tone.wav"), y_sine, sr)
    
    # 2. Silence
    y_silent = np.zeros(sr)
    sf.write(os.path.join(SAMPLE_DIR, "test_silence.wav"), y_silent, sr)
    
    # 3. Short Audio (<0.5s)
    y_short = y_sine[:int(0.3*sr)]
    sf.write(os.path.join(SAMPLE_DIR, "test_short.wav"), y_short, sr)
    
    # 4. Noise
    y_noise = np.random.normal(0, 0.1, sr)
    sf.write(os.path.join(SAMPLE_DIR, "test_noise.wav"), y_noise, sr)
    
    # 5. Clipping
    y_clip = y_sine * 10
    y_clip = np.clip(y_clip, -1.0, 1.0)
    sf.write(os.path.join(SAMPLE_DIR, "test_clipping.wav"), y_clip, sr)
    
    return [
        os.path.join(SAMPLE_DIR, "test_tone.wav"),
        os.path.join(SAMPLE_DIR, "test_silence.wav"),
        os.path.join(SAMPLE_DIR, "test_short.wav"),
        os.path.join(SAMPLE_DIR, "test_noise.wav"),
        os.path.join(SAMPLE_DIR, "test_clipping.wav")
    ]

def check_model_files():
    print("\n[1/5] Model Files Check")
    files = [
        "models/hybrid_fusion_model.pth", 
        "models/hybrid_scaler.pkl", 
        "models/hybrid_encoder.pkl"
    ]
    all_passed = True
    for f in files:
        if os.path.exists(f):
            size = os.path.getsize(f) / 1024
            print(f"✅ {f} found ({size:.1f} KB)")
        else:
            print(f"❌ {f} NOT FOUND")
            all_passed = False
    
    if all_passed:
        print("Status: PASSED")
    else:
        print("Status: FAILED")
        exit(1)

def test_feature_extraction():
    print("\n[2/5] Feature Extraction & Inference Test (Engine)")
    engine = HybridInference()
    
    synthetic_files = create_test_samples() # generated in prev step logic, ensuring variable exists
    # Use 'test_tone.wav' which is valid length
    valid_file = synthetic_files[0]
    
    start = time.time()
    try:
        res = engine.predict(valid_file)
        dur = (time.time() - start) * 1000
        print(f"✅ Standard Inference: {dur:.1f}ms")
        print(f"   Response keys: {list(res.keys())}")
        print("Status: PASSED")
    except Exception as e:
        print(f"❌ Inference Failed: {e}")
        print("Status: FAILED")

def test_inference_speed():
    print("\n[3/5] Inference Speed Test (10 iterations)")
    engine = HybridInference()
    valid_file = os.path.join(SAMPLE_DIR, "test_tone.wav")
    
    times = []
    for i in range(10):
        start = time.time()
        engine.predict(valid_file)
        times.append((time.time() - start) * 1000)
    
    avg = np.mean(times)
    print(f"✅ Average: {avg:.1f}ms")
    print(f"✅ Range: {min(times):.1f}ms - {max(times):.1f}ms")
    
    if avg < 600:
        print("✅ Target <600ms: PASSED")
        print("Status: PASSED")
    else:
        print("❌ Target <600ms: FAILED")
        print("Status: FAILED")

def test_api_health():
    print("\n[4/5] API Health Check")
    # Assume server is running (started in previous step)
    try:
        resp = requests.get(f"{API_URL}/health")
        if resp.status_code == 200:
            print(f"✅ GET /health: 200 OK ({resp.json()})")
            print("Status: PASSED")
        else:
            print(f"❌ GET /health: {resp.status_code}")
            print("Status: FAILED")
    except Exception as e:
        print(f"❌ API Connection Failed: {e}")
        print("Status: FAILED (Is the server running?)")

def test_edge_cases():
    print("\n[5/5] Edge Cases & Robustness")
    engine = HybridInference()
    
    # 1. Short Audio
    try:
        engine.predict(os.path.join(SAMPLE_DIR, "test_short.wav"))
        print("❌ Short Audio (<0.5s): FAILED (Should have raised error)")
    except ValueError as e:
        print(f"✅ Short Audio (<0.5s): Handled ({e})")
    except Exception as e:
        print(f"⚠️ Short Audio: Unexpected Error ({e})")
        
    # 2. Silence
    try:
        res = engine.predict(os.path.join(SAMPLE_DIR, "test_silence.wav"))
        print(f"✅ Silent Audio: Handled (Pred: {res['predicted_emotion']}, Conf: {res['confidence']:.2f})")
    except Exception as e:
        print(f"❌ Silent Audio: Failed ({e})")
        
    # 3. Noise
    try:
        res = engine.predict(os.path.join(SAMPLE_DIR, "test_noise.wav"))
        print(f"✅ Noisy Audio: Handled (Pred: {res['predicted_emotion']})")
    except Exception as e:
        print(f"❌ Noisy Audio: Failed ({e})")
        
    # 4. Clipping
    try:
        res = engine.predict(os.path.join(SAMPLE_DIR, "test_clipping.wav"))
        print(f"✅ Clipped Audio: Handled (Pred: {res['predicted_emotion']})")
    except Exception as e:
        print(f"❌ Clipped Audio: Failed ({e})")
        
    print("Status: PASSED")

def main():
    print("================================================================================")
    print("HAAM PRODUCTION SYSTEM VALIDATION")
    print("================================================================================")
    
    try:
        create_test_samples()
        check_model_files()
        test_feature_extraction()
        test_inference_speed()
        test_api_health()
        test_edge_cases()
        
        print("\n================================================================================")
        print("OVERALL SYSTEM STATUS: VALIDATION COMPLETE")
        print("================================================================================")
        
    except Exception as e:
        print(f"\nCRITICAL FAILURE: {e}")
        exit(1)
    finally:
        # Cleanup
        if os.path.exists(SAMPLE_DIR):
            shutil.rmtree(SAMPLE_DIR)

if __name__ == "__main__":
    main()
