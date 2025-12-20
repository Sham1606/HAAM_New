
import os
import json
import numpy as np
import librosa
import soundfile as sf
import warnings
from tqdm import tqdm
import logging
import joblib
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CALLS_DIR = r"D:\haam_framework\results\calls"
AUDIO_DIR = r"D:\haam_framework\data\cremad_samples"
MODEL_PATH = r"D:\haam_framework\models\acoustic_emotion_model.pkl"
SCALER_PATH = r"D:\haam_framework\models\feature_scaler.pkl"

def load_model():
    logger.info("Loading model and scaler...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def predict_emotion(y, sr, model, scaler):
    try:
        # Extract features EXACTLY as trained
        # Feature cols: ['pitch', 'energy', 'zcr', 'spectral_centroid']
        
        # Pitch
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=75, fmax=600)
        threshold = np.median(magnitudes)
        pitch_values = pitches[magnitudes > threshold]
        pitch_values = pitch_values[pitch_values > 0]
        pitch_mean = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0
        
        # Energy
        energy = librosa.feature.rms(y=y)
        energy_mean = float(np.mean(energy))
        
        # ZCR
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = float(np.mean(zcr))
        
        # Spectral Centroid
        sc = librosa.feature.spectral_centroid(y=y, sr=sr)
        sc_mean = float(np.mean(sc))
        
        # Create vector
        features = np.array([[pitch_mean, energy_mean, zcr_mean, sc_mean]])
        
        # Scale
        features_scaled = scaler.transform(features)
        
        # Predict
        emotion = model.predict(features_scaled)[0]
        probs = model.predict_proba(features_scaled)
        confidence = float(np.max(probs))
        
        return emotion, confidence, pitch_mean
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return "neutral", 0.0, 0.0

def process_file(json_path, model, scaler):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        call_id = data['call_id']
        audio_filename = f"{call_id}.wav"
        audio_path = os.path.join(AUDIO_DIR, audio_filename)
        
        if not os.path.exists(audio_path):
            return False
            
        # Load Audio
        y, sr = sf.read(audio_path)
        if len(y.shape) > 1: y = y.mean(axis=1)
        if sr != 16000: y = librosa.resample(y, orig_sr=sr, target_sr=16000); sr=16000
        
        # Predict
        acoustic_emotion, acoustic_conf, pitch_mean = predict_emotion(y, sr, model, scaler)
        
        # Update JSON
        # Since CREMA-D text is neutral, we override with acoustic prediction
        for seg in data['segments']:
            seg['emotion'] = acoustic_emotion
            seg['emotion_confidence'] = round(acoustic_conf, 3)
            seg['pitch_mean'] = round(pitch_mean, 2)
            
        # Recalculate Overall Metrics
        counts = {}
        processed_segments = data['segments']
        for s in processed_segments:
            e = s['emotion']
            counts[e] = counts.get(e, 0) + 1
            
        total = len(processed_segments)
        dist = {k: round(v/total, 3) for k,v in counts.items()} if total > 0 else {}
        dominant = max(counts, key=counts.get) if counts else "neutral"
        
        data['overall_metrics']['emotion_distribution'] = dist
        data['overall_metrics']['dominant_emotion'] = dominant
        data['overall_metrics']['avg_pitch'] = round(pitch_mean, 2)
        
        # Save
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to process {json_path}: {e}")
        return False

def main():
    model, scaler = load_model()
    files = [os.path.join(CALLS_DIR, f) for f in os.listdir(CALLS_DIR) if f.endswith('.json')]
    print(f"Reprocessing {len(files)} files with ML model...")
    
    success_count = 0
    for json_file in tqdm(files):
        if process_file(json_file, model, scaler):
            success_count += 1
            
    print(f"Successfully reprocessed {success_count}/{len(files)} files.")

if __name__ == "__main__":
    main()
