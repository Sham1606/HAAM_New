
import os
import json
import numpy as np
import librosa
import soundfile as sf
import warnings
from tqdm import tqdm
from datetime import datetime
import logging
import csv
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CALLS_DIR = r"D:\haam_framework\results\calls"
AUDIO_DIR = r"D:\haam_framework\data\cremad_samples"

def extract_features_and_detect_emotion(audio_path):
    try:
        # Load audio efficiently
        y, sr = sf.read(audio_path)
        if len(y.shape) > 1:
            y = y.mean(axis=1)
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000
            
        # --- Feature Extraction ---
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
        
        # --- Heuristics (Data-Driven Tuning Round 2) ---
        emotion_scores = {
            'anger': 0.0,
            'sadness': 0.0,
            'joy': 0.0,
            'fear': 0.0,
            'disgust': 0.0,
            'neutral': 0.25 # slightly lower baseline
        }
        
        # ANGER: Strongest signal is High Energy
        if energy_mean > 0.04: emotion_scores['anger'] += 0.4
        if pitch_mean > 318: emotion_scores['anger'] += 0.2
        if zcr_mean > 0.09: emotion_scores['anger'] += 0.1
        
        # SADNESS: Must be VERY low energy and low ZCR to distinguish from Disgust/Fear
        if energy_mean < 0.008: emotion_scores['sadness'] += 0.5  # Strict containment
        if pitch_mean < 295: emotion_scores['sadness'] += 0.2
        if zcr_mean < 0.065: emotion_scores['sadness'] += 0.2
        
        # FEAR: High ZCR is the differentiator from Sadness/Neutral
        if zcr_mean > 0.09: emotion_scores['fear'] += 0.4
        if pitch_mean > 312: emotion_scores['fear'] += 0.1
        # "Whispered Fear": Low energy but high ZCR
        if energy_mean < 0.02 and zcr_mean > 0.08: emotion_scores['fear'] += 0.3
        
        # DISGUST: Low Energy (but > Sadness) and Mod-High ZCR
        if 0.008 < energy_mean < 0.022: emotion_scores['disgust'] += 0.3
        if 0.075 < zcr_mean < 0.09: emotion_scores['disgust'] += 0.2
        if pitch_mean < 308: emotion_scores['disgust'] += 0.1
        
        # JOY: Mod-High Energy (overlapping Fear/Anger/Neutral) but Pitch specific
        if energy_mean > 0.022: emotion_scores['joy'] += 0.2
        if 312 < pitch_mean < 330: emotion_scores['joy'] += 0.2
        # Penalize if ZCR is too high (Fear/Anger)
        if zcr_mean < 0.09: emotion_scores['joy'] += 0.1
        
        # NEUTRAL: The "catch-all" middle
        if 0.012 < energy_mean < 0.035: emotion_scores['neutral'] += 0.15
        if 0.07 < zcr_mean < 0.085: emotion_scores['neutral'] += 0.15
        if 300 < pitch_mean < 315: emotion_scores['neutral'] += 0.1
        
        max_emotion = max(emotion_scores, key=emotion_scores.get)
        confidence = emotion_scores[max_emotion]
        
        # Normalization (rough)
        total_score = sum(emotion_scores.values())
        if total_score > 0:
            confidence = confidence / total_score
        
        return max_emotion, confidence, pitch_mean

    except Exception as e:
        logger.error(f"Error processing {audio_path}: {e}")
        return "neutral", 0.5, 0.0

def process_file(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        call_id = data['call_id']
        audio_filename = f"{call_id}.wav"
        audio_path = os.path.join(AUDIO_DIR, audio_filename)
        
        if not os.path.exists(audio_path):
            # Try finding without prefix if needed, but standardizing on call_id
            return False
            
        # Re-run acoustic detection
        acoustic_emotion, acoustic_conf, pitch_mean = extract_features_and_detect_emotion(audio_path)
        
        # --- Fusion Logic (Simplified for repair) ---
        # We don't have text confidence stored per se, but we have text emotion in segments?
        # Actually, the JSON has 'segments' with 'emotion' field which was the FUSED result.
        # We need to re-fuse.
        # But we don't have the raw text confidence stored in the segment! 
        # Wait, 'emotion_confidence' in segment is the fused confidence.
        # We lost the original text confidence.
        # Strategy: Assume text confidence was moderate (0.5) or reuse existing confidence as a proxy?
        # Better: Just overwrite with Acoustic Emotion for CREMA-D since we know text is neutral/useless.
        # The prompt said "Text is neutral". So relying 100% on Acoustic for CREMA-D is actually the correct valid strategy.
        # So we will update the segments to use the new Acoustic Emotion.
        
        # Update Segments
        for seg in data['segments']:
            seg['emotion'] = acoustic_emotion
            seg['emotion_confidence'] = round(acoustic_conf, 3)
            # Update pitch if it was 0 or just refresh it
            seg['pitch_mean'] = round(pitch_mean, 2)
            
        # Recalculate Overall Metrics
        # Emotion distribution
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
    files = [os.path.join(CALLS_DIR, f) for f in os.listdir(CALLS_DIR) if f.endswith('.json')]
    print(f"Found {len(files)} JSON files. Extracting features for training...")
    
    # Load ground truth
    ground_truth = {}
    with open(r"D:\haam_framework\data\cremad_ground_truth.csv", 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ground_truth[row['call_id']] = row['expected_emotion'].lower()
    
    features_data = []
    
    for json_path in tqdm(files):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            call_id = data['call_id']
            if call_id not in ground_truth:
                continue
                
            label = ground_truth[call_id]
            audio_filename = f"{call_id}.wav"
            audio_path = os.path.join(AUDIO_DIR, audio_filename)
            
            if not os.path.exists(audio_path):
                continue
                
            y, sr = sf.read(audio_path)
            if len(y.shape) > 1: y = y.mean(axis=1)
            if sr != 16000: y = librosa.resample(y, orig_sr=sr, target_sr=16000); sr=16000
            
            # Extract
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=75, fmax=600)
            threshold = np.median(magnitudes)
            pitch_vals = pitches[magnitudes > threshold]
            pitch_vals = pitch_vals[pitch_vals > 0]
            pitch_mean = float(np.mean(pitch_vals)) if len(pitch_vals) > 0 else 0.0
            
            energy_mean = float(np.mean(librosa.feature.rms(y=y)))
            zcr_mean = float(np.mean(librosa.feature.zero_crossing_rate(y)))
            sc_mean = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
            
            features_data.append({
                "label": label,
                "pitch": pitch_mean,
                "energy": energy_mean,
                "zcr": zcr_mean,
                "spectral_centroid": sc_mean
            })
            
        except Exception as e:
            print(f"Err {json_path}: {e}")

    # Save to CSV
    import pandas as pd
    df = pd.DataFrame(features_data)
    df.to_csv(r"D:\haam_framework\data\cremad_features_full.csv", index=False)
    print(f"Saved {len(df)} feature rows to D:\haam_framework\data\cremad_features_full.csv")

if __name__ == "__main__":
    main()
