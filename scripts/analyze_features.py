
import os
import glob
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import random
from tqdm import tqdm

CREMAD_DIR = r"D:\haam_framework\crema-d-mirror-main\AudioWAV" # Use source for ground truth filenames
CSV_OUTPUT = r"D:\haam_framework\docs\feature_analysis.csv"

def extract_features(audio_path):
    try:
        y, sr = sf.read(audio_path)
        if len(y.shape) > 1:
            y = y.mean(axis=1)
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000
            
        # PITCH
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=75, fmax=600)
        threshold = np.median(magnitudes)
        pitch_values = pitches[magnitudes > threshold]
        pitch_values = pitch_values[pitch_values > 0]
        pitch_mean = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0
        pitch_std = float(np.std(pitch_values)) if len(pitch_values) > 0 else 0.0
        
        # ENERGY
        energy = librosa.feature.rms(y=y)
        energy_mean = float(np.mean(energy))
        
        # ZCR
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = float(np.mean(zcr))
        
        # SPECTRAL CENTROID
        sc = librosa.feature.spectral_centroid(y=y, sr=sr)
        sc_mean = float(np.mean(sc))
        
        return {
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std,
            "energy_mean": energy_mean,
            "zcr_mean": zcr_mean,
            "sc_mean": sc_mean
        }
    except Exception as e:
        print(f"Error {audio_path}: {e}")
        return None

def main():
    files = glob.glob(os.path.join(CREMAD_DIR, "*.wav"))
    
    # Check if we have files in the sample dir, if not use source
    if not files:
        print("No samples found in data/cremad_samples. Creating mock list from previous runs.")
        # Fallback to source if samples dir is empty/wrong
        files = glob.glob(r"D:\haam_framework\crema-d-mirror-main\AudioWAV\*.wav")
    
    print(f"Found {len(files)} files.")
    
    # Map emotion codes
    code_map = {
        'ANG': 'anger',
        'SAD': 'sadness',
        'HAP': 'joy',
        'FEA': 'fear',
        'DIS': 'disgust',
        'NEU': 'neutral'
    }
    
    # Inspect 180 files (30 per emotion ideally, or just random)
    random.shuffle(files)
    selected_files = files[:180]
    
    results = []
    
    print("Extracting features...")
    for f in tqdm(selected_files):
        # Filename format: 1001_DFA_ANG_XX.wav
        basename = os.path.basename(f)
        parts = basename.split('_')
        if len(parts) >= 3:
            emo_code = parts[2]
            emotion = code_map.get(emo_code, 'unknown')
        else:
            emotion = 'unknown'
            
        feats = extract_features(f)
        if feats:
            feats['emotion'] = emotion
            feats['filename'] = basename
            results.append(feats)
            
    df = pd.DataFrame(results)
    df.to_csv(CSV_OUTPUT, index=False)
    
    # Print summary stats per emotion
    print("\nFeature Averages by Emotion:")
    print(df.groupby('emotion')[['pitch_mean', 'energy_mean', 'zcr_mean', 'sc_mean']].mean())
    
    print("\nStarting ranges (Min-Max) for pitching tuning:")
    for emo in df['emotion'].unique():
        subset = df[df['emotion'] == emo]
        print(f"\n{emo.upper()}:")
        print(subset[['pitch_mean', 'energy_mean', 'zcr_mean', 'sc_mean']].describe().loc[['mean', 'min', 'max']])

if __name__ == "__main__":
    main()
