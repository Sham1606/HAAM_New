import os, sys, json
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from pathlib import Path

# Add src to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.features.improved_acoustic import ImprovedAcousticExtractor

metadata_path = "data/hybrid_metadata.csv"
output_dir = "data/processed/features_20dim"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(metadata_path)
TARGET_EMOTIONS = ['neutral', 'anger', 'disgust', 'fear', 'sadness']

extractor = ImprovedAcousticExtractor(sr=16000)

print(f"Total rows in metadata: {len(df)}")
# Filter out invalid emotions
df['emotion_clean'] = df.apply(lambda r: (r.get('emotion_true') or r.get('emotion', '')).lower(), axis=1)
df = df[df['emotion_clean'].isin(TARGET_EMOTIONS)]

print(f"Rows after filtering to TARGET_EMOTIONS: {len(df)}")

import multiprocessing

def extract_one(idx_row):
    idx, row = idx_row
    dataset = row['dataset']
    call_id = str(row['call_id'])
    
    if dataset == 'CREMA-D':
        json_path = os.path.join("results/calls_cremad", f"{call_id}.json")
        feature_id = call_id
    else:
        json_path = os.path.join("results/calls_iemocap", f"{call_id}.json")
        feature_id = call_id.replace("iemocap_", "")
        
    out_file = os.path.join(output_dir, f"{feature_id}.npy")
    if os.path.exists(out_file):
        return True

    if not os.path.exists(json_path):
        return False
        
    try:
        with open(json_path) as f:
            data = json.load(f)
        
        audio_path = None
        if dataset == 'CREMA-D':
            orig = data.get('original_filename')
            if orig:
                audio_path = os.path.join("data", "CREMA-D", orig)
        else:
            # iemocap: try to rebuild the path logic from extract_20dim_features
            utt_id = data.get('metadata', {}).get('utterance_id') or data.get('utterance_id')
            if utt_id:
                sess_code = utt_id[:5]
                smap = {'Ses01': 'Session1', 'Ses02': 'Session2', 'Ses03': 'Session3', 'Ses04': 'Session4', 'Ses05': 'Session5'}
                session = smap.get(sess_code)
                if session:
                    parts = utt_id.split('_')
                    if len(parts) >= 2:
                        subfolder = '_'.join(parts[:-1])
                        path_guess = os.path.join("data", "IEMOCAP_full_release", session, "sentences", "wav", subfolder, f"{utt_id}.wav")
                        if os.path.exists(path_guess):
                            audio_path = path_guess

        if audio_path and os.path.exists(audio_path):
            y, sr = librosa.load(audio_path, sr=16000)
            feats = extractor.extract_array(y, sr)
            np.save(out_file, feats)
            return True
            
        return False
    except Exception as e:
        # print("err", e)
        return False

# Setup parallel processing
if __name__ == "__main__":
    rows = list(df.iterrows())
    print(f"Processing {len(rows)} files using {os.cpu_count() or 4} workers...")
    
    with multiprocessing.Pool(processes=min(8, os.cpu_count() or 4)) as pool:
        results = list(tqdm(pool.imap(extract_one, rows), total=len(rows)))
        
    success = sum(1 for r in results if r)
    print(f"Extracted features for {success}/{len(rows)} files.")
