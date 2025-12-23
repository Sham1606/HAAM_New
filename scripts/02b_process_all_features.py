"""
Batch process all files to extract features.
Saves features to data/processed/features_v2/{call_id}.pt
"""

import sys
sys.path.append('.')

from src.preprocessing.audio_preprocessor import AudioPreprocessor
from src.features.improved_acoustic import ImprovedAcousticExtractor
from src.features.emotion_text import EmotionTextExtractor
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import warnings
import whisper
import numpy as np

def main():
    print("="*80)
    print("PHASE 1.5: BATCH FEATURE EXTRACTION")
    print("="*80)
    
    output_dir = Path('data/processed/features_v2')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize
    print("Initializing components...")
    preprocessor = AudioPreprocessor()
    acoustic_extractor = ImprovedAcousticExtractor()
    text_extractor = EmotionTextExtractor()
    whisper_model = whisper.load_model("base")
    
    # Load real metadata
    try:
        df = pd.read_csv('data/real_metadata.csv')
        print(f"Loaded {len(df)} records from real_metadata.csv.")
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return
    
    # Filter out already processed files (Resumable)
    existing_files = set(f.stem for f in output_dir.glob('*.pt'))
    to_process = df[~df['call_id'].isin(existing_files)]
    
    print(f"Found {len(existing_files)} existing files. Processing {len(to_process)} remaining.")
    
    errors = []
    
    # robust_find_file not needed if we have full path in metadata
    for idx, row in tqdm(to_process.iterrows(), total=len(to_process), desc="Processing"):
        call_id = row['call_id']
        found_path = row.get('filepath', '')
        dataset = row['dataset']
        
        # Verify existence
        if not Path(found_path).exists():
             # Try relative if full path fails (e.g. moved folder)
             # But generating fresh metadata implies current paths.
             # Just in case:
             if dataset == 'CREMA-D':
                 # Reconstruct from base
                 pass 
             
             if not Path(found_path).exists():
                errors.append({'call_id': call_id, 'error': f'File not found: {found_path}'})
                continue
            
        try:
            # 1. Preprocess
            audio, sr = preprocessor.preprocess(found_path)
            
            # 2. Acoustic (In-Memory)
            acoustic_features = acoustic_extractor.extract_array(audio, sr=sr)
            
            # 3. Text
            # Transcribe if not in metadata or if we want fresh
            transcript = row.get('transcript', '')
            if pd.isna(transcript) or not transcript:
                # Transcribe (pass audio array directly, ensure float32)
                # Whisper strictly requires float32
                audio_32 = audio.astype(np.float32)
                res = whisper_model.transcribe(audio_32) 
                transcript = res['text'].strip()
            
            text_features = text_extractor.extract(transcript)
            
            # Save
            data = {
                'call_id': call_id,
                'acoustic': acoustic_features, # numpy array
                'text_embedding': text_features['embedding'], # numpy array
                'emotion_probs': text_features['emotion_probabilities'],
                'transcript': transcript,
                'emotion': row['emotion'],
                'dataset': dataset
            }
            
            torch.save(data, output_dir / f"{call_id}.pt")
            
        except Exception as e:
            print(f"Error processing {call_id}: {e}")
            errors.append({'call_id': call_id, 'error': str(e)})
    
    if errors:
        pd.DataFrame(errors).to_csv('results/processing_errors.csv', index=False)
        print(f"Finished with {len(errors)} errors. See results/processing_errors.csv")
    else:
        print("Finished successfully.")

if __name__ == '__main__':
    main()
