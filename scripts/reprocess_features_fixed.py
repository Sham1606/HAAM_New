"""
Reprocess ALL features with fixed extractor and unified text/audio pipeline.
This rebuilds the 10k sample dataset (v2_fixed).
"""

import sys
from pathlib import Path
# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.features.fixed_acoustic_extractor import RobustAcousticExtractor
from src.features.emotion_text import EmotionTextExtractor
import pandas as pd
import torch
import torch.nn as nn
import whisper
import numpy as np
from tqdm import tqdm
import os
import gc

def main():
    print("="*80)
    print("REPROCESSING FEATURES (v2_fixed)")
    print("="*80)
    
    # Create output directory
    output_dir = Path('data/processed/features_v2_fixed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize extractors
    print("Initializing extractors...")
    acoustic_extractor = RobustAcousticExtractor(sr=16000)
    text_extractor = EmotionTextExtractor()
    whisper_model = whisper.load_model("base")
    
    # Load metadata
    metadata_path = 'data/real_metadata_with_split.csv'
    if not os.path.exists(metadata_path):
        print(f"Error: {metadata_path} not found.")
        return
        
    df = pd.read_csv(metadata_path)
    print(f"Total samples to process: {len(df)}")
    
    # Resumption logic: Skip existing files
    existing_files = set([f.stem for f in output_dir.glob('*.pt')])
    if existing_files:
        print(f"Found {len(existing_files)} existing files. Skipping those.")
        df_to_process = df[~df['call_id'].isin(existing_files)]
    else:
        df_to_process = df
        
    print(f"Remaining samples: {len(df_to_process)}")
    
    success_count = 0
    failure_count = 0
    
    for idx, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc="Processing"):
        try:
            audio_path = row['filepath']
            if not os.path.exists(audio_path):
                # Try fallback for path separators
                audio_path = audio_path.replace('\\', '/')
                if not os.path.exists(audio_path):
                    # print(f"\nFile not found: {audio_path}")
                    failure_count += 1
                    continue
            
            # 1. Acoustic Features (Robust)
            acoustic_features = acoustic_extractor.extract_array(audio_path)
            
            # 2. Transcription (Whisper)
            transcription = whisper_model.transcribe(audio_path)
            transcript_text = transcription['text'].strip()
            
            # 3. Text Features (Emotion Probs + Embedding)
            text_res = text_extractor.extract(transcript_text)
            
            # Create feature dictionary
            # Includes text_emotion_probs for "Bug #2" fix
            features = {
                'acoustic': acoustic_features,
                'text_embedding': text_res['embedding'],
                'text_emotion_probs': np.array([
                    text_res['emotion_probabilities']['neutral'],
                    text_res['emotion_probabilities']['anger'],
                    text_res['emotion_probabilities']['disgust'],
                    text_res['emotion_probabilities']['fear'],
                    text_res['emotion_probabilities']['sadness']
                ], dtype=np.float32),
                'transcript': transcript_text,
                'dominant_emotion_text': text_res['dominant_emotion']
            }
            
            # Save
            torch.save(features, output_dir / f"{row['call_id']}.pt")
            success_count += 1
            
            # Periodically clear memory (whisper/transformers can grow)
            if idx % 100 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        except Exception as e:
            # print(f"\nError processing {row['call_id']}: {e}")
            failure_count += 1
            continue
            
    print("\n" + "="*80)
    print("REPROCESSING COMPLETE")
    print("="*80)
    print(f"Success: {success_count}/{len(df_to_process)}")
    print(f"Failures: {failure_count}")
    print(f"New dataset location: {output_dir}")

if __name__ == '__main__':
    main()
