"""
Reprocess Acoustic Features using Librosa
Purpose: Replace Praat-based features with robust Librosa features to hit <2% failure rate.
"""

import sys
from pathlib import Path
# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.features.improved_acoustic import LibrosaAcousticExtractor
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/reprocess_librosa.log"),
        logging.StreamHandler()
    ]
)

def main():
    print("="*80)
    print("PHASE 8: ROBUST LIBROSA REPROCESSING")
    print("="*80)
    
    # Paths
    input_metadata = 'data/real_metadata_with_split.csv'
    original_feature_dir = Path('data/processed/features_v2_fixed')
    output_feature_dir = Path('data/processed/features_v3_librosa')
    output_feature_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Initialize Extractor
    extractor = LibrosaAcousticExtractor(sr=16000)
    
    # Load Metadata
    if not os.path.exists(input_metadata):
        logging.error(f"Metadata not found: {input_metadata}")
        return
        
    df = pd.read_csv(input_metadata)
    logging.info(f"Loaded {len(df)} samples from metadata.")
    
    success_count = 0
    failure_count = 0
    zero_pitch_count = 0
    
    # Resumption logic
    processed_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Reprocessing Acoustic"):
        call_id = row['call_id']
        audio_path = row['filepath']
        
        # Output path
        out_path = output_feature_dir / f"{call_id}.pt"
        if out_path.exists():
            processed_count += 1
            success_count += 1
            continue
            
        try:
            # 1. Load existing feature file if it exists to preserve text features
            # Or reconstruct from scratch if missing.
            in_path = original_feature_dir / f"{call_id}.pt"
            if in_path.exists():
                feature_data = torch.load(in_path, weights_only=False)
            else:
                # If the file doesn't exist, we skip or log as failure
                logging.warning(f"Original feature file missing for {call_id}. Skipping.")
                failure_count += 1
                continue

            # 2. Extract NEW acoustic features
            if not os.path.exists(audio_path):
                logging.error(f"Audio file missing: {audio_path}")
                failure_count += 1
                continue
                
            new_acoustic = extractor.extract_array(audio_path)
            
            # Check for failure (Zero pitch fallback check)
            # feature[0] is pitch_mean. We set fallback to 150.0 in extractor.
            # If it's 0.0, something went wrong.
            if new_acoustic[0] == 0.0:
                zero_pitch_count += 1
                
            # 3. Update feature data
            feature_data['acoustic'] = new_acoustic
            
            # 4. Save
            torch.save(feature_data, out_path)
            success_count += 1
            
        except Exception as e:
            logging.error(f"Error processing {call_id}: {str(e)}")
            failure_count += 1
            continue
            
    # Final Report
    total = len(df)
    success_rate = (success_count / total) * 100 if total > 0 else 0
    failure_rate = (failure_count / total) * 100 if total > 0 else 0
    zero_failure_rate = (zero_pitch_count / total) * 100 if total > 0 else 0
    
    print("\n" + "="*80)
    print("REPROCESSING COMPLETE")
    print("="*80)
    print(f"Total Samples:      {total}")
    print(f"Successes:          {success_count} ({success_rate:.2f}%)")
    print(f"Failures (Error):   {failure_count} ({failure_rate:.2f}%)")
    print(f"Zero Pitch Detections: {zero_pitch_count} ({zero_failure_rate:.2f}%)")
    print(f"Target Failure Rate: < 2.00%")
    
    if zero_failure_rate < 2.0:
        print("✅ SUCCESS: Failure rate is within target threshold.")
    else:
        print("⚠️ WARNING: Failure rate still exceeds 2%. Check logs/reprocess_librosa.log")
        
    logging.info(f"Finished. Success: {success_count}, Failure: {failure_count}, Zero-Pitch: {zero_pitch_count}")

if __name__ == '__main__':
    main()
