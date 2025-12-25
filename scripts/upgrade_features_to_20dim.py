"""
Upgrade Features to 20-dim (v2)
Adds 8 MFCC coefficients to existing 12-dim acoustic features.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import logging
from src.features.improved_acoustic import LibrosaAcousticExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/upgrade_20dim.log"),
        logging.StreamHandler()
    ]
)

def main():
    print("="*80)
    print("PHASE 10: UPGRADING FEATURES TO 20-DIM (v2)")
    print("="*80)
    
    # Paths
    metadata_path = 'data/real_metadata_with_split.csv'
    original_dir = Path('data/processed/features_v3_librosa')
    output_dir = Path('data/processed/features_v4_20dim')
    output_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Initialize v2 Extractor
    extractor = LibrosaAcousticExtractor(sr=16000, version='v2')
    
    # Load Metadata
    if not os.path.exists(metadata_path):
        logging.error(f"Metadata not found: {metadata_path}")
        return
        
    df = pd.read_csv(metadata_path)
    logging.info(f"Processing {len(df)} samples...")
    
    success = 0
    fail = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Upgrading to 20D"):
        call_id = row['call_id']
        audio_path = row['filepath']
        out_path = output_dir / f"{call_id}.pt"
        
        if out_path.exists():
            success += 1
            continue
            
        try:
            # 1. Load existing data (to keep transcript/text embeddings)
            in_path = original_dir / f"{call_id}.pt"
            if not in_path.exists():
                logging.warning(f"Feature file missing for {call_id}. Skipping.")
                fail += 1
                continue
            
            data = torch.load(in_path, weights_only=False)
            
            # 2. Extract 20-dim features
            if not os.path.exists(audio_path):
                logging.error(f"Audio missing: {audio_path}")
                fail += 1
                continue
                
            new_acoustic = extractor.extract_array(audio_path)
            
            # Verify dimension
            if len(new_acoustic) != 20:
                logging.error(f"Dimension mismatch for {call_id}: got {len(new_acoustic)}")
                fail += 1
                continue
                
            # 3. Update and tag
            data['acoustic'] = new_acoustic
            data['version'] = 'v2'
            data['acoustic_dim'] = 20
            
            # 4. Save
            torch.save(data, out_path)
            success += 1
            
        except Exception as e:
            logging.error(f"Error {call_id}: {str(e)}")
            fail += 1
            
    logging.info(f"Upgrade Complete. Success: {success}, Fail: {fail}")
    print(f"\nCompleted Upgrade. Success: {success}, Fail: {fail}")
    print(f"Features saved to {output_dir}")

if __name__ == '__main__':
    main()
