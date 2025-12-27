"""
Split Metadata into Train/Val/Test.
Ensures data/real_metadata_with_split.csv is available for feature processing.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os

def main():
    input_path = 'data/real_metadata.csv'
    output_path = 'data/real_metadata_with_split.csv'
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Run scripts/02a_generate_real_metadata.py first.")
        return

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} records.")
    
    # Filter to 5 core emotions if needed, or keep all and filter in training
    # For consistency with Phase 1, we keep all and let trainer handle labels.
    
    # Stratified Split
    print("Creating stratified split (Train: 72%, Val: 8%, Test: 20%)...")
    try:
        # First split off Test (20%)
        train_val, test = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df['dataset']
        )
        # Then split off Val (10% of 80% = 8%)
        train, val = train_test_split(
            train_val, test_size=0.1, random_state=42
        )
        
        df.loc[train.index, 'split'] = 'train'
        df.loc[val.index, 'split'] = 'val'
        df.loc[test.index, 'split'] = 'test'
        
    except Exception as e:
        print(f"Stratification failed ({e}), falling back to simple split.")
        train_val, test = train_test_split(df, test_size=0.2, random_state=42)
        train, val = train_test_split(train_val, test_size=0.1, random_state=42)
        df.loc[train.index, 'split'] = 'train'
        df.loc[val.index, 'split'] = 'val'
        df.loc[test.index, 'split'] = 'test'

    df.to_csv(output_path, index=False)
    print(f"Saved split metadata to {output_path}")
    print(df['split'].value_counts())

if __name__ == "__main__":
    main()
