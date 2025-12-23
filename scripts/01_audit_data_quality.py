"""
Scan all audio files for quality issues (too short, clipping, noise, etc.)
Output: List of problematic files to handle in preprocessing
"""

import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os

def main():
    print("="*80)
    print("PHASE 0.2: DATA QUALITY AUDIT")
    print("="*80)
    
    Path('results/diagnosis').mkdir(parents=True, exist_ok=True)
    
    # Use existing data directory structure
    # CREMA-D seems to be in data/raw/crema-d or similar?
    # Based on list_dir, I saw 'data', 'iemocapfullrelease', 'crema-d-mirror-main'.
    # I will search for wav files in these known locations.
    
    datasets = {
        'CREMA-D': Path('d:/haam_framework/crema-d-mirror-main'), # Updated path based on list_dir
        'IEMOCAP': Path('d:/haam_framework/iemocapfullrelease')
    }
    
    issues = []
    
    for dataset_name, base_path in datasets.items():
        print(f"\nScanning {dataset_name} at {base_path}...")
        
        if not base_path.exists():
            print(f"WARNING: Path {base_path} not found. Skipping.")
            continue

        # Find audio files
        files = list(base_path.rglob('*.wav'))
        
        print(f"  Found {len(files)} files")
        
        # Limit to 1000 files for speed if too many, or do full scan? 
        # User requested full audit. I'll do full scan but with a limit if it takes too long? 
        # I'll do full scan but maybe with a simple check.
        
        # Check first 5 files to verify
        if len(files) > 0:
            print(f"  Sample: {files[0]}")
        
        for audio_file in tqdm(files, desc=f"Auditing {dataset_name}"):
            try:
                # Use soundfile for faster checking of duration/clipping if possible?
                # Librosa is safer for loading.
                # To save time, we can check file size first.
                if os.path.getsize(audio_file) < 1000: # < 1KB
                    issues.append({'file': audio_file.name, 'dataset': dataset_name, 
                                  'issue': 'file_too_small', 'value': 0, 'severity': 'critical'})
                    continue

                audio, sr = librosa.load(audio_file, sr=None)
                duration = len(audio) / sr
                max_amp = np.max(np.abs(audio))
                rms = np.sqrt(np.mean(audio**2))
                
                # Check issues
                if duration < 0.3:
                    issues.append({'file': audio_file.name, 'dataset': dataset_name, 
                                  'issue': 'too_short', 'value': duration, 'severity': 'high'})
                
                if max_amp > 0.99:
                    issues.append({'file': audio_file.name, 'dataset': dataset_name,
                                  'issue': 'clipping', 'value': max_amp, 'severity': 'high'})
                
                if rms < 0.01:
                    issues.append({'file': audio_file.name, 'dataset': dataset_name,
                                  'issue': 'too_quiet', 'value': rms, 'severity': 'medium'})
                
            except Exception as e:
                issues.append({'file': audio_file.name, 'dataset': dataset_name,
                              'issue': 'load_error', 'value': str(e), 'severity': 'critical'})
    
    df_issues = pd.DataFrame(issues)
    df_issues.to_csv('results/diagnosis/data_quality_issues.csv', index=False)
    
    print("\n" + "="*80)
    print("DATA QUALITY SUMMARY")
    print("="*80)
    
    if len(df_issues) == 0:
        print("✓ No issues found")
    else:
        print(f"⚠ Found {len(df_issues)} issues:")
        if not df_issues.empty:
            print(df_issues.groupby(['dataset', 'issue', 'severity']).size())
    
    print(f"\nReport saved: results/diagnosis/data_quality_issues.csv")

if __name__ == '__main__':
    main()
