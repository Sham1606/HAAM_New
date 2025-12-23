import sys
import os
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing.audio_preprocessor import AudioPreprocessor

def test_preprocessing():
    print("="*80)
    print("TESTING AUDIO PREPROCESSING")
    print("="*80)
    
    # 1. Setup
    preprocessor = AudioPreprocessor(target_sr=16000)
    
    # Sample files from real datasets
    crema_dir = Path(r"D:\haam_framework\crema-d-mirror-main\AudioWAV")
    iemocap_dir = Path(r"D:\haam_framework\iemocapfullrelease\1\IEMOCAP_full_release")
    
    # Pick 5 files from each
    test_files = []
    
    if crema_dir.exists():
        test_files.extend(list(crema_dir.glob('*.wav'))[:5])
    
    if iemocap_dir.exists():
        test_files.extend(list(iemocap_dir.rglob('*.wav'))[:5])
        
    print(f"Found {len(test_files)} test files.")
    
    results = []
    
    for f in test_files:
        print(f"Processing: {f.name}")
        audio, sr, meta = preprocessor.preprocess(str(f))
        
        status = "OK"
        if not audio is not None:
            status = "FAILED"
        elif meta.get('warnings'):
            status = f"WARNING: {meta['warnings']}"
            
        print(f"  Result: {status}")
        if meta.get('steps'):
            print(f"  Steps: {', '.join(meta['steps'])}")
        print(f"  Duration: {meta.get('original_duration', 0):.2f}s -> {meta.get('final_duration', 0):.2f}s")
        print("-" * 40)
        
        results.append(meta)
        
    # Analysis
    df = pd.DataFrame(results)
    if 'warnings' in df.columns:
        print("\nWarnings Summary:")
        print(df['warnings'].value_counts())
        
    print("\nTest Complete.")

if __name__ == "__main__":
    test_preprocessing()
