import pandas as pd
from pathlib import Path
import os
import sys

def robust_find_file(base_dir, target_filename):
    print(f"Scanning {base_dir} for {target_filename}...")
    try:
        if not os.path.exists(base_dir):
            print(f"  Base dir {base_dir} does not exist!")
            return None
            
        for root, dirs, files in os.walk(base_dir, topdown=True):
            if target_filename in files:
                return os.path.join(root, target_filename)
        print("  Not found in walk.")
    except Exception as e:
        print(f"  Error: {e}")
    return None

def main():
    print("DEBUG PATH FINDING")
    
    try:
        df = pd.read_csv('data/hybrid_metadata.csv')
    except:
        print("No metadata")
        return

    # Check first CREMA
    crema = df[df['dataset']=='CREMA-D'].iloc[0]
    print(f"\nTesting CREMA-D: {crema.to_dict()}")
    base = 'd:/haam_framework/crema-d-mirror-main'
    fname = crema.get('filename', str(crema['call_id']))
    if not fname.endswith('.wav'): fname += '.wav'
    
    found = robust_find_file(base, fname)
    print(f"Found: {found}")
    
    # Check first IEMOCAP
    iemocap = df[df['dataset']=='IEMOCAP'].iloc[0]
    print(f"\nTesting IEMOCAP: {iemocap.to_dict()}")
    base_i = 'd:/haam_framework/iemocapfullrelease'
    fname_i = iemocap.get('filename', str(iemocap['call_id']))
    if not fname_i.endswith('.wav'): fname_i += '.wav'
    
    found_i = robust_find_file(base_i, fname_i)
    print(f"Found: {found_i}")

if __name__ == '__main__':
    main()
