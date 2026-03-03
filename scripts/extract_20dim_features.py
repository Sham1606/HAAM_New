import os
import sys
import numpy as np
import json
import csv
import librosa
from tqdm import tqdm

# Add src to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.features.improved_acoustic import ImprovedAcousticExtractor

# Configuration
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IEMOCAP_ROOT = r"d:\haam\HAAM_New\data\IEMOCAP_full_release"
CREMAD_ROOT = r"d:\haam\HAAM_New\data\CREMA-D"
OUTPUT_DIR = os.path.join(DATA_DIR, "processed", "features_20dim")
CREMAD_MAPPING_FILE = os.path.join(DATA_DIR, "cremad_ground_truth.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_cremad_mapping():
    """Builds a map from call_id -> original_filename using ground_truth.csv"""
    mapping = {}
    if not os.path.exists(CREMAD_MAPPING_FILE):
        print(f"Warning: Mapping file not found at {CREMAD_MAPPING_FILE}")
        return mapping
        
    print(f"Loading CREMA-D mapping from {CREMAD_MAPPING_FILE}...")
    with open(CREMAD_MAPPING_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            call_id = row['call_id']
            # Reconstruct filename: 1001_DFA_ANG_XX.wav
            actor = row['actor_id']
            sentence = row['sentence']
            emo = row['emotion_code']
            intensity = row['intensity']
            filename = f"{actor}_{sentence}_{emo}_{intensity}.wav"
            mapping[call_id] = filename
    print(f"Loaded {len(mapping)} mappings.")
    return mapping

def find_iemocap_file(utterance_id):
    """Finds audio file for IEMOCAP utterance_id (SessionX/...)"""
    # Format: Ses01F_impro01_F000
    # Format: Ses01F_impro01_F000
    session_code = utterance_id[:5] # Ses01
    # Map Ses01 -> Session1
    session_map = {
        'Ses01': 'Session1', 'Ses02': 'Session2', 'Ses03': 'Session3', 'Ses04': 'Session4', 'Ses05': 'Session5'
    }
    session = session_map.get(session_code)
    if not session: return None
    
    # Heuristic to find the subfolder
    parts = utterance_id.split('_')
    if len(parts) >= 2:
        subfolder = '_'.join(parts[:-1]) # Ses01F_impro01
        
        # Try logical path first
        path_guess = os.path.join(IEMOCAP_ROOT, session, "sentences", "wav", subfolder, f"{utterance_id}.wav")
        if os.path.exists(path_guess):
            return path_guess

    return None

def process_datasets():
    extractor = ImprovedAcousticExtractor()
    cremad_map = load_cremad_mapping()
    
    processed_count = 0
    errors = 0
    skips = 0

    # Define datasets to process based on where metadata is stored
    # IEMOCAP metadata is in results/calls_iemocap
    # CREMA-D metadata is in results/calls
    
    datasets = [
        {'name': 'IEMOCAP', 'dir': os.path.join(RESULTS_DIR, "calls_iemocap")},
        {'name': 'CREMA-D', 'dir': os.path.join(RESULTS_DIR, "calls_cremad")}
    ]
    
    for ds in datasets:
        dataset_name = ds['name']
        dir_path = ds['dir']
        
        if not os.path.exists(dir_path):
            print(f"Skipping {dataset_name} (directory not found: {dir_path})")
            continue
            
        print(f"Processing {dataset_name} from {dir_path}...")
        files = [f for f in os.listdir(dir_path) if f.endswith('.json')]
        
        for fname in tqdm(files):
            json_path = os.path.join(dir_path, fname)
            
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                audio_path = None
                out_name = None
                
                if dataset_name == 'IEMOCAP':
                    # Get utterance ID
                    utt_id = data.get('metadata', {}).get('utterance_id') or data.get('utterance_id')
                    if not utt_id: continue
                    out_name = utt_id
                    
                    # Check if already processed
                    if os.path.exists(os.path.join(OUTPUT_DIR, f"{out_name}.npy")):
                        skips += 1
                        continue

                    # Find file
                    audio_path = find_iemocap_file(utt_id)
                    
                else: # CREMA-D
                    call_id = data.get('call_id')
                    if not call_id: continue
                    out_name = call_id
                    
                    if os.path.exists(os.path.join(OUTPUT_DIR, f"{out_name}.npy")):
                        skips += 1
                        continue

                    # Use original filename from JSON
                    orig_filename = data.get('original_filename')
                    if orig_filename:
                        audio_path = os.path.join(CREMAD_ROOT, orig_filename)

                # Extract
                if audio_path and os.path.exists(audio_path):
                    # Load audio (sr=16000 default)
                    y, sr = librosa.load(audio_path, sr=16000)
                    
                    # Extract 20-dim features as a proper NumPy array
                    features = extractor.extract_array(y, sr) 
                    
                    # Save
                    np.save(os.path.join(OUTPUT_DIR, f"{out_name}.npy"), features)
                    processed_count += 1
                else:
                    # Log missing files
                    if dataset_name == 'CREMA-D' and call_id in cremad_map:
                         # print(f"Missing audio: {audio_path}")
                         errors += 1
                    elif dataset_name == 'IEMOCAP' and out_name:
                         # print(f"Missing audio: {out_name}")
                         errors += 1
                    
            except Exception as e:
                with open("extraction_errors.log", "a") as log:
                    log.write(f"Error processing {fname}: {e}\n")
                    import traceback
                    traceback.print_exc(file=log)
                errors += 1

    print(f"\nCompleted extraction.")
    print(f"Processed: {processed_count}")
    print(f"Skipped (already exist): {skips}")
    print(f"Errors/Missing: {errors}")

if __name__ == "__main__":
    process_datasets()
