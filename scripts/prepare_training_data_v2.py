import os
import sys
import pickle
import numpy as np
import pandas as pd
import glob
import re
from pathlib import Path
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add project root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.features.improved_acoustic import ImprovedAcousticExtractor
# Text Extractor imported later to avoid Fork issues with CUDA

# Configuration
CREMA_DIR = Path(r"D:\haam_framework\crema-d-mirror-main\AudioWAV")
IEMOCAP_ROOT = Path(r"D:\haam_framework\iemocapfullrelease\1\IEMOCAP_full_release")
OUTPUT_FILE = Path("data/processed_features_v2.pkl")
TEMP_DIR = Path("data/temp_features")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Mappings (Same as before)
IEMOCAP_MAP = {
    'neu': 'neutral', 'hap': 'neutral', 'sad': 'sadness', 'ang': 'anger',
    'exc': 'neutral', 'fru': 'anger', 'fea': 'fear', 'sur': 'neutral',
    'dis': 'disgust', 'oth': 'neutral', 'xxx': 'neutral'
}
CREMA_MAP = {
    'ANG': 'anger', 'HAP': 'neutral', 'SAD': 'sadness',
    'NEU': 'neutral', 'FEA': 'fear', 'DIS': 'disgust'
}

# --- Parsing Helpers ---
def parse_iemocap_labels(session_dir):
    labels = {}
    label_dir = session_dir / "dialog" / "EmoEvaluation"
    if label_dir.exists():
        for label_file in label_dir.glob("*.txt"):
            with open(label_file, 'r') as f:
                content = f.read()
                pattern = re.compile(r"\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s+(Ses0\d[MF]_[A-Za-z0-9_]+)\s+([a-z]{3})\s+\[")
                for m in pattern.findall(content):
                    labels[m[2]] = IEMOCAP_MAP.get(m[3], 'neutral')
    return labels

def get_iemocap_transcript(session_dir, dialogue_id, utt_id):
    trans_file = session_dir / "dialog" / "transcriptions" / f"{dialogue_id}.txt"
    if trans_file.exists():
        with open(trans_file, 'r') as f:
            for line in f:
                if line.startswith(f"{utt_id} ["):
                    return line.split("]: ")[1].strip() if "]: " in line else ""
    return ""

# --- Worker Function for Acoustic Extraction (CPU) ---
def process_acoustic_file(args):
    """
    Extract acoustic features for a single file.
    Running in a separate process.
    """
    file_path, dataset, label, transcript = args
    try:
        # Initialize extractor independently in each process
        ext = ImprovedAcousticExtractor()
        feats = ext.extract(str(file_path))
        return {
            'file': Path(file_path).name,
            'dataset': dataset,
            'label': label,
            'transcript': transcript,
            'features': feats,
            'success': True
        }
    except Exception as e:
        return {'file': Path(file_path).name, 'success': False, 'error': str(e)}

# --- Batch Text Extraction (GPU) ---
def process_text_batches(records):
    """
    Process transcripts in batches on GPU.
    """
    import torch
    from src.features.emotion_text import EmotionTextExtractor
    
    print("\nInitializing Text Extractor on GPU...")
    text_ext = EmotionTextExtractor()
    
    batch_size = 32
    results = []
    
    # Filter valid records
    valid_records = [r for r in records if r['success']]
    
    print(f"Extracting text features for {len(valid_records)} samples...")
    
    for i in tqdm(range(0, len(valid_records), batch_size), desc="Text Batch Processing"):
        batch = valid_records[i:i+batch_size]
        transcripts = [r.get('transcript', "Speech audio.") for r in batch]
        # Replace empty transcripts
        transcripts = [t if t and len(t) > 2 else "Speech audio." for t in transcripts]
        
        # We need batch support in extractor or loop. 
        # Since EmotionTextExtractor.extract is single-item, we call it in loop here.
        # But since we are on Main Process now, it is fast on GPU.
        # Alternatively, update EmotionTextExtractor to handle list.
        # For now, looping with GPU model loaded is standard inference.
        
        for idx, item in enumerate(batch):
            try:
                txt_res = text_ext.extract(transcripts[idx])
                item['text_features'] = txt_res['emotion_probs']
                item['text_embedding'] = txt_res['embedding']
                results.append(item)
            except Exception as e:
                item['text_features'] = {}
                item['text_embedding'] = np.zeros(768)
                results.append(item)
                
    return results

def main():
    print("="*80)
    print("OPTIMIZED DATA PREPARATION (CPU Parallel + GPU Batch)")
    print("="*80)
    
    all_tasks = []
    
    # 1. Gather CREMA-D Tasks
    print("Scanning CREMA-D...")
    crema_files = list(CREMA_DIR.glob("*.wav"))
    # Limit if needed
    # crema_files = crema_files[:100] 
    
    for f in crema_files:
        parts = f.name.split('_')
        if len(parts) >= 3:
            label = CREMA_MAP.get(parts[2])
            if label:
                # CREMA has no transcripts easily available, use placeholder
                all_tasks.append((str(f), 'CREMA-D', label, "Speech audio."))

    # 2. Gather IEMOCAP Tasks
    print("Scanning IEMOCAP...")
    for session_id in range(1, 6):
        session_dir = IEMOCAP_ROOT / f"Session{session_id}"
        if not session_dir.exists(): continue
        
        labels = parse_iemocap_labels(session_dir)
        wav_root = session_dir / "sentences" / "wav"
        
        for dialog_dir in wav_root.iterdir():
            if not dialog_dir.is_dir(): continue
            for wav_file in dialog_dir.glob("*.wav"):
                utt_id = wav_file.stem
                if utt_id in labels:
                    transcript = get_iemocap_transcript(session_dir, dialog_dir.name, utt_id)
                    all_tasks.append((str(wav_file), 'IEMOCAP', labels[utt_id], transcript))

    print(f"Found {len(all_tasks)} total files to process.")
    
    # 3. Parallel Acoustic Extraction (CPU)
    # Using limited workers to avoid paging file error
    # CPU count // 2 or max 4 is safer for Windows + Heavy libraries
    cpu_count = os.cpu_count() or 4
    max_workers = max(1, min(4, int(cpu_count * 0.75)))
    print(f"Starting Acoustic Extraction with {max_workers} processes...")
    
    acoustic_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_acoustic_file, task): task for task in all_tasks}
        
        for future in tqdm(as_completed(futures), total=len(all_tasks), desc="Acoustic Extraction"):
            res = future.result()
            if res['success']:
                acoustic_results.append(res)
                
    print(f"Acoustic done. Valid samples: {len(acoustic_results)}")
    
    # 4. Text Extraction (GPU)
    # Run on main thread to access CUDA
    final_data = process_text_batches(acoustic_results)
    
    # 5. Save
    print(f"Saving {len(final_data)} records to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(final_data, f)
    print("Processing Complete.")

if __name__ == "__main__":
    # Windows support for multiprocessing
    multiprocessing.freeze_support()
    main()
