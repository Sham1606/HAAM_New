"""
Generate REAL metadata by scanning actual files on disk.
Ignores synthetic hybrid_metadata.csv.
"""

import pandas as pd
from pathlib import Path
import os
from tqdm import tqdm

def scan_crema(base_path, emotion_map):
    print(f"Scanning CREMA-D at {base_path}...")
    records = []
    # CREMA-D filenames: 1001_DFA_ANG_XX.flv (or .wav if converted)
    # Emotions: ANG, DIS, FEA, HAP, NEU, SAD
    
    # Check VideoFlash or AudioWAV
    # Based on previous `dir`, valid files are in `VideoFlash` as .flv? 
    # Or did I miss AudioWAV?
    # I'll scan recursively.
    
    files = []
    for root, dirs, filenames in os.walk(base_path):
        for f in filenames:
            if f.endswith('.flv') or f.endswith('.wav'):
                files.append(os.path.join(root, f))
                
    print(f"  Found {len(files)} files")
    
    for fpath in tqdm(files, desc="Parsing CREMA"):
        name = Path(fpath).stem
        parts = name.split('_')
        if len(parts) >= 3:
            emo_code = parts[2] # ANG
            emotion = emotion_map.get(emo_code, None)
            if emotion:
                records.append({
                    'call_id': name,
                    'filename': Path(fpath).name,
                    'filepath': fpath,
                    'dataset': 'CREMA-D',
                    'emotion': emotion,
                    'transcript': "" # Need external transcript or whisper
                })
    return records

def scan_iemocap(base_path, emotion_map):
    print(f"Scanning IEMOCAP at {base_path}...")
    records = []
    # IEMOCAP structure is complex. We scan for .wav files.
    # Filenames don't always have emotion. We usually need the .txt transcripts or EmoEvaluation files.
    # BUT, sometimes filenames imply it? No, typically stored in EmoEvaluation/*.txt.
    # However, for this quick fix, if we can't parse EmoEval, we might be stuck.
    # Wait, previous `processed/audio_wavs` had `dia0_utt0`. That is MELD, not IEMOCAP.
    # User said "IEMOCAP" but "processed" has MELD.
    # Let's focus on CREMA-D matching for now since it's easier to parse from filename.
    # For IEMOCAP, I will search for the "EmoEvaluation" folder to get labels.
    
    # Find all .txt files in EmoEvaluation
    emo_files = list(Path(base_path).rglob('*EmoEvaluation/*.txt'))
    print(f"  Found {len(emo_files)} EmoEvaluation files")
    
    # Parse EmoEval
    # Format: [START_TIME - END_TIME] TURN_NAME EMOTION [V, A, D]
    # Ses01F_impro01_F000\tneu\t[2.5000, 2.5000, 2.5000]
    
    file_map = {} # map turn_name to full path
    # First, find all wavs
    print("  Indexing wav files...")
    wav_gen = Path(base_path).rglob('*.wav')
    for w in wav_gen:
        if not w.name.startswith('.'): # ignore hidden
            file_map[w.stem] = str(w)
            
    print(f"  Indexed {len(file_map)} wavs")
            
    for ef in tqdm(emo_files, desc="Parsing IEMOCAP Labels"):
        try:
            with open(ef, 'r') as f:
                lines = f.readlines()
            for line in lines:
                if line.startswith('%') or not line.strip(): continue
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    # turn_name is usually parts[1]?? No, standard format is:
                    # [6.2901 - 8.2357]\tSes01F_impro01_F000\tneu\t[2.5000, 2.5000, 2.5000]
                    # So parts[1] is name, parts[2] is emotion
                    name = parts[1]
                    emo_code = parts[2]
                    
                    emotion = emotion_map.get(emo_code, None)
                    if emotion and name in file_map:
                        records.append({
                            'call_id': name,
                            'filename': Path(file_map[name]).name,
                            'filepath': file_map[name],
                            'dataset': 'IEMOCAP',
                            'emotion': emotion,
                            'transcript': "" 
                        })
        except Exception as e:
            # print(f"Error parsing {ef}: {e}")
            pass
            
    return records

def main():
    print("GENERATING REAL METADATA")
    
    # Map to our 5 classes + others
    # CREMA: ANG, DIS, FEA, HAP, NEU, SAD
    # IEMOCAP: ang, hap, exc, sad, neu, fru, fea, sur, dis, xxx, oth
    
    # We map 'exc' to 'hap' or 'neutral'? Usually 'hap'.
    # We only want: anger, disgust, fear, neutral, sadness. (5 classes)
    # What about 'joy'/'happy'? Diagnosis script had 'joy'. 
    # Prompt said "54.5% (Acc)".
    # Let's map HAP -> Happy. But if model is 5 classes, we might drop it or map to neutral?
    # Standard 4+Neutral: Ang, Sad, Hap, Neu (+Fea?)
    # User prompt diagnosis had: anger, disgust, fear, sadness, neutral. (NO HAPPY).
    # So I will IGNORE Happy/Excited for now, or map to Neutral? 
    # "joy -> neutral" error was high.
    # I'll include HAP as 'joy' in metadata, filtering happens in training.
    
    crema_map = {
        'ANG': 'anger', 'DIS': 'disgust', 'FEA': 'fear', 
        'HAP': 'joy', 'NEU': 'neutral', 'SAD': 'sadness'
    }
    
    iemocap_map = {
        'ang': 'anger', 'hap': 'joy', 'exc': 'joy', 'sad': 'sadness',
        'neu': 'neutral', 'fea': 'fear', 'dis': 'disgust'
    }
    
    all_records = []
    
    # Scan CREMA
    crema_records = scan_crema('d:/haam_framework/crema-d-mirror-main', crema_map)
    all_records.extend(crema_records)
    
    # Scan IEMOCAP
    # User specified: iemocapfullrelease\1\IEMOCAP_full_release
    iemocap_records = scan_iemocap('d:/haam_framework/iemocapfullrelease/1/IEMOCAP_full_release', iemocap_map)
    all_records.extend(iemocap_records)
    
    df = pd.DataFrame(all_records)
    print(f"Total records found: {len(df)}")
    print(df['dataset'].value_counts())
    print(df['emotion'].value_counts())
    
    df.to_csv('data/real_metadata.csv', index=False)
    print("Saved to data/real_metadata.csv")
    
    # Overwrite hybrid_metadata carefully? Or just update batch script to use real_metadata.
    # I'll update batch script to use `real_metadata.csv`.

if __name__ == '__main__':
    main()
