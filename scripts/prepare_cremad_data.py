
import os
import shutil
import random
import csv
from datetime import datetime, timedelta

# Configuration
CREMAD_SOURCE = r"D:\haam_framework\crema-d-mirror-main\AudioWAV"
TARGET_DIR = r"D:\haam_framework\data\cremad_samples"
GROUND_TRUTH_FILE = r"D:\haam_framework\data\cremad_ground_truth.csv"

# Mappings
EMOTION_MAP = {
    'ANG': 'anger',
    'HAP': 'joy',
    'SAD': 'sadness',
    'NEU': 'neutral',
    'FEA': 'fear',
    'DIS': 'disgust'
}

INTENSITY_PRIORITY = {'HI': 3, 'MD': 2, 'LO': 1, 'XX': 0}

def parse_filename(filename):
    # Example: 1001_DFA_ANG_HI.wav
    base = os.path.splitext(filename)[0]
    parts = base.split('_')
    
    if len(parts) >= 3:
        actor_id = int(parts[0])
        sentence = parts[1]
        emotion_code = parts[2]
        intensity = parts[3] if len(parts) > 3 else 'XX'
        return {
            'filename': filename,
            'actor_id': actor_id,
            'sentence': sentence,
            'emotion_code': emotion_code,
            'intensity': intensity,
            'expected_emotion': EMOTION_MAP.get(emotion_code)
        }
    return None

def get_agent_id(actor_id):
    if 1001 <= actor_id <= 1030:
        return 'agent_01'
    elif 1031 <= actor_id <= 1060:
        return 'agent_02'
    else:
        return 'agent_03'

def main():
    print("Processing CREMA-D dataset...")
    
    if not os.path.exists(CREMAD_SOURCE):
        print(f"Error: Source directory not found: {CREMAD_SOURCE}")
        return

    # Ensure target directory exists
    os.makedirs(TARGET_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(GROUND_TRUTH_FILE), exist_ok=True)

    # 1. Scan files
    all_files = [f for f in os.listdir(CREMAD_SOURCE) if f.endswith('.wav')]
    print(f"Found {len(all_files)} files in AudioWAV folder")

    # Group by emotion
    emotion_groups = {e: [] for e in EMOTION_MAP.values()}
    
    for fname in all_files:
        meta = parse_filename(fname)
        if meta and meta['expected_emotion']:
            emotion_groups[meta['expected_emotion']].append(meta)

    # 2. Select balanced sample (15 per emotion)
    selected_samples = []
    
    print("\nEmotion distribution (Source / Selected):")
    for emotion, samples in emotion_groups.items():
        # Sort by intensity (HI > MD > LO > XX)
        sorted_samples = sorted(samples, key=lambda x: INTENSITY_PRIORITY.get(x['intensity'], 0), reverse=True)
        
        # Select top 15
        selection = sorted_samples[:15]
        selected_samples.extend(selection)
        print(f"  {emotion:<10}: {len(samples):<5} / {len(selection)}")

    # 3. Process and Copy
    print(f"\nCopying {len(selected_samples)} files to {TARGET_DIR}...")
    
    ground_truth_rows = []
    start_date = datetime(2024, 12, 10)
    
    for i, meta in enumerate(selected_samples):
        # Generate metadata
        agent = get_agent_id(meta['actor_id'])
        
        # Random time between 9am - 5pm
        day_offset = i % 10
        hour_offset = random.randint(9, 16)
        minute_offset = random.randint(0, 59)
        dt = start_date + timedelta(days=day_offset)
        dt = dt.replace(hour=hour_offset, minute=minute_offset)
        
        timestamp = dt.isoformat()
        date_str = dt.strftime('%Y-%m-%d')
        
        # Call ID: call_YYYY-MM-DD_agent_XXX
        call_id = f"call_{date_str}_{agent}_{str(i+1).zfill(3)}"
        
        # Paths
        src_path = os.path.join(CREMAD_SOURCE, meta['filename'])
        dest_filename = f"{call_id}.wav"
        dest_path = os.path.join(TARGET_DIR, dest_filename)
        
        # Copy file
        shutil.copy2(src_path, dest_path)
        
        ground_truth_rows.append({
            'call_id': call_id,
            'agent_id': agent,
            'timestamp': timestamp,
            'date': date_str,
            'expected_emotion': meta['expected_emotion'],
            'emotion_code': meta['emotion_code'],
            'intensity': meta['intensity'],
            'actor_id': meta['actor_id'],
            'sentence': meta['sentence'],
            'audio_file': dest_filename,
            'local_path': dest_path
        })

    # 4. Save CSV
    with open(GROUND_TRUTH_FILE, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['call_id', 'agent_id', 'timestamp', 'date', 'expected_emotion', 
                      'emotion_code', 'intensity', 'actor_id', 'sentence', 'audio_file', 'local_path']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ground_truth_rows)
        
    print(f"\nGround truth saved to: {GROUND_TRUTH_FILE}")
    print("\nSummary:")
    print(f"  Total calls: {len(ground_truth_rows)}")
    
    agent_counts = {}
    for r in ground_truth_rows:
        agent_counts[r['agent_id']] = agent_counts.get(r['agent_id'], 0) + 1
        
    for agent, count in sorted(agent_counts.items()):
        print(f"  {agent}: {count} calls")
        
    print(f"  Date range: {ground_truth_rows[0]['date']} to {ground_truth_rows[-1]['date']}")

if __name__ == "__main__":
    main()
