
import os
import json
import random

# Configuration
CREMAD_SOURCE = r"d:\haam\HAAM_New\data\CREMA-D"
OUTPUT_DIR = r"d:\haam\HAAM_New\results\calls_cremad"

# Mappings
# CREMA-D codes: ANG, DES, DIS, FEA, HAP, NEU, SAD
EMOTION_MAP = {
    'ANG': 'anger',
    'HAP': 'joy',
    'SAD': 'sadness',
    'NEU': 'neutral',
    'FEA': 'fear',
    'DIS': 'disgust',
    'DES': 'disgust' # Mapping desire/disgust variant if present, usually just DIS
}

# Standardize to model allowed emotions (removed surprise if not in standard 4-6 set, but let's map widely)
# Model typically expects: neutral, anger, joy, sadness, fear, disgust.

def parse_filename(filename):
    # Example: 1001_DFA_ANG_XX.wav
    base = os.path.splitext(filename)[0]
    parts = base.split('_')
    
    if len(parts) >= 3:
        actor_id = int(parts[0])
        sentence = parts[1]
        emotion_code = parts[2]
        intensity = parts[3] if len(parts) > 3 else 'XX'
        return {
            'call_id': base,
            'actor_id': actor_id,
            'emotion_code': emotion_code,
            'intensity': intensity,
            'emotion_label': EMOTION_MAP.get(emotion_code)
        }
    return None

def main():
    print(f"Generating metadata for CREMA-D from {CREMAD_SOURCE}...")
    
    if not os.path.exists(CREMAD_SOURCE):
        print(f"Error: Source directory not found: {CREMAD_SOURCE}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    files = [f for f in os.listdir(CREMAD_SOURCE) if f.endswith('.wav')]
    print(f"Found {len(files)} WAV files.")
    
    processed_count = 0
    skipped_count = 0
    
    for fname in files:
        meta = parse_filename(fname)
        if not meta or not meta['emotion_label']:
            skipped_count += 1
            continue
            
        emotion = meta['emotion_label']
        
        # Create JSON structure compatible with training script
        json_data = {
            "call_id": meta['call_id'],
            "agent_id": f"crema_actor_{meta['actor_id']}",
            "original_filename": fname,
            "overall_metrics": {
                "emotion_distribution": {
                    emotion: 1.0  # Hardcoded confidence for ground truth
                },
                "avg_pitch": 0.0, # Will be extracted later or ignored
                "speech_rate_wpm": 0.0,
                "agent_stress_score": 0.0
            },
            "ground_truth": {
                "emotion": emotion
            },
            "metadata": {
                "dataset": "CREMA-D",
                "intensity": meta['intensity']
            }
        }
        
        out_path = os.path.join(OUTPUT_DIR, f"{meta['call_id']}.json")
        with open(out_path, 'w') as f:
            json.dump(json_data, f, indent=2)
            
        processed_count += 1
        
    print(f"Finished.")
    print(f"Generated {processed_count} JSON files in {OUTPUT_DIR}")
    print(f"Skipped {skipped_count} files (unrecognized format/emotion).")

if __name__ == "__main__":
    main()
