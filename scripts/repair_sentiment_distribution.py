
import os
import json
import glob
from pathlib import Path
from tqdm import tqdm

def repair_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    segments = data.get('segments', [])
    if not segments:
        return False

    # Soft Probability Aggregation
    overall_dist = {}
    valid_segs = 0
    
    # Initialize keys to ensure consistency
    for emo in ['neutral', 'anger', 'sadness', 'fear', 'joy', 'disgust']:
        overall_dist[emo] = 0.0

    for seg in segments:
        dist = seg.get('emotion_distribution', {})
        if dist:
            valid_segs += 1
            for emo, prob in dist.items():
                overall_dist[emo] = overall_dist.get(emo, 0.0) + prob
    
    if valid_segs > 0:
        overall_dist = {k: round(v / valid_segs, 4) for k, v in overall_dist.items()}
        
        # Update dominant emotion based on new average
        dominant = max(overall_dist, key=overall_dist.get)
        
        # Write back to metrics
        if 'overall_metrics' not in data:
            data['overall_metrics'] = {}
            
        data['overall_metrics']['emotion_distribution'] = overall_dist
        data['overall_metrics']['dominant_emotion'] = dominant
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    
    return False

def main():
    root_dir = "results"
    # Find all call_*.json recursively (covers calls and calls_iemocap)
    files = glob.glob(os.path.join(root_dir, "**", "*.json"), recursive=True)
    # Filter for call_ pattern or iemocap pattern if needed, but *.json in results subdirs is synonymous
    call_files = [f for f in files if "calls" in f and f.endswith(".json")]
    
    print(f"Found {len(call_files)} files to repair...")
    
    count = 0
    for f in tqdm(call_files):
        try:
            if repair_json(f):
                count += 1
        except Exception as e:
            print(f"Error repairing {f}: {e}")
            
    print(f"Repaired {count} files.")

if __name__ == "__main__":
    main()
