import os
import json
import pandas as pd
from tqdm import tqdm

# Paths
CREMAD_JSON_DIR = r"D:\haam_framework\results\calls"
IEMOCAP_JSON_DIR = r"D:\haam_framework\results\calls_iemocap"
OUTPUT_CSV = r"D:\haam_framework\data\hybrid_metadata.csv"
OUTPUT_SUMMARY = r"D:\haam_framework\data\hybrid_metadata_summary.txt"
CREMAD_GT_CSV = r"D:\haam_framework\data\cremad_ground_truth.csv"

def load_cremad_ground_truth():
    """Loads CREMA-D ground truth from CSV."""
    if not os.path.exists(CREMAD_GT_CSV):
        print(f"WARNING: CREMA-D ground truth file not found at {CREMAD_GT_CSV}")
        return {}
    
    try:
        df = pd.read_csv(CREMAD_GT_CSV)
        # Create a dictionary mapping call_id to expected_emotion
        gt_map = dict(zip(df['call_id'], df['expected_emotion']))
        print(f"Loaded {len(gt_map)} ground truth labels for CREMA-D.")
        return gt_map
    except Exception as e:
        print(f"Error loading CREMA-D ground truth: {e}")
        return {}

def process_cremad(gt_map):
    data = []
    if not os.path.exists(CREMAD_JSON_DIR):
        print("CREMA-D JSON directory not found.")
        return data

    files = [f for f in os.listdir(CREMAD_JSON_DIR) if f.endswith(".json")]
    
    for f in tqdm(files, desc="Processing CREMA-D"):
        try:
            with open(os.path.join(CREMAD_JSON_DIR, f), 'r') as json_file:
                content = json.load(json_file)
                
                call_id = content.get("call_id", "")
                
                # Get ground truth from map, default to 'unknown' if not found
                # CREMA-D CSV typically lowercases emotions? Let's check.
                # In CSV: 'anger', 'joy', 'neutral', 'sadness', 'disgust', 'fear'
                # HAAM emotions: 'anger', 'joy', 'neutral', 'sadness', 'disgust', 'fear'
                
                emotion_true = gt_map.get(call_id, "unknown")
                if emotion_true != "unknown":
                    emotion_true = emotion_true.lower() # Ensure lowercase
                
                row = {
                    "dataset": "CREMA-D",
                    "call_id": call_id,
                    "emotion_true": emotion_true,
                    "emotion_pred": content.get("overall_metrics", {}).get("dominant_emotion", "unknown"),
                    "confidence": content.get("segments", [{}])[0].get("emotion_confidence", 0.0), 
                    "duration": content.get("duration_seconds", 0.0),
                    "transcript": content.get("transcript", "")
                }
                data.append(row)
        except Exception as e:
            print(f"Error processing {f}: {e}")
            continue
    return data

def process_iemocap():
    data = []
    if not os.path.exists(IEMOCAP_JSON_DIR):
        print("IEMOCAP JSON directory not found.")
        return data

    files = [f for f in os.listdir(IEMOCAP_JSON_DIR) if f.endswith(".json")]
    
    for f in tqdm(files, desc="Processing IEMOCAP"):
        try:
            with open(os.path.join(IEMOCAP_JSON_DIR, f), 'r') as json_file:
                content = json.load(json_file)
                
                gt = content.get("ground_truth", {})
                emotion_true = gt.get("emotion", "unknown")
                if emotion_true:
                    emotion_true = emotion_true.lower()

                # Calculate confidence if missing
                confidence = content.get("overall_metrics", {}).get("average_confidence", 0.0)
                if confidence == 0.0 and content.get('segments'):
                     confs = [s.get('emotion_confidence', 0) for s in content['segments']]
                     if confs:
                         confidence = sum(confs) / len(confs)
                
                row = {
                    "dataset": "IEMOCAP",
                    "call_id": content.get("call_id", ""),
                    "emotion_true": emotion_true,
                    "emotion_pred": content.get("overall_metrics", {}).get("dominant_emotion", "unknown"),
                    "confidence": confidence,
                    "duration": content.get("duration_seconds", 0.0),
                    "transcript": content.get("transcript", "")
                }
                data.append(row)
        except Exception as e:
            print(f"Error processing {f}: {e}")
            continue
    return data

def main():
    print("Creating Hybrid Metadata...")
    
    # Load CREMA-D Ground Truth
    cremad_gt_map = load_cremad_ground_truth()
    
    cremad_data = process_cremad(cremad_gt_map)
    iemocap_data = process_iemocap()
    
    all_data = cremad_data + iemocap_data
    
    df = pd.DataFrame(all_data)
    
    print(f"Total samples: {len(df)}")
    if not df.empty:
        print("Dataset distribution:")
        print(df['dataset'].value_counts())
        print("\nEmotion distribution (Ground Truth):")
        print(df['emotion_true'].value_counts())
        
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved metadata to {OUTPUT_CSV}")
        
        # Save summary
        with open(OUTPUT_SUMMARY, 'w') as f:
            f.write(f"Total Samples: {len(df)}\n")
            f.write("Dataset Distribution:\n")
            f.write(df['dataset'].value_counts().to_string())
            f.write("\n\nEmotion Distribution (Ground Truth):\n")
            f.write(df['emotion_true'].value_counts().to_string())
            
    else:
        print("No data found!")

if __name__ == "__main__":
    main()
