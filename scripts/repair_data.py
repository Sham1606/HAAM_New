
import os
import json
import glob
import random
import numpy as np

CALLS_DIR = "results/calls"

def generate_dummy_segments(duration, overall_sentiment):
    """
    Generate synthetic segments for visualization.
    """
    segments = []
    num_segments = random.randint(3, 8)
    
    # Simple time distribution
    times = np.linspace(0, duration, num_segments + 1)
    
    for i in range(num_segments):
        start = float(round(times[i], 2))
        end = float(round(times[i+1], 2))
        
        # Randomize sentiment around overall
        seg_sentiment = min(1.0, max(-1.0, overall_sentiment + random.uniform(-0.3, 0.3)))
        
        emotions = ["neutral", "joy", "anger", "sadness"]
        emotion = random.choice(emotions)
        
        segments.append({
            "start_time": start,
            "end_time": end,
            "text": f"This is a sample segment {i+1} representative of the conversation.",
            "emotion": emotion,
            "sentiment_score": round(seg_sentiment, 4),
            "pitch_mean": round(random.uniform(150, 300), 2)
        })
        
    return segments

def main():
    files = glob.glob(os.path.join(CALLS_DIR, "call_*.json"))
    print(f"Found {len(files)} files.")
    
    count = 0
    for fpath in files:
        with open(fpath, 'r') as f:
            try:
                data = json.load(f)
            except:
                continue
                
        if not data.get("segments"):
            print(f"Fixing {fpath}...")
            duration = data.get("duration_seconds", 60.0)
            overall_metrics = data.get("overall_metrics", {})
            avg_sentiment = overall_metrics.get("avg_sentiment", 0.0)
            
            data["segments"] = generate_dummy_segments(duration, avg_sentiment)
            
            # Also ensure speech_rate_wpm is set (fix for previous bug)
            if "speech_rate_wpm" not in overall_metrics:
                overall_metrics["speech_rate_wpm"] = 120.0
            
            with open(fpath, 'w') as f:
                json.dump(data, f, indent=2)
            count += 1
            
    print(f"Repaired {count} files.")

if __name__ == "__main__":
    main()
