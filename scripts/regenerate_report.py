
import os
import json
import csv
import glob
from collections import defaultdict

CALLS_DIR = r"D:\haam_framework\results\calls"
GROUND_TRUTH_CSV = r"D:\haam_framework\data\cremad_ground_truth.csv"
REPORT_FILE = r"D:\haam_framework\cremad_validation_report.json"

def main():
    # 1. Load Ground Truth
    ground_truth = {}
    print("Loading ground truth...")
    if os.path.exists(GROUND_TRUTH_CSV):
        with open(GROUND_TRUTH_CSV, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ground_truth[row['call_id']] = row['expected_emotion'].lower()
    else:
        print("Ground truth file missing!")
        return

    # 2. Iterate Metrics
    print("Scanning metrics...")
    results = []
    files = glob.glob(os.path.join(CALLS_DIR, "*.json"))
    
    for f_path in files:
        try:
            with open(f_path, 'r') as f:
                data = json.load(f)
                
            call_id = data['call_id']
            # Fallback if call_id in dict is only suffix
            if call_id not in ground_truth:
                # Try finding key that ends with suffix? 
                pass
                
            expected = ground_truth.get(call_id, 'unknown')
            detected = data['overall_metrics']['dominant_emotion']
            
            results.append({
                "call_id": call_id,
                "expected": expected,
                "detected": detected,
                "metrics": data['overall_metrics']
            })
        except Exception as e:
            print(f"Error reading {f_path}: {e}")

    # 3. Calculate Stats
    total_calls = len(results)
    correct_predictions = 0
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    per_emotion_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    for res in results:
        exp = res['expected']
        det = res['detected']
        
        if exp == 'unknown':
            continue
            
        confusion_matrix[exp][det] += 1
        per_emotion_stats[exp]['total'] += 1
        
        if exp == det:
            correct_predictions += 1
            per_emotion_stats[exp]['correct'] += 1
            
    overall_accuracy = (correct_predictions / total_calls * 100) if total_calls > 0 else 0
    
    # Format per-emotion
    per_emotion_final = {}
    for emo, stats in per_emotion_stats.items():
        acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        per_emotion_final[emo] = {
            "accuracy": round(acc, 2),
            "correct": stats['correct'],
            "total": stats['total']
        }
        
    # Format confusion pairs
    confusion_pairs = {}
    for exp, det_counts in confusion_matrix.items():
        for det, count in det_counts.items():
            confusion_pairs[f"{exp} -> {det}"] = count

    # 4. Save Report
    report = {
        "summary": {
            "total_calls": total_calls,
            "correct_predictions": correct_predictions,
            "overall_accuracy": round(overall_accuracy, 2)
        },
        "per_emotion_accuracy": per_emotion_final,
        "confusion_pairs": confusion_pairs
    }
    
    with open(REPORT_FILE, 'w') as f:
        json.dump(report, f, indent=2)
        
    print(f"Generated report for {total_calls} calls. Accuracy: {overall_accuracy:.2f}%")
    print(f"Report saved to {REPORT_FILE}")

if __name__ == "__main__":
    main()
