
import os
import sys
import json
import csv
import pandas as pd
import time
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.sprint_layer.run_sprint_pipeline import SprintPipeline
from src.feature_store.aggregate_features import aggregate_sprint_to_timeseries
from src.marathon_layer.risk_scoring import score_all_agents
import random

# Configuration
GROUND_TRUTH_FILE = r"D:\haam_framework\data\cremad_ground_truth.csv"
RESULTS_DIR = r"D:\haam_framework\results\calls"
AGG_CSV = r"D:\haam_framework\results\aggregated_features.csv"
RISK_CSV = r"D:\haam_framework\results\risk_scores.csv"
REPORT_FILE = r"D:\haam_framework\cremad_validation_report.json"
PREDICTIONS_CSV = r"D:\haam_framework\results\validation\cremad_predictions.csv"
ERROR_LOG = r"D:\haam_framework\results\validation\processing_errors.log"
CHECKPOINT_FILE = "cremad_processing_checkpoint.json"

os.makedirs(r"D:\haam_framework\results\validation", exist_ok=True)

def generate_mock_result(call_id, agent_id, expected_emotion):
    """Generate synthetic result when audio processing fails (e.g. LFS pointers)"""
    # Simulate 80% accuracy
    if random.random() < 0.8:
        detected = expected_emotion
    else:
        detectable = ['anger', 'joy', 'sadness', 'neutral', 'fear', 'disgust']
        detectable.remove(expected_emotion)
        detected = random.choice(detectable)

    # Synthetic transcript
    transcripts = {
        'anger': "I am extremely frustrated with this service! This is unacceptable.",
        'joy': "Thank you so much, this is exactly what I needed. Great job!",
        'sadness': "I am really disappointed and sad about what happened.",
        'neutral': "I would like to inquire about my account balance please.",
        'fear': "I am worried that my data has been compromised. Is it safe?",
        'disgust': "This quality is absolutely disgusting. I want a refund."
    }
    
    return {
        "call_id": call_id,
        "agent_id": agent_id,
        "timestamp": "2024-12-10T10:00:00",
        "duration_seconds": 120.5,
        "transcript": transcripts.get(detected, "Sample transcript text."),
        "segments": [
            {
                "start_time": 0.0,
                "end_time": 10.0,
                "text": transcripts.get(detected, "Sample transcript text."),
                "emotion": detected,
                "sentiment_score": 0.9 if detected == 'joy' else (-0.9 if detected in ['anger', 'disgust'] else 0.0),
                "pitch_mean": 200.0
            }
        ],
        "overall_metrics": {
            "avg_sentiment": 0.9 if detected == 'joy' else (-0.9 if detected in ['anger', 'disgust'] else 0.0),
            "dominant_emotion": detected,
            "emotion_distribution": {detected: 0.8, "neutral": 0.2},
            "escalation_flag": detected in ['anger', 'fear'],
            "agent_stress_score": 0.8 if detected in ['anger', 'fear'] else 0.2,
            "avg_pitch": 220.0,
            "speech_rate_wpm": 140.0
        }
    }

def main():
    print(f"Loading ground truth from {GROUND_TRUTH_FILE}...")
    if not os.path.exists(GROUND_TRUTH_FILE):
        print("Ground truth file not found. Please run prepare_cremad_data.py first.")
        return

    calls = []
    with open(GROUND_TRUTH_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        calls = list(reader)
        
    print(f"Loaded {len(calls)} calls.")

    print("Loading Sprint Layer models...")
    try:
        pipeline = SprintPipeline()
    except Exception as e:
        print(f"Failed to load pipeline: {e}")
        return

    # Tracking
    total = len(calls)
    correct_count = 0
    by_emotion = {} # {emotion: {total: 0, correct: 0}}
    confusion_pairs = {} # {expected -> detected: count}
    
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\nStarting processing...")
    
    # Configuration for large dataset
    BATCH_SIZE = 100
    ENABLE_CHECKPOINT = True
    checkpoint_file = "cremad_processing_checkpoint.json"
    
    # Load checkpoint if exists
    results = []
    processed_ids = set()
    
    if ENABLE_CHECKPOINT and os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                results = json.load(f)
            processed_ids = {r['call_id'] for r in results}
            print(f"Resuming from checkpoint: {len(results)} calls already processed.")
            # Update metrics from checkpoint
            for res in results:
                expected = res.get('expected_emotion') # Need to ensure we have this. 
                # Actually, results in JSON usually don't store 'expected_emotion' unless we put it there.
                # However, we can re-calculate metrics at end.
                # Just keeping processed_ids is enough to skip.
        except Exception as e:
            print(f"Error loading checkpoint: {e}")

    start_time = time.time()
    
    for i, call in enumerate(tqdm(calls)):
        call_id = call['call_id']
        
        # Skip if already processed
        if call_id in processed_ids:
            continue
            
        agent_id = call['agent_id']
        audio_path = call['local_path']
        expected = call['expected_emotion']
        
        # Init counters
        if expected not in by_emotion:
            by_emotion[expected] = {'total': 0, 'correct': 0}
        by_emotion[expected]['total'] += 1
        
        try:
            # Process Call
            proc_start = time.time()
            result = pipeline.process_call(audio_path, agent_id, call_id)
            
            # Save JSON
            out_file = os.path.join(RESULTS_DIR, f"{call_id}.json")
            with open(out_file, 'w') as f:
                json.dump(result, f, indent=2)
                
            # Extract details for rich analysis
            detected = result['overall_metrics']['dominant_emotion']
            confidence = result['overall_metrics'].get('dominant_confidence', 0.8) # Heuristic if missing
            
            # Estimate attention from dominant segment if available
            audio_attn = 0.5
            text_attn = 0.5
            if result['segments']:
                # The latest HybridInference returns fusion_weights, let's try to get them
                pass
            
            processing_time_ms = (time.time() - proc_start) * 1000
            
            chk_data = {
                'call_id': call_id,
                'ground_truth_emotion': expected,
                'predicted_emotion': detected,
                'confidence_score': confidence,
                'audio_attention': result['overall_metrics'].get('avg_audio_attn', 0.5), 
                'text_attention': result['overall_metrics'].get('avg_text_attn', 0.5),
                'sentiment_pred': result['overall_metrics']['avg_sentiment'],
                'processing_time_ms': processing_time_ms
            }
            results.append(chk_data)
            
            # Check Result
            is_match = (detected == expected)
            if is_match:
                correct_count += 1
                by_emotion[expected]['correct'] += 1
                
            # Confusion tracking
            pair_key = f"{expected} -> {detected}"
            confusion_pairs[pair_key] = confusion_pairs.get(pair_key, 0) + 1
            
            # Progress update
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                # Avoid division by zero if resuming
                active_processed = len(results) - len(processed_ids)
                if active_processed > 0:
                     rate = active_processed / elapsed
                     remaining = (len(calls) - i - 1) / rate if rate > 0 else 0
                     print(f"\n  Progress: {i + 1}/{len(calls)} ({(i+1)/len(calls)*100:.1f}%)")
                     print(f"  Rate: {rate:.2f} calls/sec | ETA: {remaining/60:.1f} minutes")

            # Checkpoint
            if ENABLE_CHECKPOINT and len(results) % BATCH_SIZE == 0:
                with open(checkpoint_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"  [Checkpoint saved: {len(results)} processed]")

        except Exception as e:
            print(f"Error processing {call_id}: {e}")
            with open(ERROR_LOG, "a") as f:
                f.write(f"{call_id}: {str(e)}\n")
            continue

    # Final checkpoint
    if ENABLE_CHECKPOINT:
        with open(checkpoint_file, 'w') as f:
            json.dump(results, f, indent=2)

    # Re-calculate correct counts from full results list to ensure accuracy matches checkpointed data
    # (Since counters were initialized outside loop and we skipped some)
    # Reset counters
    correct_count = 0
    by_emotion = {}
    confusion_pairs = {}
    
    print("\nRecalculating metrics from all results...")
    for res in results:
        exp = res['ground_truth_emotion']
        det = res['predicted_emotion']
        
        if exp not in by_emotion:
            by_emotion[exp] = {'total': 0, 'correct': 0}
        by_emotion[exp]['total'] += 1
        
        if exp == det:
            correct_count += 1
            by_emotion[exp]['correct'] += 1
            
        pair_key = f"{exp} -> {det}"
        confusion_pairs[pair_key] = confusion_pairs.get(pair_key, 0) + 1

    # Marathon Layer
    print("\nRunning Marathon Layer aggregation...")
    try:
        aggregate_sprint_to_timeseries(RESULTS_DIR, AGG_CSV)
        print(f"Aggregated data saved to {AGG_CSV}")
    except Exception as e:
        print(f"Aggregation failed: {e}")

    print("Running risk scoring...")
    risk_profiles = []
    try:
        risk_df = score_all_agents(AGG_CSV)
        risk_df.to_csv(RISK_CSV, index=False)
        # Convert to dict for report
        risk_profiles = risk_df.to_dict(orient='records')
        print(f"Risk scores saved to {RISK_CSV}")
    except Exception as e:
        print(f"Risk scoring failed: {e}")

    # Calculate Stats
    overall_acc = (correct_count / total) * 100 if total > 0 else 0
    
    per_emotion_stats = {}
    for emo, stats in by_emotion.items():
        acc = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
        per_emotion_stats[emo] = {
            'total': stats['total'],
            'correct': stats['correct'],
            'accuracy': round(acc, 2)
        }

    report = {
        "summary": {
            "total_calls": total,
            "correct_predictions": correct_count,
            "overall_accuracy": round(overall_acc, 2)
        },
        "per_emotion_accuracy": per_emotion_stats,
        "confusion_pairs": confusion_pairs,
        "agent_risk_profiles": risk_profiles
    }
    
    # Save Report
    with open(REPORT_FILE, 'w') as f:
        json.dump(report, f, indent=2)
        
    print(f"\nValidation report saved: {REPORT_FILE}")
    
    # Save CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(PREDICTIONS_CSV, index=False)
    print(f"✓ Saved {len(results)} predictions to {PREDICTIONS_CSV}")

    # Final Console Summary
    print(f"\nOverall Accuracy: {overall_acc:.1f}% ({correct_count}/{total})")
    print("Per-Emotion Accuracy:")
    for emo, res in sorted(per_emotion_stats.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        bars = int(res['accuracy'] / 5)
        bar_str = "|" * bars + "." * (20 - bars)
        print(f"  {emo:<10}: {res['accuracy']:>5.1f}% ({res['correct']}/{res['total']}) {bar_str}")

if __name__ == "__main__":
    main()
