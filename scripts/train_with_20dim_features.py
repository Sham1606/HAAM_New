"""
Ablation Study: 12-dim vs 20-dim Acoustic Features
Runs sequential training sessions and compares recall for Fear/Disgust.
"""

import subprocess
import json
import sys
from pathlib import Path
import pandas as pd

def run_training(dim, feature_dir, results_suffix):
    cmd = [
        sys.executable, "scripts/05_train_improved_model.py",
        "--acoustic-dim", str(dim),
        "--feature-dir", feature_dir,
        "--epochs", "20" # Faster for ablation
    ]
    print(f"\n>>> Running Training: {dim} Dimensions using {feature_dir}...")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"ERROR: Training failed for {dim} dimensions.")
        return None
    
    # Load metrics
    metrics_path = Path("results/improved/metrics.json")
    if metrics_path.exists():
        try:
            with open(metrics_path, "r") as f:
                data = json.load(f)
            return data
        except json.JSONDecodeError:
            print(f"ERROR: Could not decode metrics.json for {dim} dimensions.")
            return None
    return None

def main():
    results = {}
    
    # 1. Baseline (12-dim)
    # Using features_v2_fixed which correctly has 12 dims
    res_12 = run_training(12, "data/processed/features_v2_fixed", "12d")
    if res_12:
        results['12d'] = res_12
        
    # 2. Upgraded (20-dim)
    # Ensure upgrade script has been run!
    if not Path("data/processed/features_v4_20dim").exists():
        print("\nERROR: data/processed/features_v4_20dim not found. Run scripts/upgrade_features_to_20dim.py first.")
        # Try to proceed with just 12d if available
    else:
        res_20 = run_training(20, "data/processed/features_v4_20dim", "20d")
        if res_20:
            results['20d'] = res_20
        
    # 3. Comparison Report
    print("\n" + "="*80)
    print("ABLATION STUDY REPORT: 12D VS 20D")
    print("="*80)
    print(f"{'Metric':<20} | {'12D Baseline':<15} | {'20D Upgraded':<15} | {'Gain':<10}")
    print("-" * 70)
    
    metrics_to_show = ['test_accuracy', 'best_val_accuracy']
    for m in metrics_to_show:
        v1 = results['12d'].get(m, 0) if '12d' in results else 0
        v2 = results['20d'].get(m, 0) if '20d' in results else 0
        gain = v2 - v1
        print(f"{m:<10} | {v1:<15.4f} | {v2:<15.4f} | {gain:+.4f}")

    # Save comparison to docs/
    Path('docs').mkdir(exist_ok=True)
    with open('docs/mfcc_ablation_report.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nFull report saved to docs/mfcc_ablation_report.json")

if __name__ == '__main__':
    main()
