"""
Analyze current model failures using existing predictions in hybrid_metadata.csv
Output: Confusion matrices, error analysis, accuracy breakdown
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import json

def main():
    print("="*80)
    print("PHASE 0.1: BASELINE MODEL DIAGNOSIS")
    print("="*80)
    
    # Create output directory
    Path('results/diagnosis').mkdir(parents=True, exist_ok=True)
    
    # Load metadata with predictions
    metadata_path = 'data/hybrid_metadata.csv'
    if not Path(metadata_path).exists():
        print(f"ERROR: {metadata_path} not found.")
        return
        
    df_results = pd.read_csv(metadata_path)
    print(f"Loaded {len(df_results)} records from {metadata_path}")
    
    # Filter only rows with predictions if needed (though head showed they have them)
    if 'emotion_pred' not in df_results.columns:
        print("ERROR: 'emotion_pred' column missing in metadata.")
        return
        
    df_results = df_results.dropna(subset=['emotion_pred', 'emotion_true'])
    
    # Normalize emotion keys just in case
    df_results['emotion_true'] = df_results['emotion_true'].str.lower()
    df_results['emotion_pred'] = df_results['emotion_pred'].str.lower()
    
    # Add 'correct' column
    df_results['correct'] = df_results['emotion_true'] == df_results['emotion_pred']
    
    # Analysis 1: Per-dataset accuracy
    print("\n1. ACCURACY BY DATASET:")
    print("-"*40)
    datasets = df_results['dataset'].unique()
    for dataset in datasets:
        acc = df_results[df_results['dataset'] == dataset]['correct'].mean() * 100
        print(f"  {dataset}: {acc:.2f}%")
    
    # Analysis 2: Confusion matrices
    emotions = sorted(df_results['emotion_true'].unique())
    print(f"\nEmotions found: {emotions}")
    
    for dataset in datasets:
        subset = df_results[df_results['dataset'] == dataset]
        cm = confusion_matrix(subset['emotion_true'], subset['emotion_pred'], labels=emotions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=emotions, yticklabels=emotions)
        plt.title(f'Baseline Confusion Matrix - {dataset}')
        plt.ylabel('True Emotion')
        plt.xlabel('Predicted Emotion')
        plt.tight_layout()
        plt.savefig(f'results/diagnosis/baseline_cm_{dataset}.png', dpi=150)
        plt.close()
    
    # Analysis 3: Error analysis
    errors = df_results[~df_results['correct']]
    print("\n2. MOST CONFUSED PAIRS:")
    print("-"*40)
    for dataset in datasets:
        subset = errors[errors['dataset'] == dataset]
        if len(subset) == 0:
            print(f"\n{dataset}: No errors found.")
            continue
            
        pairs = subset.groupby(['emotion_true', 'emotion_pred']).size().sort_values(ascending=False).head(5)
        print(f"\n{dataset}:")
        for (true_e, pred_e), count in pairs.items():
            print(f"  {true_e} -> {pred_e}: {count} errors")
    
    # Analysis 4: Duration correlation
    print("\n3. DURATION vs ACCURACY:")
    print("-"*40)
    # Ensure duration is numeric
    df_results['duration'] = pd.to_numeric(df_results['duration'], errors='coerce')
    
    short = df_results[df_results['duration'] < 1.5]['correct'].mean() * 100
    medium = df_results[(df_results['duration'] >= 1.5) & (df_results['duration'] < 5)]['correct'].mean() * 100
    long = df_results[df_results['duration'] >= 5]['correct'].mean() * 100
    print(f"  Short (<1.5s): {short:.1f}%")
    print(f"  Medium (1.5-5s): {medium:.1f}%")
    print(f"  Long (>5s): {long:.1f}%")
    
    # Save results
    df_results.to_csv('results/diagnosis/baseline_predictions_analyzed.csv', index=False)
    
    summary = {
        'overall_accuracy': float(df_results['correct'].mean()),
        'per_dataset': {d: float(df_results[df_results['dataset']==d]['correct'].mean()) for d in datasets},
        'per_emotion': {e: float(df_results[df_results['emotion_true']==e]['correct'].mean()) 
                       for e in emotions}
    }
    
    with open('results/diagnosis/baseline_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*80)
    print("DIAGNOSIS COMPLETE")
    print("="*80)
    print(f"Overall: {summary['overall_accuracy']*100:.2f}%")
    for d, acc in summary['per_dataset'].items():
        print(f"{d}: {acc*100:.2f}%")
    print("\nResults saved to: results/diagnosis/")

if __name__ == '__main__':
    main()
