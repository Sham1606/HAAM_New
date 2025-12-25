import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import json

def analyze():
    print("PHASE 7.1: DEEP ERROR ANALYSIS")
    print("="*80)
    
    input_path = 'results/diagnosis/test_predictions_deep.csv'
    if not Path(input_path).exists():
        print(f"Error: {input_path} not found. Run generate_test_predictions.py first.")
        return
        
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} test results.")
    
    # 1. Overall & Per-Dataset Accuracy
    print("\n1. ACCURACY ANALYSIS")
    overall_acc = df['correct'].mean() * 100
    print(f"Overall Test Accuracy: {overall_acc:.2f}%")
    
    dataset_acc = df.groupby('dataset')['correct'].mean() * 100
    for ds, acc in dataset_acc.items():
        print(f"  {ds}: {acc:.2f}%")
        
    # 2. Confusion Analysis (Focus on Fear)
    print("\n2. EMOTION CONFUSION (RECALL)")
    emotions = sorted(df['emotion_true'].unique())
    for e in emotions:
        subset = df[df['emotion_true'] == e]
        recall = subset['correct'].mean() * 100
        print(f"  {e:10} Recall: {recall:.1f}% ({len(subset)} samples)")
        
    # 3. Top Confused Pairs
    print("\n3. TOP CONFUSED PAIRS (Where model is wrong)")
    wrong = df[df['correct'] == 0]
    conf_pairs = wrong.groupby(['emotion_true', 'emotion_pred']).size().sort_values(ascending=False).head(10)
    for (true, pred), count in conf_pairs.items():
        print(f"  True: {true:8} -> Pred: {pred:8} | Count: {count}")

    # 4. Feature Health Audit (Zeros)
    print("\n4. FEATURE QUALITY AUDIT (Zero-Extraction Count)")
    for col in ['pitch_is_zero', 'jitter_is_zero', 'shimmer_is_zero']:
        zero_count = df[col].sum()
        zero_pct = (zero_count / len(df)) * 100
        print(f"  {col:15}: {zero_count} samples ({zero_pct:.1f}%)")

    # 5. Modality Weighting Analysis
    print("\n5. MODALITY WEIGHTING (Acoustic vs Text)")
    avg_acoustic = df['weight_acoustic'].mean()
    avg_text = df['weight_text'].mean()
    print(f"  Average Acoustic Weight: {avg_acoustic:.2f}")
    print(f"  Average Text Weight:     {avg_text:.2f}")
    
    # Correlation between correct and weights
    corr_acoustic = df['correct'].corr(df['weight_acoustic'])
    print(f"  Correlation (Correct vs Acoustic Weight): {corr_acoustic:.3f}")

    # 6. Confusion Matrix Visualization
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(df['emotion_true'], df['emotion_pred'], labels=emotions)
    # Normalize by row (recall)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=emotions, yticklabels=emotions)
    plt.title('Normalized Confusion Matrix (Recall)')
    plt.ylabel('True Emotion')
    plt.xlabel('Predicted Emotion')
    plt.tight_layout()
    plt.savefig('results/diagnosis/deep_cm_normalized.png', dpi=150)
    print("\nSaved confusion matrix to results/diagnosis/deep_cm_normalized.png")

    # Save summary json
    summary = {
        'overall_acc': float(overall_acc),
        'dataset_acc': dataset_acc.to_dict(),
        'per_emotion_recall': {e: float(df[df['emotion_true']==e]['correct'].mean()) for e in emotions},
        'feature_zeros': {col: int(df[col].sum()) for col in ['pitch_is_zero', 'jitter_is_zero', 'shimmer_is_zero']}
    }
    with open('results/diagnosis/deep_analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == '__main__':
    analyze()
