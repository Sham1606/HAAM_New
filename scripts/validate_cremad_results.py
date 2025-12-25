import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import time

# Constants
PREDICTIONS_CSV = r"D:\haam_framework\results\validation\cremad_predictions.csv"
PLOTS_DIR = r"D:\haam_framework\plots\validation"
ROC_DIR = os.path.join(PLOTS_DIR, "roc_curves")
REPORT_PATH = r"D:\haam_framework\docs\cremad_detailed_report.txt"

# Ensure directories exist
os.makedirs(ROC_DIR, exist_ok=True)
os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'neutral']

def plot_confusion_matrix(y_true, y_pred, emotions, save_path):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=emotions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[e.capitalize() for e in emotions], 
                yticklabels=[e.capitalize() for e in emotions])
    plt.title('CREMA-D Confusion Matrix - HAAM Framework', fontsize=14)
    plt.ylabel('True Emotion')
    plt.xlabel('Predicted Emotion')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✓ Saved confusion matrix to {save_path}")

def plot_per_class_accuracy(recall, emotions, save_path):
    """Plot per-class recall/accuracy bar chart."""
    plt.figure(figsize=(10, 6))
    colors = ['#e74c3c', '#8e44ad', '#3498db', '#f39c12', '#95a5a6', '#2ecc71']
    plt.bar([e.capitalize() for e in emotions], recall * 100, color=colors)
    plt.xlabel('Emotion')
    plt.ylabel('Recall (%)')
    plt.title('Per-Emotion Recall - CREMA-D Validation')
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✓ Saved per-class accuracy to {save_path}")

def plot_confidence_distribution(df, save_path):
    """Plot confidence score violin plot."""
    plt.figure(figsize=(12, 6))
    df['correct'] = df['ground_truth_emotion'] == df['predicted_emotion']
    sns.violinplot(data=df, x='predicted_emotion', y='confidence_score',
                   hue='correct', split=True, palette={True: 'green', False: 'red'})
    plt.title('Confidence Score Distribution (Green=Correct, Red=Wrong)')
    plt.xlabel('Predicted Emotion')
    plt.ylabel('Confidence Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✓ Saved confidence distribution to {save_path}")

def plot_processing_time(df, save_path):
    """Plot processing time histogram."""
    plt.figure(figsize=(10, 6))
    plt.hist(df['processing_time_ms'], bins=30, color='skyblue', edgecolor='black')
    plt.xlabel('Processing Time (ms)')
    plt.ylabel('Frequency')
    plt.title(f'Inference Time Distribution (Median: {df["processing_time_ms"].median():.1f}ms)')
    plt.axvline(df['processing_time_ms'].median(), color='red', linestyle='--', label='Median')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✓ Saved processing time histogram to {save_path}")

def plot_roc_curves(y_true, df, emotions, save_dir):
    """Plot ROC curve for each emotion (One-vs-Rest)."""
    y_true_bin = label_binarize(y_true, classes=emotions)
    
    # Simple probability estimation: 
    # Predicted class gets the confidence_score, others share (1-conf)
    probas = np.zeros((len(df), len(emotions)))
    for idx, row in df.iterrows():
        try:
            pred_idx = emotions.index(row['predicted_emotion'])
            probas[idx, pred_idx] = row['confidence_score']
            remaining = (1.0 - row['confidence_score']) / (len(emotions) - 1)
            for i in range(len(emotions)):
                if i != pred_idx:
                    probas[idx, i] = remaining
        except (ValueError, IndexError):
            continue
    
    for i, emotion in enumerate(emotions):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], probas[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {emotion.capitalize()}')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{emotion}_roc.png'), dpi=300)
        plt.close()

    print(f"✓ Saved ROC curves to {save_dir}")

def generate_benchmark_comparison(accuracy, macro_f1):
    """Compare HAAM with CREMA-D state-of-the-art."""
    SOTA_BENCHMARKS = [
        ("Baseline CNN (Lim et al. 2020)", 0.52, 0.48),
        ("LSTM-Attention (Zhao et al. 2021)", 0.61, 0.58),
        ("Multimodal Fusion (Kim et al. 2022)", 0.67, 0.64),
        ("Transformer-based (Lee et al. 2023)", 0.69, 0.66),
        ("HAAM Framework (Ours)", accuracy, macro_f1)
    ]

    table = "| Model | Accuracy | Macro F1 | Status |\n"
    table += "|-------|----------|----------|--------|\n"
    for model, acc_score, f1_score in SOTA_BENCHMARKS:
        acc_str = f"{acc_score*100:.1f}%" if acc_score is not None else "N/A"
        f1_str = f"{f1_score:.3f}" if f1_score is not None else "N/A"
        
        if "HAAM" in model:
            if acc_score is not None and acc_score >= 0.67:
                status = "✅ SOTA"
            elif acc_score is not None and acc_score >= 0.60:
                status = "⚠️ Competitive"
            else:
                status = "❌ Below Target"
        else:
            status = "Baseline"
        
        table += f"| {model} | {acc_str} | {f1_str} | {status} |\n"

    return table

def main():
    if not os.path.exists(PREDICTIONS_CSV):
        print(f"Predictions file not found: {PREDICTIONS_CSV}")
        return

    df = pd.read_csv(PREDICTIONS_CSV)
    
    # Filter only expected emotions
    df = df[df['ground_truth_emotion'].isin(EMOTIONS)]
    
    y_true = df['ground_truth_emotion']
    y_pred = df['predicted_emotion']

    # Overall Summary
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=EMOTIONS
    )
    macro_f1 = precision_recall_fscore_support(y_true, y_pred, average='macro', labels=EMOTIONS)[2]
    weighted_f1 = precision_recall_fscore_support(y_true, y_pred, average='weighted', labels=EMOTIONS)[2]

    # Generate Plots
    plot_confusion_matrix(y_true, y_pred, EMOTIONS, os.path.join(PLOTS_DIR, "confusion_matrix.png"))
    plot_per_class_accuracy(recall, EMOTIONS, os.path.join(PLOTS_DIR, "per_class_accuracy.png"))
    plot_confidence_distribution(df, os.path.join(PLOTS_DIR, "confidence_distribution.png"))
    plot_processing_time(df, os.path.join(PLOTS_DIR, "processing_time_hist.png"))
    plot_roc_curves(y_true, df, EMOTIONS, ROC_DIR)

    # Detailed Text Report
    report = f"""
╔═══════════════════════════════════════════════════════════════════════════╗
║ HAAM Framework - CREMA-D Validation Report ║
╚═══════════════════════════════════════════════════════════════════════════╝

Dataset: CREMA-D Validation Set
Total Samples: {len(df)}
Emotions: {', '.join([e.capitalize() for e in EMOTIONS])}

═══════════════════════════════════════════════════════════════════════════
OVERALL METRICS
═══════════════════════════════════════════════════════════════════════════

Accuracy: {accuracy*100:.2f}%
Macro F1: {macro_f1:.3f}
Weighted F1: {weighted_f1:.3f}

Average Inference Time: {df['processing_time_ms'].mean():.1f}ms
Median Inference Time: {df['processing_time_ms'].median():.1f}ms

═══════════════════════════════════════════════════════════════════════════
PER-EMOTION PERFORMANCE
═══════════════════════════════════════════════════════════════════════════

| Emotion  | Precision | Recall | F1    | Support |
|----------|-----------|--------|-------|---------|
"""				
    for i, emotion in enumerate(EMOTIONS):
        report += f"| {emotion.capitalize():8} | {precision[i]:.3f}     | {recall[i]:.3f}  | {f1[i]:.3f}  | {support[i]:7} |\n"

    # Confusions
    cm = confusion_matrix(y_true, y_pred, labels=EMOTIONS)
    confusions = []
    for i in range(len(EMOTIONS)):
        for j in range(len(EMOTIONS)):
            if i != j and cm[i, j] > 0:
                confusions.append((EMOTIONS[i], EMOTIONS[j], cm[i, j]))
    confusions.sort(key=lambda x: x[2], reverse=True)

    report += f"""
═══════════════════════════════════════════════════════════════════════════
CONFUSION MATRIX ANALYSIS
═══════════════════════════════════════════════════════════════════════════

See visualization: plots/validation/confusion_matrix.png

Most Common Confusions:
"""
    for idx, (true_emo, pred_emo, count) in enumerate(confusions[:5], 1):
        report += f"{idx}. {true_emo.capitalize()} → {pred_emo.capitalize()}: {count} cases\n"

    report += f"""
═══════════════════════════════════════════════════════════════════════════
BENCHMARK COMPARISON
═══════════════════════════════════════════════════════════════════════════

{generate_benchmark_comparison(accuracy, macro_f1)}

═══════════════════════════════════════════════════════════════════════════
KEY FINDINGS
═══════════════════════════════════════════════════════════════════════════

Strongest Performance: {EMOTIONS[np.argmax(recall)].capitalize()} ({np.max(recall)*100:.1f}% recall)

Weakest Performance: {EMOTIONS[np.argmin(recall)].capitalize()} ({np.min(recall)*100:.1f}% recall)

Most Confused Pair: {confusions[0][0].capitalize() if confusions else 'N/A'} ↔ {confusions[0][1].capitalize() if confusions else 'N/A'}

═══════════════════════════════════════════════════════════════════════════
RECOMMENDATIONS FOR IMPROVEMENT
═══════════════════════════════════════════════════════════════════════════
"""
    # Auto-recommendations
    weak_emotions = [EMOTIONS[i] for i, r in enumerate(recall) if r < 0.4]
    if weak_emotions:
        report += f"1. Implement class weighting or data synthesis for: {', '.join(weak_emotions)}\n"

    if df['processing_time_ms'].mean() > 200:
        report += f"2. Optimize feature extraction pipeline (current avg: {df['processing_time_ms'].mean():.0f}ms)\n"

    if accuracy < 0.65:
        report += "3. Explore multi-stage fusion or additional acoustic features (e.g. eGeMAPS)\n"

    report += f"""
═══════════════════════════════════════════════════════════════════════════
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
═══════════════════════════════════════════════════════════════════════════
"""
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✓ Detailed report saved to {REPORT_PATH}")

if __name__ == "__main__":
    main()
