
import json
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import logging

# Configuration
DATA_DIR = r"D:\haam_framework\data"
METADATA_FILE = os.path.join(DATA_DIR, "hybrid_metadata.csv")
OUTPUT_DIR = r"D:\haam_framework\results\analysis"
METRICS_FILE = os.path.join(OUTPUT_DIR, "evaluation_metrics.json")

EMOTIONS = ['neutral', 'joy', 'sadness', 'anger', 'fear', 'disgust']

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=EMOTIONS)
    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title(f'Confusion Matrix: {title}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()

def calculate_metrics(df, dataset_name):
    y_true = df['emotion_true']
    y_pred = df['emotion_pred']
    
    # Check if we have ground truth
    if y_true.iloc[0] == 'unknown':
        logger.warning(f"No ground truth for {dataset_name}, skipping metrics.")
        return {}

    # Overall metrics
    wa = accuracy_score(y_true, y_pred)
    
    # Classification Report
    report = classification_report(y_true, y_pred, target_names=EMOTIONS, output_dict=True, zero_division=0)
    
    # Calculate Unweighted Accuracy (UA)
    class_accuracies = []
    cm = confusion_matrix(y_true, y_pred, labels=EMOTIONS)
    for i in range(len(EMOTIONS)):
        # Accuracy = TP / Total True
        total = cm[i, :].sum()
        if total > 0:
            acc = cm[i, i] / total
            class_accuracies.append(acc)
    ua = np.mean(class_accuracies) if class_accuracies else 0.0
    
    return {
        "weighted_accuracy": wa,
        "unweighted_accuracy": ua,
        "macro_avg": report['macro avg'],
        "weighted_avg": report['weighted avg'],
        "per_emotion": {e: report[e] for e in EMOTIONS if e in report}
    }

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(METADATA_FILE):
        logger.error("Metadata file not found. Run create_hybrid_metadata.py first.")
        return

    df = pd.read_csv(METADATA_FILE)
    
    # Filter out unknowns
    df = df[df['emotion_true'] != 'unknown']
    
    results = {}
    
    # CREMA-D
    logger.info("Evaluating CREMA-D...")
    df_crema = df[df['dataset'] == 'CREMA-D']
    if not df_crema.empty:
        results['crema_d'] = calculate_metrics(df_crema, 'CREMA-D')
        plot_confusion_matrix(df_crema['emotion_true'], df_crema['emotion_pred'], 
                            'CREMA-D', 'confusion_matrix_crema_d.png')
        
    # IEMOCAP
    logger.info("Evaluating IEMOCAP...")
    df_iemocap = df[df['dataset'] == 'IEMOCAP']
    if not df_iemocap.empty:
        results['iemocap'] = calculate_metrics(df_iemocap, 'IEMOCAP')
        plot_confusion_matrix(df_iemocap['emotion_true'], df_iemocap['emotion_pred'], 
                            'IEMOCAP', 'confusion_matrix_iemocap.png')
        
    # Combined
    logger.info("Evaluating Combined...")
    results['combined'] = calculate_metrics(df, 'Combined')
    plot_confusion_matrix(df['emotion_true'], df['emotion_pred'], 
                        'Hybrid Dataset', 'confusion_matrix_combined.png')
    
    # Save Metrics
    with open(METRICS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
        
    # Generate Comparison Chart
    if 'crema_d' in results and 'iemocap' in results:
        datasets = ['CREMA-D', 'IEMOCAP', 'Combined']
        wa_scores = [results['crema_d']['weighted_accuracy'], 
                     results['iemocap']['weighted_accuracy'], 
                     results['combined']['weighted_accuracy']]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(datasets, wa_scores, color=['#3498db', '#2ecc71', '#9b59b6'])
        plt.ylim(0, 1.0)
        plt.title('Overall Accuracy Comparison')
        plt.ylabel('Weighted Accuracy')
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1%}',
                     ha='center', va='bottom')
                     
        plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_comparison.png'), dpi=300)
        plt.close()

    logger.info(f"Evaluation complete. Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
