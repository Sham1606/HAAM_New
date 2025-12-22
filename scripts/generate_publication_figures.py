
import os
import json
import logging
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

RESULTS_DIR = r"D:\haam_framework\results"
METRICS_FILE = os.path.join(RESULTS_DIR, "hybrid_model_metrics.json")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

def setup_style():
    """Set aesthetic style for plots."""
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.family'] = 'sans-serif'

def plot_confusion_matrix(data):
    """Figure 1: Confusion Matrix"""
    matrix = np.array(data['confusion_matrix'])
    classes = data['classes']
    
    plt.figure(figsize=(10, 8))
    
    # Normalize for color mapping but show counts
    # Or just raw counts
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    
    plt.title("Confusion Matrix - Hybrid Fusion Model\n(Test Set, N=1540)", fontsize=14, pad=20)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrix.png"))
    plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrix.pdf"))
    logger.info("✓ Confusion matrix saved to results/figures/confusion_matrix.png/pdf")
    plt.close()

def plot_per_class_performance(data):
    """Figure 2: Per-Class Performance"""
    report = data['classification_report']
    classes = data['classes']
    
    records = []
    for cls in classes:
        metrics = report.get(cls, {})
        records.append({'Emotion': cls, 'Metric': 'Precision', 'Score': metrics.get('precision', 0)})
        records.append({'Emotion': cls, 'Metric': 'Recall', 'Score': metrics.get('recall', 0)})
        records.append({'Emotion': cls, 'Metric': 'F1-Score', 'Score': metrics.get('f1-score', 0)})
        
    df = pd.DataFrame(records)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Emotion', y='Score', hue='Metric', data=df, palette='viridis')
    
    plt.title("Per-Class Performance Metrics\n(Hybrid Fusion Model)", fontsize=14, pad=20)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "per_class_performance.png"))
    plt.savefig(os.path.join(FIGURES_DIR, "per_class_performance.pdf"))
    logger.info("✓ Per-class performance saved to results/figures/per_class_performance.png/pdf")
    plt.close()

def plot_class_distribution(data):
    """Figure 3: Class Distribution"""
    report = data['classification_report']
    classes = data['classes']
    
    counts = [report[cls]['support'] for cls in classes]
    labels = [f"{cls.capitalize()} ({int(c)})" for cls, c in zip(classes, counts)]
    
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette('husl', len(classes))
    
    plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
    plt.title("Test Set Class Distribution\n(N=1540 samples)", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "class_distribution.png"))
    logger.info("✓ Class distribution saved to results/figures/class_distribution.png")
    plt.close()

def main():
    print("Generating publication figures...")
    
    if not os.path.exists(METRICS_FILE):
        logger.error(f"Metrics file not found: {METRICS_FILE}")
        return
        
    with open(METRICS_FILE, 'r') as f:
        data = json.load(f)
        
    setup_style()
    plot_confusion_matrix(data)
    plot_per_class_performance(data)
    plot_class_distribution(data)
    
    print("\nAll figures generated successfully!")

if __name__ == "__main__":
    main()
