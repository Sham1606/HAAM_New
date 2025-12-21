
import json
import os
import pandas as pd
import numpy as np
from scipy import stats
import logging

# Configuration
DATA_DIR = r"D:\haam_framework\data"
METADATA_FILE = os.path.join(DATA_DIR, "hybrid_metadata.csv")
OUTPUT_DIR = r"D:\haam_framework\results\analysis"
REPORT_FILE = os.path.join(OUTPUT_DIR, "hybrid_validation_report.txt")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def main():
    if not os.path.exists(METADATA_FILE):
        return

    df = pd.read_csv(METADATA_FILE)
    df = df[df['emotion_true'] != 'unknown'] # Filter unknown ground truth
    
    # Prepare groups
    crema_correct = (df[df['dataset'] == 'CREMA-D']['emotion_true'] == df[df['dataset'] == 'CREMA-D']['emotion_pred']).astype(int)
    iemocap_correct = (df[df['dataset'] == 'IEMOCAP']['emotion_true'] == df[df['dataset'] == 'IEMOCAP']['emotion_pred']).astype(int)
    
    if len(crema_correct) == 0 or len(iemocap_correct) == 0:
        logger.warning("Insufficient data for statistical tests")
        return

    # 1. T-test on Accuracy (using binary correctness vectors)
    t_stat, p_val = stats.ttest_ind(crema_correct, iemocap_correct)
    effect_size = cohens_d(crema_correct, iemocap_correct)
    
    # 2. Chi-square on Emotion Distribution
    contingency = pd.crosstab(df['dataset'], df['emotion_true'])
    chi2, p_val_chi, dof, expected = stats.chi2_contingency(contingency)
    
    # Generate Report
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write("HAAM FRAMEWORK: HYBRID VALIDATION REPORT\n")
        f.write("========================================\n\n")
        
        f.write("1. DATASET COMPARISON\n")
        f.write(f"   CREMA-D Samples: {len(crema_correct)}\n")
        f.write(f"   IEMOCAP Samples: {len(iemocap_correct)}\n")
        f.write(f"   CREMA-D Accuracy: {crema_correct.mean():.2%}\n")
        f.write(f"   IEMOCAP Accuracy: {iemocap_correct.mean():.2%}\n\n")
        
        f.write("2. STATISTICAL SIGNIFICANCE TESTS\n")
        f.write("   A. Accuracy Difference (T-Test)\n")
        f.write(f"      T-Statistic: {t_stat:.4f}\n")
        f.write(f"      P-Value:     {p_val:.4e}\n")
        f.write(f"      Effect Size: {effect_size:.4f} (Cohen's d)\n")
        sig = "SIGNIFICANT" if p_val < 0.05 else "NOT SIGNIFICANT"
        f.write(f"      Result:      {sig}\n\n")
        
        f.write("   B. Emotion Distribution (Chi-Square)\n")
        f.write(f"      Chi2 Stat:   {chi2:.4f}\n")
        f.write(f"      P-Value:     {p_val_chi:.4e}\n")
        f.write(f"      Result:      {'SIGNIFICANT' if p_val_chi < 0.05 else 'NOT SIGNIFICANT'}\n\n")
        
        f.write("3. CONFIDENCE ANALYSIS\n")
        avg_conf_crema = df[df['dataset'] == 'CREMA-D']['confidence'].mean()
        avg_conf_iemocap = df[df['dataset'] == 'IEMOCAP']['confidence'].mean()
        f.write(f"   Avg Confidence CREMA-D: {avg_conf_crema:.3f}\n")
        f.write(f"   Avg Confidence IEMOCAP: {avg_conf_iemocap:.3f}\n\n")
        
        f.write("4. CONCLUSION\n")
        if p_val < 0.05 and t_stat > 0:
            f.write("   CREMA-D performance is significantly higher than IEMOCAP.\n")
            f.write("   This suggests the model performs better on short, single utterances than conversations.\n")
        elif p_val < 0.05 and t_stat < 0:
            f.write("   IEMOCAP performance is significantly higher than CREMA-D.\n")
        else:
            f.write("   No significant difference in performance between datasets.\n")
    
    logger.info(f"Validation report saved to {REPORT_FILE}")

if __name__ == "__main__":
    main()
