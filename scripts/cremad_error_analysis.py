"""
CREMA-D Error Analysis Script
Analyzes top misclassifications and generates actionable insights.
"""

import pandas as pd
import numpy as np
import json
import os

# Constants
PREDICTIONS_CSV = r"D:\haam_framework\results\validation\cremad_predictions.csv"
ERROR_CSV = r"D:\haam_framework\results\validation\error_analysis.csv"
ERROR_REPORT = r"D:\haam_framework\docs\error_analysis_report.md"

def categorize_error(row):
    """Classify error type based on confidence and patterns."""
    if row['confidence_score'] > 0.8:
        return "high_confidence_error"
    elif row['confidence_score'] < 0.5:
        return "low_confidence_error"
    elif row['confusion_pair'] in ['fear->sadness', 'sadness->fear', 'anger->disgust', 'disgust->anger']:
        return "boundary_confusion"
    else:
        return "general_error"

def determine_modality_bias(row):
    """Determine if error was due to over-reliance on audio or text."""
    if row['audio_attention'] > 0.7:
        return "audio_dominant"
    elif row['text_attention'] > 0.7:
        return "text_dominant"
    else:
        return "balanced"

def analyze_confusion_patterns(errors):
    """Analyze and explain confusion patterns."""
    confusion_counts = errors['confusion_pair'].value_counts()
    insights = ""
    for pair, count in confusion_counts.head(3).items():
        true_emo, pred_emo = pair.split('->')
        insights += f"- **{pair}**: {count} cases. "
        
        if pair in ['fear->sadness', 'sadness->fear']:
            insights += "These emotions share low arousal characteristics. Consider adding energy-based features.\n"
        elif pair in ['anger->disgust', 'disgust->anger']:
            insights += "Both are negative high-arousal emotions. Improve vocal quality features (e.g., roughness).\n"
        else:
            insights += "Review acoustic and linguistic patterns for this pair.\n"
    return insights

def analyze_modality_recommendations(errors):
    """Provide modality-specific recommendations."""
    modality_counts = errors['modality_bias'].value_counts()
    recs = ""
    if 'audio_dominant' in modality_counts and modality_counts['audio_dominant'] > len(errors) * 0.4:
        recs += "- Model over-relies on audio features. Increase text attention weight during training.\n"
    if 'text_dominant' in modality_counts and modality_counts['text_dominant'] > len(errors) * 0.4:
        recs += "- Model over-relies on text features. Improve acoustic feature extraction robustness.\n"
    if not recs:
        recs = "- Modality balance is good. No immediate attention mechanism adjustments needed.\n"
    return recs

def generate_error_report(df, errors, top_errors):
    """Generate detailed markdown error report."""
    total_samples = len(df)
    error_count = len(errors)
    correct_count = total_samples - error_count
    
    report = f"""# CREMA-D Error Analysis Report

## 1. Overview
Total Samples: {total_samples}

- **Errors**: {error_count} ({error_count/total_samples*100:.1f}%)
- **Correct**: {correct_count} ({correct_count/total_samples*100:.1f}%)

## 2. Error Distribution by Type
"""
    error_type_counts = errors['error_type'].value_counts()
    for error_type, count in error_type_counts.items():
        report += f"- **{error_type.replace('_', ' ').title()}**: {count} ({count/error_count*100:.1f}%)\n"

    report += """
## 3. Most Common Confusion Pairs
"""
    confusion_counts = errors['confusion_pair'].value_counts().head(5)
    for idx, (pair, count) in enumerate(confusion_counts.items(), 1):
        report += f"{idx}. **{pair}**: {count} errors\n"

    report += """
## 4. Modality Bias Analysis
"""
    modality_counts = errors['modality_bias'].value_counts()
    for modality, count in modality_counts.items():
        report += f"- **{modality.replace('_', ' ').title()}**: {count} errors ({count/error_count*100:.1f}%)\n"

    report += """
## 5. Top-20 Surprising Misclassifications (Ranked by Confidence)
"""
    for idx, row in top_errors.iterrows():
        report += f"""
### Error #{idx + 1}
**Sample ID**: `{row['call_id']}`
- **Ground Truth**: {row['ground_truth_emotion'].capitalize()}
- **Predicted**: {row['predicted_emotion'].capitalize()} (Confidence: {row['confidence_score']:.2f})
- **Attention Weights**: Audio {row['audio_attention']:.2f}, Text {row['text_attention']:.2f}
- **Error Type**: {row['error_type'].replace('_', ' ').title()}
- **Analysis**: {"Model over-relied on " + row['modality_bias'].replace('_', ' ') + " cues."}
"""

    report += f"""
## 6. Insights & Recommendations

### 6.1 Confusion Patterns
{analyze_confusion_patterns(errors)}

### 6.2 Modality Recommendations
{analyze_modality_recommendations(errors)}

### 6.3 Action Items
1. Implement class weighting for underperforming emotions.
2. Balance audio-text attention mechanism (e.g. Entropy-based gating).
3. Add arousal-based features to distinguish similar emotions.
4. Validate ground truth labels for high-confidence errors.
"""
    with open(ERROR_REPORT, 'w', encoding='utf-8') as f:
        f.write(report)

def analyze_errors():
    """Comprehensive error analysis pipeline."""
    if not os.path.exists(PREDICTIONS_CSV):
        print(f"Predictions file not found: {PREDICTIONS_CSV}")
        return

    df = pd.read_csv(PREDICTIONS_CSV)
    df['correct'] = df['ground_truth_emotion'] == df['predicted_emotion']
    errors = df[~df['correct']].copy()
    
    if len(errors) == 0:
        print("No errors found in predictions!")
        return

    # 1. Categorize
    errors['confusion_pair'] = errors.apply(
        lambda row: f"{row['ground_truth_emotion']}->{row['predicted_emotion']}", axis=1
    )
    errors['error_type'] = errors.apply(categorize_error, axis=1)
    errors['modality_bias'] = errors.apply(determine_modality_bias, axis=1)
    
    # Sort for top errors
    errors = errors.sort_values('confidence_score', ascending=False)
    top_errors = errors.head(20).copy().reset_index()

    # 2. Export CSV
    errors[['call_id', 'ground_truth_emotion', 'predicted_emotion', 'confidence_score',
            'audio_attention', 'text_attention', 'error_type', 'confusion_pair', 
            'modality_bias']].to_csv(ERROR_CSV, index=False)

    # 3. Report
    generate_error_report(df, errors, top_errors)

    print("✓ Error analysis complete")
    print(f"  - CSV: {ERROR_CSV}")
    print(f"  - Report: {ERROR_REPORT}")

if __name__ == "__main__":
    os.makedirs(os.path.dirname(ERROR_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(ERROR_REPORT), exist_ok=True)
    analyze_errors()
