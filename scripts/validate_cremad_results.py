
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Configuration
REPORT_FILE = r"D:\haam_framework\cremad_validation_report.json"
DOCS_DIR = r"D:\haam_framework\docs"

def main():
    print("Generating validation visualizations...")
    
    if not os.path.exists(REPORT_FILE):
        print(f"Report file not found: {REPORT_FILE}")
        return
        
    os.makedirs(DOCS_DIR, exist_ok=True)
    
    with open(REPORT_FILE, 'r') as f:
        report = json.load(f)
        
    summary = report['summary']
    per_emotion = report['per_emotion_accuracy']
    confusion = report['confusion_pairs']
    
    # NOTE: The following statistical analysis assumes 'validation_data' is available,
    # which should be a list of dictionaries like [{'expected': 'happy', 'detected': 'sad', 'match': False}, ...]
    # This variable is not defined in the provided context. You may need to load it from the report.
    # For example: validation_data = report.get('predictions', [])
    
    # 1. Confusion Matrix Heatmap
    # Parse    # Confusion Matrix
    print("\nConfusion Matrix:")
    
    # Get all emotions
    # This part assumes 'validation_data' is available. If not, it will cause an error.
    # As a placeholder, using emotions from per_emotion if validation_data is not present.

    # Construct validation_data list for detailed analysis
    # Construct validation_data list for detailed analysis
    validation_data = []
    
    # Reconstruct synthetic validation_data from per_emotion_stats for CI calculation
    for emo, stats in per_emotion.items():
        total = stats['total']
        correct = stats['correct']
        incorrect = total - correct
        
        # Append correct instances
        for _ in range(correct):
            validation_data.append({'expected': emo, 'detected': emo, 'match': True})
            
        # Append incorrect instances (we don't know *what* they were detected as, only that they were wrong)
        # But for CONFIDENCE INTERVALS on ACCURACY, we only need Match/NoMatch.
        for _ in range(incorrect):
            validation_data.append({'expected': emo, 'detected': 'unknown', 'match': False})
            
    # For Chi-Square on the Matrix, we need the full matrix from confusion_pairs
    unique_emotions = sorted(per_emotion.keys())
    matrix_data = {e: {d: 0 for d in unique_emotions} for e in unique_emotions}
    
    for pair, count in confusion.items():
        parts = pair.split(' -> ')
        if len(parts) == 2:
            exp, det = parts
            if exp in matrix_data and det in matrix_data[exp]:
                matrix_data[exp][det] = count
                
    conf_matrix = pd.DataFrame(matrix_data).T.fillna(0) # Rows=Expected, Cols=Detected
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # Statistical Significance Analysis (Chi-Square)
    try:
        from scipy import stats
        import numpy as np
        print("\nStatistical Significance Analysis:")
        print("-" * 70)

        # Chi-square test
        total_obs = conf_matrix.sum().sum()
        if total_obs > 0:
            expected_random = total_obs / len(unique_emotions)
            
            chi2_statistic = 0
            for emotion in unique_emotions:
                observed = conf_matrix.loc[emotion, emotion]
                # Goodness of fit for "Diagonal vs Random"? 
                # Or standard Chi-Square test of independence for the whole matrix?
                # The user requested "Chi-square test for overall performance" comparing diagonal to random.
                # observed diagonal vs expected random diagonal.
                chi2_statistic += (observed - expected_random) ** 2 / expected_random

            p_value = 1 - stats.chi2.cdf(chi2_statistic, df=len(unique_emotions)-1)

            print(f"Chi-square statistic: {chi2_statistic:.2f}")
            print(f"P-value: {p_value:.4f}")

            if p_value < 0.001:
                print("✅ Results are HIGHLY statistically significant (p < 0.001)")
            elif p_value < 0.05:
                print("✅ Results are statistically significant (p < 0.05)")
            else:
                print("⚠️ Results may not be statistically significant")
            
        # 95% Confidence Intervals
        print("\n95% Confidence Intervals:")
        print("-" * 70)

        for emotion in unique_emotions:
            stats_emo = per_emotion.get(emotion, {'total':0, 'correct':0})
            n = stats_emo['total']
            correct = stats_emo['correct']
            
            if n == 0:
                print(f"{emotion:10s}: No data")
                continue
                
            accuracy = correct / n
            
            # Wilson score interval
            z = 1.96
            denominator = 1 + z**2 / n
            centre = (accuracy + z**2 / (2*n)) / denominator
            adjustment = z * np.sqrt((accuracy * (1 - accuracy) + z**2 / (4*n)) / n) / denominator
            
            ci_lower = max(0, centre - adjustment)
            ci_upper = min(1, centre + adjustment)
            
            print(f"{emotion:10s}: {accuracy:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
            
    except ImportError:
        print("\nSkipping statistical analysis (scipy or numpy not installed)")
    except Exception as e:
        print(f"\nStats error: {e}")
    # End of new statistical analysis block
    
    # Original confusion matrix heatmap generation (using df_cm from earlier)
    # If validation_data was present, df_cm would need to be created from conf_matrix
    # For now, assuming the original df_cm generation logic is still desired for the plot
    # and the new conf_matrix is for console output.
    
    # Re-creating df_cm for the plot, ensuring it's always available
    emotions = sorted(per_emotion.keys())
    matrix_data = {e: {d: 0 for d in emotions} for e in emotions} # Initialize with all emotions
    
    for pair, count in confusion.items():
        parts = pair.split(' -> ')
        if len(parts) == 2:
            expected, detected = parts
            if expected in matrix_data and detected in matrix_data[expected]:
                matrix_data[expected][detected] = count
                
    df_cm = pd.DataFrame(matrix_data).T # Transpose so Rows=Expected, Cols=Detected
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Emotion Detection Confusion Matrix')
    plt.ylabel('Expected Emotion')
    plt.xlabel('Detected Emotion')
    
    cm_path = os.path.join(DOCS_DIR, "cremad_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"✅ Confusion matrix saved: {cm_path}")
    
    # 2. Accuracy Bar Chart
    # Prepare data
    emo_labels = []
    accuracies = []
    
    # Sort by accuracy descending
    sorted_stats = sorted(per_emotion.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    for emo, stats in sorted_stats:
        emo_labels.append(emo)
        accuracies.append(stats['accuracy'])
        
    plt.figure(figsize=(10, 6))
    colors = ['green' if acc >= 70 else 'red' for acc in accuracies]
    bars = plt.bar(emo_labels, accuracies, color=colors)
    
    # Add threshold line
    plt.axhline(y=70, color='gray', linestyle='--', alpha=0.7, label='Threshold (70%)')
    
    # Add values on top
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}%',
                 ha='center', va='bottom')
                 
    plt.title('Emotion Detection Accuracy by Category')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 110)
    plt.legend()
    
    acc_path = os.path.join(DOCS_DIR, "cremad_accuracy_chart.png")
    plt.savefig(acc_path, dpi=300)
    plt.close()
    print(f"✅ Accuracy chart saved: {acc_path}")
    
    # 3. Detailed Text Report
    report_path = os.path.join(DOCS_DIR, "cremad_detailed_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("CREMA-D VALIDATION REPORT\n")
        f.write("=========================\n\n")
        
        f.write("1. DATASET SUMMARY\n")
        f.write(f"   Total Calls Processed: {summary['total_calls']}\n")
        f.write(f"   Correct Predictions:   {summary['correct_predictions']}\n")
        f.write(f"   Overall Accuracy:      {summary['overall_accuracy']}%\n\n")
        
        f.write("2. PER-EMOTION RESULTS\n")
        for emo, stats in sorted_stats:
            acc = stats['accuracy']
            bars = int(acc / 5)
            bar_str = "█" * bars + "░" * (20 - bars)
            f.write(f"   {emo:<10} : {acc:>5.1f}% ({stats['correct']}/{stats['total']}) {bar_str}\n")
        f.write("\n")
        
        f.write("3. COMMON MISCLASSIFICATIONS (Top 10)\n")
        sorted_conf = sorted(confusion.items(), key=lambda x: x[1], reverse=True)
        for pair, count in sorted_conf[:10]:
            f.write(f"   {pair:<20} : {count} times\n")
        f.write("\n")
        
        f.write("4. AGENT RISK PROFILES\n")
        risks = report.get('agent_risk_profiles', [])
        if risks:
            for r in risks:
                f.write(f"   Agent {r['agent_id']}: Risk Level = {r['risk_level']} (Score: {r['risk_score']})\n")
        else:
            f.write("   No risk profiles generated.\n")
            
        f.write("\n5. ASSESSMENT\n")
        if summary['overall_accuracy'] >= 70:
            f.write("   STATUS: PASSED. The model meets the 70% accuracy threshold.\n")
        else:
            f.write("   STATUS: FAILED. The model is below the 70% accuracy threshold.\n")
            
        best_emo = sorted_stats[0][0]
        worst_emo = sorted_stats[-1][0]
        f.write(f"   Best Performance: {best_emo}\n")
        f.write(f"   Worst Performance: {worst_emo}\n")

    print(f"✅ Detailed report saved: {report_path}")
    print("\nValidation complete!")

if __name__ == "__main__":
    main()
