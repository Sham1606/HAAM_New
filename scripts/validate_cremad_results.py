
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
    
    # 1. Confusion Matrix Heatmap
    # Parse confusion pairs into a DataFrame
    # Key format: "Expected -> Detected"
    emotions = sorted(per_emotion.keys())
    matrix_data = {e: {d: 0 for d in emotions} for e in emotions}
    
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
