import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_ablation_results(csv_path, out_dir):
    df = pd.read_csv(csv_path)
    os.makedirs(out_dir, exist_ok=True)

    # Melt for plotting
    df_melted = df.melt(id_vars="mode",
                        value_vars=["sentiment_acc", "emotion_acc"],
                        var_name="Metric", value_name="Accuracy")
    df_melted["Metric"] = df_melted["Metric"].replace({
        "sentiment_acc": "Sentiment Accuracy",
        "emotion_acc": "Emotion Accuracy"
    })

    # --- Accuracy Plot ---
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df_melted, x="mode", y="Accuracy", hue="Metric", palette="Set2")
    plt.title("Ablation Study: Accuracy by Mode")
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.xlabel("Mode")
    plt.legend()
    acc_plot_path = os.path.join(out_dir, "ablation_accuracy.png")
    plt.savefig(acc_plot_path)
    plt.close()
    print(f"✅ Accuracy plot saved: {acc_plot_path}")

    # --- F1 Score Plot ---
    df_melted_f1 = df.melt(id_vars="mode",
                           value_vars=["sentiment_f1", "emotion_f1"],
                           var_name="Metric", value_name="F1 Score")
    df_melted_f1["Metric"] = df_melted_f1["Metric"].replace({
        "sentiment_f1": "Sentiment F1",
        "emotion_f1": "Emotion F1"
    })

    plt.figure(figsize=(8, 6))
    sns.barplot(data=df_melted_f1, x="mode", y="F1 Score", hue="Metric", palette="Set1")
    plt.title("Ablation Study: F1 Score by Mode")
    plt.ylim(0, 1.0)
    plt.ylabel("F1 Score")
    plt.xlabel("Mode")
    plt.legend()
    f1_plot_path = os.path.join(out_dir, "ablation_f1.png")
    plt.savefig(f1_plot_path)
    plt.close()
    print(f"✅ F1 plot saved: {f1_plot_path}")

if __name__ == "__main__":
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.dirname(os.path.dirname(script_dir))
    except NameError:
        PROJECT_ROOT = os.getcwd()

    csv_path = os.path.join(PROJECT_ROOT, "ablation_results.csv")
    out_dir = os.path.join(PROJECT_ROOT, "plots")
    plot_ablation_results(csv_path, out_dir)
