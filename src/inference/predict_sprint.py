import torch
import os
import sys
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.embedding_dataset import MELDEmbeddingDataset
from models.fusion_model import MultimodalFusionModel


# -------------------------------
# Confusion Matrix Plot
def plot_confusion(y_true, y_pred, labels, title, filename):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# -------------------------------
# Per-class Attention Bar Chart
def plot_attention_bar(df, label_col, title, filename):
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x=label_col, y="avg_audio_attn", color="darkorange", label="Audio")
    sns.barplot(data=df, x=label_col, y="avg_text_attn", color="dodgerblue", label="Text",
                bottom=df["avg_audio_attn"])
    plt.ylabel("Attention Share")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# -------------------------------
# Combined Ablation Attention Plot
# -------------------------------
def plot_ablation_comparison(attention_dfs, label_col, title, filename):
    merged = None
    for mode, df in attention_dfs.items():
        df = df.copy()
        df["mode"] = mode
        merged = df if merged is None else pd.concat([merged, df], axis=0)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=merged, x=label_col, y="avg_audio_attn",
                hue="mode", dodge=True, palette="Set2")
    plt.ylabel("Avg Audio Attention")
    plt.title(title + " — Audio Attention across Modes")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"📊 Ablation comparison plot saved to {filename}")


# -------------------------------
# Evaluate Function
# -------------------------------
def evaluate_mode(model, test_dataset, device, mode, project_root):
    sentiment_preds, sentiment_true = [], []
    emotion_preds, emotion_true = [], []
    rows = []

    model.eval()
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            sample = test_dataset[idx]
            audio = sample['audio_embedding'].unsqueeze(0).to(device)
            text = sample['text_embedding'].unsqueeze(0).to(device)

            s_true = sample['sentiment_label'].item()
            e_true = sample['emotion_label'].item()

            s_logit, e_logit, attn = model(audio, text, mode=mode)

            s_pred = torch.argmax(s_logit, dim=1).item()
            e_pred = torch.argmax(e_logit, dim=1).item()

            sentiment_true.append(s_true)
            sentiment_preds.append(s_pred)
            emotion_true.append(e_true)
            emotion_preds.append(e_pred)

            audio_weight = attn[0, 0].item()

            rows.append({
                "Dialogue_ID": sample['dialogue_id'],
                "Utterance_ID": sample['utterance_id'],
                "True_Emotion": test_dataset.emotion_mapping_inv[e_true],
                "Pred_Emotion": test_dataset.emotion_mapping_inv[e_pred],
                "True_Sentiment": test_dataset.sentiment_mapping_inv[s_true],
                "Pred_Sentiment": test_dataset.sentiment_mapping_inv[s_pred],
                "Audio_Attention": audio_weight,
                "Text_Attention": 1 - audio_weight
            })

    # Save predictions
    df = pd.DataFrame(rows)
    out_csv = os.path.join(project_root, f"predictions_with_attention_{mode}.csv")
    df.to_csv(out_csv, index=False)

    # Per-class attention (Emotion)
    emo_attention = df.groupby("True_Emotion").agg(
        avg_audio_attn=("Audio_Attention", "mean"),
        avg_text_attn=("Text_Attention", "mean"),
        n_samples=("True_Emotion", "count")
    ).reset_index()

    sent_attention = df.groupby("True_Sentiment").agg(
        avg_audio_attn=("Audio_Attention", "mean"),
        avg_text_attn=("Text_Attention", "mean"),
        n_samples=("True_Sentiment", "count")
    ).reset_index()

    emo_csv = os.path.join(project_root, f"per_class_emotion_attention_{mode}.csv")
    sent_csv = os.path.join(project_root, f"per_class_sentiment_attention_{mode}.csv")
    emo_attention.to_csv(emo_csv, index=False)
    sent_attention.to_csv(sent_csv, index=False)

    emo_plot = os.path.join(project_root, f"per_class_emotion_attention_{mode}.png")
    sent_plot = os.path.join(project_root, f"per_class_sentiment_attention_{mode}.png")
    plot_attention_bar(emo_attention, "True_Emotion", f"Per-class Attention (Emotion) - {mode}", emo_plot)
    plot_attention_bar(sent_attention, "True_Sentiment", f"Per-class Attention (Sentiment) - {mode}", sent_plot)

    return emo_attention, sent_attention


# -------------------------------
# Main
# -------------------------------
if __name__ == '__main__':
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.dirname(os.path.dirname(script_dir))
    except NameError:
        PROJECT_ROOT = os.getcwd()

    MODEL_PATH = os.path.join(PROJECT_ROOT, 'saved_models', 'sprint_model_v5_best.pth')

    # Model config
    AUDIO_DIM = 768
    TEXT_DIM = 768
    HIDDEN_DIM = 1024
    DROPOUT = 0.3
    NUM_SENTIMENT_CLASSES = 3
    NUM_EMOTION_CLASSES = 7

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = MultimodalFusionModel(
        audio_input_dim=AUDIO_DIM,
        text_input_dim=TEXT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_sentiment_classes=NUM_SENTIMENT_CLASSES,
        num_emotion_classes=NUM_EMOTION_CLASSES,
        dropout_rate=DROPOUT
    ).to(device)

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file not found at {MODEL_PATH}")
        exit()

    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    print("✅ Model loaded.")

    # Load test dataset
    test_dataset = MELDEmbeddingDataset(PROJECT_ROOT, data_type='test')
    print(f"Loaded {len(test_dataset)} test samples.")

    # Run for all modes
    modes = ["fusion", "audio", "text"]
    emo_results, sent_results = {}, {}

    for mode in modes:
        print(f"\n🔎 Evaluating in mode: {mode}")
        emo_df, sent_df = evaluate_mode(model, test_dataset, device, mode, PROJECT_ROOT)
        emo_results[mode] = emo_df
        sent_results[mode] = sent_df

    # Combined Ablation Plots
    emo_ablation_plot = os.path.join(PROJECT_ROOT, "per_class_emotion_attention_ablation.png")
    sent_ablation_plot = os.path.join(PROJECT_ROOT, "per_class_sentiment_attention_ablation.png")

    plot_ablation_comparison(emo_results, "True_Emotion", "Ablation Study (Emotion)", emo_ablation_plot)
    plot_ablation_comparison(sent_results, "True_Sentiment", "Ablation Study (Sentiment)", sent_ablation_plot)

    print(f"\n📊 Ablation study completed. Results saved:\n"
          f"  Emotion comparison -> {emo_ablation_plot}\n"
          f"  Sentiment comparison -> {sent_ablation_plot}")
