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
# Combined Plot (Sentiment + Emotion) with Normalized Y-axis
# -------------------------------

def plot_combined_attention(sent_df, emo_df, combined_filename):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    max_y = 1.0  # normalized scale (0 to 1)

    # Sentiment plot
    sns.barplot(data=sent_df, x="True_Sentiment", y="avg_audio_attn",
                color="darkorange", label="Audio", ax=axes[0])
    sns.barplot(data=sent_df, x="True_Sentiment", y="avg_text_attn",
                color="dodgerblue", label="Text",
                bottom=sent_df["avg_audio_attn"], ax=axes[0])
    axes[0].set_title("Per-class Attention (Sentiment)")
    axes[0].set_ylabel("Attention Share")
    axes[0].set_ylim(0, max_y)
    axes[0].legend()

    # Emotion plot
    sns.barplot(data=emo_df, x="True_Emotion", y="avg_audio_attn",
                color="darkorange", label="Audio", ax=axes[1])
    sns.barplot(data=emo_df, x="True_Emotion", y="avg_text_attn",
                color="dodgerblue", label="Text",
                bottom=emo_df["avg_audio_attn"], ax=axes[1])
    axes[1].set_title("Per-class Attention (Emotion)")
    axes[1].set_ylabel("")
    axes[1].set_ylim(0, max_y)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(combined_filename)
    plt.close()
    print(f"📊 Normalized combined attention plot saved to: {combined_filename}")


# -------------------------------
# Main
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
    NUM_CONTEXT_CLASSES = 4

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

            s_logit, e_logit, attn = model(audio, text)

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
    out_csv = os.path.join(PROJECT_ROOT, "predictions_with_attention.csv")
    df.to_csv(out_csv, index=False)

    # -------------------------------
    # Per-class average attention
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

    # Save CSVs
    emo_csv = os.path.join(PROJECT_ROOT, "per_class_emotion_attention.csv")
    sent_csv = os.path.join(PROJECT_ROOT, "per_class_sentiment_attention.csv")
    emo_attention.to_csv(emo_csv, index=False)
    sent_attention.to_csv(sent_csv, index=False)

    # Save plots
    emo_plot = os.path.join(PROJECT_ROOT, "per_class_emotion_attention.png")
    sent_plot = os.path.join(PROJECT_ROOT, "per_class_sentiment_attention.png")
    combined_plot = os.path.join(PROJECT_ROOT, "per_class_attention_combined.png")

    plot_attention_bar(emo_attention, "True_Emotion", "Per-class Attention (Emotion)", emo_plot)
    plot_attention_bar(sent_attention, "True_Sentiment", "Per-class Attention (Sentiment)", sent_plot)
    plot_combined_attention(sent_attention, emo_attention, combined_plot)

    print(f"📊 All attention plots saved:\n  {sent_plot}\n  {emo_plot}\n  {combined_plot}")
