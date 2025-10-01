import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

# Add project path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.embedding_dataset import MELDEmbeddingDataset
from models.fusion_model import MultimodalFusionModel
from training.train_sprint_model import compute_class_weights, train_model


def run_ablation(mode, project_root, model_save_dir,
                 batch_size=32, lr=1e-5, epochs=20, hidden_dim=1024, dropout=0.3,
                 emotion_loss_weight=3.0, patience=10):

    print(f"\n========== Running Ablation: {mode.upper()} ==========")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    train_dataset = MELDEmbeddingDataset(project_root, data_type="train")
    val_dataset = MELDEmbeddingDataset(project_root, data_type="dev")
    test_dataset = MELDEmbeddingDataset(project_root, data_type="test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Class weights
    sentiment_labels = [s["sentiment_label"].item() for s in train_dataset]
    emotion_labels = [s["emotion_label"].item() for s in train_dataset]
    sentiment_weights = compute_class_weights(sentiment_labels, 3).to(device)
    emotion_weights = compute_class_weights(emotion_labels, 7).to(device)

    # Model
    model = MultimodalFusionModel(
        audio_input_dim=768, text_input_dim=768, hidden_dim=hidden_dim,
        num_sentiment_classes=3, num_emotion_classes=7, dropout_rate=dropout
    ).to(device)

    sentiment_criterion = nn.CrossEntropyLoss(weight=sentiment_weights)
    emotion_criterion = nn.CrossEntropyLoss(weight=emotion_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    # Train
    model_path = os.path.join(model_save_dir, f"sprint_model_{mode}.pth")
    train_model(model, train_loader, val_loader, optimizer,
                sentiment_criterion, emotion_criterion,
                torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5),
                epochs, device, model_path, emotion_loss_weight, patience, mode=mode)

    # Evaluate
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    sent_true, sent_pred, emo_true, emo_pred = [], [], [], []

    with torch.no_grad():
        for batch in test_loader:
            audio, text = batch["audio_embedding"].to(device), batch["text_embedding"].to(device)
            s_labels, e_labels = batch["sentiment_label"].to(device), batch["emotion_label"].to(device)

            s_logits, e_logits, _ = model(audio, text, mode=mode)
            s_preds = torch.argmax(s_logits, dim=1).cpu().numpy()
            e_preds = torch.argmax(e_logits, dim=1).cpu().numpy()

            sent_true.extend(s_labels.cpu().numpy())
            sent_pred.extend(s_preds)
            emo_true.extend(e_labels.cpu().numpy())
            emo_pred.extend(e_preds)

    sent_acc = accuracy_score(sent_true, sent_pred)
    sent_f1 = f1_score(sent_true, sent_pred, average="weighted")
    emo_acc = accuracy_score(emo_true, emo_pred)
    emo_f1 = f1_score(emo_true, emo_pred, average="weighted")

    print(f"Sentiment: Acc={sent_acc:.4f}, F1={sent_f1:.4f}")
    print(f"Emotion:   Acc={emo_acc:.4f}, F1={emo_f1:.4f}")

    return {
        "mode": mode,
        "sentiment_acc": sent_acc,
        "sentiment_f1": sent_f1,
        "emotion_acc": emo_acc,
        "emotion_f1": emo_f1
    }


if __name__ == "__main__":
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.dirname(os.path.dirname(script_dir))
    except NameError:
        PROJECT_ROOT = os.getcwd()

    MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "saved_models")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    results = []
    for mode in ["fusion", "audio", "text"]:
        res = run_ablation(mode, PROJECT_ROOT, MODEL_SAVE_DIR, epochs=10, patience=5)
        results.append(res)

    # Save results
    df = pd.DataFrame(results)
    out_csv = os.path.join(PROJECT_ROOT, "ablation_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n✅ Ablation results saved to {out_csv}")
    print(df)
