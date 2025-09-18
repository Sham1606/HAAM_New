import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
import time
# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.embedding_dataset import MELDEmbeddingDataset
from models.fusion_model import MultimodalFusionModel

# -----------------------
# Focal loss for imbalanced emotion classes
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

# -----------------------
def compute_class_weights(labels, num_classes):
    class_counts = np.bincount(labels, minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1e-5)
    class_weights = class_weights / class_weights.sum() * num_classes
    return torch.tensor(class_weights, dtype=torch.float32)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def train_model(model, train_loader, val_loader, optimizer, sentiment_criterion,
                emotion_criterion, scheduler, num_epochs, device, model_save_path,
                emotion_weight, patience=15, augment_audio=False):
    best_val_accuracy = 0.0
    epochs_without_improvement = 0
    history = {"epoch": [], "train_sent_acc": [], "train_emo_acc": [], "val_sent_acc": [], "val_emo_acc": [], "avg_attn_audio": []}

    for epoch in range(num_epochs):
        t0 = time.time()
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        model.train()
        train_sent_loss = 0.0
        train_emo_loss = 0.0
        train_sent_preds, train_sent_labels = [], []
        train_emo_preds, train_emo_labels = [], []
        attn_accum = []

        for batch in train_loader:
            audio_embeddings = batch['audio_embedding'].to(device)
            text_embeddings = batch['text_embedding'].to(device)
            sentiment_labels = batch['sentiment_label'].to(device)
            emotion_labels = batch['emotion_label'].to(device)

            # small Gaussian noise augmentation on audio embeddings (optional)
            if augment_audio:
                audio_embeddings = audio_embeddings + (0.01 * torch.randn_like(audio_embeddings).to(device))

            sentiment_logits, emotion_logits, attn_w = model(audio_embeddings, text_embeddings)

            loss_sent = sentiment_criterion(sentiment_logits, sentiment_labels)
            loss_emo = emotion_criterion(emotion_logits, emotion_labels)
            loss = loss_sent + (loss_emo * emotion_weight)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_sent_loss += loss_sent.item()
            train_emo_loss += loss_emo.item()

            train_sent_preds.extend(torch.argmax(sentiment_logits, dim=1).cpu().numpy())
            train_sent_labels.extend(sentiment_labels.cpu().numpy())
            train_emo_preds.extend(torch.argmax(emotion_logits, dim=1).cpu().numpy())
            train_emo_labels.extend(emotion_labels.cpu().numpy())

            attn_accum.append(attn_w[:,0].detach().cpu().numpy())  # audio weights

        train_sent_acc = accuracy_score(train_sent_labels, train_sent_preds) if len(train_sent_labels)>0 else 0.0
        train_emo_acc = accuracy_score(train_emo_labels, train_emo_preds) if len(train_emo_labels)>0 else 0.0
        avg_attn_audio = float(np.concatenate(attn_accum).mean()) if len(attn_accum)>0 else 0.0

        print(f"Train | Sent Loss: {train_sent_loss/len(train_loader):.4f}, Acc: {train_sent_acc:.4f} | Emo Loss: {train_emo_loss/len(train_loader):.4f}, Acc: {train_emo_acc:.4f}")
        print(f"        Avg audio attention (train): {avg_attn_audio:.3f}")

        # Validation
        model.eval()
        val_sent_loss = 0.0
        val_emo_loss = 0.0
        val_sent_preds, val_sent_labels = [], []
        val_emo_preds, val_emo_labels = [], []
        val_attn_accum = []

        with torch.no_grad():
            for batch in val_loader:
                audio_embeddings = batch['audio_embedding'].to(device)
                text_embeddings = batch['text_embedding'].to(device)
                sentiment_labels = batch['sentiment_label'].to(device)
                emotion_labels = batch['emotion_label'].to(device)

                sentiment_logits, emotion_logits, attn_w = model(audio_embeddings, text_embeddings)

                loss_sent = sentiment_criterion(sentiment_logits, sentiment_labels)
                loss_emo = emotion_criterion(emotion_logits, emotion_labels)

                val_sent_loss += loss_sent.item()
                val_emo_loss += loss_emo.item()

                val_sent_preds.extend(torch.argmax(sentiment_logits, dim=1).cpu().numpy())
                val_sent_labels.extend(sentiment_labels.cpu().numpy())
                val_emo_preds.extend(torch.argmax(emotion_logits, dim=1).cpu().numpy())
                val_emo_labels.extend(emotion_labels.cpu().numpy())

                val_attn_accum.append(attn_w[:,0].detach().cpu().numpy())

        val_sent_acc = accuracy_score(val_sent_labels, val_sent_preds) if len(val_sent_labels)>0 else 0.0
        val_emo_acc = accuracy_score(val_emo_labels, val_emo_preds) if len(val_emo_labels)>0 else 0.0
        avg_attn_val = float(np.concatenate(val_attn_accum).mean()) if len(val_attn_accum)>0 else 0.0

        print(f"Val   | Sent Loss: {val_sent_loss/len(val_loader):.4f}, Acc: {val_sent_acc:.4f} | Emo Loss: {val_emo_loss/len(val_loader):.4f}, Acc: {val_emo_acc:.4f}")
        print(f"        Avg audio attention (val): {avg_attn_val:.3f}")

        # Scheduler step (CosineAnnealingLR step uses epoch)
        if scheduler is not None:
            scheduler.step()

        combined_val_acc = (val_sent_acc + val_emo_acc) / 2.0
        history["epoch"].append(epoch+1)
        history["train_sent_acc"].append(train_sent_acc)
        history["train_emo_acc"].append(train_emo_acc)
        history["val_sent_acc"].append(val_sent_acc)
        history["val_emo_acc"].append(val_emo_acc)
        history["avg_attn_audio"].append(avg_attn_val)

        # checkpoint
        if combined_val_acc > best_val_accuracy:
            best_val_accuracy = combined_val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"-> New best model saved to {model_save_path} (combined acc={best_val_accuracy:.4f})")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        t1 = time.time()
        print(f"Epoch time: {(t1-t0):.1f}s")

    # Save training history
    hist_path = os.path.join(os.path.dirname(model_save_path), "training_history.json")
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {hist_path}")

if __name__ == '__main__':
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.dirname(os.path.dirname(script_dir))
    except NameError:
        PROJECT_ROOT = os.getcwd()

    MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, 'saved_models')
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # Hyperparameters
    BATCH_SIZE = 32
    LR = 2e-5
    NUM_EPOCHS = 50
    HIDDEN_DIM = 1024
    DROPOUT = 0.3
    EMOTION_LOSS_WEIGHT = 3.0
    PATIENCE = 15
    AUGMENT_AUDIO = True  # small Gaussian noise augmentation during training

    NUM_SENTIMENT_CLASSES = 3
    NUM_EMOTION_CLASSES = 7

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Datasets
    train_dataset = MELDEmbeddingDataset(project_root=PROJECT_ROOT, data_type='train')
    val_dataset = MELDEmbeddingDataset(project_root=PROJECT_ROOT, data_type='dev')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Class weights
    sentiment_labels = [sample['sentiment_label'].item() for sample in train_dataset]
    emotion_labels = [sample['emotion_label'].item() for sample in train_dataset]
    sentiment_weights = compute_class_weights(sentiment_labels, NUM_SENTIMENT_CLASSES).to(device)
    emotion_weights = compute_class_weights(emotion_labels, NUM_EMOTION_CLASSES).to(device)

    # Model
    model = MultimodalFusionModel(
        hidden_dim=HIDDEN_DIM,
        num_sentiment_classes=NUM_SENTIMENT_CLASSES,
        num_emotion_classes=NUM_EMOTION_CLASSES,
        dropout_rate=DROPOUT
    ).to(device)
    model.apply(init_weights)

    # Losses & optimizer
    sentiment_criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=sentiment_weights)
    emotion_criterion = FocalLoss(gamma=2.0, weight=emotion_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)

    # Scheduler: Cosine annealing
    scheduler = CosineAnnealingLR(optimizer, T_max=10)

    model_path = os.path.join(MODEL_SAVE_DIR, 'sprint_model_v5_best.pth')
    train_model(model, train_loader, val_loader, optimizer,
                sentiment_criterion, emotion_criterion, scheduler,
                NUM_EPOCHS, device, model_path, EMOTION_LOSS_WEIGHT,
                patience=PATIENCE, augment_audio=AUGMENT_AUDIO)

    print("\n--- Training Complete! ---")
