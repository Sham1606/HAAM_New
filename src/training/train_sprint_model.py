import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from sklearn.metrics import accuracy_score
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.embedding_dataset import MELDEmbeddingDataset
from models.fusion_model import MultimodalFusionModel


def compute_class_weights(labels, num_classes):
    class_counts = np.bincount(labels, minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1e-5)
    class_weights = class_weights / class_weights.sum() * num_classes
    return torch.tensor(class_weights, dtype=torch.float32)


def train_model(model, train_loader, val_loader, optimizer,
                sentiment_criterion, emotion_criterion,
                scheduler, num_epochs, device,
                model_save_path, emotion_weight,
                patience=10, mode="fusion"):
    best_val_accuracy = 0.0
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} | Mode: {mode} ---")

        # Training
        model.train()
        train_sent_loss, train_emot_loss = 0.0, 0.0
        train_sent_preds, train_sent_labels = [], []
        train_emot_preds, train_emot_labels = [], []

        for batch in train_loader:
            audio = batch['audio_embedding'].to(device)
            text = batch['text_embedding'].to(device)
            sent_labels = batch['sentiment_label'].to(device)
            emot_labels = batch['emotion_label'].to(device)

            sent_logits, emot_logits, _ = model(audio, text, mode=mode)

            loss_sent = sentiment_criterion(sent_logits, sent_labels)
            loss_emot = emotion_criterion(emot_logits, emot_labels)
            total_loss = loss_sent + (loss_emot * emotion_weight)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_sent_loss += loss_sent.item()
            train_emot_loss += loss_emot.item()
            train_sent_preds.extend(torch.argmax(sent_logits, dim=1).cpu().numpy())
            train_sent_labels.extend(sent_labels.cpu().numpy())
            train_emot_preds.extend(torch.argmax(emot_logits, dim=1).cpu().numpy())
            train_emot_labels.extend(emot_labels.cpu().numpy())

        train_sent_acc = accuracy_score(train_sent_labels, train_sent_preds)
        train_emot_acc = accuracy_score(train_emot_labels, train_emot_preds)
        print(f"Train | Sent Loss: {train_sent_loss/len(train_loader):.4f}, "
              f"Acc: {train_sent_acc:.4f} | Emot Loss: {train_emot_loss/len(train_loader):.4f}, "
              f"Acc: {train_emot_acc:.4f}")

        # Validation
        model.eval()
        val_sent_loss, val_emot_loss = 0.0, 0.0
        val_sent_preds, val_sent_labels = [], []
        val_emot_preds, val_emot_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                audio = batch['audio_embedding'].to(device)
                text = batch['text_embedding'].to(device)
                sent_labels = batch['sentiment_label'].to(device)
                emot_labels = batch['emotion_label'].to(device)

                sent_logits, emot_logits, _ = model(audio, text, mode=mode)

                loss_sent = sentiment_criterion(sent_logits, sent_labels)
                loss_emot = emotion_criterion(emot_logits, emot_labels)

                val_sent_loss += loss_sent.item()
                val_emot_loss += loss_emot.item()
                val_sent_preds.extend(torch.argmax(sent_logits, dim=1).cpu().numpy())
                val_sent_labels.extend(sent_labels.cpu().numpy())
                val_emot_preds.extend(torch.argmax(emot_logits, dim=1).cpu().numpy())
                val_emot_labels.extend(emot_labels.cpu().numpy())

        val_sent_acc = accuracy_score(val_sent_labels, val_sent_preds)
        val_emot_acc = accuracy_score(val_emot_labels, val_emot_preds)
        print(f"Val   | Sent Loss: {val_sent_loss/len(val_loader):.4f}, "
              f"Acc: {val_sent_acc:.4f} | Emot Loss: {val_emot_loss/len(val_loader):.4f}, "
              f"Acc: {val_emot_acc:.4f}")

        val_total_loss = val_sent_loss + (val_emot_loss * emotion_weight)
        scheduler.step(val_total_loss)

        current_val_acc = (val_sent_acc + val_emot_acc) / 2
        if current_val_acc > best_val_accuracy:
            best_val_accuracy = current_val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"-> New best model saved to {model_save_path} "
                  f"with combined accuracy: {best_val_accuracy:.4f}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after {epoch + 1} epochs.")
                break


if __name__ == '__main__':
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.dirname(os.path.dirname(script_dir))
    except NameError:
        PROJECT_ROOT = os.getcwd()

    MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, 'saved_models')
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # -------------------
    # Hyperparameters
    # -------------------
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-5
    NUM_EPOCHS = 50
    HIDDEN_DIM = 1024
    DROPOUT = 0.3
    EMOTION_LOSS_WEIGHT = 3.0
    PATIENCE = 15

    AUDIO_DIM = 768
    TEXT_DIM = 768
    NUM_SENTIMENT_CLASSES = 3
    NUM_EMOTION_CLASSES = 7

    # Choose mode: "fusion", "audio", "text"
    MODE = "text"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, Mode: {MODE}")

    # Load datasets
    train_dataset = MELDEmbeddingDataset(PROJECT_ROOT, data_type='train')
    val_dataset = MELDEmbeddingDataset(PROJECT_ROOT, data_type='dev')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Compute class weights
    sentiment_labels = [s['sentiment_label'].item() for s in train_dataset]
    emotion_labels = [s['emotion_label'].item() for s in train_dataset]
    sentiment_weights = compute_class_weights(sentiment_labels, NUM_SENTIMENT_CLASSES).to(device)
    emotion_weights = compute_class_weights(emotion_labels, NUM_EMOTION_CLASSES).to(device)

    # Model
    model = MultimodalFusionModel(
        audio_input_dim=AUDIO_DIM,
        text_input_dim=TEXT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_sentiment_classes=NUM_SENTIMENT_CLASSES,
        num_emotion_classes=NUM_EMOTION_CLASSES,
        dropout_rate=DROPOUT
    ).to(device)

    # Losses & Optimizer
    sentiment_criterion = nn.CrossEntropyLoss(weight=sentiment_weights)
    emotion_criterion = nn.CrossEntropyLoss(weight=emotion_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Save path per mode
    model_path = os.path.join(MODEL_SAVE_DIR, f"sprint_model_v5_best_{MODE}.pth")

    # Train
    train_model(model, train_loader, val_loader, optimizer,
                sentiment_criterion, emotion_criterion,
                scheduler, NUM_EPOCHS, device,
                model_path, EMOTION_LOSS_WEIGHT,
                patience=PATIENCE, mode=MODE)

    print("\n--- Training Complete! ---")
