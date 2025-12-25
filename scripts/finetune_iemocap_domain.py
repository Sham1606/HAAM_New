import os
import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import json
import joblib
import argparse
from sklearn.preprocessing import StandardScaler

from src.models.attention_fusion_model import AttentionFusionNetwork
from src.utils.data_augmentation import augment_batch # If using raw audio

# Re-using HybridDataset but with IEMOCAP specific logic if needed
class FinetuneDataset(Dataset):
    def __init__(self, df, feature_dir, emotion_map, scaler=None, augment=False):
        self.df = df.copy()
        self.feature_dir = Path(feature_dir)
        self.emotion_map = emotion_map
        self.scaler = scaler
        self.augment = augment
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feature_path = self.feature_dir / f"{row['call_id']}.pt"
        
        try:
            features = torch.load(feature_path, weights_only=False)
            acoustic = features['acoustic']
            
            if self.scaler:
                acoustic = self.scaler.transform(acoustic.reshape(1, -1))[0]
            
            # Simple feature-level augmentation if raw audio isn't used
            if self.augment and np.random.random() > 0.5:
                noise = np.random.normal(0, 0.05, acoustic.shape)
                acoustic = acoustic + noise
                
            acoustic = torch.tensor(acoustic, dtype=torch.float32)
            text = torch.tensor(features['text_embedding'], dtype=torch.float32).squeeze()
            
            emotion = row['emotion'].lower()
            label = self.emotion_map.get(emotion, 0)
            
            return acoustic, text, label
        except Exception as e:
            return torch.zeros(12), torch.zeros(768), 0

def evaluate(model, loader, device, num_classes=5):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for acoustic, text, labels in loader:
            acoustic, text, labels = acoustic.to(device), text.to(device), labels.to(device)
            outputs, _ = model(acoustic, text)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return accuracy_score(all_labels, all_preds), all_preds, all_labels

def main():
    parser = argparse.ArgumentParser(description="IEMOCAP Domain Fine-tuning")
    parser.add_argument("--pretrained-model", type=str, required=True, help="Path to CREMA-D checkpoint")
    parser.add_argument("--iemocap-data", type=str, default="data/processed/features_v3_librosa", help="Feature directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--freeze-encoder", type=bool, default=True)
    parser.add_argument("--save-path", type=str, default="saved_models/iemocap_finetuned.pth")
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Fine-tuning on {DEVICE}...")

    emotion_map = {'neutral': 0, 'anger': 1, 'disgust': 2, 'fear': 3, 'sadness': 4}
    
    # 1. Load Data
    df = pd.read_csv('data/real_metadata_with_split.csv')
    df_iemocap = df[df['dataset'] == 'IEMOCAP']
    
    if len(df_iemocap) == 0:
        print("No IEMOCAP data found in metadata!")
        return

    train_df = df_iemocap[df_iemocap['split'] == 'train']
    val_df = df_iemocap[df_iemocap['split'] == 'val']
    test_df = df_iemocap[df_iemocap['split'] == 'test']
    
    # 2. Setup Model
    model = AttentionFusionNetwork(num_classes=5).to(DEVICE)
    model.load_state_dict(torch.load(args.pretrained_model, map_location=DEVICE))
    print(f"Loaded pre-trained weights from {args.pretrained_model}")

    if args.freeze_encoder:
        model.freeze_encoders()
        
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Load Scaler from CREMA-D training
    scaler = None
    if os.path.exists('models/improved/scaler.pkl'):
        scaler = joblib.load('models/improved/scaler.pkl')

    train_ds = FinetuneDataset(train_df, args.iemocap_data, emotion_map, scaler=scaler, augment=True)
    val_ds = FinetuneDataset(val_df, args.iemocap_data, emotion_map, scaler=scaler, augment=False)
    test_ds = FinetuneDataset(test_df, args.iemocap_data, emotion_map, scaler=scaler, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # 3. Baseline measurement
    print("\nMeasuring Baseline on IEMOCAP...")
    base_acc, _, _ = evaluate(model, test_loader, DEVICE)
    print(f"Baseline IEMOCAP Accuracy: {base_acc:.2%}")

    # 4. Training Loop
    print("\nStarting Fine-tuning Stage...")
    best_acc = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for acoustic, text, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            acoustic, text, labels = acoustic.to(DEVICE), text.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs, _ = model(acoustic, text)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        val_acc, _, _ = evaluate(model, val_loader, DEVICE)
        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f} Val Acc: {val_acc:.2%}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            print(f"Saved best model with Val Acc: {val_acc:.2%}")

    # 5. Final Evaluation
    print("\nFinal Evaluation on IEMOCAP Test Set...")
    model.load_state_dict(torch.load(args.save_path))
    test_acc, preds, labels = evaluate(model, test_loader, DEVICE)
    print(f"Post-Adaptation Accuracy: {test_acc:.2%}")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=list(emotion_map.keys())))

    # Create directories if missing
    os.makedirs("results/domain_adaptation", exist_ok=True)
    os.makedirs("plots/domain_adaptation", exist_ok=True)

    # Save metrics
    results = {
        "baseline_acc": base_acc,
        "finetuned_acc": test_acc,
        "improvement": test_acc - base_acc
    }
    with open("results/domain_adaptation/finetuned_iemocap_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
