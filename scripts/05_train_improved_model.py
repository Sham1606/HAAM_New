"""
Train the Improved Hybrid Model using Attention Fusion and Balanced Sampling.
Loads features from data/processed/features_v2/
"""

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
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import json
import joblib
import argparse
from sklearn.preprocessing import StandardScaler

from src.models.attention_fusion_model import AttentionFusionNetwork
from src.models.focal_loss import FocalLoss
from src.utils.balanced_sampler import get_balanced_sampler

class HybridDataset(Dataset):
    def __init__(self, df, feature_dir, emotion_map, scaler=None, acoustic_dim=12):
        self.df = df.copy()
        self.feature_dir = Path(feature_dir)
        self.emotion_map = emotion_map
        self.scaler = scaler
        self.acoustic_dim = acoustic_dim
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feature_path = self.feature_dir / f"{row['call_id']}.pt"
        
        try:
            features = torch.load(feature_path, weights_only=False)
            
            # Acoustic (12 dim - Robust)
            acoustic = features['acoustic']
            if self.scaler:
                acoustic = self.scaler.transform(acoustic.reshape(1, -1))[0]
            
            # Data Augmentation: Random Noise (only during training)
            if self.df.iloc[idx]['split'] == 'train' and np.random.random() > 0.5:
                noise = np.random.normal(0, 0.01, acoustic.shape)
                acoustic = acoustic + noise
                
            acoustic = torch.tensor(acoustic, dtype=torch.float32)
            
            # Text (768 dim)
            text = torch.tensor(features['text_embedding'], dtype=torch.float32)
            if text.dim() > 1:
                text = text.squeeze()
            
            # Label
            emotion = row['emotion_true'] if 'emotion_true' in row else row['emotion']
            if isinstance(emotion, str):
                emotion = emotion.lower()
            label = self.emotion_map.get(emotion, 0)
            
            return acoustic, text, label
            
        except Exception as e:
            # print(f"Error loading {feature_path}: {e}")
            return torch.zeros(self.acoustic_dim), torch.zeros(768), 0

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for acoustic, text, labels in loader:
        acoustic, text, labels = acoustic.to(device), text.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs, _ = model(acoustic, text)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    return running_loss / len(loader), correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for acoustic, text, labels in loader:
            acoustic, text, labels = acoustic.to(device), text.to(device), labels.to(device)
            
            outputs, _ = model(acoustic, text)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return running_loss / len(loader), correct / total, all_preds, all_labels

def calculate_class_weights(df, emotion_map):
    labels = [emotion_map.get(e.lower(), 0) for e in df['emotion']]
    counts = np.bincount(labels, minlength=len(emotion_map))
    total = len(labels)
    # Inverse frequency weights: n_samples / (n_classes * bincount)
    weights = total / (len(emotion_map) * counts)
    # Normalize weights so they sum to n_classes
    weights = weights / weights.sum() * len(emotion_map)
    return torch.tensor(weights, dtype=torch.float32)

def plot_class_recall(history, emotions, save_path):
    plt.figure(figsize=(12, 6))
    for i, emo in enumerate(emotions):
        recalls = [h[i] for h in history['val_class_recall']]
        plt.plot(recalls, label=emo)
    plt.title('Per-Class Recall over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train Improved HAAM Model")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--use-focal-loss", action="store_true")
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--acoustic-dim", type=int, default=12)
    parser.add_argument("--feature-dir", type=str, default='data/processed/features_v3_librosa')
    args = parser.parse_args()

    print("="*80)
    print("PHASE 4: IMPROVED MODEL TRAINING (CLASS IMBALANCE MITIGATION)")
    print("="*80)
    
    # Config
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")
    print(f"Loss Function: {'Focal Loss' if args.use_focal_loss else 'Weighted CrossEntropy'}")
    
    # Paths
    feature_dir = Path(args.feature_dir)
    ACOUSTIC_DIM = args.acoustic_dim
    Path('models/improved').mkdir(parents=True, exist_ok=True)
    Path('results/improved').mkdir(parents=True, exist_ok=True)
    
    emotion_map = {'neutral': 0, 'anger': 1, 'disgust': 2, 'fear': 3, 'sadness': 4}
    curr_emotions = sorted(list(emotion_map.keys()))
    
    # Load Metadata
    try:
        if Path('data/real_metadata_with_split.csv').exists():
            df = pd.read_csv('data/real_metadata_with_split.csv')
            print("Loaded metadata with split.")
        else:
            print("Split file not found, creating from real_metadata.csv...")
            df = pd.read_csv('data/real_metadata.csv')
            # Normalize emotion (already done in generation but safe check)
            df['emotion'] = df['emotion'].str.lower()
            df = df[df['emotion'].isin(emotion_map.keys())]
            
            # Stratify by dataset first
            train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['dataset'])
            train, val = train_test_split(train, test_size=0.1, random_state=42)
            
            df.loc[train.index, 'split'] = 'train'
            df.loc[val.index, 'split'] = 'val'
            df.loc[test.index, 'split'] = 'test'
            df.to_csv('data/real_metadata_with_split.csv', index=False)
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return

    # Check for features
    available_files = set(f.stem for f in feature_dir.glob('*.pt'))
    print(f"Features available: {len(available_files)}")
    
    # Filter DF to those with features
    df = df[df['call_id'].isin(available_files)]
    print(f"Records with features: {len(df)}")
    
    if len(df) < 50:
        print("Not enough features extracted yet to train. Please wait for extraction to complete.")
        return

    df_train = df[df['split']=='train']
    df_val = df[df['split']=='val']
    df_test = df[df['split']=='test']
    
    print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
    
    # 1. Fit Scaler on Training Data
    print("Fitting scaler on training features...")
    all_acoustic = []
    for cid in tqdm(df_train['call_id'], desc="Loading features for scaling"):
        try:
            ft = torch.load(feature_dir / f"{cid}.pt")
            all_acoustic.append(ft['acoustic'])
        except: continue
    
    scaler = StandardScaler()
    scaler.fit(np.array(all_acoustic))
    joblib.dump(scaler, 'models/improved/scaler.pkl')
    print("  Scaler saved.")

    # Datasets
    train_ds = HybridDataset(df_train, feature_dir, emotion_map, scaler=scaler, acoustic_dim=ACOUSTIC_DIM)
    val_ds = HybridDataset(df_val, feature_dir, emotion_map, scaler=scaler, acoustic_dim=ACOUSTIC_DIM)
    test_ds = HybridDataset(df_test, feature_dir, emotion_map, scaler=scaler, acoustic_dim=ACOUSTIC_DIM)
    
    # Compute dynamic class weights
    print("Calculating dynamic class weights...")
    class_weights = calculate_class_weights(df_train, emotion_map).to(DEVICE)
    print(f"  Class Weights: {class_weights.cpu().numpy()}")

    # Sampler
    train_labels = [emotion_map.get(e.lower(), 0) for e in df_train['emotion']]
    sampler, _ = get_balanced_sampler(train_labels)
    
    # Loaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    model = AttentionFusionNetwork(acoustic_dim=ACOUSTIC_DIM, num_classes=5).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) # Standard LR
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    # Loss Selection
    if args.use_focal_loss:
        criterion = FocalLoss(alpha=class_weights, gamma=args.gamma)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Train
    best_val_f1 = 0
    patience = 12
    counter = 0
    history = {
        'train_loss': [], 'val_loss': [], 
        'train_acc': [], 'val_acc': [],
        'val_macro_f1': [],
        'val_class_recall': [] # List of tuples [rec_0, rec_1, ...]
    }
    
    print("\nStarting Training...")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, preds, labels = evaluate(model, val_loader, criterion, DEVICE)
        
        # Calculate Per-Class Metrics
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None, labels=[0,1,2,3,4])
        macro_f1 = f1_score(labels, preds, average='macro')
        
        scheduler.step(macro_f1)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_macro_f1'].append(macro_f1)
        history['val_class_recall'].append(recall)
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Macro-F1: {macro_f1:.4f} Acc: {val_acc:.4f}")
        for i, emo in enumerate(curr_emotions):
            print(f"  - {emo:8s} Recall: {recall[i]:.2f} F1: {f1[i]:.2f}")
        
        # Checkpoint based on Macro F1
        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            counter = 0
            torch.save(model.state_dict(), 'models/improved/best_model.pth')
            print(f"  Saved improved model (Macro-F1: {macro_f1:.4f})")
        else:
            counter += 1
            if counter >= patience:
                print("  Early stopping!")
                break
    
    # Save Recall Plot
    plot_class_recall(history, curr_emotions, 'results/improved/class_recall_epochs.png')
                
    # Final Evaluation
    print("\nFinal Evaluation on Test Set...")
    model.load_state_dict(torch.load('models/improved/best_model.pth'))
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, DEVICE)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Report
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=curr_emotions))
    
    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=curr_emotions, yticklabels=curr_emotions)
    plt.title(f'Improved Model Confusion Matrix (Acc: {test_acc:.2%})')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('results/improved/confusion_matrix.png')
    
    # Save Metrics
    results = {
        'test_accuracy': test_acc,
        'best_val_accuracy': best_val_acc,
        'history': history
    }
    with open('results/improved/metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print("\nResults saved to results/improved/")

if __name__ == '__main__':
    main()
