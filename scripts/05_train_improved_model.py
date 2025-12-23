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
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import json
import joblib
from sklearn.preprocessing import StandardScaler

from src.models.attention_fusion_model import AttentionFusionNetwork
from src.utils.balanced_sampler import get_balanced_sampler

class HybridDataset(Dataset):
    def __init__(self, df, feature_dir, emotion_map, scaler=None):
        self.df = df.copy()
        self.feature_dir = Path(feature_dir)
        self.emotion_map = emotion_map
        self.scaler = scaler
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feature_path = self.feature_dir / f"{row['call_id']}.pt"
        
        try:
            features = torch.load(feature_path)
            
            # Acoustic (12 dim)
            acoustic = features['acoustic']
            if self.scaler:
                acoustic = self.scaler.transform(acoustic.reshape(1, -1))[0]
            
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
            return torch.zeros(12), torch.zeros(768), 0

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

def main():
    print("="*80)
    print("PHASE 4: IMPROVED MODEL TRAINING")
    print("="*80)
    
    # Config
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 0.0001 # Lower LR for stability
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")
    
    # Paths
    feature_dir = Path('data/processed/features_v2')
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
    train_ds = HybridDataset(df_train, feature_dir, emotion_map, scaler=scaler)
    val_ds = HybridDataset(df_val, feature_dir, emotion_map, scaler=scaler)
    test_ds = HybridDataset(df_test, feature_dir, emotion_map, scaler=scaler)
    
    # Sampler
    # Get labels for training set for balancing
    train_labels = [emotion_map.get(e, 0) for e in df_train['emotion']]
    sampler, class_weights = get_balanced_sampler(train_labels)
    # print(f"Class weights (inverse freq): {class_weights}")
    
    # Loaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    model = AttentionFusionNetwork(num_classes=5).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) # Higher start LR
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    criterion = nn.CrossEntropyLoss()
    
    # Train
    best_val_acc = 0
    patience = 12 # More patience for scheduler to work
    counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    print("\nStarting Training...")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        
        scheduler.step(val_acc)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            torch.save(model.state_dict(), 'models/improved/best_model.pth')
            # print("  Saved improved model")
        else:
            counter += 1
            if counter >= patience:
                print("  Early stopping!")
                break
                
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
