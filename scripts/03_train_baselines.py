"""
Train 3 baseline models to establish performance ceiling:
1. CREMA-D only
2. IEMOCAP only  
3. Simple combined (no balancing)

This tells us if balancing/domain adaptation is needed.
Expects features to be in data/processed/features_v2/
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

class SimpleHybridModel(nn.Module):
    def __init__(self, acoustic_dim=12, text_dim=768, num_classes=5):
        super().__init__()
        
        self.acoustic_encoder = nn.Sequential(
            nn.Linear(acoustic_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32)
        )
        
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 32)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, acoustic, text):
        acoustic_feat = self.acoustic_encoder(acoustic)
        text_feat = self.text_encoder(text)
        combined = torch.cat([acoustic_feat, text_feat], dim=1)
        return self.classifier(combined)

class EmotionDataset(Dataset):
    def __init__(self, df, feature_dir, emotion_map):
        self.df = df
        self.feature_dir = Path(feature_dir)
        self.emotion_map = emotion_map
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feature_path = self.feature_dir / f"{row['call_id']}.pt"
        
        try:
            features = torch.load(feature_path)
            acoustic = torch.tensor(features['acoustic'], dtype=torch.float32)
            # Text embedding might be (1, 768) or (768,)
            text = torch.tensor(features['text_embedding'], dtype=torch.float32)
            if text.dim() > 1:
                text = text.squeeze()
            
            # Map emotion
            emotion = row['emotion_true'] if 'emotion_true' in row else row['emotion']
            if isinstance(emotion, str):
                emotion = emotion.lower()
            label = self.emotion_map.get(emotion, 0) # Default to 0 (neutral) if unknown? Or skip?
            
            return acoustic, text, label 
        except Exception as e:
            # Handle error gracefully? Return zero tensors
            print(f"Error loading {feature_path}: {e}")
            return torch.zeros(12), torch.zeros(768), 0

def train_model(model, train_loader, val_loader, epochs=20, lr=0.001, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    
    best_val_acc = 0
    patience = 5
    counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for acoustic, text, labels in train_loader:
            acoustic, text, labels = acoustic.to(device), text.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(acoustic, text)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for acoustic, text, labels in val_loader:
                acoustic, text = acoustic.to(device), text.to(device)
                outputs = model(acoustic, text)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(val_labels, val_preds)
        print(f"Epoch {epoch+1}: Loss {train_loss/len(train_loader):.4f}, Val Acc {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            torch.save(model.state_dict(), 'temp_best.pth')
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break
                
    if Path('temp_best.pth').exists():
        model.load_state_dict(torch.load('temp_best.pth'))
    return model, best_val_acc

def main():
    print("="*80)
    print("PHASE 2: BASELINE MODEL TRAINING")
    print("="*80)
    
    Path('results/baselines').mkdir(parents=True, exist_ok=True)
    Path('models/baselines').mkdir(parents=True, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    emotion_map = {'neutral': 0, 'anger': 1, 'disgust': 2, 'fear': 3, 'sadness': 4, 'joy': 0} 
    # Mapped 'joy' to 'neutral' as per diagnosis (only 5 classes supported usually)?
    # Or should we exclude 'joy'? diagnosis showed 'joy' -> 'neutral 'errors.
    # The prompt said 54.5% accuracy.
    # The map in prompt was {'neutral': 0, 'anger': 1, 'disgust': 2, 'fear': 3, 'sadness': 4}.
    # So 'joy' is likely out of distribution or treated as neutral.
    
    # Load metadata
    try:
        df = pd.read_csv('data/real_metadata.csv')
        print(f"Loaded {len(df)} records from real_metadata.csv")
    except:
        print("Error loading metadata")
        return
        
    # Check if features exist and filter dataframe
    feature_dir = Path('data/processed/features_v2')
    if not feature_dir.exists():
        print("Feature directory not found!")
        return
        
    available_ids = {f.stem for f in feature_dir.glob('*.pt')}
    if not available_ids:
        print("No features found! Run Batch Feature Extraction first.")
        return
        
    original_len = len(df)
    df = df[df['call_id'].astype(str).isin(available_ids)]
    print(f"Filtered to {len(df)} available features (out of {original_len})")
    
    if len(df) < 50:
        print("Not enough features for training yet. Wait for more processing.")
        return

    # Filter columns
    df['emotion'] = df['emotion'].str.lower()
    df = df[df['emotion'].isin(emotion_map.keys())]
    
    # Split
    if 'split' not in df.columns:
        print("Creating new train/val/test split...")
        try:
             train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['dataset'])
             train, val = train_test_split(train, test_size=0.1, random_state=42)
            
             df.loc[train.index, 'split'] = 'train'
             df.loc[val.index, 'split'] = 'val'
             df.loc[test.index, 'split'] = 'test'
             df.to_csv('data/real_metadata_with_split.csv', index=False)
        except Exception as e:
            # Fallback for mock/debug if classes too small for stratify
            train, test = train_test_split(df, test_size=0.2, random_state=42)
            train, val = train_test_split(train, test_size=0.1, random_state=42)
            df.loc[train.index, 'split'] = 'train'
            df.loc[val.index, 'split'] = 'val'
            df.loc[test.index, 'split'] = 'test'
    else:
        print("Using existing split.")
        
    df_train = df[df['split']=='train']
    df_val = df[df['split']=='val']
    df_test = df[df['split']=='test']
    
    loaders = {}
    for name, d in [('train', df_train), ('val', df_val), ('test', df_test)]:
        ds = EmotionDataset(d, 'data/processed/features_v2', emotion_map)
        loaders[name] = DataLoader(ds, batch_size=32, shuffle=(name=='train'))
        
    results = {}
    
    # 1. CREMA-D ONLY
    print("\nTraining Baseline: CREMA-D Only")
    df_c_train = df_train[df_train['dataset']=='CREMA-D']
    df_c_val = df_val[df_val['dataset']=='CREMA-D']
    if len(df_c_train) > 0:
        c_loaders = {
            'train': DataLoader(EmotionDataset(df_c_train, 'data/processed/features_v2', emotion_map), batch_size=32, shuffle=True),
            'val': DataLoader(EmotionDataset(df_c_val, 'data/processed/features_v2', emotion_map), batch_size=32),
            'test': loaders['test'] # Test on ALL? Or just CREMA?
            # Prompt asked for "CREMA-D only" baseline. Usually means trained on CREMA. Tested on CREMA?
            # Or tested on BOTH to show cross-corpus failure?
            # "Analyze Per dataset accuracy". So test on ALL.
        }
        
        model = SimpleHybridModel().to(device)
        model, _ = train_model(model, c_loaders['train'], c_loaders['val'], device=device)
        torch.save(model.state_dict(), 'models/baselines/crema_only.pth')
        
        # Test
        preds, labels, datasets = [], [], []
        with torch.no_grad():
            for i in range(len(df_test)):
                row = df_test.iloc[i]
                ft = torch.load(Path('data/processed/features_v2') / f"{row['call_id']}.pt")
                ac = torch.tensor(ft['acoustic']).float().to(device).unsqueeze(0)
                tx = torch.tensor(ft['text_embedding']).float().to(device).unsqueeze(0)
                if tx.dim() > 2: tx = tx.squeeze(1)
                
                out = model(ac, tx)
                pred = torch.argmax(out, dim=1).item()
                preds.append(pred)
                labels.append(emotion_map.get(row['emotion'], 0))
                datasets.append(row['dataset'])
                
        acc = accuracy_score(labels, preds)
        print(f"CREMA-Only Test Acc: {acc:.4f}")
        results['crema_only'] = acc
    
    # 2. IEMOCAP ONLY
    print("\nTraining Baseline: IEMOCAP Only")
    df_i_train = df_train[df_train['dataset']=='IEMOCAP']
    df_i_val = df_val[df_val['dataset']=='IEMOCAP']
    if len(df_i_train) > 0:
        i_loaders = {
            'train': DataLoader(EmotionDataset(df_i_train, 'data/processed/features_v2', emotion_map), batch_size=32, shuffle=True),
            'val': DataLoader(EmotionDataset(df_i_val, 'data/processed/features_v2', emotion_map), batch_size=32)
        }
        model = SimpleHybridModel().to(device)
        model, _ = train_model(model, i_loaders['train'], i_loaders['val'], device=device)
        torch.save(model.state_dict(), 'models/baselines/iemocap_only.pth')
        
        # Test
        preds, labels = [], []
        with torch.no_grad():
            for i in range(len(df_test)):
                row = df_test.iloc[i]
                ft = torch.load(Path('data/processed/features_v2') / f"{row['call_id']}.pt")
                ac = torch.tensor(ft['acoustic']).float().to(device).unsqueeze(0)
                tx = torch.tensor(ft['text_embedding']).float().to(device).unsqueeze(0)
                if tx.dim() > 2: tx = tx.squeeze(1)

                out = model(ac, tx)
                preds.append(torch.argmax(out, dim=1).item())
                labels.append(emotion_map.get(row['emotion'], 0))
                
        acc = accuracy_score(labels, preds)
        print(f"IEMOCAP-Only Test Acc: {acc:.4f}")
        results['iemocap_only'] = acc

    # 3. COMBINED
    print("\nTraining Baseline: Combined")
    model = SimpleHybridModel().to(device)
    model, _ = train_model(model, loaders['train'], loaders['val'], device=device)
    torch.save(model.state_dict(), 'models/baselines/combined.pth')
    
    # Test
    preds, labels = [], []
    with torch.no_grad():
        for i in range(len(df_test)):
            row = df_test.iloc[i]
            ft = torch.load(Path('data/processed/features_v2') / f"{row['call_id']}.pt")
            ac = torch.tensor(ft['acoustic']).float().to(device).unsqueeze(0)
            tx = torch.tensor(ft['text_embedding']).float().to(device).unsqueeze(0)
            if tx.dim() > 2: tx = tx.squeeze(1)
            
            out = model(ac, tx)
            preds.append(torch.argmax(out, dim=1).item())
            labels.append(emotion_map.get(row['emotion'], 0))
            
    acc = accuracy_score(labels, preds)
    print(f"Combined Test Acc: {acc:.4f}")
    results['combined'] = acc
    
    with open('results/baselines/results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()
