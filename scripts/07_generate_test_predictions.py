import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
import joblib

import sys
sys.path.append(r'd:\haam_framework')

import importlib.util
from src.models.attention_fusion_model import AttentionFusionNetwork

# Load HybridDataset from the script with numeric prefix
spec = importlib.util.spec_from_file_location("train_mod", r"d:\haam_framework\scripts\05_train_improved_model.py")
train_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mod)
HybridDataset = train_mod.HybridDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_and_save():
    print("PHASE 7.1: GENERATING DIAGNOSTIC DATA")
    
    # Paths
    metadata_path = 'data/real_metadata_with_split.csv'
    feature_dir = Path('data/processed/features_v2')
    model_path = 'models/improved/best_model.pth'
    scaler_path = 'models/improved/scaler.pkl'
    
    # Load metadata
    df = pd.read_csv(metadata_path)
    test_df = df[df['split'] == 'test'].copy()
    print(f"Test samples: {len(test_df)}")
    
    # Load Scaler
    scaler = joblib.load(scaler_path)
    
    emotion_map = {'neutral': 0, 'anger': 1, 'disgust': 2, 'fear': 3, 'sadness': 4}
    
    # Dataset & Loader
    test_ds = HybridDataset(test_df, feature_dir, emotion_map, scaler=scaler)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    # Model
    model = AttentionFusionNetwork(acoustic_dim=12, num_classes=5).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    emotion_map_rev = {0: 'neutral', 1: 'anger', 2: 'disgust', 3: 'fear', 4: 'sadness'}
    
    results = []
    
    print("Running inference on test set...")
    idx = 0
    with torch.no_grad():
        for acoustic, text, labels in tqdm(test_loader):
            acoustic, text = acoustic.to(DEVICE), text.to(DEVICE)
            logits, weights = model(acoustic, text)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            # Move to CPU
            preds = preds.cpu().numpy()
            labels = labels.cpu().numpy()
            weights = weights.cpu().numpy()
            probs = probs.cpu().numpy()
            acoustic_np = acoustic.cpu().numpy()
            
            for i in range(len(preds)):
                sample_row = test_df.iloc[idx]
                
                res = {
                    'call_id': sample_row['call_id'],
                    'emotion_true': emotion_map_rev[labels[i]],
                    'emotion_pred': emotion_map_rev[preds[i]],
                    'correct': int(preds[i] == labels[i]),
                    'conf': float(np.max(probs[i])),
                    'weight_acoustic': float(weights[i, 0]),
                    'weight_text': float(weights[i, 1]),
                    'dataset': sample_row['dataset'],
                    'duration': sample_row.get('duration', 0)
                }
                
                # Add individual probabilities
                for ei, name in emotion_map_rev.items():
                    res[f'prob_{name}'] = float(probs[i, ei])
                
                # Add acoustic feature health (check for zeros)
                # Feature order: pitch_mean, pitch_std, pitch_range, pitch_slope, jitter, shimmer, hnr, rms_mean, rms_std, speech_rate, zcr, spectral_centroid
                res['pitch_is_zero'] = int(acoustic_np[i, 0] == 0)
                res['jitter_is_zero'] = int(acoustic_np[i, 4] == 0)
                res['shimmer_is_zero'] = int(acoustic_np[i, 5] == 0)
                
                results.append(res)
                idx += 1
                
    # Save to CSV
    res_df = pd.DataFrame(results)
    output_path = 'results/diagnosis/test_predictions_deep.csv'
    Path('results/diagnosis').mkdir(parents=True, exist_ok=True)
    res_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == '__main__':
    evaluate_and_save()
