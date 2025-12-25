"""
Feature Quality Audit
Purpose: Check if features are extracting correctly
Expected: Find 2-3 critical bugs that explain the 52.7% plateau.
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os

def audit_acoustic_features():
    """Check acoustic feature quality across all test samples"""
    
    print("="*80)
    print("ACOUSTIC FEATURE QUALITY AUDIT")
    print("="*80)
    
    # Load metadata
    metadata_path = 'data/real_metadata_with_split.csv'
    if not os.path.exists(metadata_path):
        print(f"Error: {metadata_path} not found.")
        return None
        
    df = pd.read_csv(metadata_path)
    df_test = df[df['split'] == 'test'].copy()
    features_dir = Path('data/processed/features_v2')
    
    print(f"Processing {len(df_test)} test samples...")
    
    # Collect features from all samples
    feature_stats = []
    
    for idx, row in df_test.iterrows():
        try:
            feature_path = features_dir / f"{row['call_id']}.pt"
            if not feature_path.exists():
                continue
                
            features = torch.load(feature_path, weights_only=False)
            acoustic = features['acoustic']  # Should be 12-dim array
            
            # Map features (order from ImprovedAcousticExtractor)
            # 0:pitch_mean, 1:pitch_std, 2:pitch_range, 3:pitch_slope,
            # 4:jitter, 5:shimmer, 6:hnr,
            # 7:rms_mean, 8:rms_std,
            # 9:speech_rate, 10:zero_crossing_rate,
            # 11:spectral_centroid
            
            feature_stats.append({
                'call_id': row['call_id'],
                'dataset': row['dataset'],
                'emotion': row['emotion'],
                'pitch_mean': float(acoustic[0]),
                'pitch_std': float(acoustic[1]),
                'pitch_range': float(acoustic[2]),
                'pitch_slope': float(acoustic[3]),
                'jitter': float(acoustic[4]),
                'shimmer': float(acoustic[5]),
                'hnr': float(acoustic[6]),
                'rms_mean': float(acoustic[7]),
                'rms_std': float(acoustic[8]),
                'speech_rate': float(acoustic[9]),
                'zcr': float(acoustic[10]),
                'spectral_centroid': float(acoustic[11])
            })
            
        except Exception as e:
            # print(f"Error loading {row['call_id']}: {e}")
            continue
    
    if not feature_stats:
        print("No features found to audit.")
        return None
        
    df_features = pd.DataFrame(feature_stats)
    
    # Analysis 1: Check for zeros (extraction failures)
    print("\n1. ZERO VALUE ANALYSIS (Extraction Failures)")
    print("-"*80)
    
    feature_names = ['pitch_mean', 'pitch_std', 'pitch_range', 'pitch_slope',
                     'jitter', 'shimmer', 'hnr', 'rms_mean', 'rms_std',
                     'speech_rate', 'zcr', 'spectral_centroid']
    
    for feat in feature_names:
        zero_count = (df_features[feat] == 0).sum()
        zero_pct = (zero_count / len(df_features)) * 100
        
        status = "❌ CRITICAL" if zero_pct > 30 else "⚠️ WARNING" if zero_pct > 10 else "✓"
        print(f"{feat:20s}: {zero_pct:5.1f}% zeros ({zero_count}/{len(df_features)}) {status}")
    
    # Analysis 2: Check for outliers
    print("\n2. OUTLIER ANALYSIS (Unrealistic Values)")
    print("-"*80)
    
    # Pitch should be 60-500 Hz for human speech
    invalid_pitch = ((df_features['pitch_mean'] < 50) | (df_features['pitch_mean'] > 500)).sum()
    print(f"Invalid pitch values (<50 or >500 Hz): {invalid_pitch} samples ({invalid_pitch/len(df_features)*100:.1f}%)")
    
    # Jitter/Shimmer should be small
    invalid_jitter = (df_features['jitter'] > 0.1).sum()
    invalid_shimmer = (df_features['shimmer'] > 0.2).sum()
    print(f"Excessive jitter (>0.1): {invalid_jitter} samples")
    print(f"Excessive shimmer (>0.2): {invalid_shimmer} samples")
    
    # HNR should be 0-35 dB
    invalid_hnr = ((df_features['hnr'] < -5) | (df_features['hnr'] > 40)).sum()
    print(f"Invalid HNR: {invalid_hnr} samples")
    
    # Analysis 3: Feature distribution by emotion
    print("\n3. FEATURE VARIANCE BY EMOTION (Means)")
    print("-"*80)
    
    for emotion in sorted(df_features['emotion'].unique()):
        emotion_df = df_features[df_features['emotion'] == emotion]
        if len(emotion_df) > 0:
            print(f"  {emotion:8s}: Pitch={emotion_df['pitch_mean'].mean():.1f} | RMS={emotion_df['rms_mean'].mean():.4f} | Samples={len(emotion_df)}")
    
    # Analysis 4: Dataset comparison
    print("\n4. DATASET COMPARISON")
    print("-"*80)
    
    for dataset in sorted(df_features['dataset'].unique()):
        dataset_df = df_features[df_features['dataset'] == dataset]
        avg_zeros = (dataset_df[feature_names] == 0).sum(axis=1).mean()
        print(f"  {dataset:10s}: Samples={len(dataset_df)} | Avg Pitch={dataset_df['pitch_mean'].mean():.1f} Hz | Avg Zeros={avg_zeros:.1f}/12")
    
    # Visualization
    Path('results/diagnostics_v2.1').mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Feature Distribution Audit (Across Emotions)', fontsize=16, fontweight='bold')
    
    for idx, feat in enumerate(feature_names):
        ax = axes[idx // 4, idx % 4]
        for emotion in sorted(df_features['emotion'].unique()):
            emotion_df = df_features[df_features['emotion'] == emotion]
            ax.hist(emotion_df[feat], alpha=0.5, label=emotion, bins=20)
        
        ax.set_title(feat)
        if idx == 0: ax.legend(fontsize='small')
    
    plt.tight_layout()
    plt.savefig('results/diagnostics_v2.1/feature_distributions.png', dpi=150)
    print(f"\n✓ Saved: results/diagnostics_v2.1/feature_distributions.png")
    
    # Save CSV
    df_features.to_csv('results/diagnostics_v2.1/feature_audit.csv', index=False)
    print(f"✓ Saved: results/diagnostics_v2.1/feature_audit.csv")
    
    return df_features

def audit_text_features():
    """Check text emotion probabilities"""
    
    print("\n" + "="*80)
    print("TEXT FEATURE QUALITY AUDIT")
    print("="*80)
    
    metadata_path = 'data/real_metadata_with_split.csv'
    df = pd.read_csv(metadata_path)
    df_test = df[df['split'] == 'test'].copy()
    features_dir = Path('data/processed/features_v2')
    
    text_stats = []
    
    for idx, row in df_test.iterrows():
        try:
            feature_path = features_dir / f"{row['call_id']}.pt"
            if not feature_path.exists(): continue
            
            features = torch.load(feature_path, weights_only=False)
            
            # Need to check if text_emotion_probs exists
            # In our previous iterations, it might just be text_embedding
            # But the user script expects 'text_emotion_probs'
            # Let's check what's actually there.
            
            if 'text_emotion_probs' in features:
                probs = features['text_emotion_probs']
                text_stats.append({
                    'call_id': row['call_id'],
                    'true_emotion': row['emotion'],
                    'neutral_prob': probs[0],
                    'anger_prob': probs[1],
                    'disgust_prob': probs[2],
                    'fear_prob': probs[3],
                    'sadness_prob': probs[4],
                    'predicted_emotion': ['neutral', 'anger', 'disgust', 'fear', 'sadness'][np.argmax(probs)]
                })
        except:
            continue
    
    if len(text_stats) == 0:
        print("⚠️ WARNING: No text emotion probabilities found in feature files!")
        print("This means the current extractor only saves embeddings, not class probabilities.")
        return None
    
    df_text = pd.DataFrame(text_stats)
    
    # Check if stuck on neutral
    neutral_dominant = (df_text['neutral_prob'] > 0.7).sum()
    print(f"1. NEUTRAL BIAS CHECK")
    print(f"   Samples with neutral > 0.7: {neutral_dominant}/{len(df_text)} ({neutral_dominant/len(df_text)*100:.1f}%)")
    
    # Variance
    print(f"\n2. PROBABILITY VARIANCE")
    for emo in ['neutral', 'anger', 'disgust', 'fear', 'sadness']:
        col = f'{emo}_prob'
        print(f"   {emo:10s}: Mean={df_text[col].mean():.3f} | Std={df_text[col].std():.3f}")
    
    # Text-only accuracy
    correct = (df_text['predicted_emotion'] == df_text['true_emotion']).sum()
    acc = correct / len(df_text)
    print(f"\n3. TEXT-ONLY ACCURACY: {acc*100:.1f}%")
    
    df_text.to_csv('results/diagnostics_v2.1/text_audit.csv', index=False)
    return df_text

if __name__ == '__main__':
    Path('results/diagnostics_v2.1').mkdir(parents=True, exist_ok=True)
    audit_acoustic_features()
    audit_text_features()
    print("\nAudit Complete.")
