"""
Extract 12 core acoustic features proven to correlate with emotions:
- Prosody (4): pitch_mean, pitch_std, pitch_range, pitch_slope
- Voice Quality (3): jitter, shimmer, hnr
- Energy (2): rms_mean, rms_std  
- Rhythm (2): speech_rate, zero_crossing_rate
- Spectral (1): spectral_centroid
"""

import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call
import warnings
from pathlib import Path

class ImprovedAcousticExtractor:
    def __init__(self, sr=16000):
        self.sr = sr
    
    def extract(self, audio_path):
        """Extract all 12 features from Path or Array. Kept for back-compat but prefers array."""
        try:
             # Just delegate
             if isinstance(audio_path, str) or isinstance(audio_path, Path):
                 audio, sr = librosa.load(audio_path, sr=self.sr)
             else:
                 # Assume it's a tuple (audio, sr) or just audio
                 # This method signature is messy, better use extract_from_signal
                 return self.extract_from_signal(audio_path, self.sr)
                 
             return self.extract_from_signal(audio, sr)
        except Exception:
             return self.get_empty_features()

    def get_empty_features(self):
        return {k: 0.0 for k in [
                'pitch_mean', 'pitch_std', 'pitch_range', 'pitch_slope',
                'jitter', 'shimmer', 'hnr',
                'rms_mean', 'rms_std',
                'speech_rate', 'zero_crossing_rate',
                'spectral_centroid'
            ]}

    def extract_from_signal(self, audio, sr):
        """Extract features directly from numpy array (12 dimensions)"""
        features = {}
        
        # 1-4: Prosody
        try:
            f0, _, _ = librosa.pyin(audio, fmin=65, fmax=1000, sr=sr, frame_length=2048, hop_length=1024)
            f0_clean = f0[~np.isnan(f0)]
            if len(f0_clean) > 5:
                features['pitch_mean'] = float(np.mean(f0_clean))
                features['pitch_std'] = float(np.std(f0_clean))
                features['pitch_range'] = float(np.max(f0_clean) - np.min(f0_clean))
                features['pitch_slope'] = float(np.polyfit(np.arange(len(f0_clean)), f0_clean, 1)[0])
            else:
                features.update({'pitch_mean': 0, 'pitch_std': 0, 'pitch_range': 0, 'pitch_slope': 0})
        except:
            features.update({'pitch_mean': 0, 'pitch_std': 0, 'pitch_range': 0, 'pitch_slope': 0})
        
        # 5-7: Voice Quality
        try:
            snd = parselmouth.Sound(audio, sampling_frequency=sr)
            point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)
            jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)
            
            features['jitter'] = float(jitter) if not np.isnan(jitter) else 0
            features['shimmer'] = float(shimmer) if not np.isnan(shimmer) else 0
            features['hnr'] = float(hnr) if not np.isnan(hnr) else 0
        except:
            features.update({'jitter': 0, 'shimmer': 0, 'hnr': 0})
        
        # 8-9: Energy
        rms = librosa.feature.rms(y=audio)
        features['rms_mean'] = float(np.mean(rms))
        features['rms_std'] = float(np.std(rms))
        
        # 10-11: Rhythm/ZCR
        zcr = librosa.feature.zero_crossing_rate(audio)
        features['zero_crossing_rate'] = float(np.mean(zcr))
        
        # 12: Spectral
        try:
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
            features['spectral_centroid'] = float(np.mean(spectral_centroids))
        except:
            features['spectral_centroid'] = 0

        # Rhythm (speech rate)
        try:
            onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
            peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10)
            duration = len(audio) / sr
            features['speech_rate'] = float((len(peaks) / duration) * 60) if duration > 0 else 0
        except:
            features['speech_rate'] = 0
        
        return features
    
    def extract_array(self, audio_input, sr=None):
        """Return as numpy array (12 dimensions)"""
        if sr is None: sr = self.sr
        if isinstance(audio_input, str) or isinstance(audio_input, Path):
             features = self.extract(audio_input)
        else:
             features = self.extract_from_signal(audio_input, sr)
             
        feature_order = ['pitch_mean', 'pitch_std', 'pitch_range', 'pitch_slope',
                        'jitter', 'shimmer', 'hnr',
                        'rms_mean', 'rms_std',
                        'speech_rate', 'zero_crossing_rate',
                        'spectral_centroid']
        return np.array([features[f] for f in feature_order], dtype=np.float32)
