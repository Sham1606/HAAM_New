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

class LibrosaAcousticExtractor:
    """
    Robust acoustic feature extractor using purely librosa.
    v1: 12-dim (Prosody, Energy, Spectral)
    v2: 20-dim (v1 + Top 8 MFCCs)
    """
    
    def __init__(self, sr=16000, version='v2'):
        self.sr = sr
        self.version = version
        self.v1_names = [
            'pitch_mean', 'pitch_std', 'pitch_range', 'pitch_slope',
            'rms_mean', 'rms_std', 'zero_crossing_rate',
            'spectral_centroid', 'spectral_rolloff', 'spectral_flatness',
            'speech_rate', 'spectral_bandwidth'
        ]
        self.mfcc_names = [f'mfcc_{i}' for i in range(8)]
        
    @property
    def feature_names(self):
        if self.version == 'v1':
            return self.v1_names
        return self.v1_names + self.mfcc_names

    def extract(self, audio_input):
        """Extract features from audio path or signal array."""
        try:
            if isinstance(audio_input, (str, Path)):
                if not Path(audio_input).exists():
                    return self.get_empty_features()
                audio, sr = librosa.load(audio_input, sr=self.sr)
            else:
                audio = audio_input
                sr = self.sr

            if len(audio) < int(0.1 * sr):
                audio = np.pad(audio, (0, int(0.1 * sr) - len(audio)), mode='reflect')
            
            return self._extract_features(audio, sr)
        except Exception:
            return self.get_empty_features()

    def _extract_features(self, audio, sr):
        features = {}
        
        # 1-4: Prosody (F12)
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
            )
            f0_clean = f0[~np.isnan(f0)]
            if len(f0_clean) > 0:
                features['pitch_mean'] = float(np.mean(f0_clean))
                features['pitch_std'] = float(np.std(f0_clean))
                features['pitch_range'] = float(np.max(f0_clean) - np.min(f0_clean))
                x = np.arange(len(f0_clean))
                features['pitch_slope'] = float(np.polyfit(x, f0_clean, 1)[0]) if len(f0_clean) > 1 else 0.0
            else:
                features.update({'pitch_mean': 150.0, 'pitch_std': 0.0, 'pitch_range': 0.0, 'pitch_slope': 0.0})
        except:
            features.update({'pitch_mean': 150.0, 'pitch_std': 0.0, 'pitch_range': 0.0, 'pitch_slope': 0.0})

        # 5-6: Energy
        rms = librosa.feature.rms(y=audio)
        features['rms_mean'] = float(np.mean(rms))
        features['rms_std'] = float(np.std(rms))

        # 7: ZCR
        zcr = librosa.feature.zero_crossing_rate(audio)
        features['zero_crossing_rate'] = float(np.mean(zcr))

        # 8-10: Spectral
        sc = librosa.feature.spectral_centroid(y=audio, sr=sr)
        features['spectral_centroid'] = float(np.mean(sc))
        sr_val = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        features['spectral_rolloff'] = float(np.mean(sr_val))
        sf = librosa.feature.spectral_flatness(y=audio)
        features['spectral_flatness'] = float(np.mean(sf))

        # 11: Rhythm
        try:
            onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
            peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10)
            duration = len(audio) / sr
            features['speech_rate'] = float((len(peaks) / duration) * 60) if duration > 0.05 else 0.0
        except:
            features['speech_rate'] = 0.0

        # 12: Bandwidth
        sb = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        features['spectral_bandwidth'] = float(np.mean(sb))

        # 13-20: Top 8 MFCCs (v2 only)
        if self.version == 'v2':
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            # Select 8 coefficients with highest variance across time
            variances = np.var(mfccs, axis=1)
            top_8_indices = np.argsort(variances)[-8:]
            top_8_indices.sort() # Keep order for consistency
            
            for i, idx in enumerate(top_8_indices):
                features[f'mfcc_{i}'] = float(np.mean(mfccs[idx]))

        return features

    def get_empty_features(self):
        return {k: 0.0 for k in self.feature_names}

    def extract_array(self, audio_input):
        features = self.extract(audio_input)
        return np.array([features[f] for f in self.feature_names], dtype=np.float32)

class ImprovedAcousticExtractor(LibrosaAcousticExtractor):
    def __init__(self, sr=16000, version='v2'):
        super().__init__(sr=sr, version=version)
    
    def extract_from_signal(self, audio, sr):
        return self._extract_features(audio, sr)

    def extract_array(self, audio_input, sr=None):
        return super().extract_array(audio_input)
