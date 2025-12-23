"""
Normalize audio: trim silence, fix volume, remove DC offset, resample
Critical for consistent feature extraction
"""

import librosa
import numpy as np
from scipy import signal
import soundfile as sf
import warnings

class AudioPreprocessor:
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr
    
    def preprocess(self, audio_path):
        """
        Preprocess audio file
        
        Returns:
            audio: normalized audio array
            sr: sample rate
        """
        # Load
        # Use try-except block for robustness
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                audio, sr = librosa.load(audio_path, sr=None)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            raise e
        
        # Remove DC offset
        audio = audio - np.mean(audio)
        
        # Trim silence
        # top_db=20 is a good default, maybe adjustable
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        # Resample
        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr
        
        # Normalize volume to -3dB
        max_amp = np.max(np.abs(audio))
        if max_amp > 0:
            audio = audio * (0.7 / max_amp)
        
        # High-pass filter (remove low-frequency noise <80Hz)
        try:
            sos = signal.butter(5, 80, 'hp', fs=sr, output='sos')
            audio = signal.sosfilt(sos, audio)
        except Exception as e:
            # Fallback if filter fails (e.g. too short audio)
            pass
        
        return audio, sr

if __name__ == '__main__':
    # Test
    # Create a dummy file or use a known one if testing directly
    pass
