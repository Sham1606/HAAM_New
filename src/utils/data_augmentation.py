import numpy as np
import librosa
import random
import torch

def add_background_noise(y, snr_db=15):
    """
    Add synthetic Gaussian noise to the audio signal.
    """
    signal_power = np.mean(y**2)
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), len(y))
    return y + noise

def time_stretch_audio(y, rate=None):
    """
    Change the speed/duration of the audio signal.
    rate > 1: faster, rate < 1: slower
    """
    if rate is None:
        rate = random.uniform(0.9, 1.1)
    return librosa.effects.time_stretch(y, rate=rate)

def pitch_shift_audio(y, sr, steps=None):
    """
    Shift the pitch of the audio signal in semitones.
    """
    if steps is None:
        steps = random.uniform(-2, 2)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)

def augment_batch(acoustic_batch, sr=16000):
    """
    Apply random augmentations to a batch of raw audio signals.
    Note: This expects raw audio signals, not pre-extracted features.
    """
    augmented = []
    for y in acoustic_batch:
        choice = random.choice(['noise', 'stretch', 'pitch', 'none'])
        if choice == 'noise':
            y = add_background_noise(y, snr_db=random.uniform(10, 20))
        elif choice == 'stretch':
            y = time_stretch_audio(y, rate=random.uniform(0.9, 1.1))
        elif choice == 'pitch':
            y = pitch_shift_audio(y, sr=sr, steps=random.uniform(-2, 2))
        augmented.append(y)
    return augmented

if __name__ == "__main__":
    # Test augmentation
    import soundfile as sf
    test_file = "data/cremad_samples/1001_DFA_ANG_XX.wav" # Replace with actual path
    try:
        y, sr = librosa.load(test_file, sr=16000)
        y_noise = add_background_noise(y, snr_db=15)
        y_stretch = time_stretch_audio(y, rate=1.1)
        y_pitch = pitch_shift_audio(y, sr, steps=2)
        
        print(f"Original length: {len(y)}")
        print(f"Stretched length: {len(y_stretch)}")
        # sf.write('test_noise.wav', y_noise, sr)
    except Exception as e:
        print(f"Test failed: {e}")
