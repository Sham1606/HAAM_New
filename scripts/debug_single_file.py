import sys
import os
from pathlib import Path

# Project root
sys.path.append('.')

# Patching
sys.modules['tensorflow'] = None
import transformers.utils.import_utils
import transformers.modeling_utils
def no_op_check(): pass
transformers.utils.import_utils.check_torch_load_is_safe = no_op_check
transformers.modeling_utils.check_torch_load_is_safe = no_op_check

import torch
import numpy as np
import pandas as pd
import whisper
from src.preprocessing.audio_preprocessor import AudioPreprocessor
from src.features.improved_acoustic import ImprovedAcousticExtractor
from src.features.emotion_text import EmotionTextExtractor

def debug_file(path, call_id="test_id"):
    print(f"\nProcessing {path}...")
    
    try:
        # 1. Preprocess
        print("  1. Preprocessing...")
        prep = AudioPreprocessor()
        audio, sr = prep.preprocess(path)
        print(f"     Audio: {audio.shape}, SR: {sr}, Dtype: {audio.dtype}")
        
        # 2. Acoustic
        print("  2. Acoustic Extraction...")
        ac_ext = ImprovedAcousticExtractor()
        ac_feats = ac_ext.extract_array(audio, sr=sr)
        print(f"     Acoustic feats: {ac_feats}")
        
        # 3. Whisper
        print("  3. Whisper Transcription...")
        w_model = whisper.load_model("base")
        # Ensure float32
        audio_32 = audio.astype(np.float32)
        res = w_model.transcribe(audio_32)
        transcript = res['text'].strip()
        print(f"     Transcript: {transcript}")
        
        # 4. Text
        print("  4. Text Extraction...")
        t_ext = EmotionTextExtractor()
        t_feats = t_ext.extract(transcript)
        print(f"     Text Emotion: {t_feats['dominant_emotion']}")
        
        # 5. Save Test
        print("  5. Save Test...")
        data = {
            'call_id': call_id,
            'acoustic': ac_feats,
            'text_embedding': t_feats['embedding'],
            'emotion_probs': t_feats['emotion_probabilities'],
            'transcript': transcript,
            'emotion': 'neutral',
            'dataset': 'test'
        }
        test_out = Path('data/processed/features_v2/test_file.pt')
        test_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, test_out)
        print(f"     Success! File saved to {test_out}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # Test specific file 1089
    sample_path = "d:/haam_framework/crema-d-mirror-main/AudioWAV/1089_WSI_NEU_XX.wav"
    debug_file(sample_path)
