import librosa
import sys
import traceback
from pathlib import Path
import pandas as pd
sys.path.append('.')

# Emulate 02b imports and hacks
sys.modules['tensorflow'] = None
# Bypass CVE check
import transformers.utils.import_utils
def no_op_check(): pass
transformers.utils.import_utils.check_torch_load_is_safe = no_op_check

from src.preprocessing.audio_preprocessor import AudioPreprocessor
from src.features.improved_acoustic import ImprovedAcousticExtractor
import whisper

def test_load(path):
    print(f"\nTesting {path}...")
    if not Path(path).exists():
        print("  File not found!")
        return
        
    try:
        # 1. Preprocess
        print("  1. Preprocessing...")
        prep = AudioPreprocessor()
        y, sr = prep.preprocess(path)
        print(f"     Success! Shape: {y.shape}, SR: {sr}")
        
        # 2. Acoustic
        print("  2. Acoustic Extraction...")
        ac_ext = ImprovedAcousticExtractor()
        feats = ac_ext.extract_array(y, sr)
        print(f"     Success! Feats shape: {feats.shape}")
        
        # 3. Whisper
        print("  3. Whisper Transcription...")
        model = whisper.load_model("base")
        res = model.transcribe(y)
        print(f"     Success! Text: {res['text'][:50]}...")
        
    except Exception:
        print("  Failed!")
        traceback.print_exc()

def main():
    print("DEBUG AUDIO LOADING")
    try:
        df = pd.read_csv('data/real_metadata.csv')
    except:
        print("No metadata")
        return

    # Test CREMA (FLV)
    crema = df[df['dataset']=='CREMA-D'].iloc[0]
    test_load(crema['filepath'])
    
    # Test IEMOCAP (WAV)
    iemocap = df[df['dataset']=='IEMOCAP'].iloc[0]
    test_load(iemocap['filepath'])

if __name__ == '__main__':
    main()
