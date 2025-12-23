"""
Test complete feature extraction pipeline on sample files
Verify all features extract correctly
"""

import sys
sys.path.append('.')

from src.preprocessing.audio_preprocessor import AudioPreprocessor
from src.features.improved_acoustic import ImprovedAcousticExtractor
from src.features.emotion_text import EmotionTextExtractor
import pandas as pd
import warnings
# Try importing whisper, but don't fail immediately if not present (though required for text)
try:
    import whisper
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False
    print("WARNING: Whisper not found. Text features will be limited.")

def main():
    print("="*80)
    print("PHASE 1: FEATURE EXTRACTION TEST")
    print("="*80)
    
    # Initialize
    print("Initializing components...")
    preprocessor = AudioPreprocessor()
    acoustic_extractor = ImprovedAcousticExtractor()
    text_extractor = EmotionTextExtractor()
    
    if HAS_WHISPER:
        print("Loading Whisper model...")
        whisper_model = whisper.load_model("base")
    
    # Test on 10 samples from hybrid_metadata.csv
    try:
        df_test = pd.read_csv('data/hybrid_metadata.csv')
        samples = df_test.sample(n=10)
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return
    
    results = []
    
    # Needs to locate actual files.
    # Metadata has 'dataset' and 'filename' (or call_id?)
    # The 'filename' in metadata usually matches the file on disk or needs path construction.
    # Let's try to find them.
    
    base_paths = {
        'CREMA-D': 'd:/haam_framework/crema-d-mirror-main', 
        'IEMOCAP': 'd:/haam_framework/iemocapfullrelease'
    }
    
    for idx, row in samples.iterrows():
        print(f"\n[{idx+1}] ID: {row.get('call_id', 'Unknown')}")
        
        # Construct path
        filename = row.get('filename', str(row['call_id']))
        if not filename.endswith('.wav'):
            filename += '.wav'
            
        dataset = row['dataset']
        base = base_paths.get(dataset, '')
        
        # Use rglob to find file if path is not exact
        # This is slow but robust for testing
        found_path = None
        if base:
            import pathlib
            matches = list(pathlib.Path(base).rglob(filename))
            if matches:
                found_path = str(matches[0])
            else:
                 # Try using 'audio_wavs' in processed if raw not found
                 matches = list(pathlib.Path('data/processed/audio_wavs').rglob(filename))
                 if matches:
                     found_path = str(matches[0])
        
        if not found_path:
            print(f"  ? File not found locally: {filename}")
            results.append({'filename': filename, 'success': False, 'error': 'File not found'})
            continue
            
        print(f"  Found: {found_path}")
        
        try:
            # Preprocess
            audio, sr = preprocessor.preprocess(found_path)
            print(f"  ✓ Preprocessed: {len(audio)/sr:.2f}s at {sr}Hz")
            
            # Acoustic features
            acoustic = acoustic_extractor.extract(found_path) # Extractor also reloads audio, which is inefficient but robust
            zero_count = sum(1 for v in acoustic.values() if v == 0)
            print(f"  ✓ Acoustic: {len(acoustic)} features ({zero_count} zeros)")
            
            # Transcribe
            transcript = ""
            if HAS_WHISPER:
                transcript_result = whisper_model.transcribe(found_path)
                transcript = transcript_result['text'].strip()
                print(f"  ✓ Transcript: \"{transcript[:40]}...\"")
            else:
                transcript = row.get('transcript', "")
                print(f"  ✓ Transcript (from meta): \"{transcript[:40]}...\"")
            
            # Text features
            text = text_extractor.extract(transcript)
            print(f"  ✓ Text emotion: {text['dominant_emotion']} ({text['confidence']:.2f})")
            
            results.append({
                'filename': filename,
                'success': True,
                'zero_acoustic_features': zero_count,
                **acoustic
            })
            
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({'filename': filename, 'success': False, 'error': str(e)})
    
    df_results = pd.DataFrame(results)
    df_results.to_csv('results/diagnosis/feature_test.csv', index=False)
    
    print("\n" + "="*80)
    print("FEATURE TEST SUMMARY")
    print("="*80)
    if len(df_results) > 0:
        success_rate = df_results['success'].mean() * 100
        print(f"Success rate: {success_rate:.1f}%")
        
        successful = df_results[df_results['success']]
        if len(successful) > 0:
            avg_zeros = successful['zero_acoustic_features'].mean()
            print(f"Avg zero features: {avg_zeros:.1f}/12")
            if avg_zeros > 4:
                print("⚠ Warning: Many features extracting as zero!")
    else:
        print("No results generated.")

if __name__ == '__main__':
    main()
