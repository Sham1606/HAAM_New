import os
import sys
import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Add project root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing.audio_preprocessor import AudioPreprocessor
from src.features.improved_acoustic import ImprovedAcousticExtractor
from src.features.emotion_text import EmotionTextExtractor

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
CACHE_FILE = Path("data/processed_features_v2.pkl")
CREMA_DIR = Path(r"D:\haam_framework\crema-d-mirror-main\AudioWAV")
IEMOCAP_DIR = Path(r"D:\haam_framework\iemocapfullrelease\1\IEMOCAP_full_release")

EMOTIONS = ['anger', 'disgust', 'fear', 'neutral', 'sadness']

def get_ground_truth_label(filename, dataset):
    """Parse filename to get ground truth emotion."""
    filename = str(filename).lower()
    if dataset == 'CREMA-D':
        # 1001_DFA_ANG_XX.wav
        if '_ang_' in filename: return 'anger'
        if '_dis_' in filename: return 'disgust'
        if '_fea_' in filename: return 'fear'
        if '_hap_' in filename: return 'neutral' # Mapping Happy -> Neutral per plan? Or ignore? Plan said "Joy dropped" or "mapped". Check emotion_text.py maps Joy->Neutral. I'll map Happy->Neutral here for consistency, OR Keep it as 'joy' and filter later.
        # Let's map HAP -> neutral for now to match 5-class target.
        if '_neu_' in filename: return 'neutral'
        if '_sad_' in filename: return 'sadness'
    elif dataset == 'IEMOCAP':
        # Depends on directory structure or label file.
        # Parsing filename is unreliable for IEMOCAP, usually need the label file.
        # But for 'Baseline', maybe we skip IEMOCAP if parsing is hard without the label map loaded.
        # DIAGNOSIS script loaded metadata. 
        # I should use the 'data/hybrid_metadata.csv' if expected, or re-parse.
        # Since I don't have the full label map loaded here easily, I will skip IEMOCAP for the *Baseline script* if it takes too long to implement parsing, 
        # OR I rely on the filename if it contains the label (some versions do?). 
        # Actually, let's look at `scripts/process_iemocap_dataset.py` logic. It parses a text file.
        # To make this script standalone and robust, I should rely on a PRE-BUILT metadata csv if possible.
        # If not, I'll stick to CREMA-D for the initial baseline validation to prove feature power.
        pass
    
    return None

def extract_dataset(limit=None):
    """Extract features for datasets."""
    print("Initializing components...")
    preprocessor = AudioPreprocessor(target_sr=16000)
    acoustic_ext = ImprovedAcousticExtractor()
    text_ext = EmotionTextExtractor()
    
    data = []
    
    # 1. CREMA-D
    print(f"Scanning CREMA-D: {CREMA_DIR}")
    files = list(CREMA_DIR.glob("*.wav"))
    if limit: files = files[:limit]
    
    for f in tqdm(files, desc="Processing CREMA-D"):
        label = get_ground_truth_label(f.name, 'CREMA-D')
        if not label: continue
        
        # Audio Preprocess
        # We process in memory without saving to disk to save time/space
        # But `extract` takes a path. 
        # AudioPreprocessor returns numpy array. ImprovedAcousticExtractor takes path.
        # I should update AcousticExtractor to take array optionally, or save temp.
        # For now, let's just use the raw file path for AcousticExtractor (it loads with librosa),
        # but Preprocessor includes normalization steps that are CRITICAL.
        # So I MUST save temp or modify extractor.
        # Modified Plan: Preprocess -> Save Temp -> Extract.
        
        try:
            # Preprocess
            audio, sr, meta = preprocessor.preprocess(str(f))
            
            # Save to temporary buffer/file?
            # Or pass audio directly if updated. 
            # `ImprovedAcousticExtractor` uses `librosa.load(path)`.
            # I will assume for Baseline we just run extraction on raw files (time constraint) 
            # BUT Preprocessor fixes Clipping. 
            # Let's Skip Preprocessor for this specific 'Baseline' speed run? 
            # NO, Quality is the goal.
            # I'll create a temp dir.
            temp_path = Path(f"temp_processing/{f.name}")
            temp_path.parent.mkdir(exist_ok=True)
            import soundfile as sf
            sf.write(temp_path, audio, sr)
            
            # Extract Acoustic
            feats = acoustic_ext.extract(str(temp_path)) # Using temp processed file
            
            # Extract Text (Mocking for now as we don't have transcripts in filename)
            # CREMA-D has fixed sentences.
            # 12 sentences. Map:
            # IEO: It would show...
            # TIE: That is exactly...
            # etc.
            # Mapping map:
            sentence_code = f.name.split('_')[1]
            sentences = {
                'IEO': "It would show you have the lowest grade of intelligence.",
                'TIE': "That is exactly what happened.",
                'HPL': "We have to become the doctor's patients.",
                'NEU': "I'm neutral.", # Fallback
                'ANG': "I'm angry.",
                # ... Populate full list or use placeholder
                # Ideally use Whisper. But Whisper is slow.
                # For baseline, let's use a Placeholder or skip text features for CREMA-D 
                # OR use the Text Extractor on the *generic* sentence?
            }
            # For strict baseline, let's ignore Text features or set to uniform, 
            # OR run Whisper on the temp file.
            # running Whisper is slow.
            # I will Skip Text Features for this fast baseline script and focus on Acoustic performance.
            # Or use dummy text features.
            
            # Let's try to get meaningful text feats if possible. 
            # I'll use a dummy transcript "Speech audio" to get a neutral vector, 
            # just to verify the PIPELINE handles the concatenation.
            text_feats = text_ext.extract("Speech audio.")
            
            # Combine
            # 12 acoustic + 5 text probs
            vec = []
            # Acoustic (sorted keys)
            ac_keys = sorted(feats.keys())
            for k in ac_keys: vec.append(feats[k])
            
            # Text
            # txt_keys = sorted(text_feats['emotion_probs'].keys())
            # for k in txt_keys: vec.append(text_feats['emotion_probs'][k])
            
            # Store
            data.append({
                'file': f.name,
                'dataset': 'CREMA-D',
                'label': label,
                'features': feats, # Dictionary
                'text_features': text_feats['emotion_probs']
            })
            
            # Cleanup
            temp_path.unlink()
            
        except Exception as e:
            print(f"Failed {f.name}: {e}")
            continue
            
    # Save
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(data, f)
    
    return data

def train_baselines():
    if CACHE_FILE.exists():
        print(f"Loading cached features from {CACHE_FILE}...")
        with open(CACHE_FILE, 'rb') as f:
            data = pickle.load(f)
    else:
        print("Cache not found. Extracting dataset (limit=500 for speed)...")
        data = extract_dataset(limit=500) # Limit for speed test
        
    print(f"Loaded {len(data)} samples.")
    
    # Prepare Arrays
    X = []
    y = []
    
    for item in data:
        # Flatten features
        f = item['features']
        # Ensure sorting
        ac_vals = [f[k] for k in sorted(f.keys())]
        
        # Optional: Add Text features if available
        # t = item['text_features']
        # tx_vals = [t[k] for k in sorted(t.keys())]
        
        # for now, only Acoustic to test Phase 2.1
        X.append(ac_vals)
        y.append(item['label'])
        
    X = np.array(X)
    y = np.array(y)
    
    # Encode Labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print("\nTraining Baseline Models (Acoustic Only)...")
    
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100),
        'MLP': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)
    }
    
    results = {}
    
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = acc
        print(f"\n{name} Accuracy: {acc*100:.2f}%")
        print(classification_report(y_test, preds, target_names=le.classes_))
        
    print("\nSummary:")
    for k, v in results.items():
        print(f"{k}: {v*100:.2f}%")

if __name__ == "__main__":
    train_baselines()
