import torch
import librosa
import os
import numpy as np
import pandas as pd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, DistilBertTokenizer, DistilBertModel
from pyannote.audio import Pipeline
import warnings
import json
from moviepy.video.io.VideoFileClip import VideoFileClip
from dotenv import load_dotenv

# --- Load Environment Variables ---
load_dotenv()

# --- Suppress Warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module='huggingface_hub.utils._token')
warnings.filterwarnings("ignore", category=FutureWarning)

# --- 1. INITIALIZE MODELS ---
print("Initializing models...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ASR Model
asr_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)

# Diarization Model
hf_token = os.getenv('HAAM')
if hf_token is None:
    print("\n--- AUTHENTICATION ERROR ---")
    print("Hugging Face token not found in .env file.")
    exit()

try:
    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token).to(device)
except Exception as e:
    print(f"\n--- MODEL DOWNLOAD FAILED ---\nError: {e}")
    print("\nPlease ensure you have accepted the terms on BOTH of these Hugging Face pages:")
    print("1. https://huggingface.co/pyannote/speaker-diarization-3.1")
    print("2. https://huggingface.co/pyannote/segmentation-3.0")
    exit()

# Text Encoder Model
text_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
text_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
print(f"Models loaded and sent to {device}.")


# --- 2. HELPER FUNCTIONS ---
def extract_audio_from_mp4(mp4_path, wav_path):
    if os.path.exists(wav_path): return True
    try:
        with VideoFileClip(mp4_path) as video:
            video.audio.write_audiofile(wav_path, codec='pcm_s16le', fps=16000, logger=None)
        return True
    except Exception as e:
        print(f"  - Error extracting audio from {mp4_path}: {e}")
        return False

def extract_audio_features(segment, sample_rate):
    try:
        mfccs = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=13).mean(axis=1).tolist()
        pitches, magnitudes = librosa.piptrack(y=segment, sr=sample_rate)
        pitch = pitches[magnitudes > 0].mean() if np.any(magnitudes > 0) else 0.0
        energy = librosa.feature.rms(y=segment)[0].mean()
        return {"mfccs": mfccs, "pitch": float(pitch), "energy": float(energy)}
    except Exception:
        return {"mfccs": [], "pitch": 0.0, "energy": 0.0}

def get_text_embeddings(text, tokenizer, model):
    if not text or not isinstance(text, str): return []
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten().tolist()
    except Exception as e:
        print(f"  - Error getting text embeddings for text '{text}': {e}")
        return []

# --- 3. MAIN PROCESSING FUNCTION ---
def process_meld_split(video_dir, csv_path, output_folder):
    """
    Processes a MELD split (train, dev, or test) using explicit paths for videos and CSV.
    """
    split_name = os.path.basename(video_dir)
    print(f"\nStarting MELD processing for '{split_name}' split...")
    
    audio_output_dir = os.path.join(output_folder, 'audio')
    json_output_dir = os.path.join(output_folder, 'json')
    os.makedirs(audio_output_dir, exist_ok=True)
    os.makedirs(json_output_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"  - ERROR: CSV file not found at: {csv_path}")
        return

    print(f"  - Using CSV file: {csv_path}")
    df = pd.read_csv(csv_path)

    total_rows = len(df)
    for index, row in df.iterrows():
        dialogue_id = row['Dialogue_ID']
        utterance_id = row['Utterance_ID']
        
        print(f"\rProcessing {index + 1}/{total_rows}: dia{dialogue_id}_utt{utterance_id}.mp4...", end="")

        mp4_filename = f"dia{dialogue_id}_utt{utterance_id}.mp4"
        mp4_filepath = os.path.join(video_dir, mp4_filename)
        
        json_filename = f"dia{dialogue_id}_utt{utterance_id}.json"
        json_filepath = os.path.join(json_output_dir, json_filename)

        if os.path.exists(json_filepath):
            continue

        if not os.path.exists(mp4_filepath):
            print(f"\n  - MP4 file not found: {mp4_filename}. Skipping.")
            continue

        wav_filename = f"dia{dialogue_id}_utt{utterance_id}.wav"
        wav_filepath = os.path.join(audio_output_dir, wav_filename)
        if not extract_audio_from_mp4(mp4_filepath, wav_filepath):
            continue
            
        audio_input, sample_rate = librosa.load(wav_filepath, sr=16000)
        audio_features = extract_audio_features(audio_input, sample_rate)
        transcript = row['Utterance']
        text_embeddings = get_text_embeddings(transcript, text_tokenizer, text_model)
        
        final_data = {
            "dialogue_id": int(dialogue_id),
            "utterance_id": int(utterance_id),
            "transcript": transcript,
            "sentiment": row['Sentiment'],
            "emotion": row['Emotion'],
            "audio_features": audio_features,
            "text_embeddings": text_embeddings
        }
        
        with open(json_filepath, 'w') as f:
            json.dump(final_data, f, indent=4)
    print(f"\nFinished processing for '{split_name}' split.")


# --- 4. MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # --- CORRECTED PATH LOGIC ---
    # This robustly finds the project root directory by navigating up from the script's location.
    # This ensures that the script can be run from any directory.
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
    except NameError:
        # Fallback for interactive environments where __file__ is not defined
        project_root = os.getcwd()

    
    # Paths to your raw data, constructed from the project root
    train_video_path = os.path.join(project_root, 'data', 'raw', 'MELD.Raw', 'train')
    dev_video_path = os.path.join(project_root, 'data', 'raw', 'MELD.Raw', 'dev')
    test_video_path = os.path.join(project_root, 'data', 'raw', 'MELD.Raw', 'test')
    
    train_csv_path = os.path.join(project_root, 'data', 'raw', 'train_sent_emo.csv')
    dev_csv_path = os.path.join(project_root, 'data', 'raw', 'dev_sent_emo.csv')
    # The MELD test set CSV is named 'test_sent_emo.csv'
    test_csv_path = os.path.join(project_root, 'data', 'raw', 'test_sent_emo.csv') 

    # Path to save the processed output
    processed_output_path = os.path.join(project_root, 'data', 'processed')

    print(f"Project Root Detected: {project_root}")

    # --- PROCESS EACH DATA SPLIT ---
    process_meld_split(train_video_path, train_csv_path, processed_output_path)
    process_meld_split(dev_video_path, dev_csv_path, processed_output_path)
    process_meld_split(test_video_path, test_csv_path, processed_output_path)
    
    print("\n--- All MELD processing complete! ---")
