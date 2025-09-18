import torch
import librosa
import os
import numpy as np
import pandas as pd
from transformers import (
    WavLMModel, Wav2Vec2FeatureExtractor,
    BertTokenizer, BertModel
)
import warnings
import json
from dotenv import load_dotenv
from tqdm import tqdm
import subprocess

# --- Setup ---
load_dotenv()
warnings.filterwarnings("ignore", category=FutureWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Check for ffmpeg ---
try:
    subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
except (subprocess.CalledProcessError, FileNotFoundError):
    print("Error: ffmpeg is not installed or not in PATH.")
    exit(1)

# --- 1. INITIALIZE MODELS ---
print("Initializing models...")

# Audio Embedding Model (WavLM-base-plus)
audio_processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
audio_model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(device)

# Text Embedding Model (BERT-base)
text_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text_model = BertModel.from_pretrained('bert-base-uncased').to(device)

print(f"Models loaded and sent to {device}.")

# --- 2. HELPER FUNCTIONS ---
def extract_audio_from_mp4(mp4_path, wav_path):
    """Extracts audio from MP4 using ffmpeg (mono, 16kHz)."""
    if os.path.exists(wav_path):
        return True
    try:
        cmd = ["ffmpeg", "-y", "-i", mp4_path, "-ac", "1", "-ar", "16000", "-vn", wav_path]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=20)
        return True
    except subprocess.TimeoutExpired:
        print(f"  - WARNING: ffmpeg timeout for {mp4_path}")
        return False
    except subprocess.CalledProcessError as e:
        print(f"  - WARNING: ffmpeg failed for {mp4_path}, Error: {e}")
        return False

def get_audio_embedding(audio_input, processor, model):
    """Generates audio embedding from raw waveform using WavLM-base-plus."""
    if audio_input is None or len(audio_input) == 0:
        return []
    try:
        inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model(inputs.input_values)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten().tolist()
    except Exception as e:
        print(f"  - Error getting audio embedding: {e}")
        return []

def get_text_embedding(text, tokenizer, model):
    """Generates text embedding from input string using BERT-base."""
    if not text or not isinstance(text, str):
        return []
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten().tolist()
    except Exception as e:
        print(f"  - Error getting text embedding: {e}")
        return []

def process_meld_split(video_path, csv_path, json_output_dir, audio_output_dir):
    """Processes MELD split (train/dev/test) into JSON embeddings with resume support."""
    if not os.path.exists(csv_path):
        print(f"Error: CSV not found at {csv_path}")
        return
    df = pd.read_csv(csv_path)
    os.makedirs(json_output_dir, exist_ok=True)
    os.makedirs(audio_output_dir, exist_ok=True)

    total_files = len(df)
    print(f"Processing {total_files} files for {os.path.basename(csv_path)}...")

    for i, row in tqdm(df.iterrows(), total=total_files):
        dialogue_id = row['Dialogue_ID']
        utterance_id = row['Utterance_ID']
        mp4_filename = f"dia{dialogue_id}_utt{utterance_id}.mp4"
        mp4_filepath = os.path.join(video_path, mp4_filename)

        json_filename = f"dia{dialogue_id}_utt{utterance_id}.json"
        json_filepath = os.path.join(json_output_dir, json_filename)
        wav_filename = f"dia{dialogue_id}_utt{utterance_id}.wav"
        wav_filepath = os.path.join(audio_output_dir, wav_filename)

        # ✅ Resume support: Skip if JSON already exists
        if os.path.exists(json_filepath):
            continue
        if not os.path.exists(wav_filepath):
            if not extract_audio_from_mp4(mp4_filepath, wav_filepath):
                continue  # skip if ffmpeg failed

        if not os.path.exists(mp4_filepath):
            print(f"  - WARNING: MP4 not found: {mp4_filename}. Skipping.")
            continue

        audio_embedding = []
        if extract_audio_from_mp4(mp4_filepath, wav_filepath):
            try:
                audio_input, _ = librosa.load(wav_filepath, sr=16000)
            except Exception as e:
                print(f"  - WARNING: librosa failed for {wav_filepath}, Error: {e}")
                audio_input = []
            audio_embedding = get_audio_embedding(audio_input, audio_processor, audio_model)

        text_embedding = get_text_embedding(row['Utterance'], text_tokenizer, text_model)

        # Fallbacks
        if not audio_embedding:
            audio_embedding = [0.0] * 768  # WavLM-base-plus dim
        if not text_embedding:
            text_embedding = [0.0] * 768   # BERT-base dim

        final_data = {
            "dialogue_id": int(dialogue_id),
            "utterance_id": int(utterance_id),
            "transcript": row['Utterance'],
            "sentiment": row['Sentiment'],
            "emotion": row['Emotion'],
            "audio_embedding": audio_embedding,
            "text_embedding": text_embedding
        }

        with open(json_filepath, 'w') as f:
            json.dump(final_data, f)

        # Debug progress every 100 utterances
        if i % 100 == 0:
            print(f"Processed {i}/{total_files} utterances...")

    json_count = len([f for f in os.listdir(json_output_dir) if f.endswith('.json')])
    print(f"\nFinished {os.path.basename(csv_path)}: CSV rows = {len(df)}, JSON = {json_count}")


# --- 4. MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
    except NameError:
        project_root = os.getcwd()

    train_video_path = os.path.join(project_root, 'data', 'raw', 'MELD.Raw', 'train')
    dev_video_path = os.path.join(project_root, 'data', 'raw', 'MELD.Raw', 'dev')
    test_video_path = os.path.join(project_root, 'data', 'raw', 'MELD.Raw', 'test')

    train_csv_path = os.path.join(project_root, 'data', 'raw', 'train_sent_emo.csv')
    dev_csv_path = os.path.join(project_root, 'data', 'raw', 'dev_sent_emo.csv')
    test_csv_path = os.path.join(project_root, 'data', 'raw', 'test_sent_emo.csv')

    processed_output_path = os.path.join(project_root, 'data', 'processed')
    audio_output_path = os.path.join(processed_output_path, 'audio_wavs')

    print(f"Project Root: {project_root}")

    process_meld_split(train_video_path, train_csv_path, os.path.join(processed_output_path, 'json_embeddings'), os.path.join(audio_output_path, 'train'))
    process_meld_split(dev_video_path, dev_csv_path, os.path.join(processed_output_path, 'json_embeddings'), os.path.join(audio_output_path, 'dev'))
    process_meld_split(test_video_path, test_csv_path, os.path.join(processed_output_path, 'json_embeddings'), os.path.join(audio_output_path, 'test'))

    print("\n--- All MELD processing complete! ---")
