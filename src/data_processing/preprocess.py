import torch
import librosa
import os
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, DistilBertTokenizer, DistilBertModel
from pyannote.audio import Pipeline
import warnings
import sys
from huggingface_hub import HfFolder  # For secure token handling
import json  # For output saving

sys.modules["torchvision"] = None
# Suppress a specific warning from huggingface_hub
warnings.filterwarnings("ignore", category=UserWarning, module='huggingface_hub.utils._token')

# --- 1. INITIALIZE MODELS (do this once) ---
# This section loads the pre-trained AI models that will act as our "workers".

# Worker 1: The Transcriber (Automatic Speech Recognition)
# This model listens to audio and converts it to text.
print("Loading ASR (Transcriber) model...")
# Set device before loading models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
asr_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)  # Move to device
print("ASR model loaded.")

# Worker 2: The Detective (Speaker Diarization)
# This model listens to a conversation and identifies who is speaking and when.
print("Loading Diarization (Detective) model...")

# Secure token handling: Assumes you've run `huggingface-cli login` or set HF_TOKEN env var.
hf_token =  os.getenv("HF_TOKEN")

diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=hf_token
)
# Send the pipeline to a GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
diarization_pipeline = diarization_pipeline.to(device)
print(f"Diarization model loaded and sent to {device}.")

# Worker 3: Text Encoder (DistilBERT for embeddings)
print("Loading Text Encoder model...")
text_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
text_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
print("Text Encoder loaded.")

# --- 2. AUDIO FEATURE EXTRACTION FUNCTION ---
def extract_audio_features(segment, sample_rate):
    """
    Extracts acoustic features from an audio segment.
    
    Args:
        segment (np.array): Audio segment.
        sample_rate (int): Sampling rate.
    
    Returns:
        dict: Extracted features (MFCC, pitch, energy).
    """
    try:
        # MFCC (Mel-Frequency Cepstral Coefficients)
        mfccs = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=13)
        mfcc_mean = mfccs.mean(axis=1).tolist()
        
        # Pitch (using YIN algorithm for robustness)
        pitches, magnitudes = librosa.piptrack(y=segment, sr=sample_rate)
        pitch = pitches[magnitudes > 0].mean() if np.any(magnitudes > 0) else 0.0
        
        # Energy (RMS - Root Mean Square)
        energy = librosa.feature.rms(y=segment)[0].mean()
        
        return {
            "mfccs": mfcc_mean,
            "pitch": float(pitch),
            "energy": float(energy)
        }
    except Exception as e:
        print(f"Error extracting audio features: {e}")
        return {"mfccs": [], "pitch": 0.0, "energy": 0.0}

# --- 3. DEFINE THE PROCESSING FUNCTION ---
def process_audio_file(file_path, output_file='output.json'):
    """
    Processes a single audio file to perform diarization, transcription, feature extraction, and text encoding.
    This is the main "assembly line" function.
    
    Args:
        file_path (str): The path to the .wav audio file.
        output_file (str): Path to save the JSON output.
    
    Returns:
        list: A list of dictionaries with speaker, times, transcript, features, and embeddings.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return []

    print(f"\nProcessing file: {file_path}...")
    
    # Step 1: Load the audio file.
    try:
        audio_input, sample_rate = librosa.load(file_path, sr=16000)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return []

    # Step 2: Perform Speaker Diarization.
    print("Performing speaker diarization...")
    diarization_input = {"waveform": torch.from_numpy(audio_input).unsqueeze(0).to(device), "sample_rate": sample_rate}
    
    try:
        diarization = diarization_pipeline(diarization_input)
    except Exception as e:
        print(f"Error during diarization: {e}")
        return []

    processed_segments = []

    # Step 3: Process each speaker segment.
    print("Transcribing and extracting features for segments...")
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_time = turn.start
        end_time = turn.end

        # Extract the audio segment
        start_idx = int(start_time * sample_rate)
        end_idx = int(end_time * sample_rate)
        segment = audio_input[start_idx:end_idx]

        if len(segment) == 0:
            print(f"Skipping empty segment at {start_time:.2f}s - {end_time:.2f}s")
            continue

        # Validate segment length
        if len(segment) < 16000:  # Minimum 1 second at 16kHz
            print(f"Warning: Segment too short ({len(segment)} samples) at {start_time:.2f}s - {end_time:.2f}s")
            continue

        # Transcribe the segment
        transcription = ""
        try:
            # Ensure input_values is on the correct device
            input_values = asr_processor(segment, return_tensors="pt", padding="longest", sampling_rate=16000).input_values.to(device)
            with torch.no_grad():
                logits = asr_model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = asr_processor.batch_decode(predicted_ids)[0].strip().lower()
            if not transcription:
                print(f"Warning: Empty transcription for segment at {start_time:.2f}s - {end_time:.2f}s")
        except Exception as e:
            print(f"Error during transcription at {start_time:.2f}s - {end_time:.2f}s: {e}")

        # Extract audio features
        audio_features = extract_audio_features(segment, sample_rate)

        # Get text embeddings (if transcription is not empty)
        text_embeddings = []
        if transcription:
            try:
                inputs = text_tokenizer(transcription, return_tensors="pt", truncation=True, max_length=512).to(device)
                with torch.no_grad():
                    outputs = text_model(**inputs)
                text_embeddings = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy().flatten().tolist()
            except Exception as e:
                print(f"Error getting text embeddings at {start_time:.2f}s - {end_time:.2f}s: {e}")

        # Store the structured information
        segment_info = {
            "speaker": speaker,
            "start_time": round(start_time, 2),
            "end_time": round(end_time, 2),
            "transcript": transcription,
            "audio_features": audio_features,
            "text_embeddings": text_embeddings
        }
        processed_segments.append(segment_info)
        print(f"  - {speaker} ({start_time:.2f}s - {end_time:.2f}s): {transcription}")

    # Save to JSON
    if processed_segments:
        try:
            with open(output_file, 'w') as f:
                json.dump(processed_segments, f, indent=4)
            print(f"Results saved to {output_file}")
        except Exception as e:
            print(f"Error saving output: {e}")

    return processed_segments

# --- 4. EXAMPLE USAGE ---
if __name__ == "__main__":
    # Replace with your audio file path
    example_file = 'HARVARD_raw/Harvard list 02.wav'
    results = process_audio_file(example_file, output_file='processed_output.json')
    
    if results:
        print("\n--- Processing Complete ---")
        print(f"Found {len(results)} segments.")
    else:
        print("\n--- Processing Failed ---")
        print("Please check the file path and ensure all models and tokens are set up correctly.")