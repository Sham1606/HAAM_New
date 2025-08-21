import torch
import librosa
import os
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pyannote.audio import Pipeline
import warnings
import sys
sys.modules["torchvision"] = None
# Suppress a specific warning from huggingface_hub
warnings.filterwarnings("ignore", category=UserWarning, module='huggingface_hub.utils._token')


# --- 1. INITIALIZE MODELS (do this once) ---
# This section loads the pre-trained AI models that will act as our "workers".

# Worker 1: The Transcriber (Automatic Speech Recognition)
# This model listens to audio and converts it to text.
print("Loading ASR (Transcriber) model...")
asr_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
print("ASR model loaded.")

# Worker 2: The Detective (Speaker Diarization)
# This model listens to a conversation and identifies who is speaking and when.
print("Loading Diarization (Detective) model...")

# --- IMPORTANT: SECURITY WARNING ---
# You have hardcoded your Hugging Face token below.
# This is okay for a private script, but DO NOT share this code publicly
# or commit it to a public Git repository, as it exposes your personal token.
# A safer method for future projects is to log in via the terminal:
# `huggingface-cli login`
# and then you can remove the `use_auth_token` argument.
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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" # Your token
)
# Send the pipeline to a GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
diarization_pipeline = diarization_pipeline.to(device)
print(f"Diarization model loaded and sent to {device}.")


# --- 2. DEFINE THE PROCESSING FUNCTION ---
def process_audio_file(file_path):
    """
    Processes a single audio file to perform diarization and transcription.
    This is the main "assembly line" function.
    
    Args:
        file_path (str): The path to the .wav audio file.
    
    Returns:
        list: A list of dictionaries, each containing a speaker, start/end times, and transcript.
              Returns an empty list if the file cannot be processed.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return []

    print(f"\nProcessing file: {file_path}...")
    
    # Step 1: Load the audio file.
    # Librosa ensures it's in the correct format (16kHz mono) for our models.
    try:
        audio_input, sample_rate = librosa.load(file_path, sr=16000)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return []

    # Step 2: Perform Speaker Diarization.
    print("Performing speaker diarization...")
    # The input for diarization must be a PyTorch tensor on the correct device.
    diarization_input = {"waveform": torch.from_numpy(audio_input).unsqueeze(0).to(device), "sample_rate": sample_rate}
    
    try:
        diarization = diarization_pipeline(diarization_input)
    except Exception as e:
        print(f"Error during diarization: {e}")
        return []

    processed_segments = []

    # Step 3: Transcribe each speaker segment.
    print("Transcribing segments...")
    # The loop iterates through the timeline created by the diarization model.
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        # Get the start and end times of the speaker's turn
        start_time = turn.start
        end_time = turn.end

        # Extract the audio segment for this specific turn from the original audio
        segment = audio_input[int(start_time * sample_rate):int(end_time * sample_rate)]

        if len(segment) == 0:
            continue

        # Prepare the audio segment for the ASR model
        input_values = asr_processor(segment, return_tensors="pt", padding="longest", sampling_rate=16000).input_values

        # Get the model's predictions (logits)
        logits = asr_model(input_values).logits

        # Decode the predictions to get the final transcribed text
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = asr_processor.batch_decode(predicted_ids)[0]

        # Store the structured information
        segment_info = {
            "speaker": speaker,
            "start_time": round(start_time, 2),
            "end_time": round(end_time, 2),
            "transcript": transcription.strip().lower() # Clean up whitespace and convert to lower
        }
        processed_segments.append(segment_info)
        print(f"  - {speaker} ({start_time:.2f}s - {end_time:.2f}s): {segment_info['transcript']}")

    return processed_segments

# --- 3. EXAMPLE USAGE ---
if __name__ == "__main__":
    # --- IMPORTANT ---
    # Replace this with the actual path to YOUR audio file.
    # The file should be a .wav file.
    example_file = 'HARVARD_raw/Harvard list 02.wav' 
    
    results = process_audio_file(example_file)
    
    if results:
        print("\n--- Processing Complete ---")
        print(f"Found {len(results)} segments.")
        # In a real application, you would save these 'results' to a JSON file or database.
        # For example:
        import json
        with open('output.json', 'w') as f:
            json.dump(results, f, indent=4)
    else:
        print("\n--- Processing Failed ---")
        print("Please check the file path and ensure all models and tokens are set up correctly.")

