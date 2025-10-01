import os, sys
import torch
import torchaudio
import subprocess
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2Model, AutoTokenizer, AutoModel, pipeline

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.models.fusion_model import MultimodalFusionModel

# ----------------------------
# CONFIG
# ----------------------------
AUDIO_DIM = 768
TEXT_DIM = 768
HIDDEN_DIM = 1024
NUM_SENTIMENT_CLASSES = 3
NUM_EMOTION_CLASSES = 7
DROPOUT = 0.3

sentiment_mapping = {0: "positive", 1: "negative", 2: "neutral"}
emotion_mapping = {0: "neutral", 1: "joy", 2: "sadness", 3: "fear", 4: "anger", 5: "surprise", 6: "disgust"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Audio Conversion (WebM → WAV)
# ----------------------------
def convert_to_wav(input_path):
    output_path = input_path.rsplit(".", 1)[0] + "_converted.wav"
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ac", "1", "-ar", "16000",
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path

# ----------------------------
# Real Speech-to-Text with Whisper
# ----------------------------
def transcribe_audio(audio_path):
    print("🎙️ Running Whisper for transcription...")
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=0 if torch.cuda.is_available() else -1)
    result = asr(audio_path)
    transcript = result["text"].strip()
    return transcript if transcript else "Could not transcribe."

# ----------------------------
# Main pipeline
# ----------------------------
def process_audio_file(audio_path, model_path):
    # Step 1: Convert to WAV
    wav_path = convert_to_wav(audio_path)

    # Step 2: Transcript
    transcript = transcribe_audio(wav_path)

    # Step 3: Text embedding (BERT)
    text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    text_model = AutoModel.from_pretrained("bert-base-uncased").to(device)

    text_inputs = text_tokenizer(transcript, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_features = text_model(**text_inputs.to(device)).last_hidden_state.mean(dim=1).cpu()

    # Dummy audio embedding (⚡ you can replace with wav2vec2 if needed)
    audio_features = torch.zeros((1, AUDIO_DIM))

    # Step 4: Load Sprint model
    model = MultimodalFusionModel(
        audio_input_dim=AUDIO_DIM,
        text_input_dim=TEXT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_sentiment_classes=NUM_SENTIMENT_CLASSES,
        num_emotion_classes=NUM_EMOTION_CLASSES,
        dropout_rate=DROPOUT
    ).to(device)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Step 5: Prediction
    with torch.no_grad():
        sent_logits, emo_logits, attn = model(audio_features.to(device), text_features.to(device))

    sent_pred = torch.argmax(torch.softmax(sent_logits, dim=1), dim=1).item()
    emo_pred = torch.argmax(torch.softmax(emo_logits, dim=1), dim=1).item()

    # Step 5: Handle attention safely
    if attn is not None and not torch.isnan(attn).any():
        audio_attn = float(attn[0, 0].item())
        text_attn = float(attn[0, 1].item())
        total = audio_attn + text_attn
        if total > 0:
            audio_attn = round((audio_attn / total) * 100, 2)
            text_attn = round((text_attn / total) * 100, 2)
        else:
            audio_attn, text_attn = 50.0, 50.0
    else:
        audio_attn, text_attn = 50.0, 50.0
    return {
        "transcript": transcript,
        "sentiment": sentiment_mapping[sent_pred],
        "emotion": emotion_mapping[emo_pred],
        "attention": {
            "audio": audio_attn,
            "text": text_attn
        }
    }