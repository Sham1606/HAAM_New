import os
import sys
import torch
import torchaudio
import tempfile
import subprocess
import pandas as pd

from transformers import Wav2Vec2Processor, Wav2Vec2Model, AutoTokenizer, AutoModel

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.fusion_model import MultimodalFusionModel

# ----------------------------
# CONFIG
# ----------------------------
VIDEO_PATH = r"D:\My_Data_Science\fpo\haam_framework\data\raw\MELD.Raw\dev\dia5_utt10.mp4"
CSV_PATH = r"D:\My_Data_Science\fpo\haam_framework\data\raw\dev_sent_emo.csv"
MODEL_PATH = r"D:\My_Data_Science\fpo\haam_framework\saved_models\sprint_model_v5_best_fusion.pth"

AUDIO_DIM = 768
TEXT_DIM = 768
HIDDEN_DIM = 1024
NUM_SENTIMENT_CLASSES = 3
NUM_EMOTION_CLASSES = 7
DROPOUT = 0.3

# Label mappings
sentiment_mapping = {0: "positive", 1: "negative", 2: "neutral"}
emotion_mapping = {0: "neutral", 1: "joy", 2: "sadness", 3: "fear", 4: "anger", 5: "surprise", 6: "disgust"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Step 1: Extract Dialogue_ID, Utterance_ID from filename
# ----------------------------
filename = os.path.basename(VIDEO_PATH)
parts = filename.replace(".mp4", "").split("_")
dialogue_id = int(parts[0].replace("dia", ""))
utterance_id = int(parts[1].replace("utt", ""))

print(f"🎬 Testing file → Dialogue_ID={dialogue_id}, Utterance_ID={utterance_id}")

# ----------------------------
# Step 2: Load transcript from CSV
# ----------------------------
df = pd.read_csv(CSV_PATH)
row = df[(df["Dialogue_ID"] == dialogue_id) & (df["Utterance_ID"] == utterance_id)]

if row.empty:
    print("❌ Error: Could not find transcript in CSV!")
    transcript = "Dummy transcript."
else:
    transcript = row.iloc[0]["Utterance"]

print(f"📝 Transcript: {transcript}")

# ----------------------------
# Step 3: Extract audio using ffmpeg
# ----------------------------
print("🔉 Extracting audio from video with ffmpeg...")
tmp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

cmd = [
    "ffmpeg", "-y", "-i", VIDEO_PATH,
    "-ar", "16000", "-ac", "1",  # resample to 16kHz mono
    tmp_audio
]
subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Load with torchaudio
waveform, sr = torchaudio.load(tmp_audio)
print(f"✅ Extracted audio shape: {waveform.shape}, Sample rate: {sr}")

# ----------------------------
# Step 4: Load embedding models
# ----------------------------
print("📥 Loading embedding models...")
audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)

text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text_model = AutoModel.from_pretrained("bert-base-uncased").to(device)

# ----------------------------
# Step 5: Audio embedding
# ----------------------------
inputs = audio_processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
with torch.no_grad():
    audio_features = audio_model(**inputs.to(device)).last_hidden_state.mean(dim=1).cpu().numpy()

# ----------------------------
# Step 6: Text embedding
# ----------------------------
inputs = text_tokenizer(transcript, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    text_features = text_model(**inputs.to(device)).last_hidden_state.mean(dim=1).cpu().numpy()

# ----------------------------
# Step 7: Load Sprint model
# ----------------------------
print("🧠 Loading Sprint model...")
model = MultimodalFusionModel(
    audio_input_dim=AUDIO_DIM,
    text_input_dim=TEXT_DIM,
    hidden_dim=HIDDEN_DIM,
    num_sentiment_classes=NUM_SENTIMENT_CLASSES,
    num_emotion_classes=NUM_EMOTION_CLASSES,
    dropout_rate=DROPOUT
).to(device)

state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.eval()

# ----------------------------
# Step 8: Prediction
# ----------------------------
audio_tensor = torch.tensor(audio_features, dtype=torch.float32).to(device)
text_tensor = torch.tensor(text_features, dtype=torch.float32).to(device)

with torch.no_grad():
    sent_logits, emo_logits, attn = model(audio_tensor, text_tensor)

sent_pred = torch.argmax(torch.softmax(sent_logits, dim=1), dim=1).item()
emo_pred = torch.argmax(torch.softmax(emo_logits, dim=1), dim=1).item()

# ----------------------------
# Step 9: Results
# ----------------------------
print("\n--- Predictions for Single Video ---")
print(f"Transcript: {transcript}")
print(f"Predicted Sentiment: {sentiment_mapping[sent_pred]}")
print(f"Predicted Emotion: {emotion_mapping[emo_pred]}")
print(f"Attention → Audio: {attn[0,0].item()*100:.2f}%, Text: {attn[0,1].item()*100:.2f}%")

# Cleanup
os.remove(tmp_audio)
