
import os
import sys
import logging
import time
import numpy as np
import torch
import joblib
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.hybrid_fusion import HybridFusionNetwork

logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = r"D:\haam_framework\models\hybrid_fusion_model.pth"
SCALER_PATH = r"D:\haam_framework\models\hybrid_scaler.pkl"
ENCODER_PATH = r"D:\haam_framework\models\hybrid_encoder.pkl"

TARGET_EMOTIONS = ['neutral', 'anger', 'disgust', 'fear', 'sadness']

class HybridInference:
    def __init__(self):
        self.device = "cpu" # Force CPU for consistency
        logger.info("Loading inference models...")
        
        # 1. Load PyTorch Model
        self.model = HybridFusionNetwork(n_acoustic=3, n_text=5, n_classes=5)
        if os.path.exists(MODEL_PATH):
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(self.device)))
            self.model.to(self.device)
            self.model.eval()
            logger.info("✅ Hybrid Fusion Model loaded")
        else:
            logger.error(f"❌ Model not found at {MODEL_PATH}")
            # Raise or handle? For service, we might want to fail startup
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

        # 2. Load Scaler & Encoder
        if os.path.exists(SCALER_PATH) and os.path.exists(ENCODER_PATH):
            self.scaler = joblib.load(SCALER_PATH)
            self.encoder = joblib.load(ENCODER_PATH)
            logger.info("✅ Scaler & Encoder loaded")
        else:
            logger.error("❌ Scaler or Encoder not found")
            raise FileNotFoundError("Scaler or Encoder not found")

        # 3. Load Whisper (ASR)
        logger.info("Loading Whisper (tiny)...")
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        self.whisper = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
        self.whisper.config.forced_decoder_ids = None

        # 4. Load Sentiment Model
        logger.info("Loading Sentiment Model...")
        self.sentiment_pipe = pipeline(
            "text-classification", 
            model="j-hartmann/emotion-english-distilroberta-base", 
            return_all_scores=True
        )
        logger.info("Inference Service Ready.")

    def extract_acoustic_features(self, y, sr):
        """
        Extracts: pitch_mean, speech_rate_wpm, agent_stress_score (metrics only, stress logic in predict)
        """
        # 1. Pitch
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=75, fmax=600)
        threshold = np.median(magnitudes)
        pitch_values = pitches[magnitudes > threshold]
        pitch_values = pitch_values[pitch_values > 0]
        pitch_mean = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0

        # Return raw measures for prediction logic
        rms = librosa.feature.rms(y=y)
        
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        speech_rate_proxy = float(tempo[0] * 0.5) if len(tempo) > 0 else 0.0

        return {
            "pitch_mean": pitch_mean,
            "tempo_proxy": speech_rate_proxy,
            "rms_std": float(np.std(rms)),
            "rms_mean": float(np.mean(rms))
        }

    def predict(self, audio_path):
        start_time = time.time()
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load Audio
        y, sr = librosa.load(audio_path, sr=16000)
        duration = librosa.get_duration(y=y, sr=sr)
        
        if duration < 0.5:
             raise ValueError("Audio processing error: Audio too short (<0.5s)")
             
        # 1. Transcribe (Whisper)
        input_features = self.processor(y, sampling_rate=16000, return_tensors="pt").input_features
        predicted_ids = self.whisper.generate(input_features)
        transcript = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        if not transcript: transcript = "."

        # 2. Text Features (Sentiment)
        text_features = [0.2] * 5
        try:
            scores = self.sentiment_pipe(transcript[:512])[0]
            score_map = {item['label']: item['score'] for item in scores}
            
            # targets: neutral, anger, disgust, fear, sadness
            val_neutral = score_map.get('neutral', 0.0) + score_map.get('surprise', 0.0) + score_map.get('joy', 0.0)
            val_anger = score_map.get('anger', 0.0)
            val_disgust = score_map.get('disgust', 0.0)
            val_fear = score_map.get('fear', 0.0)
            val_sadness = score_map.get('sadness', 0.0)
            
            raw_vec = [val_neutral, val_anger, val_disgust, val_fear, val_sadness]
            total = sum(raw_vec)
            if total > 0:
                text_features = [v/total for v in raw_vec]
        except Exception as e:
            logger.warning(f"Sentiment extraction failed: {e}")

        # 3. Acoustic Features & Stress Heuristic
        ac_raw = self.extract_acoustic_features(y, sr)
        
        # WPM from transcript
        word_count = len(transcript.split())
        wpm = (word_count / duration) * 60 if duration > 0 else 0.0
        
        # Stress Heuristic
        stress_score = 0.2
        if ac_raw['pitch_mean'] > 250 and wpm > 150:
            stress_score += 0.3
        
        acoustic_vec = [ac_raw['pitch_mean'], wpm, stress_score]
        
        # 4. Predict
        acoustic_np = np.array([acoustic_vec])
        acoustic_scaled = self.scaler.transform(acoustic_np)
        acoustic_tensor = torch.FloatTensor(acoustic_scaled).to(self.device)
        text_tensor = torch.FloatTensor([text_features]).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(acoustic_tensor, text_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            
        # Results
        top_indices = probs.argsort()[::-1][:3]
        top_3 = [(TARGET_EMOTIONS[i], float(probs[i])) for i in top_indices]
        
        predicted_idx = top_indices[0]
        predicted_emotion = TARGET_EMOTIONS[predicted_idx]
        confidence = float(probs[predicted_idx])
        
        inference_time = time.time() - start_time
        
        return {
            "predicted_emotion": predicted_emotion,
            "confidence": confidence,
            "transcript": transcript,
            "top_3_predictions": top_3,
            "acoustic_features": {
                "pitch": round(ac_raw['pitch_mean'], 2),
                "speech_rate": round(wpm, 2),
                "stress": round(stress_score, 2)
            },
            "sentiment_distribution": {
                k: round(v, 3) for k, v in zip(TARGET_EMOTIONS, text_features)
            },
            "inference_time_ms": round(inference_time * 1000, 2)
        }
