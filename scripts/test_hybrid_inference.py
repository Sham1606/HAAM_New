
import os
import sys
import argparse
import logging
import time
import numpy as np
import torch
import joblib
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.hybrid_fusion import HybridFusionNetwork

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = r"D:\haam_framework\models\hybrid_fusion_model.pth"
SCALER_PATH = r"D:\haam_framework\models\hybrid_scaler.pkl"
ENCODER_PATH = r"D:\haam_framework\models\hybrid_encoder.pkl"

TARGET_EMOTIONS = ['neutral', 'anger', 'disgust', 'fear', 'sadness']

class HybridInference:
    def __init__(self):
        self.device = "cpu" # Force CPU for now to ensure consistency, can use cuda if needed
        logger.info("Loading models...")
        
        # 1. Load PyTorch Model
        self.model = HybridFusionNetwork(n_acoustic=3, n_text=5, n_classes=5)
        # Load state dict
        if os.path.exists(MODEL_PATH):
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(self.device)))
            self.model.to(self.device)
            self.model.eval()
            logger.info("✅ Hybrid Fusion Model loaded")
        else:
            logger.error(f"❌ Model not found at {MODEL_PATH}")
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
        logger.info("System Ready.")

    def extract_acoustic_features(self, y, sr):
        """
        Extracts: pitch_mean, speech_rate_wpm, agent_stress_score
        """
        # 1. Pitch
        # training used piptrack
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=75, fmax=600)
        threshold = np.median(magnitudes)
        pitch_values = pitches[magnitudes > threshold]
        pitch_values = pitch_values[pitch_values > 0]
        pitch_mean = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0

        # 2. Speech Rate
        # training used: word_count / duration 
        # But for raw audio without transcript knowledge at feature time?
        # Inference prompt says: "librosa.beat.tempo() -> multiply by 0.5 for WPM approximation"
        # Wait, the training script used: speech_rate_wps = word_count / duration
        # If I change extraction method, inputs will be skewed.
        # Ideally, I should transcribe FIRST, get word count, then calc rate.
        # Let's do that in the main predict method.
        # But if I MUST use signal processing:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        # tempo is BPM. Speech rate WPM approx = BPM * 0.5? Rough proxy.
        # To match training data (which used real word count), getting word count from Whisper is SAFER.
        # I will use Whisper transcript for WPM to be consistent with training.
        speech_rate_proxy = float(tempo[0] * 0.5) if len(tempo) > 0 else 0.0

        # 3. Stress
        # training: pitch > 250 and rate > 150 -> +0.3 etc.
        # The prompt asks for: "stress = np.std(rms) / (np.mean(rms) + 1e-6)"
        # The training script used a HEURISTIC based on pitch/rate/emotion.
        # If I use RMS-based stress now, the model might be confused.
        # BUT the plan says "Use training features exact match".
        # Training script:
        # agent_stress_score = 0.2
        # if acoustic_features['pitch_mean'] > 250 and speech_rate_wps * 60 > 150: agent_stress_score += 0.3
        # if dominant_emotion in ['anger', 'fear', 'sadness']: agent_stress_score += 0.2
        
        # This relies on the PREDICTION (dominant_emotion). Circular dependency?
        # No, 'dominant_emotion' in training data came from the text/audio pipeline output.
        # For inference, we can calculate a "Base Stress" from Pitch/Rate, but we can't use the final emotion yet.
        # I will implement the heuristics part that DOESN'T depend on emotion (Pitch/Rate) to be safe, 
        # OR use the Prompt's suggested RMS method if the user explicitly requested "Match training exactly" which contradicts the Prompt's "Use RMS".
        # User "Match training exactly" overrides Prompt's "Use RMS".
        # So I will calculate stress based on Pitch & Rate.
        
        rms = librosa.feature.rms(y=y)
        stress_score = 0.2 # Base
        # We need rate first. I'll return intermediate vals.
        
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
        
        if not transcript:
            transcript = "." # Safe fallback

        # 2. Text Features (Sentiment)
        text_features = [0.0] * 5
        try:
            scores = self.sentiment_pipe(transcript[:512])[0] # List of dicts
            # Map to 5 target emotions
            # DistilRoBERTa labels: joy, neutral, sadness, anger, fear, disgust, surprise
            score_map = {item['label']: item['score'] for item in scores}
            
            # targets: neutral, anger, disgust, fear, sadness
            # mapping:
            # neutral <- neutral + surprise + joy (since joy is dropped target, maybe map to neutral or ignore?)
            # User said: "Handle 'joy' and 'surprise' by mapping to 'neutral'"
            
            val_neutral = score_map.get('neutral', 0.0) + score_map.get('surprise', 0.0) + score_map.get('joy', 0.0)
            val_anger = score_map.get('anger', 0.0)
            val_disgust = score_map.get('disgust', 0.0)
            val_fear = score_map.get('fear', 0.0)
            val_sadness = score_map.get('sadness', 0.0)
            
            raw_vec = [val_neutral, val_anger, val_disgust, val_fear, val_sadness]
            total = sum(raw_vec)
            if total > 0:
                text_features = [v/total for v in raw_vec]
            else:
                text_features = [0.2] * 5 # Fallback
                
        except Exception as e:
            logger.warning(f"Sentiment extraction failed: {e}")
            text_features = [0.2] * 5

        # 3. Acoustic Features & Stress Heuristic
        ac_raw = self.extract_acoustic_features(y, sr)
        
        # Calculate WPM from transcript
        word_count = len(transcript.split())
        wpm = (word_count / duration) * 60 if duration > 0 else 0.0
        
        # Stress Heuristic (Match Training)
        stress_score = 0.2
        if ac_raw['pitch_mean'] > 250 and wpm > 150:
            stress_score += 0.3
        # Note: We can't add the "dominant_emotion" component of stress yet as we haven't predicted it.
        # We'll use this partial stress score.
        
        acoustic_vec = [ac_raw['pitch_mean'], wpm, stress_score]
        
        # 4. Prepare Tensor
        # Scale Acoustic
        acoustic_np = np.array([acoustic_vec])
        acoustic_scaled = self.scaler.transform(acoustic_np)
        acoustic_tensor = torch.FloatTensor(acoustic_scaled).to(self.device)
        
        text_tensor = torch.FloatTensor([text_features]).to(self.device)
        
        # 5. Predict
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

def main():
    parser = argparse.ArgumentParser(description="HAAM Hybrid Inference")
    parser.add_argument("audio_path", help="Path to audio file")
    args = parser.parse_args()
    
    try:
        engine = HybridInference()
        print("\n" + "="*80)
        print("HYBRID EMOTION PREDICTION")
        print("="*80)
        print(f"Audio: {args.audio_path}")
        
        result = engine.predict(args.audio_path)
        
        print(f"\nTranscript: \"{result['transcript']}\"")
        print(f"\nPredicted Emotion: {result['predicted_emotion'].upper()}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Inference Time: {result['inference_time_ms']} ms")
        
        print("\nTop 3 Predictions:")
        for emo, prob in result['top_3_predictions']:
            print(f"  {emo}: {prob:.1%}")
            
        print("\nAcoustic Features:")
        print(f"  pitch: {result['acoustic_features']['pitch']} Hz")
        print(f"  speech_rate: {result['acoustic_features']['speech_rate']} WPM")
        print(f"  stress: {result['acoustic_features']['stress']}")
        
        print("\nSentiment Distribution:")
        for k, v in result['sentiment_distribution'].items():
            print(f"  {k}: {v:.3f}")
            
        print("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
