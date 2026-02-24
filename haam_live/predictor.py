
import torch
import numpy as np
import os
import sys
import whisper # Official OpenAI Whisper
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.models.improved_hybrid_model import ImprovedHybridModel
from src.features.improved_acoustic import ImprovedAcousticExtractor
from src.features.emotion_text import EmotionTextExtractor
from .utils import get_logger, EMOTION_MAP

logger = get_logger("Predictor")

class HaamPredictor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Components
        logger.info("Loading Acoustic Extractor...")
        self.acoustic_extractor = ImprovedAcousticExtractor(sr=16000)
        
        logger.info("Loading Text Emotion Extractor...")
        self.text_extractor = EmotionTextExtractor()
        
        logger.info("Loading Whisper (tiny)...")
        # Load tiny model for speed
        self.whisper_model = whisper.load_model("tiny", device=self.device)
        
        # Model
        logger.info("Loading Hybrid Fusion Model...")
        self.model = self._load_model()
        
    def _load_model(self):
        # Config matches ImprovedHybridModel defaults
        model = ImprovedHybridModel(
            n_acoustic=20, # v2 extractor has 20 features
            n_text_emb=768,
            n_text_probs=5,
            n_classes=5
        ).to(self.device)
        model.eval()
        
        # Load weights if available, else warn
        model_path = os.path.join(PROJECT_ROOT, "saved_models", "hybrid_fusion_model.pth")
        if os.path.exists(model_path):
             try:
                 model.load_state_dict(torch.load(model_path, map_location=self.device))
                 logger.info("✅ Loaded hybrid_fusion_model.pth")
             except Exception as e:
                 logger.error(f"❌ Failed to load weights: {e}")
        else:
             logger.warning("⚠️ Weights not found. Using initialized model for pipeline demo.")
        
        return model

    def predict(self, audio_data: np.ndarray):
        """
        Run full inference pipeline on raw audio.
        """
        # 1. Acoustic Features
        try:
            features = self.acoustic_extractor.extract_array(audio_data)
        except Exception as e:
            logger.error(f"Acoustic extraction failed: {e}")
            features = np.zeros(20, dtype=np.float32)

        # 2. Transcription
        try:
            # Whisper expects float32 tensor
            audio_tensor = torch.from_numpy(audio_data).float()
            
            # Pad/Trim to 30s as whisper expects? No, verify usage.
            # whisper.transcribe() handles raw audio if passed as numpy array
            result = self.whisper_model.transcribe(audio_data.astype(np.float32))
            transcript = result['text'].strip()
        except Exception as e:
            logger.error(f"Whisper failed: {e}")
            transcript = ""

        if not transcript:
            transcript = "..."

        # 3. Text Features
        text_result = self.text_extractor.extract(transcript)
        embedding = text_result['embedding']
        
        # [neutral, anger, disgust, fear, sadness]
        ordered_probs_5 = [
            text_result['emotion_probabilities'].get('neutral', 0),
            text_result['emotion_probabilities'].get('anger', 0),
            text_result['emotion_probabilities'].get('disgust', 0),
            text_result['emotion_probabilities'].get('fear', 0),
            text_result['emotion_probabilities'].get('sadness', 0)
        ]

        # 4. Model Inference
        with torch.no_grad():
            t_ac = torch.tensor(features).unsqueeze(0).to(self.device).float()
            t_emb = torch.tensor(embedding).unsqueeze(0).to(self.device).float()
            t_probs = torch.tensor(ordered_probs_5).unsqueeze(0).to(self.device).float()
            
            output_logits, attn_weights = self.model(t_ac, t_emb, t_probs)
            
            # Process Attention for XAI
            attn_val = attn_weights.cpu().numpy()[0]
            w_audio = float(attn_val[0])
            w_text = float(attn_val[1])
            
            softmax_probs = torch.nn.functional.softmax(output_logits, dim=1).cpu().numpy()[0]
            
            pred_idx = np.argmax(softmax_probs)
            confidence = float(softmax_probs[pred_idx])
            
            idx_to_str = {0: "neutral", 1: "anger", 2: "disgust", 3: "fear", 4: "sadness"}
            predicted_emotion = idx_to_str.get(pred_idx, "unknown")
            
            idx_to_str = {0: "neutral", 1: "anger", 2: "disgust", 3: "fear", 4: "sadness"}
            predicted_emotion = idx_to_str.get(pred_idx, "unknown")

        return {
            "emotion": predicted_emotion,
            "confidence": round(confidence, 2),
            "transcript": transcript,
            "attention": {
                "audio": round(w_audio, 2), 
                "text": round(w_text, 2)
            }
        }
