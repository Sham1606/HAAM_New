
import torch
import numpy as np
import os
import sys
import logging
# import whisper # Moved to lazy load or try/except
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.models.improved_hybrid_model import ImprovedHybridModel
from src.features.improved_acoustic import ImprovedAcousticExtractor
from src.features.emotion_text import EmotionTextExtractor

logger = logging.getLogger(__name__)

class LivePredictor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Components
        logger.info("Loading Acoustic Extractor...")
        self.acoustic_extractor = ImprovedAcousticExtractor(sr=16000)
        
        logger.info("Loading Text Emotion Extractor...")
        self.text_extractor = EmotionTextExtractor()
        
        logger.info("Loading Whisper (tiny)...")
        try:
            import whisper
            self.whisper_model = whisper.load_model("tiny", device=self.device)
            self.has_whisper = True
        except ImportError:
            logger.error("Whisper not installed. Falling back to dummy transcription.")
            self.has_whisper = False
        except Exception as e:
            logger.error(f"Error loading Whisper: {e}")
            self.has_whisper = False
        
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
        
        # Load weights
        # Priority: Check for hybrid_fusion_model.pth, else sprint_model_v5 (if compatible architecture), else init
        model_path = os.path.join(PROJECT_ROOT, "saved_models", "hybrid_fusion_model.pth")
        
        if os.path.exists(model_path):
             try:
                 state = torch.load(model_path, map_location=self.device)
                 model.load_state_dict(state)
                 logger.info("✅ Loaded hybrid_fusion_model.pth")
             except Exception as e:
                 logger.error(f"❌ Failed to load weights from {model_path}: {e}")
        else:
             logger.warning(f"⚠️ Weights not found at {model_path}. Using initialized model for demo.")
        
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
        transcript = "..."
        if self.has_whisper:
            try:
                # Whisper expects float32
                result = self.whisper_model.transcribe(audio_data.astype(np.float32), fp16=False)
                transcript = result['text'].strip()
            except Exception as e:
                logger.error(f"Whisper failed: {e}")

        if not transcript:
            transcript = "..."

        # 3. Text Features
        text_result = self.text_extractor.extract(transcript)
        embedding = text_result['embedding']
        
        # [neutral, anger, disgust, fear, sadness]
        # Map from emotion_probabilities dict
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
            
            output = self.model(t_ac, t_emb, t_probs)
            # Model returns (logits, attn_weights) tuple
            if isinstance(output, tuple):
                output_logits, attn_weights = output
            else:
                output_logits = output
                attn_weights = None

            softmax_probs = torch.nn.functional.softmax(output_logits, dim=1).cpu().numpy()[0]
            
            pred_idx = np.argmax(softmax_probs)
            confidence = float(softmax_probs[pred_idx])
            
            idx_to_str = {0: "neutral", 1: "anger", 2: "disgust", 3: "fear", 4: "sadness"}
            predicted_emotion = idx_to_str.get(pred_idx, "unknown")

            # Use real attention weights if available
            if attn_weights is not None:
                aw = attn_weights.cpu().numpy()[0]
                attn_audio = round(float(aw[0]), 3)
                attn_text  = round(float(aw[1]), 3)
            else:
                attn_audio = 0.5
                attn_text  = 0.5

        return {
            "emotion": predicted_emotion,
            "confidence": round(confidence, 2),
            "transcript": transcript,
            "attention": {
                "audio": attn_audio,
                "text":  attn_text,
            }
        }
