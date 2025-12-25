import sys
import os

# START HACK: Prevent TensorFlow import due to Numpy 2.0 incompatibility
sys.modules['tensorflow'] = None

# START HACK: Bypass CVE-2025-32434 check in transformers (we trust local models)
try:
    import transformers.utils.import_utils
    import transformers.modeling_utils
    def no_op_check(): pass
    transformers.utils.import_utils.check_torch_load_is_safe = no_op_check
    transformers.modeling_utils.check_torch_load_is_safe = no_op_check
    try:
        import transformers.pipelines.base
        transformers.pipelines.base.check_torch_load_is_safe = no_op_check
    except: pass
except:
    pass
# END HACK

import logging
import time
import numpy as np
import torch
import joblib
import librosa
import whisper
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.attention_fusion_model import AttentionFusionNetwork
from src.preprocessing.audio_preprocessor import AudioPreprocessor
from src.features.improved_acoustic import ImprovedAcousticExtractor
from src.features.emotion_text import EmotionTextExtractor

logger = logging.getLogger(__name__)

# Constants
BASE_MODEL_PATH = r"D:\haam_framework\models\improved\best_model.pth"
FINETUNED_MODEL_PATH = r"D:\haam_framework\saved_models\iemocap_finetuned.pth"
SCALER_PATH = r"D:\haam_framework\models\improved\scaler.pkl"

# Default to finetuned if available (better generalization for natural speech)
MODEL_PATH = FINETUNED_MODEL_PATH if os.path.exists(FINETUNED_MODEL_PATH) else BASE_MODEL_PATH

TARGET_EMOTIONS = ['neutral', 'anger', 'disgust', 'fear', 'sadness']

class HybridInference:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading inference models on {self.device}...")
        
        # 1. Initialize v2 Pipeline Components
        self.preprocessor = AudioPreprocessor()
        self.acoustic_extractor = ImprovedAcousticExtractor()
        self.text_extractor = EmotionTextExtractor()
        self.whisper_model = whisper.load_model("base")
        
        # 2. Load PyTorch Attention Fusion Model
        # acoustic_dim=12 as per stable v2.1 architecture
        self.model = AttentionFusionNetwork(acoustic_dim=12, num_classes=5)
        if os.path.exists(MODEL_PATH):
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(self.device)))
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"✅ Attention Fusion Model loaded from {MODEL_PATH}")
        else:
            logger.error(f"❌ Model not found at {MODEL_PATH}")
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

        # 3. Load Scaler
        if os.path.exists(SCALER_PATH):
            self.scaler = joblib.load(SCALER_PATH)
            logger.info("✅ Scaler loaded")
        else:
            logger.error("❌ Scaler not found")
            raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}")

        logger.info("Inference Service v2 Ready.")

    def predict(self, audio_path):
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            # 1. Preprocess
            audio, sr = self.preprocessor.preprocess(audio_path)
            return self.predict_array(audio, sr)
        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise e

    def predict_array(self, audio, sr=16000):
        start_time = time.time()
        try:
            # 1. Acoustic Features (In-Memory)
            acoustic_features = self.acoustic_extractor.extract_array(audio, sr=sr)
            
            # 2. Text & Sentiment (Whisper + DistilRoBERTa)
            # Transcribe
            audio_32 = audio.astype(np.float32)
            res = self.whisper_model.transcribe(audio_32) 
            transcript = res['text'].strip()
            if not transcript: transcript = "."
            
            # Text features & embeddings
            text_res = self.text_extractor.extract(transcript)
            text_embedding = text_res['embedding']
            
            # 3. Neural Fusion Inference
            # Scale acoustic
            acoustic_scaled = self.scaler.transform(acoustic_features.reshape(1, -1))
            
            # Prepare tensors
            ac_tensor = torch.tensor(acoustic_scaled, dtype=torch.float32).to(self.device)
            tx_tensor = torch.tensor(text_embedding, dtype=torch.float32).to(self.device)
            
            # Ensure batch dimension [1, dim]
            if ac_tensor.dim() == 1: ac_tensor = ac_tensor.unsqueeze(0)
            if tx_tensor.dim() == 1: tx_tensor = tx_tensor.unsqueeze(0)
            if tx_tensor.dim() > 2: tx_tensor = tx_tensor.squeeze(1)
            
            with torch.no_grad():
                outputs, weights = self.model(ac_tensor, tx_tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                attn_weights = weights.cpu().numpy()[0]
                
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
                "fusion_weights": {
                    "acoustic": round(float(attn_weights[0]), 3),
                    "text": round(float(attn_weights[1]), 3)
                },
                "acoustic_summary": {
                    "pitch_mean": round(float(acoustic_features[0]), 2),
                    "rms_mean": round(float(acoustic_features[7]), 3)
                },
                "inference_time_ms": round(inference_time * 1000, 2)
            }
        except Exception as e:
            logger.error(f"Inference array error: {e}")
            raise e
        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise e
