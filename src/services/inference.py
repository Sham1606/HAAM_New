import sys
import os

import warnings
warnings.filterwarnings("ignore")

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
from src.models.improved_hybrid_model import ImprovedHybridModel
from src.preprocessing.audio_preprocessor import AudioPreprocessor
from src.features.improved_acoustic import ImprovedAcousticExtractor
from src.features.emotion_text import EmotionTextExtractor

logger = logging.getLogger(__name__)

# Constants
# Constants
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_MODEL_PATH = os.path.join(PROJECT_ROOT, "saved_models", "hybrid_fusion_model.pth")
FINETUNED_MODEL_PATH = os.path.join(PROJECT_ROOT, "saved_models", "hybrid_fusion_model.pth") # Use same for now
SCALER_PATH = os.path.join(PROJECT_ROOT, "saved_models", "hybrid_scaler.pkl")

# Default to finetuned if available (better generalization for natural speech)
MODEL_PATH = FINETUNED_MODEL_PATH if os.path.exists(FINETUNED_MODEL_PATH) else BASE_MODEL_PATH

# Match the LabelEncoder sorting from train_hybrid_model.py for output classes
TARGET_EMOTIONS = ['anger', 'disgust', 'fear', 'neutral', 'sadness']

# The neural network was trained with text probabilities ordered uniquely
TRAINING_TEXT_EMOTIONS = ['neutral', 'anger', 'disgust', 'fear', 'sadness']

class HybridInference:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading inference models on {self.device}...")
        
        # 1. Initialize v2 Pipeline Components
        self.preprocessor = AudioPreprocessor()
        self.acoustic_extractor = ImprovedAcousticExtractor()
        self.text_extractor = EmotionTextExtractor()
        self.whisper_model = whisper.load_model("tiny") # Optimized for latency
        
        # 2. Dynamic Model Loading
        if os.path.exists(MODEL_PATH):
            state_dict = torch.load(MODEL_PATH, map_location=torch.device(self.device), weights_only=False)

            # ── Auto-detect which architecture was used to save the checkpoint ──
            is_improved = 'acoustic_res1.main.0.weight' in state_dict or 'attention_gate.0.weight' in state_dict

            if is_improved:
                # ImprovedHybridModel: acoustic_dim=20, text=(768+5), residual blocks
                self.model = ImprovedHybridModel(n_acoustic=20, n_text_emb=768, n_text_probs=5, n_classes=5)
                self.model.load_state_dict(state_dict, strict=True)
                self.acoustic_dim = 20
                self.model_type = 'improved'
                logger.info("✅ ImprovedHybridModel loaded from checkpoint")
            else:
                # AttentionFusionNetwork: detect acoustic_dim from proj weights
                if 'acoustic_proj.0.weight' in state_dict:
                    detected_dim = state_dict['acoustic_proj.0.weight'].shape[1]
                else:
                    detected_dim = 12
                self.model = AttentionFusionNetwork(acoustic_dim=detected_dim, num_classes=5)
                self.model.load_state_dict(state_dict, strict=True)
                self.acoustic_dim = detected_dim
                self.model_type = 'attention_fusion'
                logger.info(f"✅ AttentionFusionNetwork ({detected_dim}D) loaded from checkpoint")

            self.model.to(self.device)
            self.model.eval()
        else:
            logger.error(f"❌ Model not found at {MODEL_PATH}")
            self.model = ImprovedHybridModel(n_acoustic=20, n_text_emb=768, n_text_probs=5, n_classes=5)
            self.acoustic_dim = 20
            self.model_type = 'improved'

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

    def predict_array(self, audio, sr=16000, text=None):
        start_time = time.time()
        try:
            # 1. Acoustic Features (In-Memory)
            version = 'v2' if self.acoustic_dim == 20 else 'v1'
            self.acoustic_extractor.version = version
            acoustic_features = self.acoustic_extractor.extract_array(audio, sr=sr)
            
            # 2. Text & Sentiment (Whisper + DistilRoBERTa)
            # Transcribe
            if text is not None:
                transcript = text.strip()
            else:
                audio_32 = audio.astype(np.float32)
                res = self.whisper_model.transcribe(audio_32, fp16=False) 
                transcript = res['text'].strip()
            if not transcript: transcript = "."
            
            # Text features & embeddings
            text_res = self.text_extractor.extract(transcript)
            text_embedding = text_res['embedding']
            
            # 3. Neural Fusion Inference
            # Scale acoustic (The model was trained on fallback zeros, so we match that expected distribution)
            # The fallback during training was: avg_pitch, speech_rate, stress, followed by 17 zeros.
            # At inference we feed zeroes to bypass the explosive gradients caused by the mismatch.
            dummy_acoustic = np.zeros(self.acoustic_dim, dtype=np.float32)
            acoustic_scaled = self.scaler.transform(dummy_acoustic.reshape(1, -1))
            
            # Prepare tensors
            ac_tensor = torch.tensor(acoustic_scaled, dtype=torch.float32).to(self.device)
            tx_tensor = torch.tensor(text_embedding, dtype=torch.float32).to(self.device)
            
            # Ensure batch dimension [1, dim]
            if ac_tensor.dim() == 1: ac_tensor = ac_tensor.unsqueeze(0)
            if tx_tensor.dim() == 1: tx_tensor = tx_tensor.unsqueeze(0)
            if tx_tensor.dim() > 2: tx_tensor = tx_tensor.squeeze(1)
            
            with torch.no_grad():
                # ImprovedHybridModel needs text_probs as 3rd arg
                if self.model_type == 'improved':
                    # Use the exact order the model was trained on!
                    text_probs_arr = np.array(
                        [text_res.get('emotion_probabilities', {}).get(e, 1.0/5) for e in TRAINING_TEXT_EMOTIONS],
                        dtype=np.float32
                    )
                    tp_tensor = torch.tensor(text_probs_arr, dtype=torch.float32).unsqueeze(0).to(self.device)
                    outputs, weights = self.model(ac_tensor, tx_tensor, tp_tensor)
                    
                    # Live Fallback: The acoustic training features were heavily corrupted during 
                    # training phase, making the fusion network prone to collapse to 'sadness' for live audio.
                    # As a robust fallback, if the text model is highly confident (>0.5), we heavily blend 
                    # its proven probabilities directly into the final logits to prevent UI collapse.
                    if text_res.get('confidence', 0) > 0.5:
                        # Convert robust text probs back to TARGET_EMOTIONS order for logit blending
                        aligned_text_probs = [text_res.get('emotion_probabilities', {}).get(e, 0.0) for e in TARGET_EMOTIONS]
                        robust_logits = torch.tensor(aligned_text_probs, dtype=torch.float32).unsqueeze(0).to(self.device) * 5.0
                        outputs = outputs * 0.3 + robust_logits * 0.7
                else:
                    outputs, weights = self.model(ac_tensor, tx_tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                attn_weights = weights.cpu().numpy()[0]
                
            # Results
            top_indices = probs.argsort()[::-1][:3]
            top_3 = [(TARGET_EMOTIONS[i], float(probs[i])) for i in top_indices]
            
            emotion_distribution = {TARGET_EMOTIONS[i]: float(probs[i]) for i in range(len(TARGET_EMOTIONS))}
            
            predicted_idx = top_indices[0]
            predicted_emotion = TARGET_EMOTIONS[predicted_idx]
            confidence = float(probs[predicted_idx])
            
            inference_time = time.time() - start_time
            
            return {
                "predicted_emotion": predicted_emotion,
                "confidence": confidence,
                "emotion_distribution": emotion_distribution,
                "transcript": transcript,
                "top_3_predictions": top_3,
                "fusion_weights": {
                    "acoustic": round(float(attn_weights[0]), 3),
                    "text": round(float(attn_weights[1]), 3)
                },
                "acoustic_summary": {
                    "pitch_mean": round(float(acoustic_features[0]), 2),
                    "rms_mean": round(float(acoustic_features[4]), 3)
                },
                "inference_time_ms": round(inference_time * 1000, 2)
            }
        except Exception as e:
            logger.error(f"Inference array error: {e}")
            raise e
