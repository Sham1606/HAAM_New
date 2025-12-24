
import os
import json
import argparse
import logging
import time
import numpy as np
import librosa
import torch
import whisper
# transformers.pipeline is imported lazily where needed to avoid importing TensorFlow
# (prevents NumPy C-extension errors at module import time)
from datetime import datetime
import warnings
import soundfile as sf
import joblib


# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants for models
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
# Using base model as requested
WHISPER_MODEL_SIZE = "base"

# Import HybridInference from services
try:
    from src.services.inference import HybridInference
except ImportError:
    # Handle direct script execution vs module import
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from services.inference import HybridInference

class SprintPipeline:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Load unified inference engine (v2)
        logger.info("Initializing HybridInference v2 engine...")
        self.inference_engine = HybridInference()
        
        # We share the models where possible
        self.whisper_model = self.inference_engine.whisper_model
        
        # Keep sentiment for Cardiff roberta if specifically needed for "sentiment_score"
        # but for emotion we use the unified engine.
        logger.info("SprintPipeline v2 ready.")

    def extract_acoustic_features(self, y, sr):
        """
        Extract acoustic features using librosa: Pitch, Speech Rate, MFCCs.
        Returns a dictionary of features.
        """
        try:
            # Pitch extraction (using piptrack)
            # Fmin ~ 75Hz, Fmax ~ 600Hz for human speech
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=75, fmax=600)
            
            # Select pitches with high magnitude
            threshold = np.median(magnitudes)
            pitch_values = pitches[magnitudes > threshold]
            pitch_values = pitch_values[pitch_values > 0] # non-zero
            
            if len(pitch_values) > 0:
                pitch_mean = float(np.mean(pitch_values))
                pitch_std = float(np.std(pitch_values))
            else:
                pitch_mean = 0.0
                pitch_std = 0.0

            # MFCCs (Mean of 13 coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = float(np.mean(mfccs))

            # Duration
            duration = librosa.get_duration(y=y, sr=sr)
            
            return {
                "pitch_mean": pitch_mean,
                "pitch_std": pitch_std,
                "mfcc_mean": mfcc_mean,
                "duration_seconds": duration
            }
        except Exception as e:
            logger.error(f"Error extracting acoustic features: {e}")
            return {
                "pitch_mean": 0.0,
                "pitch_std": 0.0,
                "mfcc_mean": 0.0,
                "duration_seconds": 0.0
            }

    def transcribe_audio(self, audio_path):
        """
        Transcribe audio using Whisper.
        Returns the full result dict from Whisper (text + segments).
        """
        try:
            result = self.whisper_model.transcribe(audio_path)
            return result
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            raise
    


    def analyze_text_segment(self, text, y_segment=None, sr=16000):
        """
        v2 Hybrid analysis: Uses unified AttentionFusion engine.
        """
        if y_segment is None or len(y_segment) < 100:
             # Fallback if no audio for this segment
             return "neutral", 0.0, 0.5

        try:
            # We use predict_array which handles everything (Whisper internal is skipped if we pass text)
            # Actually, predict_array currently RE-TRANSCRIBES. 
            # To be efficient, we might want a version that takes text.
            # But for simplicity and consistency, let's just use the robust predict_array.
            
            res = self.inference_engine.predict_array(y_segment, sr=sr)
            
            # Note: predict_array returns confidence, transcript, etc.
            # SprintPipeline expects: emotion, sentiment_score, confidence
            
            # We'll use a placeholder sentiment score as predict_array focuses on emotion
            sentiment_score = 0.0 
            if res['predicted_emotion'] == 'anger': sentiment_score = -0.6
            elif res['predicted_emotion'] == 'sadness': sentiment_score = -0.4
            
            return res['predicted_emotion'], sentiment_score, res['confidence']
            
        except Exception as e:
            logger.warning(f"Segment analysis failed: {e}")
            return "neutral", 0.0, 0.5

    def load_audio_robust(self, audio_path, target_sr=16000):
        """
        Load audio using soundfile to avoid ffmpeg dependency issues.
        Returns: y (np.array), sr (int)
        """
        try:
            y, sr = sf.read(audio_path)
            # Convert to mono if stereo
            if len(y.shape) > 1:
                y = y.mean(axis=1)
            
            # Resample if needed
            if sr != target_sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
                
            return y, sr
        except Exception as e:
            logger.error(f"Robust load failed: {e}")
            raise

    def process_call(self, audio_path, agent_id, call_id):
        """
        Main processing function for a single call.
        """
        logger.info(f"Processing call {call_id} for agent {agent_id}...")
        start_time = time.time()

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # 1. Load Audio & Extract Acoustic Features
        logger.info("Extracting acoustic features...")
        try:
            # Load with robust loader
            y, sr = self.load_audio_robust(audio_path, target_sr=16000) 
            acoustic_features = self.extract_acoustic_features(y, sr)
        except Exception as e:
            import traceback
            logger.error(f"Failed to load audio: {e}")
            logger.error(traceback.format_exc())
            raise

        # 2. Transcribe
        logger.info("Transcribing audio...")
        # Dictionary expected by transcribe_audio used to take path. 
        # But self.whisper_model.transcribe can take array.
        # Let's adjust transcribe_audio to take audio_path OR array.
        # Or just call model directly here?
        # Let's check transcribe_audio signature. It uses self.whisper_model.transcribe(path).
        # We can pass 'y' (float32 array) to it instead of path.
        transcription_result = self.whisper_model.transcribe(y.astype(np.float32))
        full_transcript = transcription_result['text']
        whisper_segments = transcription_result['segments']

        # 3. Analyze Segments
        logger.info("Analyzing segments...")
        processed_segments = []
        emotion_counts = {}
        total_sentiment = 0.0
        
        # Word count for speech rate
        word_count = len(full_transcript.split())
        speech_rate_wps = word_count / acoustic_features['duration_seconds'] if acoustic_features['duration_seconds'] > 0 else 0

        for seg in whisper_segments:
            text = seg['text'].strip()
            if not text:
                continue
            
            # Extract audio segment for acoustic analysis
            seg_start = seg['start']
            seg_end = seg['end']
            start_sample = int(seg_start * sr)
            end_sample = int(seg_end * sr)
            
            # Get audio segment safely
            y_seg = None
            if start_sample < len(y) and end_sample <= len(y) and end_sample > start_sample:
                y_seg = y[start_sample:end_sample]
            
            # Analyze with HYBRID approach (text + acoustics)
            emotion, sentiment, confidence = self.analyze_text_segment(text, y_seg, sr)
            
            # Track emotion distribution
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            total_sentiment += sentiment
            
            # Extract segment-level pitch for metrics
            seg_pitch = acoustic_features['pitch_mean']  # Default to overall
            if y_seg is not None and len(y_seg) > 0:
                pitches_seg, _ = librosa.piptrack(y=y_seg, sr=sr, fmin=75, fmax=600)
                pv_seg = pitches_seg[pitches_seg > 0]
                if len(pv_seg) > 0:
                    seg_pitch = float(np.mean(pv_seg))
            
            processed_segments.append({
                "start_time": round(seg_start, 2),
                "end_time": round(seg_end, 2),
                "text": text,
                "emotion": emotion,
                "emotion_confidence": round(confidence, 3),
                "sentiment_score": round(sentiment, 4),
                "pitch_mean": round(seg_pitch, 2)
            })

        # 4. Aggregate Metrics
        num_segments = len(processed_segments)
        avg_sentiment = total_sentiment / num_segments if num_segments > 0 else 0.0
        
        # Emotion distribution
        total_emotions = sum(emotion_counts.values())
        emotion_dist = {k: round(v / total_emotions, 3) for k, v in emotion_counts.items()} if total_emotions > 0 else {}
        dominant_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "neutral"

        # Logic for flags
        escalation_flag = False
        # If angry > 30% or sentiment very negative
        if emotion_dist.get('anger', 0) > 0.3 or avg_sentiment < -0.5:
            escalation_flag = True

        # Agent stress score (Placeholder heuristic)
        # Normal pitch ~ 100-200Hz male, 200-300Hz female.
        # High pitch + High speed -> Stress
        agent_stress_score = 0.2
        if acoustic_features['pitch_mean'] > 250 and speech_rate_wps * 60 > 150: # >150 wpm
            agent_stress_score += 0.3
        if dominant_emotion in ['anger', 'fear', 'sadness']:
            agent_stress_score += 0.2
            
        overall_metrics = {
            "avg_sentiment": round(avg_sentiment, 4),
            "dominant_emotion": dominant_emotion,
            "emotion_distribution": emotion_dist,
            "escalation_flag": escalation_flag,
            "agent_stress_score": round(agent_stress_score, 2),
            "speech_rate_wpm": round(speech_rate_wps * 60, 2),
            "avg_pitch": round(acoustic_features['pitch_mean'], 2)
        }

        # 5. Construct Output
        output_data = {
            "call_id": call_id,
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": round(acoustic_features['duration_seconds'], 2),
            "transcript": full_transcript.strip(),
            "segments": processed_segments,
            "overall_metrics": overall_metrics
        }
        
        elapsed = time.time() - start_time
        logger.info(f"Processing complete in {elapsed:.2f}s")
        return output_data

def main():
    parser = argparse.ArgumentParser(description="HAAM Sprint Layer - Single Call Processor")
    parser.add_argument("--audio", required=True, help="Path to audio file (wav/mp3)")
    parser.add_argument("--agent", required=True, help="Agent ID")
    parser.add_argument("--call", required=True, help="Call ID")
    parser.add_argument("--output_dir", default="results/calls", help="Directory to save JSON output")
    
    args = parser.parse_args()
    
    # Validation
    if not os.path.exists(args.audio):
        logger.error(f"Input file does not exist: {args.audio}")
        return

    # Ensure output dir exists
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"call_{args.call}.json")

    try:
        pipeline = SprintPipeline()
        result = pipeline.process_call(args.audio, args.agent, args.call)
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        
    except Exception as e:
        logger.critical(f"Pipeline failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
