
import os
import json
import argparse
import logging
import time
import numpy as np
import librosa
import torch
import whisper
from transformers import pipeline
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

class SprintPipeline:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Load models on initialization to avoid reloading for every call
        logger.info("Loading Whisper model...")
        self.whisper_model = whisper.load_model(WHISPER_MODEL_SIZE, device=self.device)
        
        logger.info("Loading Emotion model...")
        try:
            self.emotion_classifier = pipeline(
                "text-classification", 
                model=EMOTION_MODEL, 
                return_all_scores=True, 
                device=0 if self.device == "cuda" else -1
            )
        except Exception as e:
            logger.error(f"Failed to load emotion model: {e}")
            raise

        logger.info("Loading Sentiment model...")
        try:
            self.sentiment_classifier = pipeline(
                "sentiment-analysis", 
                model=SENTIMENT_MODEL, 
                tokenizer=SENTIMENT_MODEL,
                return_all_scores=True,
                device=0 if self.device == "cuda" else -1
            )
        except Exception as e:
            logger.error(f"Failed to load sentiment model: {e}")
            raise

        # Load trained acoustic emotion model
        logger.info("Loading Acoustic Emotion Model...")
        try:
            model_path = r"D:\haam_framework\models\acoustic_emotion_model.pkl"
            scaler_path = r"D:\haam_framework\models\feature_scaler.pkl"
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.acoustic_model = joblib.load(model_path)
                self.acoustic_scaler = joblib.load(scaler_path)
                logger.info("Acoustic model loaded successfully")
            else:
                logger.warning(f"Acoustic model not found at {model_path}")
                self.acoustic_model = None
                self.acoustic_scaler = None
        except Exception as e:
            logger.error(f"Failed to load acoustic model: {e}")
            self.acoustic_model = None
            self.acoustic_scaler = None

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
    
    def get_sentiment_score(self, scores):
        """
        Calculate a single sentiment score from probabilities.
        Mapping: Negative (-1), Neutral (0), Positive (1)
        """
        # scores is a list of dicts: [{'label': 'positive', 'score': 0.9}, ...]
        score_map = {s['label']: s['score'] for s in scores}
        
        # Cardiff model labels: positive, negative, neutral
        pos = score_map.get('positive', 0.0)
        neg = score_map.get('negative', 0.0)
        # neutral = score_map.get('neutral', 0.0)
        
        # Composite score
        return pos - neg

    def predict_emotion_from_acoustics(self, y, sr):
        """
        Predict emotion using trained ML model.
        """
        if self.acoustic_model is None:
            return 'neutral', 0.5
        
        try:
            # Extract features exactly as trained
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=75, fmax=600)
            threshold = np.median(magnitudes)
            pitch_values = pitches[magnitudes > threshold]
            pitch_values = pitch_values[pitch_values > 0]
            
            pitch_mean = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0 # Changed to 0.0 to match training extraction logic
            
            energy = librosa.feature.rms(y=y)
            energy_mean = float(np.mean(energy))
            
            zcr = librosa.feature.zero_crossing_rate(y)
            zcr_mean = float(np.mean(zcr))
            
            sc = librosa.feature.spectral_centroid(y=y, sr=sr)
            sc_mean = float(np.mean(sc))
            
            # Create feature vector (pitch, energy, zcr, spectral_centroid)
            # Note: Training script used ['pitch', 'energy', 'zcr', 'spectral_centroid']
            # Reorder if necessary. train_acoustic_model.py used: feature_cols = ['pitch', 'energy', 'zcr', 'spectral_centroid']
            features = np.array([[pitch_mean, energy_mean, zcr_mean, sc_mean]])
            
            # Scale features
            features_scaled = self.acoustic_scaler.transform(features)
            
            # Predict
            emotion = self.acoustic_model.predict(features_scaled)[0]
            probabilities = self.acoustic_model.predict_proba(features_scaled)
            confidence = float(np.max(probabilities))
            
            return emotion, confidence
            
        except Exception as e:
            logger.warning(f"Acoustic prediction failed: {e}")
            return 'neutral', 0.5

    def analyze_text_segment(self, text, y_segment=None, sr=16000):
        """
        Hybrid emotion analysis: ML acoustic model + Text model fusion.
        """
        if not text.strip():
            return "neutral", 0.0, 0.5
        
        # TEXT EMOTION
        text_emotion = "neutral"
        text_confidence = 0.5
        
        try:
            emotions = self.emotion_classifier(text[:512])[0]
            max_emotion = max(emotions, key=lambda x: x['score'])
            text_emotion = max_emotion['label']
            text_confidence = max_emotion['score']
        except Exception as e:
            logger.warning(f"Text emotion failed: {e}")
        
        # ACOUSTIC EMOTION (ML Model)
        acoustic_emotion = "neutral"
        acoustic_confidence = 0.5
        
        if y_segment is not None and len(y_segment) > 100:
            acoustic_emotion, acoustic_confidence = self.predict_emotion_from_acoustics(y_segment, sr)
        
        # FUSION
        acoustic_weight = 0.6
        text_weight = 0.4
        
        if text_confidence < 0.3:
            acoustic_weight = 0.8
            text_weight = 0.2
        
        # Map text labels
        text_emotion_mapped = {
            'joy': 'joy',
            'anger': 'anger',
            'sadness': 'sadness',
            'fear': 'fear',
            'disgust': 'disgust',
            'surprise': 'neutral',
            'neutral': 'neutral'
        }.get(text_emotion, 'neutral')
        
        # Decision logic
        if text_emotion_mapped == acoustic_emotion:
            final_emotion = acoustic_emotion
            final_confidence = max(text_confidence, acoustic_confidence)
        else:
            text_score = text_confidence * text_weight
            acoustic_score = acoustic_confidence * acoustic_weight
            
            if acoustic_score > text_score:
                final_emotion = acoustic_emotion
                final_confidence = acoustic_confidence
            else:
                final_emotion = text_emotion_mapped
                final_confidence = text_confidence
        
        # SENTIMENT
        sentiment_score = 0.0
        try:
            sentiments = self.sentiment_classifier(text[:512])[0]
            sentiment_score = self.get_sentiment_score(sentiments)
        except:
            pass
        
        return final_emotion, sentiment_score, final_confidence

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
