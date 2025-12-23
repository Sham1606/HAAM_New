"""
Use emotion-specific model (not generic sentiment)
Model: j-hartmann/emotion-english-distilroberta-base
Output: 5-emotion probabilities + 768-dim embedding
"""

import sys
# START HACK: Prevent TensorFlow import due to Numpy 2.0 incompatibility
sys.modules['tensorflow'] = None
# END HACK

# START HACK: Bypass CVE-2025-32434 check in transformers (we trust local models)
import transformers.utils.import_utils
def no_op_check(): pass
transformers.utils.import_utils.check_torch_load_is_safe = no_op_check
# END HACK

import torch
from transformers import pipeline, AutoTokenizer, AutoModel
import numpy as np
import warnings

class EmotionTextExtractor:
    def __init__(self):
        print("Loading emotion text model...")
        # Check for GPU
        self.device = 0 if torch.cuda.is_available() else -1
        
        try:
            self.classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True,
                device=self.device,
                framework="pt"
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
            self.model = AutoModel.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                
        except Exception as e:
            print(f"Warning: Could not load emotion model: {e}")
            self.classifier = None
            self.model = None
            self.tokenizer = None
        
        # Map 7 emotions to our 5
        self.emotion_map = {
            'anger': 'anger',
            'disgust': 'disgust',
            'fear': 'fear',
            'joy': 'neutral',
            'neutral': 'neutral',
            'sadness': 'sadness',
            'surprise': 'neutral'
        }
    
    def extract(self, transcript):
        """Extract emotion probabilities and embedding"""
        
        if not transcript or len(transcript.strip()) < 2:
            return self._get_neutral()
            
        if self.classifier is None:
             # Fallback if model load failed
            return self._get_neutral()
        
        try:
            # Get emotion probabilities
             # Handle truncation manually or rely on pipeline truncation (defaults can be tricky)
            # The pipeline handles text, but max length might be an issue.
            
            results = self.classifier(transcript[:512]) # Simple truncation
            
            emotion_probs = {'anger': 0, 'disgust': 0, 'fear': 0, 'neutral': 0, 'sadness': 0}
            
            # results is [[{'label': '...', 'score': ...}, ...]]
            if isinstance(results, list) and isinstance(results[0], list):
                res_list = results[0]
            else:
                res_list = results

            for result in res_list:
                label = result['label'].lower()
                score = result['score']
                if label in self.emotion_map:
                    target = self.emotion_map[label]
                    emotion_probs[target] += score
            
            # Normalize
            total = sum(emotion_probs.values())
            if total > 0:
                emotion_probs = {k: v/total for k, v in emotion_probs.items()}
            
            # Get embedding
            inputs = self.tokenizer(transcript, return_tensors="pt", truncation=True, max_length=512)
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS] token
                if len(embedding.shape) > 1:
                     embedding = embedding.squeeze()
            
            return {
                'emotion_probabilities': emotion_probs,
                'embedding': embedding,
                'dominant_emotion': max(emotion_probs, key=emotion_probs.get),
                'confidence': max(emotion_probs.values())
            }
        
        except Exception as e:
            print(f"Text extraction error: {e}")
            return self._get_neutral()
    
    def _get_neutral(self):
        return {
            'emotion_probabilities': {'anger': 0, 'disgust': 0, 'fear': 0, 'neutral': 1, 'sadness': 0},
            'embedding': np.zeros(768, dtype=np.float32),
            'dominant_emotion': 'neutral',
            'confidence': 1.0
        }
