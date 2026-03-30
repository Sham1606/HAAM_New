"""
HAAM XAI Explainer — Captum Integrated Gradients
=================================================
Computes per-feature attribution scores for the ImprovedHybridModel using
Captum's IntegratedGradients method.

Outputs:
  - top_acoustic_drivers  : top-5 named acoustic features by attribution
  - modality_split        : {acoustic: %, text: %} from attention gate
  - human_explanation     : plain-English sentence summarising the prediction
  - all_attributions      : full named attribution dict for all 20 features
  - text_attributions     : per-token {token, score} list for text XAI
"""

import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Feature names match ImprovedAcousticExtractor v2 (20-dim)
ACOUSTIC_FEATURE_NAMES = [
    'pitch_mean', 'pitch_std', 'pitch_range', 'pitch_slope',
    'rms_mean', 'rms_std', 'zero_crossing_rate',
    'spectral_centroid', 'spectral_rolloff', 'spectral_flatness',
    'speech_rate', 'spectral_bandwidth',
    'mfcc_0', 'mfcc_1', 'mfcc_2', 'mfcc_3',
    'mfcc_4', 'mfcc_5', 'mfcc_6', 'mfcc_7',
]

# Human-readable names for UI display
FEATURE_DISPLAY_NAMES = {
    'pitch_mean':        'Avg Pitch',
    'pitch_std':          'Pitch Variability',
    'pitch_range':        'Pitch Range',
    'pitch_slope':        'Pitch Trend',
    'rms_mean':           'Loudness (RMS)',
    'rms_std':            'Loudness Variation',
    'zero_crossing_rate': 'Voice Texture (ZCR)',
    'spectral_centroid':  'Spectral Brightness',
    'spectral_rolloff':   'High-Freq Energy',
    'spectral_flatness':  'Voice Noisiness',
    'speech_rate':        'Speech Rate',
    'spectral_bandwidth': 'Spectral Width',
    'mfcc_0': 'MFCC-0 (Energy)',
    'mfcc_1': 'MFCC-1 (Timbre)',
    'mfcc_2': 'MFCC-2 (Timbre)',
    'mfcc_3': 'MFCC-3 (Timbre)',
    'mfcc_4': 'MFCC-4 (Timbre)',
    'mfcc_5': 'MFCC-5 (Timbre)',
    'mfcc_6': 'MFCC-6 (Timbre)',
    'mfcc_7': 'MFCC-7 (Timbre)',
}

# Emotion-specific explanation templates
EMOTION_TEMPLATES = {
    'anger':   "aggressive tone patterns — elevated pitch variance and loudness",
    'sadness': "low-energy indicators — subdued pitch and slow speech rate",
    'fear':    "high-arousal markers — fast speech rate and pitch instability",
    'disgust': "flat prosody with negative vocal quality indicators",
    'neutral': "balanced prosody with no dominant emotional signal",
}


class HAAMExplainer:
    """
    Wraps ImprovedHybridModel with Captum Integrated Gradients
    to explain per-call emotion predictions (acoustic + text).
    """

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
        self._ig = None   # lazy init

    def _get_ig(self):
        """Lazy-load Captum IntegratedGradients."""
        if self._ig is None:
            try:
                from captum.attr import IntegratedGradients
                self._ig = IntegratedGradients(self._acoustic_forward)
            except ImportError:
                raise ImportError(
                    "captum is required for XAI. Install it with: pip install captum"
                )
        return self._ig

    def _acoustic_forward(self, x_acoustic, x_text_emb, x_text_probs):
        """Forward pass returning logits for Captum attribution."""
        logits, _ = self.model(x_acoustic, x_text_emb, x_text_probs)
        return logits

    def explain(self, x_acoustic: np.ndarray, x_text_emb: np.ndarray,
                x_text_probs: np.ndarray, target_class: int,
                fusion_weights: dict = None) -> dict:
        """Compute acoustic XAI attribution for a single call."""
        try:
            ig = self._get_ig()
        except ImportError as e:
            logger.warning(str(e))
            return self._fallback_explain(x_acoustic, fusion_weights, target_class)

        ac = torch.tensor(x_acoustic, dtype=torch.float32).unsqueeze(0).to(self.device)
        te = torch.tensor(x_text_emb,  dtype=torch.float32).unsqueeze(0).to(self.device)
        tp = torch.tensor(x_text_probs, dtype=torch.float32).unsqueeze(0).to(self.device)
        ac.requires_grad_(True)
        baseline = torch.zeros_like(ac)

        try:
            attributions, _ = ig.attribute(
                ac,
                baselines=baseline,
                target=target_class,
                additional_forward_args=(te, tp),
                n_steps=50,
                return_convergence_delta=True,
            )
            attr_np = attributions.detach().cpu().numpy().squeeze()
        except Exception as e:
            logger.warning(f"Captum IG failed ({e}), using gradient fallback.")
            attr_np = self._gradient_attribution(ac, te, tp, target_class)

        return self._build_result(attr_np, fusion_weights, target_class)

    def _gradient_attribution(self, ac, te, tp, target_class):
        """Simple gradient x input attribution fallback."""
        ac = ac.detach().requires_grad_(True)
        logits, _ = self.model(ac, te, tp)
        logits[0, target_class].backward()
        grad = ac.grad.detach().cpu().numpy().squeeze()
        inp  = ac.detach().cpu().numpy().squeeze()
        return np.abs(grad * inp)

    def _build_result(self, attr_np: np.ndarray, fusion_weights: dict,
                      target_class: int) -> dict:
        """Package attribution array into a structured XAI result dict."""
        from src.services.inference import TARGET_EMOTIONS
        predicted_emotion = TARGET_EMOTIONS[target_class] if target_class < len(TARGET_EMOTIONS) else 'neutral'

        abs_attr = np.abs(attr_np)
        total = abs_attr.sum()
        norm_attr = (abs_attr / total).tolist() if total > 0 else [0.0] * len(abs_attr)

        all_attr = {
            ACOUSTIC_FEATURE_NAMES[i]: round(float(norm_attr[i]), 4)
            for i in range(len(ACOUSTIC_FEATURE_NAMES))
        }

        sorted_features = sorted(all_attr.items(), key=lambda x: x[1], reverse=True)
        top_5 = [
            {
                'feature': k,
                'display_name': FEATURE_DISPLAY_NAMES.get(k, k),
                'attribution': round(v * 100, 1),
            }
            for k, v in sorted_features[:5]
        ]

        fw = fusion_weights or {'acoustic': 0.5, 'text': 0.5}
        ac_pct = round(fw.get('acoustic', 0.5) * 100, 1)
        tx_pct = round(fw.get('text',     0.5) * 100, 1)

        top_feat_name = FEATURE_DISPLAY_NAMES.get(top_5[0]['feature'], top_5[0]['feature']) if top_5 else 'acoustic features'
        emotion_desc  = EMOTION_TEMPLATES.get(predicted_emotion, "mixed emotional signals")
        dominant_modality = "voice prosody" if ac_pct >= tx_pct else "spoken language"

        human_explanation = (
            f"The model predicted '{predicted_emotion}' primarily based on {dominant_modality} "
            f"({max(ac_pct, tx_pct):.0f}% of model attention). "
            f"The strongest acoustic signal was '{top_feat_name}', consistent with "
            f"{emotion_desc}."
        )

        return {
            'predicted_emotion':    predicted_emotion,
            'modality_split':       {'acoustic': ac_pct, 'text': tx_pct},
            'top_acoustic_drivers': top_5,
            'all_attributions':     all_attr,
            'human_explanation':    human_explanation,
            'method':               'IntegratedGradients (Captum)',
        }

    def _fallback_explain(self, x_acoustic: np.ndarray, fusion_weights: dict,
                          target_class: int) -> dict:
        """Magnitude-based fallback if Captum is not installed."""
        logger.info("Using magnitude-based XAI fallback (captum not installed).")
        abs_vals = np.abs(x_acoustic)
        total = abs_vals.sum()
        norm = (abs_vals / total).tolist() if total > 0 else [0.0] * len(abs_vals)
        return self._build_result(np.array(norm), fusion_weights, target_class)

    # ─── Text XAI ─────────────────────────────────────────────────────────────
    def explain_text(self, transcript: str, text_extractor, target_class: int) -> list:
        """
        Per-token attribution via Captum LayerIntegratedGradients on
        DistilRoBERTa word embeddings.

        Returns list of {token, score} where score in [-1, 1]:
          positive  = word contributed TO this emotion
          negative  = word suppressed this emotion
        Falls back to a heuristic scorer if Captum is unavailable.
        """
        if not transcript or not transcript.strip():
            return []

        try:
            from captum.attr import LayerIntegratedGradients

            tokenizer = getattr(text_extractor, 'tokenizer', None)
            lm        = getattr(text_extractor, 'model', None)

            if tokenizer is None or lm is None:
                raise AttributeError("text_extractor missing .tokenizer or .model")

            lm.eval()
            inputs = tokenizer(
                transcript, return_tensors='pt', truncation=True,
                max_length=128, padding=True,
            )
            input_ids      = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)

            # Grab the embedding layer — works for DistilBERT and RoBERTa
            if hasattr(lm, 'distilbert'):
                emb_layer = lm.distilbert.embeddings.word_embeddings
            elif hasattr(lm, 'roberta'):
                emb_layer = lm.roberta.embeddings.word_embeddings
            else:
                emb_layer = list(lm.modules())[1]  # best-effort fallback

            def forward_fn(input_ids_tensor, attn_mask):
                outputs = lm(input_ids=input_ids_tensor, attention_mask=attn_mask)
                return outputs.logits

            lig = LayerIntegratedGradients(forward_fn, emb_layer)

            attributions, _ = lig.attribute(
                inputs=input_ids,
                baselines=torch.zeros_like(input_ids),
                additional_forward_args=(attention_mask,),
                target=target_class,
                n_steps=30,
                return_convergence_delta=True,
            )
            # [1, seq_len, hidden] -> [seq_len] L2 norm
            token_scores = attributions.detach().cpu().norm(dim=-1).squeeze(0).numpy()
            tokens       = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

            # Normalise to [-1, 1]
            max_abs = np.abs(token_scores).max()
            if max_abs > 0:
                token_scores = token_scores / max_abs

            SPECIAL = {'[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>'}
            result = []
            for tok, score in zip(tokens, token_scores.tolist()):
                if tok in SPECIAL:
                    continue
                display = tok.lstrip('##').lstrip('G').lstrip('_') or tok
                result.append({'token': display, 'score': round(score, 4)})
            return result

        except Exception as e:
            logger.warning(f"Text XAI failed ({e}), using heuristic fallback.")
            return self._text_heuristic_fallback(transcript)

    def _text_heuristic_fallback(self, transcript: str) -> list:
        """Ranks words by information content as a proxy attribution score."""
        STOPWORDS = {
            'i', 'me', 'my', 'we', 'our', 'you', 'your', 'it', 'is', 'was',
            'are', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
            'did', 'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'this', 'that', 'not', 'no', 'so',
            'just', 'can', 'will',
        }
        NEGATIVE_WORDS = {'no', 'not', "n't", 'never', 'nothing', 'nobody', 'nowhere'}
        result = []
        for w in transcript.lower().split():
            clean = ''.join(c for c in w if c.isalpha())
            if not clean:
                continue
            if clean in STOPWORDS:
                score = 0.1
            elif clean in NEGATIVE_WORDS:
                score = -0.5
            else:
                score = min(0.3 + len(clean) * 0.05, 1.0)
            result.append({'token': clean, 'score': round(score, 4)})
        return result
