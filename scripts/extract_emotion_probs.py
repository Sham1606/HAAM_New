"""
Extract emotion-specific probabilities using j-hartmann/emotion-english-distilroberta-base.
This replaces the heuristic emotion_distribution from the JSON metadata.

The model maps text -> 7 emotions (anger, disgust, fear, joy, neutral, sadness, surprise).
We select only our 5 target emotions and re-normalize to get proper 5-dim probability vectors.

Saves: data/processed/emotion_probs/{feature_id}.npy  (shape: [5])

Run ONCE before retraining:
    python scripts/extract_emotion_probs.py
"""

import os
import sys
import json
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

CREMAD_DIR   = os.path.join(PROJECT_ROOT, "results", "calls_cremad")
IEMOCAP_DIR  = os.path.join(PROJECT_ROOT, "results", "calls_iemocap")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "data", "processed", "emotion_probs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Our 5 target emotions (ordered) — must match training script
TARGET_EMOTIONS = ["neutral", "anger", "disgust", "fear", "sadness"]

# Map from model output labels -> our labels (model uses lowercase)
LABEL_MAP = {
    "neutral":  "neutral",
    "anger":    "anger",
    "disgust":  "disgust",
    "fear":     "fear",
    "sadness":  "sadness",
    "joy":      None,       # exclude
    "surprise": None,       # exclude
}

CREMAD_SENTENCES = {
    "IEO": "It's eleven o'clock.",
    "TIE": "That is exactly what happened.",
    "IOM": "I'm on my way to the meeting.",
    "IWW": "I wonder what this is about.",
    "TAI": "The airplane is almost full.",
    "MTI": "Maybe tomorrow it will be cold.",
    "IWL": "I would like a new alarm clock.",
    "ITH": "I think I have a doctor's appointment.",
    "DFA": "Don't forget a jacket.",
    "ITS": "I think I've seen this before.",
    "TSI": "The surface is slick.",
    "WSI": "We'll stop in a couple of minutes.",
}


def load_pipeline():
    """Load the emotion classification pipeline."""
    print("Loading j-hartmann/emotion-english-distilroberta-base ...")
    from transformers import pipeline
    pipe = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,       # return all label scores
        device=-1         # CPU
    )
    print("  Model loaded!")
    return pipe


def text_to_probs(pipe, text: str) -> np.ndarray:
    """Returns a 5-dim normalized probability vector for the 5 target emotions."""
    if not text or text.strip() in ("", "...", "unknown"):
        return np.array([0.2] * 5, dtype=np.float32)  # uniform prior

    results = pipe(text[:512])[0]   # list of {label, score}
    score_map = {r["label"].lower(): r["score"] for r in results}

    probs = np.array(
        [score_map.get(e, 0.0) for e in TARGET_EMOTIONS],
        dtype=np.float32
    )
    total = probs.sum()
    if total > 0:
        probs = probs / total   # re-normalize after excluding joy/surprise
    else:
        probs = np.array([0.2] * 5, dtype=np.float32)

    return probs


def process_dir(pipe, results_dir: str, prefix_strip: str = "", dataset_name: str = ""):
    """Process all JSONs in a results dir and save emotion probs."""
    files = [f for f in os.listdir(results_dir) if f.endswith(".json")]
    done = skipped = errors = 0

    for fname in tqdm(files, desc=dataset_name):
        call_id = fname.replace(".json", "")
        feature_id = call_id.replace(prefix_strip, "") if prefix_strip else call_id
        out_path = os.path.join(OUTPUT_DIR, f"{feature_id}.npy")

        if os.path.exists(out_path):
            skipped += 1
            continue

        try:
            with open(os.path.join(results_dir, fname)) as f:
                data = json.load(f)

            # Get transcript
            text = (
                data.get("transcript")
                or data.get("metadata", {}).get("transcript_ground_truth", "")
                or ""
            )

            # CREMA-D fallback: use sentence code text
            if not text and dataset_name == "CREMA-D":
                parts = call_id.split("_")
                sent_code = parts[1] if len(parts) >= 2 else ""
                text = CREMAD_SENTENCES.get(sent_code, "")

            probs = text_to_probs(pipe, text)
            np.save(out_path, probs)
            done += 1

        except Exception as e:
            errors += 1

    print(f"  {dataset_name}: Processed={done}, Skipped={skipped}, Errors={errors}")


def main():
    pipe = load_pipeline()

    # CREMA-D (call_id is the feature_id directly)
    process_dir(pipe, CREMAD_DIR, prefix_strip="", dataset_name="CREMA-D")

    # IEMOCAP (feature_id = call_id without 'iemocap_' prefix)
    if os.path.exists(IEMOCAP_DIR):
        process_dir(pipe, IEMOCAP_DIR, prefix_strip="iemocap_", dataset_name="IEMOCAP")

    total = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".npy")])
    print(f"\n✅ Done. Total emotion prob files: {total}")
    print(f"   Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
