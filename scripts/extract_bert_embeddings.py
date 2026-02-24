"""
Extract real BERT (bert-base-uncased) embeddings for all training samples.
Saves 768-dim CLS token embeddings to data/processed/bert_embeddings/{call_id}.npy

CREMA-D: Uses fixed sentence texts mapped from sentence codes.
IEMOCAP: Reads transcript from JSON metadata.

Run ONCE before training:
    python scripts/extract_bert_embeddings.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# ── Paths ──────────────────────────────────────────────────────────────────────
CREMAD_RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results", "calls_cremad")
IEMOCAP_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "calls_iemocap")
CREMAD_GT_CSV       = os.path.join(PROJECT_ROOT, "data", "cremad_ground_truth.csv")
OUTPUT_DIR          = os.path.join(PROJECT_ROOT, "data", "processed", "bert_embeddings")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── CREMA-D sentence code → text ───────────────────────────────────────────────
# CREMA-D uses 12 fixed sentences (IEO, TIE, IOM, IWW, TAI, MTI, IWL, ITH, DFA, ITS, TSI, WSI)
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


def get_bert_model():
    """Lazily load BERT tokenizer and model."""
    print("Loading BERT (bert-base-uncased)...")
    from transformers import BertTokenizer, BertModel
    import torch
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"  BERT loaded on {device}")
    return tokenizer, model, device


def encode_text(tokenizer, model, device, text: str) -> np.ndarray:
    """Returns 768-dim CLS embedding for a piece of text."""
    import torch
    if not text or text.strip() == "":
        text = "unknown"
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]  # [768]
    return cls_emb.astype(np.float32)


def process_cremad(tokenizer, model, device):
    """Extract BERT embeddings for all CREMA-D JSON files."""
    print("\n── CREMA-D ──")
    files = [f for f in os.listdir(CREMAD_RESULTS_DIR) if f.endswith(".json")]
    done = skipped = errors = 0

    for fname in tqdm(files, desc="CREMA-D"):
        call_id = fname.replace(".json", "")
        out_path = os.path.join(OUTPUT_DIR, f"{call_id}.npy")
        if os.path.exists(out_path):
            skipped += 1
            continue

        try:
            with open(os.path.join(CREMAD_RESULTS_DIR, fname)) as f:
                data = json.load(f)

            # Look for transcript in JSON, else map sentence code
            text = data.get("transcript", "")
            if not text:
                # Extract sentence code from call_id (e.g. 1001_DFA_ANG_XX → DFA)
                parts = call_id.split("_")
                sent_code = parts[1] if len(parts) >= 2 else ""
                text = CREMAD_SENTENCES.get(sent_code, "")

            emb = encode_text(tokenizer, model, device, text)
            np.save(out_path, emb)
            done += 1
        except Exception as e:
            errors += 1

    print(f"  Processed: {done}, Skipped: {skipped}, Errors: {errors}")


def process_iemocap(tokenizer, model, device):
    """Extract BERT embeddings for all IEMOCAP JSON files."""
    if not os.path.exists(IEMOCAP_RESULTS_DIR):
        print("\nIEMOCAP results directory not found - skipping.")
        return

    print("\n── IEMOCAP ──")
    files = [f for f in os.listdir(IEMOCAP_RESULTS_DIR) if f.endswith(".json")]
    done = skipped = errors = 0

    for fname in tqdm(files, desc="IEMOCAP"):
        call_id = fname.replace(".json", "")
        # File saved as utterance_id (strip iemocap_ prefix)
        feature_id = call_id.replace("iemocap_", "")
        out_path = os.path.join(OUTPUT_DIR, f"{feature_id}.npy")
        if os.path.exists(out_path):
            skipped += 1
            continue

        try:
            with open(os.path.join(IEMOCAP_RESULTS_DIR, fname)) as f:
                data = json.load(f)

            # Try to get transcript
            text = (
                data.get("transcript")
                or data.get("text")
                or data.get("metadata", {}).get("transcript", "")
                or ""
            )
            emb = encode_text(tokenizer, model, device, text)
            np.save(out_path, emb)
            done += 1
        except Exception as e:
            errors += 1

    print(f"  Processed: {done}, Skipped: {skipped}, Errors: {errors}")


def main():
    tokenizer, model, device = get_bert_model()
    process_cremad(tokenizer, model, device)
    process_iemocap(tokenizer, model, device)
    total = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".npy")])
    print(f"\n✅ Done. Total embeddings in {OUTPUT_DIR}: {total}")


if __name__ == "__main__":
    main()
