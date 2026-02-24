"""
Rebuild hybrid_metadata.csv from ALL JSON files in results/calls_cremad and results/calls_iemocap.

This replaces the partial CSV (12,271 rows) with a full one covering every extracted sample (~17k).

Run ONCE:
    python scripts/rebuild_metadata.py
"""

import os
import sys
import json
import csv
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

CREMAD_DIR   = os.path.join(PROJECT_ROOT, "results", "calls_cremad")
IEMOCAP_DIR  = os.path.join(PROJECT_ROOT, "results", "calls_iemocap")
OUTPUT_CSV   = os.path.join(PROJECT_ROOT, "data", "hybrid_metadata.csv")

TARGET_EMOTIONS = {"neutral", "anger", "disgust", "fear", "sadness", "joy"}

# CREMA-D fixed sentence code → full text (for transcript column)
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

rows = []

# ── CREMA-D ────────────────────────────────────────────────────────────────────
cremad_files = [f for f in os.listdir(CREMAD_DIR) if f.endswith(".json")]
print(f"CREMA-D: {len(cremad_files)} JSON files")
skipped_cremad = 0

for fname in tqdm(cremad_files, desc="CREMA-D"):
    try:
        with open(os.path.join(CREMAD_DIR, fname)) as f:
            data = json.load(f)

        call_id = data.get("call_id", fname.replace(".json", ""))
        emotion  = (data.get("ground_truth", {}).get("emotion") or "").lower()

        if not emotion or emotion not in TARGET_EMOTIONS:
            skipped_cremad += 1
            continue

        # Transcript from JSON or sentence code
        transcript = data.get("transcript", "")
        if not transcript:
            parts = call_id.split("_")
            sent_code = parts[1] if len(parts) >= 2 else ""
            transcript = CREMAD_SENTENCES.get(sent_code, "")

        metrics    = data.get("overall_metrics", {})
        duration   = metrics.get("duration_seconds", 0.0) or round(
            data.get("duration_seconds", 0.0), 2)
        dom_emo    = metrics.get("dominant_emotion", emotion)

        rows.append({
            "dataset":     "CREMA-D",
            "call_id":     call_id,
            "emotion_true": emotion,
            "emotion_pred": dom_emo,
            "confidence":  round(metrics.get("agent_stress_score", 0.5), 3),
            "duration":    round(float(duration), 2),
            "transcript":  transcript,
        })
    except Exception as e:
        skipped_cremad += 1

print(f"  Loaded: {len(rows)}, Skipped: {skipped_cremad}")

# ── IEMOCAP ────────────────────────────────────────────────────────────────────
cremad_count = len(rows)

if os.path.exists(IEMOCAP_DIR):
    iemocap_files = [f for f in os.listdir(IEMOCAP_DIR) if f.endswith(".json")]
    print(f"\nIEMOCAP: {len(iemocap_files)} JSON files")
    skipped_iemocap = 0

    for fname in tqdm(iemocap_files, desc="IEMOCAP"):
        try:
            with open(os.path.join(IEMOCAP_DIR, fname)) as f:
                data = json.load(f)

            call_id = data.get("call_id", fname.replace(".json", ""))
            emotion  = (data.get("ground_truth", {}).get("emotion") or "").lower()

            if not emotion or emotion not in TARGET_EMOTIONS:
                skipped_iemocap += 1
                continue

            metrics   = data.get("overall_metrics", {})
            duration  = data.get("duration_seconds", 0.0) or metrics.get("duration_seconds", 0.0)
            transcript = (
                data.get("transcript")
                or data.get("metadata", {}).get("transcript_ground_truth", "")
                or ""
            )
            dom_emo   = metrics.get("dominant_emotion", emotion)

            rows.append({
                "dataset":      "IEMOCAP",
                "call_id":      call_id,
                "emotion_true": emotion,
                "emotion_pred": dom_emo,
                "confidence":   round(metrics.get("agent_stress_score", 0.5), 3),
                "duration":     round(float(duration), 2),
                "transcript":   transcript,
            })
        except Exception as e:
            skipped_iemocap += 1

    iemocap_added = len(rows) - cremad_count
    print(f"  Loaded: {iemocap_added}, Skipped: {skipped_iemocap}")
else:
    print("\nIEMOCAP directory not found - skipping.")

# ── Write CSV ──────────────────────────────────────────────────────────────────
print(f"\nWriting {len(rows)} rows to {OUTPUT_CSV} ...")
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["dataset","call_id","emotion_true","emotion_pred","confidence","duration","transcript"])
    writer.writeheader()
    writer.writerows(rows)

# Summary
from collections import Counter
emotion_counts = Counter(r["emotion_true"] for r in rows)
dataset_counts = Counter(r["dataset"] for r in rows)

print(f"\n✅ hybrid_metadata.csv rebuilt with {len(rows)} total samples")
print(f"   Datasets : {dict(dataset_counts)}")
print(f"   Emotions : {dict(emotion_counts)}")
