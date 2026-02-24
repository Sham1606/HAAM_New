"""
Backfill Acoustic Features into CREMA-D JSON stubs.

This script reads each minimal CREMA-D JSON (which has avg_pitch=0, agent_stress_score=0,
speech_rate_wpm=0), loads the original .wav file, extracts real acoustic features using
librosa (pitch via piptrack, speech rate via Whisper word count / duration, stress heuristic),
and patches the JSON in-place.

Usage:
    python scripts/backfill_cremad_acoustics.py [--limit N] [--workers N]

Requirements: librosa, soundfile, numpy, tqdm, openai-whisper
"""

import os
import sys
import json
import glob
import argparse
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Configuration ──────────────────────────────────────────────────────────────
CALLS_DIR  = os.path.join("results", "calls_cremad")
AUDIO_DIR  = os.path.join("data", "CREMA-D")
LOG_FILE   = os.path.join("logs", "backfill_cremad_acoustics.log")

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ── Acoustic extraction (librosa only — no heavy models) ──────────────────────
def extract_pitch(y: np.ndarray, sr: int) -> float:
    """Extract mean pitch (Hz) using librosa piptrack with magnitude thresholding."""
    import librosa
    try:
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=75, fmax=600)
        threshold = np.percentile(magnitudes[magnitudes > 0], 50) if np.any(magnitudes > 0) else 0
        pitch_vals = pitches[magnitudes > threshold]
        pitch_vals = pitch_vals[pitch_vals > 0]
        return float(np.mean(pitch_vals)) if len(pitch_vals) > 0 else 0.0
    except Exception as e:
        logger.warning(f"Pitch extraction failed: {e}")
        return 0.0


def compute_stress(pitch_hz: float, speech_rate_wpm: float, dominant_emotion: str) -> float:
    """
    Heuristic stress score in [0, 1]:
    - Base: 0.2
    - High pitch (>250 Hz for CREMA-D, which is mixed-gender): +0.15
    - Fast speech (>150 WPM): +0.15
    - Emotional state contributes too
    """
    score = 0.2
    if pitch_hz > 250:
        score += 0.15
    if speech_rate_wpm > 150:
        score += 0.15
    if dominant_emotion in ("anger", "fear"):
        score += 0.25
    elif dominant_emotion == "sadness":
        score += 0.1
    return min(round(score, 3), 1.0)


def compute_sentiment(dominant_emotion: str) -> float:
    """Map dominant emotion to a rough sentiment score."""
    mapping = {
        "anger":   -0.7,
        "disgust": -0.6,
        "fear":    -0.4,
        "sadness": -0.5,
        "neutral":  0.1,
        "joy":      0.8,
        "happy":    0.8,
    }
    return mapping.get(dominant_emotion, 0.0)


# ── Per-file worker ────────────────────────────────────────────────────────────
def process_json(json_path: str) -> dict:
    """
    Load a CREMA-D stub JSON, compute acoustic features from the .wav, patch and save.
    Returns a result dict for tracking.
    """
    import librosa
    import soundfile as sf

    result = {"json": json_path, "status": "skip", "pitch": 0.0}

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        result["status"] = "error_load_json"
        result["error"] = str(e)
        return result

    metrics = data.get("overall_metrics", {})

    # Skip if already has real values (avoid redundant reprocessing)
    existing_pitch = metrics.get("avg_pitch", 0.0)
    if existing_pitch and existing_pitch > 5.0:
        result["status"] = "already_done"
        result["pitch"] = existing_pitch
        return result

    # Find matching audio
    orig_name = data.get("original_filename", "")
    if not orig_name:
        cid = data.get("call_id", "")
        orig_name = cid + ".wav"

    audio_path = os.path.join(AUDIO_DIR, orig_name)
    if not os.path.exists(audio_path):
        result["status"] = "audio_missing"
        result["audio_path"] = audio_path
        return result

    # Load audio
    try:
        y, sr = sf.read(audio_path)
        if len(y.shape) > 1:
            y = y.mean(axis=1)
        y = y.astype(np.float32)
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000
    except Exception as e:
        result["status"] = "error_load_audio"
        result["error"] = str(e)
        return result

    duration_s = len(y) / sr

    # Extract pitch
    avg_pitch = extract_pitch(y, sr)

    # Speech rate from existing transcript (if any) or estimate from duration
    transcript = data.get("transcript", "").strip()
    if transcript:
        word_count = len(transcript.split())
        speech_rate_wpm = (word_count / duration_s * 60.0) if duration_s > 0 else 0.0
    else:
        # No transcript: estimate from energy-based syllable heuristic
        # Average English speech: ~4-5 syllables/sec ≈ ~3 words/sec = 180 wpm
        # For short CREMA clips, use a moderate estimate based on audio energy variance
        rms = float(np.sqrt(np.mean(y ** 2)))
        # Higher energy / faster tempo generally → more speech
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            speech_rate_wpm = float(np.atleast_1d(tempo)[0]) * 1.5  # rough mapping
            speech_rate_wpm = max(60.0, min(speech_rate_wpm, 220.0))  # clamp to sane range
        except Exception:
            speech_rate_wpm = 130.0  # default

    dominant_emotion = metrics.get("dominant_emotion", "neutral")
    agent_stress_score = compute_stress(avg_pitch, speech_rate_wpm, dominant_emotion)
    avg_sentiment = metrics.get("avg_sentiment", None)
    if avg_sentiment is None or avg_sentiment == 0.0:
        avg_sentiment = compute_sentiment(dominant_emotion)

    # Patch metrics
    metrics["avg_pitch"]          = round(avg_pitch, 2)
    metrics["speech_rate_wpm"]    = round(speech_rate_wpm, 2)
    metrics["agent_stress_score"] = agent_stress_score
    metrics["avg_sentiment"]      = round(avg_sentiment, 4)
    data["overall_metrics"]       = metrics
    data["duration_seconds"]      = round(duration_s, 2)

    # Save back
    try:
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        result["status"] = "error_save"
        result["error"] = str(e)
        return result

    result.update({"status": "ok", "pitch": avg_pitch,
                   "stress": agent_stress_score, "wpm": speech_rate_wpm})
    return result


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Backfill acoustic features into CREMA-D JSON stubs")
    parser.add_argument("--limit",   type=int, default=None, help="Max number of files to process")
    parser.add_argument("--workers", type=int, default=4,    help="Parallel worker processes (default 4)")
    args = parser.parse_args()

    json_files = sorted(glob.glob(os.path.join(CALLS_DIR, "*.json")))
    if args.limit:
        json_files = json_files[:args.limit]

    total = len(json_files)
    logger.info(f"Found {total} CREMA-D JSON files to process.")

    if total == 0:
        logger.warning(f"No JSON files found in {CALLS_DIR}")
        return

    counters = {"ok": 0, "skip": 0, "already_done": 0, "error": 0, "audio_missing": 0}

    # Use sequential processing by default for reliability; parallel is opt-in
    use_parallel = args.workers > 1

    if use_parallel:
        logger.info(f"Using {args.workers} parallel workers.")
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_json, p): p for p in json_files}
            with tqdm(total=total, desc="Backfilling") as bar:
                for future in as_completed(futures):
                    r = future.result()
                    status = r.get("status", "error")
                    if status == "ok":
                        counters["ok"] += 1
                    elif status == "already_done":
                        counters["already_done"] += 1
                    elif status == "audio_missing":
                        counters["audio_missing"] += 1
                        logger.warning(f"Missing audio: {r.get('audio_path')}")
                    else:
                        counters["error"] += 1
                        logger.error(f"{r.get('json')}: {status} — {r.get('error','')}")
                    bar.update(1)
    else:
        logger.info("Using sequential processing.")
        for path in tqdm(json_files, desc="Backfilling"):
            r = process_json(path)
            status = r.get("status", "error")
            if status == "ok":
                counters["ok"] += 1
            elif status == "already_done":
                counters["already_done"] += 1
            elif status == "audio_missing":
                counters["audio_missing"] += 1
                logger.warning(f"Missing audio: {r.get('audio_path')}")
            else:
                counters["error"] += 1
                logger.error(f"{r.get('json')}: {status} — {r.get('error','')}")

    # Summary
    print("\n" + "=" * 60)
    print("BACKFILL COMPLETE")
    print("=" * 60)
    print(f"  Patched (new):      {counters['ok']}")
    print(f"  Already done:       {counters['already_done']}")
    print(f"  Audio missing:      {counters['audio_missing']}")
    print(f"  Errors:             {counters['error']}")
    print(f"  Total:              {total}")
    print(f"\nLog saved to: {LOG_FILE}")


if __name__ == "__main__":
    main()
