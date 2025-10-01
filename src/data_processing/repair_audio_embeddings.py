# repair_audio_embeddings.py
import os
import json
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from moviepy import VideoFileClip

# ------------------------------
# Utility: Load audio from MP4
# ------------------------------
def load_audio_from_mp4(mp4_path, target_sr=16000):
    try:
        clip = VideoFileClip(mp4_path)
        audio = clip.audio.to_soundarray(fps=target_sr)
        clip.close()

        if audio is None:
            raise RuntimeError("No audio track found")

        # Stereo → Mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        return torch.tensor(audio).unsqueeze(0), target_sr
    except Exception as e:
        raise RuntimeError(f"Error reading {mp4_path}: {e}")

# ------------------------------
# Regenerate embeddings
# ------------------------------
def regenerate_embeddings(project_root):
    json_dir = os.path.join(project_root, "data", "processed", "json_embeddings")
    raw_base = os.path.join(project_root, "data", "raw", "MELD.Raw")

    os.makedirs(json_dir, exist_ok=True)
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    corrupted_log = os.path.join(results_dir, "corrupted_audio_files.txt")

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    model.eval()

    repaired, skipped = 0, 0
    corrupted_files = []

    for json_file in os.listdir(json_dir):
        if not json_file.endswith(".json"):
            continue

        json_path = os.path.join(json_dir, json_file)
        with open(json_path, "r") as f:
            data = json.load(f)

        # Skip if already has valid audio embedding
        audio_emb = data.get("audio_embedding", [])
        if audio_emb and any(val != 0.0 for val in audio_emb):
            continue

        dialogue_id = data.get("dialogue_id")
        utterance_id = data.get("utterance_id")
        mp4_file = f"dia{dialogue_id}_utt{utterance_id}.mp4"

        # Try each split folder (train/dev/test)
        found_path = None
        for split in ["train", "dev", "test"]:
            candidate = os.path.join(raw_base, split, mp4_file)
            if os.path.exists(candidate):
                found_path = candidate
                break

        if not found_path:
            print(f"❌ Missing audio file: {mp4_file}")
            corrupted_files.append(mp4_file)
            skipped += 1
            continue

        try:
            waveform, sr = load_audio_from_mp4(found_path, target_sr=16000)
            inputs = feature_extractor(
                waveform.squeeze().numpy(), sampling_rate=sr, return_tensors="pt"
            )
            with torch.no_grad():
                emb = model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()

            data["audio_embedding"] = emb
            with open(json_path, "w") as f:
                json.dump(data, f)

            repaired += 1
            print(f"✅ Repaired embedding for {json_file}")

        except Exception as e:
            print(f"⚠️ Skipping corrupted {json_file}: {e}")
            corrupted_files.append(mp4_file)
            skipped += 1

    # Save corrupted file list
    if corrupted_files:
        with open(corrupted_log, "w") as f:
            for fname in corrupted_files:
                f.write(fname + "\n")
        print(f"\n⚠️ {len(corrupted_files)} corrupted files logged at {corrupted_log}")

    print(f"\n🎉 Repair process complete! Fixed {repaired} files, skipped {skipped} corrupted files.")

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.dirname(os.path.dirname(script_dir))
    except NameError:
        PROJECT_ROOT = os.getcwd()

    regenerate_embeddings(PROJECT_ROOT)
