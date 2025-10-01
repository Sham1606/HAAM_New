import os
import json
import numpy as np

def check_missing_embeddings(project_root):
    json_dir = os.path.join(project_root, "data", "processed", "json_embeddings")
    if not os.path.exists(json_dir):
        raise FileNotFoundError(f"❌ JSON embeddings folder not found at {json_dir}")

    print(f"🔎 Scanning JSON embeddings in {json_dir}...")

    missing_audio = []
    total_files = 0

    for fname in os.listdir(json_dir):
        if not fname.endswith(".json"):
            continue
        total_files += 1
        fpath = os.path.join(json_dir, fname)

        try:
            with open(fpath, "r") as f:
                data = json.load(f)

            audio_emb = np.array(data.get("audio_embedding", []), dtype=np.float32)

            # Check if missing or placeholder (all zeros or empty)
            if audio_emb.size == 0 or not np.any(audio_emb):
                missing_audio.append(fname)

        except Exception as e:
            print(f"⚠️ Error reading {fname}: {e}")
            missing_audio.append(fname)

    print(f"\n📊 Total JSONs scanned: {total_files}")
    print(f"❌ Missing/placeholder audio embeddings: {len(missing_audio)}")

    # Save results
    out_path = os.path.join(project_root, "results", "missing_audio_embeddings.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        for item in missing_audio:
            f.write(item + "\n")

    print(f"✅ List of missing audio embeddings saved to: {out_path}")


if __name__ == "__main__":
    # Adjust PROJECT_ROOT one level up from src/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(script_dir))

    check_missing_embeddings(PROJECT_ROOT)
