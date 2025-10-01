import os
import json
import numpy as np
from tqdm import tqdm

def check_embeddings_health(project_root):
    json_dir = os.path.join(project_root, "data", "processed", "json_embeddings")

    total_files = 0
    missing_audio = 0
    missing_text = 0
    corrupted_files = []

    for file in tqdm(os.listdir(json_dir), desc="Scanning JSON embeddings"):
        if not file.endswith(".json"):
            continue

        total_files += 1
        file_path = os.path.join(json_dir, file)

        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            audio_emb = np.array(data.get("audio_embedding", []))
            text_emb = np.array(data.get("text_embedding", []))

            if audio_emb.size == 0 or not np.any(audio_emb):
                missing_audio += 1
                corrupted_files.append(file)

            if text_emb.size == 0 or not np.any(text_emb):
                missing_text += 1
                corrupted_files.append(file)

        except Exception as e:
            print(f"❌ Error reading {file}: {e}")
            corrupted_files.append(file)

    # Summary
    print("\n====== Embedding Health Report ======")
    print(f"Total JSON files scanned: {total_files}")
    print(f"Missing/placeholder audio embeddings: {missing_audio} ({missing_audio/total_files:.2%})")
    print(f"Missing/placeholder text embeddings: {missing_text} ({missing_text/total_files:.2%})")

    if corrupted_files:
        corrupted_log = os.path.join(project_root, "corrupted_embeddings.txt")
        with open(corrupted_log, "w") as f:
            for cf in set(corrupted_files):
                f.write(cf + "\n")
        print(f"⚠️ Corrupted or missing embeddings logged to {corrupted_log}")
    else:
        print("✅ No corrupted files detected!")

if __name__ == "__main__":
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.dirname(os.path.dirname(script_dir))
    except NameError:
        PROJECT_ROOT = os.getcwd()

    check_embeddings_health(PROJECT_ROOT)
