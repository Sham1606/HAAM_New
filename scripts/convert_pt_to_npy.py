import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

v3_dir = Path("data/processed/features_v3_librosa")
out_dir = Path("data/processed/features_20dim")
out_dir.mkdir(parents=True, exist_ok=True)

pt_files = list(v3_dir.glob("*.pt"))
print(f"Converting {len(pt_files)} files...")

for pt_file in tqdm(pt_files):
    try:
        data = torch.load(pt_file, map_location='cpu', weights_only=False)
        arr = np.array(data['acoustic'], dtype=np.float32)
        
        # Determine the name. In v3, files are {call_id}.pt
        # In features_20dim, files are {feature_id}.npy (where 'iemocap_' prefix has been removed by train_hybrid_model.py logic, 
        # wait! train_hybrid_model.py does: feature_id = call_id.replace("iemocap_", "").
        # We should save using exactly the call_id or what train expects.
        # It expects `feature_id.npy` where feature_id = call_id.replace("iemocap_", "").
        name = pt_file.stem
        feature_id = name.replace("iemocap_", "")
        
        np.save(out_dir / f"{feature_id}.npy", arr)
    except Exception as e:
        print(e)

print("Done.")
