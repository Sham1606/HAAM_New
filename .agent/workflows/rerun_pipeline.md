---
description: Rerun the entire HAAM pipeline on a new system to achieve 65-70% accuracy
---

1. Configure Paths
Verify that the `crema-d-mirror-main` and `iemocapfullrelease` paths in `scripts/02a_generate_real_metadata.py` match your new system's directory structure.

2. Generate Metadata
// turbo
`python scripts/02a_generate_real_metadata.py`
This creates `data/real_metadata.csv` by scanning your local audio files.

3. Initialize Data Splits
// turbo
`python scripts/03_train_baselines.py`
Run this briefly (or let it fail) to ensure `data/real_metadata_with_split.csv` is generated periodically, OR simply run the training script later which also handles missing splits.

4. Robust Feature Extraction
// turbo
`python scripts/reprocess_features_fixed.py`
This converts audio into 12-dim robust features (pYIN) and text sentiment. **This is the most critical step.**

5. Train Improved Model
// turbo
`python scripts/05_train_improved_model.py`
Trains the Attention Fusion model using the fixed features and Class Weighting for 'Fear'.

6. Verify Accuracy
`python scripts/08_deep_error_analysis.py`
Generates the final performance report to verify the 65-70% accuracy target.
