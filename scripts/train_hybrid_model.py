"""
train_hybrid_model.py  ―  v2  (accuracy target: 65-70%)

Key improvements over v1
─────────────────────────
1. Proper 70 / 15 / 15  train / val / test  split (no data leakage)
2. SMOTE oversampling on *train only* to fix class imbalance
3. FocalLoss(γ=2) + label_smoothing=0.1  instead of plain CrossEntropy
4. AdamW + CosineAnnealingLR  instead of Adam + ReduceLROnPlateau
5. Mixup augmentation  (α=0.2) in training loop
6. Stronger Gaussian noise  (std=0.05) for acoustic branch
7. Lower dropout  (0.25) – architecture is now wider, less need for heavy dropout
8. Correct early stopping on *validation* accuracy (not test)
9. Fully saves confusion matrix + per-class metrics
"""

import os, sys, json, logging
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.models.improved_hybrid_model import ImprovedHybridModel
from src.models.focal_loss import FocalLoss

# ── Paths ────────────────────────────────────────────────────────────────────
HYBRID_METADATA_PATH = os.path.join(PROJECT_ROOT, "data",  "hybrid_metadata.csv")
CREMAD_RESULTS_DIR   = os.path.join(PROJECT_ROOT, "results", "calls_cremad")
IEMOCAP_RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results", "calls_iemocap")
FEATURE_DIR          = os.path.join(PROJECT_ROOT, "data", "processed", "features_20dim")
BERT_DIR             = os.path.join(PROJECT_ROOT, "data", "processed", "bert_embeddings")
EMO_PROBS_DIR        = os.path.join(PROJECT_ROOT, "data", "processed", "emotion_probs")
MODEL_SAVE_PATH      = os.path.join(PROJECT_ROOT, "saved_models", "hybrid_fusion_model.pth")
SCALER_SAVE_PATH     = os.path.join(PROJECT_ROOT, "saved_models", "hybrid_scaler.pkl")
ENCODER_SAVE_PATH    = os.path.join(PROJECT_ROOT, "saved_models", "hybrid_encoder.pkl")
METRICS_SAVE_PATH    = os.path.join(PROJECT_ROOT, "results",   "hybrid_model_metrics.json")
HISTORY_SAVE_PATH    = os.path.join(PROJECT_ROOT, "saved_models", "training_history.json")

os.makedirs(os.path.dirname(MODEL_SAVE_PATH),  exist_ok=True)
os.makedirs(os.path.dirname(METRICS_SAVE_PATH), exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s  %(message)s')
log = logging.getLogger(__name__)

TARGET_EMOTIONS = ['neutral', 'anger', 'disgust', 'fear', 'sadness']

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class HybridDataset(Dataset):
    def __init__(self, X_acoustic, X_text, y, augment=False, noise_std=0.05):
        self.X_acoustic = torch.FloatTensor(X_acoustic)
        self.X_text     = torch.FloatTensor(X_text)
        self.y          = torch.LongTensor(y)
        self.augment    = augment
        self.noise_std  = noise_std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        ac = self.X_acoustic[idx].clone()
        if self.augment:
            ac += torch.randn_like(ac) * self.noise_std
        return ac, self.X_text[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    log.info("Loading hybrid metadata …")
    df = pd.read_csv(HYBRID_METADATA_PATH)

    records, skipped = [], 0
    for _, row in df.iterrows():
        dataset  = row['dataset']
        call_id  = str(row['call_id'])
        emotion  = (row.get('emotion_true') or row.get('emotion', '')).lower()

        if emotion == 'joy' or emotion not in TARGET_EMOTIONS:
            continue

        # ── JSON path ─────────────────────────────────────────────────────
        if dataset == 'CREMA-D':
            json_path  = os.path.join(CREMAD_RESULTS_DIR,  f"{call_id}.json")
            feature_id = call_id
        else:
            json_path  = os.path.join(IEMOCAP_RESULTS_DIR, f"{call_id}.json")
            feature_id = call_id.replace("iemocap_", "")

        if not os.path.exists(json_path):
            skipped += 1
            continue

        # ── 20-dim acoustic features ───────────────────────────────────────
        npy_path   = os.path.join(FEATURE_DIR, f"{feature_id}.npy")
        acoustic   = None
        if os.path.exists(npy_path):
            try:
                arr = np.load(npy_path, allow_pickle=False)
                if arr.shape[0] == 20:
                    acoustic = arr.tolist()
            except Exception:
                pass

        if acoustic is None:
            # Fallback: read from JSON and pad to 20 dims
            try:
                with open(json_path) as f:
                    data    = json.load(f)
                m           = data.get('overall_metrics', {})
                acoustic = [
                    m.get('avg_pitch',           0.0) or 0.0,
                    m.get('speech_rate_wpm',      0.0) or 0.0,
                    m.get('agent_stress_score',   0.0) or 0.0,
                ] + [0.0] * 17      # pad to 20
            except Exception:
                skipped += 1
                continue

        # ── Text features (768 BERT + 5 emotion probs) ─────────────────────
        bert_path  = os.path.join(BERT_DIR,      f"{feature_id}.npy")
        probs_path = os.path.join(EMO_PROBS_DIR, f"{feature_id}.npy")

        try:
            bert_emb  = (np.load(bert_path,  allow_pickle=False).tolist()
                         if os.path.exists(bert_path) else [0.0] * 768)
        except Exception:
            bert_emb  = [0.0] * 768

        if dataset == 'IEMOCAP' and os.path.exists(probs_path):
            try:
                emo_probs = np.load(probs_path, allow_pickle=False).tolist()
            except Exception:
                emo_probs = [0.2] * 5
        else:
            # CREMA-D: derive probs from JSON emotion_distribution
            try:
                with open(json_path) as f:
                    data = json.load(f)
                dist      = data.get('overall_metrics', {}).get('emotion_distribution', {})
                raw       = [dist.get(e, 0.0) for e in TARGET_EMOTIONS]
                total     = sum(raw) or 1.0
                emo_probs = [v / total for v in raw]
            except Exception:
                emo_probs = [0.2] * 5

        text_full = bert_emb + emo_probs   # 773-dim

        records.append({
            'acoustic' : acoustic,
            'text'     : text_full,
            'label'    : emotion,
            'dataset'  : dataset,
        })

    log.info(f"Loaded {len(records)} samples  (skipped {skipped})")
    class_dist = Counter(r['label'] for r in records)
    log.info(f"Class distribution: {dict(sorted(class_dist.items()))}")

    X_a = np.array([r['acoustic'] for r in records])
    X_t = np.array([r['text']     for r in records])
    y   = [r['label'] for r in records]
    return X_a, X_t, y


# ─────────────────────────────────────────────────────────────────────────────
# Splits  70 / 15 / 15
# ─────────────────────────────────────────────────────────────────────────────
def three_way_split(X_a, X_t, y_enc):
    # First: 70 train / 30 temp
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    for tr, tmp in sss1.split(X_a, y_enc):
        X_a_tr, X_t_tr, y_tr = X_a[tr], X_t[tr], y_enc[tr]
        X_a_tmp, X_t_tmp, y_tmp = X_a[tmp], X_t[tmp], y_enc[tmp]

    # Second: 50-50 of temp → 15 val / 15 test
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    for val_idx, tst_idx in sss2.split(X_a_tmp, y_tmp):
        X_a_val, X_t_val, y_val = X_a_tmp[val_idx], X_t_tmp[val_idx], y_tmp[val_idx]
        X_a_tst, X_t_tst, y_tst = X_a_tmp[tst_idx], X_t_tmp[tst_idx], y_tmp[tst_idx]

    log.info(f"Split → train={len(y_tr)}  val={len(y_val)}  test={len(y_tst)}")
    return (X_a_tr,  X_t_tr,  y_tr,
            X_a_val, X_t_val, y_val,
            X_a_tst, X_t_tst, y_tst)


# ─────────────────────────────────────────────────────────────────────────────
# Mixup
# ─────────────────────────────────────────────────────────────────────────────
def mixup_batch(ac, text, labels, n_classes, alpha=0.2):
    """Mixup augmentation on a single batch."""
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(ac.size(0), device=ac.device)
    mixed_ac   = lam * ac   + (1 - lam) * ac[idx]
    mixed_text = lam * text + (1 - lam) * text[idx]
    # Soft one-hot targets
    y_a = torch.zeros(ac.size(0), n_classes, device=ac.device).scatter_(1, labels.unsqueeze(1), 1.0)
    y_b = y_a[idx]
    soft_y = lam * y_a + (1 - lam) * y_b
    return mixed_ac, mixed_text, soft_y


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train_model():
    # ── Load & encode ─────────────────────────────────────────────────────
    X_a, X_t, y_raw = load_data()
    if len(X_a) == 0:
        log.error("No data found.")
        return

    # Map strings directly to TARGET_EMOTIONS indexes
    y_enc = np.array([TARGET_EMOTIONS.index(lab) for lab in y_raw])
    n_classes = len(TARGET_EMOTIONS)
    log.info(f"Classes: {TARGET_EMOTIONS}")

    # Create dummy encoder object for backward compatibility in saving
    class MockEncoder:
        @property
        def classes_(self):
            return np.array(TARGET_EMOTIONS)
    encoder = MockEncoder()

    # ── 70/15/15 split ────────────────────────────────────────────────────
    (X_a_tr, X_t_tr, y_tr,
     X_a_val, X_t_val, y_val,
     X_a_tst, X_t_tst, y_tst) = three_way_split(X_a, X_t, y_enc)

    # ── Scale acoustic on train stats only ───────────────────────────────
    scaler       = StandardScaler()
    X_a_tr_sc    = scaler.fit_transform(X_a_tr)
    X_a_val_sc   = scaler.transform(X_a_val)
    X_a_tst_sc   = scaler.transform(X_a_tst)

    # ── SMOTE on training acoustic features ──────────────────────────────
    log.info("Applying SMOTE to balance training classes …")
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=42, k_neighbors=min(5, min(Counter(y_tr).values()) - 1))
        # SMOTE on 20-dim acoustic (faster, and text embeddings are already rich)
        X_a_sm, y_tr_sm = sm.fit_resample(X_a_tr_sc, y_tr)
        # For text: repeat rows by re-indexing (SMOTE gives us new indices implicitly)
        # We'll treat SMOTE indices as nearest-neighbour resampling on both modalities
        # Simple approach: re-run SMOTE with full 20+773=793 dims
        X_full_tr = np.hstack([X_a_tr_sc, X_t_tr])
        X_full_sm, y_tr_sm = sm.fit_resample(X_full_tr, y_tr)
        X_a_sm   = X_full_sm[:, :20]
        X_t_sm   = X_full_sm[:, 20:]
    except Exception as e:
        log.warning(f"SMOTE failed ({e}) - using class-weighted loss only.")
        X_a_sm, X_t_sm, y_tr_sm = X_a_tr_sc, X_t_tr, y_tr

    log.info(f"After SMOTE: {Counter(y_tr_sm)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Class weights (backup to SMOTE) ──────────────────────────────────
    counts = Counter(y_tr_sm)
    total  = sum(counts.values())
    class_weights = torch.tensor(
        [total / (n_classes * counts.get(i, 1)) for i in range(n_classes)],
        dtype=torch.float, device=device
    )
    log.info(f"Class weights: { {c: f'{w:.2f}' for c, w in zip(encoder.classes_, class_weights.cpu())} }")

    # ── Model ─────────────────────────────────────────────────────────────
    model = ImprovedHybridModel(
        n_acoustic=20, n_text_emb=768, n_text_probs=5,
        n_classes=n_classes, dropout=0.25
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model params: {total_params:,}")

    # ── Loss: Focal with label smoothing ──────────────────────────────────
    criterion = FocalLoss(gamma=2.0, alpha=class_weights, label_smoothing=0.1)

    # ── Optimiser + scheduler ─────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    max_epochs = 120
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    # ── DataLoaders ───────────────────────────────────────────────────────
    train_ds  = HybridDataset(X_a_sm,    X_t_sm,   y_tr_sm, augment=True,  noise_std=0.05)
    val_ds    = HybridDataset(X_a_val_sc,X_t_val,  y_val,   augment=False)
    test_ds   = HybridDataset(X_a_tst_sc,X_t_tst,  y_tst,   augment=False)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  drop_last=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=0)

    # ── History tracking ──────────────────────────────────────────────────
    history = {k: [] for k in ['epoch', 'train_loss', 'train_emo_acc',
                                'val_emo_acc', 'val_emo_acc', 'avg_attn_audio']}

    best_val_acc = 0.0
    patience     = 15
    no_improve   = 0

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss, correct_tr, total_tr = 0.0, 0, 0
        attn_sums = []

        for ac, text_in, labels in train_loader:
            ac, text_in, labels = ac.to(device), text_in.to(device), labels.to(device)
            text_emb   = text_in[:, :768]
            text_probs = text_in[:, 768:]

            # Mixup (50% of the time)
            if np.random.random() < 0.5:
                ac_m, text_m, soft_y = mixup_batch(ac, text_emb, labels, n_classes)
                optimizer.zero_grad()
                logits, attn = model(ac_m, text_m, text_probs)
                # Soft-label CE (manual, since FocalLoss expects hard labels for p_t)
                log_p = torch.nn.functional.log_softmax(logits, dim=1)
                loss  = -(soft_y * log_p).sum(dim=1).mean()
            else:
                optimizer.zero_grad()
                logits, attn = model(ac, text_emb, text_probs)
                loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            preds = logits.argmax(1)
            correct_tr += (preds == labels).sum().item()
            total_tr   += labels.size(0)
            attn_sums.append(attn[:, 0].mean().item())

        scheduler.step()

        train_acc   = correct_tr / total_tr
        avg_loss    = running_loss / len(train_loader)
        avg_attn_a  = np.mean(attn_sums)

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        correct_v, total_v = 0, 0
        with torch.no_grad():
            for ac, text_in, labels in val_loader:
                ac, text_in, labels = ac.to(device), text_in.to(device), labels.to(device)
                logits, _ = model(ac, text_in[:, :768], text_in[:, 768:])
                correct_v += (logits.argmax(1) == labels).sum().item()
                total_v   += labels.size(0)
        val_acc = correct_v / total_v

        lr_now = optimizer.param_groups[0]['lr']
        log.info(
            f"Epoch {epoch:3d}/{max_epochs}  "
            f"Loss {avg_loss:.4f}  TrainAcc {train_acc:.4f}  "
            f"ValAcc {val_acc:.4f}  lr={lr_now:.2e}  ã_audio={avg_attn_a:.3f}"
        )

        # Record history
        history['epoch'].append(epoch)
        history['train_loss'].append(round(avg_loss, 5))
        history['train_emo_acc'].append(round(train_acc, 5))
        history['val_emo_acc'].append(round(val_acc, 5))
        history['avg_attn_audio'].append(round(avg_attn_a, 5))

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve   = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            log.info(f"  ✓ Saved best model (val={val_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info(f"Early stopping at epoch {epoch}. Best val={best_val_acc:.4f}")
                break

    # ── Final test evaluation ─────────────────────────────────────────────
    log.info("Loading best model for test evaluation …")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for ac, text_in, labels in test_loader:
            ac, text_in, labels = ac.to(device), text_in.to(device), labels.to(device)
            logits, _ = model(ac, text_in[:, :768], text_in[:, 768:])
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    test_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    log.info(f"\n{'='*60}")
    log.info(f"FINAL TEST ACCURACY: {test_acc:.4f}  ({test_acc*100:.2f}%)")
    log.info(f"{'='*60}")

    cls_names = encoder.classes_.tolist()
    report    = classification_report(all_labels, all_preds, target_names=cls_names, output_dict=True)
    cm        = confusion_matrix(all_labels, all_preds).tolist()

    print("\n" + classification_report(all_labels, all_preds, target_names=cls_names))

    # ── Save metrics ──────────────────────────────────────────────────────
    metrics = {
        "test_accuracy"        : round(test_acc, 6),
        "best_val_accuracy"    : round(best_val_acc, 6),
        "classification_report": report,
        "confusion_matrix"     : cm,
        "class_names"          : cls_names,
        "training_samples"     : len(y_tr_sm),
        "val_samples"          : len(y_val),
        "test_samples"         : len(y_tst),
    }
    with open(METRICS_SAVE_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Metrics saved → {METRICS_SAVE_PATH}")

    with open(HISTORY_SAVE_PATH, 'w') as f:
        json.dump(history, f, indent=2)
    log.info(f"History saved → {HISTORY_SAVE_PATH}")

    joblib.dump(scaler,  SCALER_SAVE_PATH)
    joblib.dump(encoder, ENCODER_SAVE_PATH)
    log.info("Scaler + Encoder saved.")
    log.info("Training complete.")


if __name__ == "__main__":
    train_model()
