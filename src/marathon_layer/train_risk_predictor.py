import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    'total_calls', 'avg_sentiment', 'anger_pct', 'sadness_pct',
    'fear_pct', 'joy_pct', 'avg_stress_score', 'engagement_score',
    'sentiment_trend_7d', 'anger_trend_7d', 'duration_trend_7d', 'workload_spike'
]
SEQ_LEN = 14       # 14-day sliding window
INPUT_DIM = len(FEATURE_COLS)   # 12


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class BurnoutDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences   # list of np.array [SEQ_LEN, INPUT_DIM]
        self.labels = labels         # list of float {0, 1}

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor([self.labels[idx]])
        )


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

class LSTMRiskPredictor(nn.Module):
    """
    Bidirectional LSTM that ingests a SEQ_LEN-day window of agent features
    and outputs a burnout probability [0, 1].
    """
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]   # last time step
        return self.fc(last_out)


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic data generation
# Uses same risk thresholds as risk_scoring.py to create plausible labels
# ──────────────────────────────────────────────────────────────────────────────

def generate_synthetic_data(n_agents=80, days_per_agent=60, seed=42):
    """
    Generate synthetic daily agent feature sequences with burnout labels.
    Label = 1 when agent is in a burnout pattern (high stress + sentiment decline).
    """
    rng = np.random.default_rng(seed)
    sequences, labels = [], []

    for agent in range(n_agents):
        # Random baseline personality
        base_sentiment = rng.uniform(-0.2, 0.4)
        base_stress = rng.uniform(0.1, 0.5)
        burnout_agent = rng.random() < 0.35   # 35% of agents experience burnout

        rows = []
        for day in range(days_per_agent):
            # Burnout agents progressively worsen
            progress = day / days_per_agent
            sentiment_drift = -0.4 * progress if burnout_agent else 0.0
            stress_drift = 0.3 * progress if burnout_agent else 0.0

            row = {
                'total_calls':        rng.integers(8, 30),
                'avg_sentiment':      np.clip(base_sentiment + sentiment_drift + rng.normal(0, 0.1), -1, 1),
                'anger_pct':          np.clip(0.1 + stress_drift + rng.normal(0, 0.05), 0, 1),
                'sadness_pct':        np.clip(0.08 + stress_drift * 0.5 + rng.normal(0, 0.04), 0, 1),
                'fear_pct':           np.clip(0.05 + rng.normal(0, 0.03), 0, 1),
                'joy_pct':            np.clip(0.2 - sentiment_drift * 0.5 + rng.normal(0, 0.05), 0, 1),
                'avg_stress_score':   np.clip(base_stress + stress_drift + rng.normal(0, 0.07), 0, 1),
                'engagement_score':   np.clip(0.6 - stress_drift * 0.8 + rng.normal(0, 0.1), 0, 1),
                'sentiment_trend_7d': sentiment_drift * 0.5 + rng.normal(0, 0.05),
                'anger_trend_7d':     stress_drift * 0.3 + rng.normal(0, 0.03),
                'duration_trend_7d':  rng.normal(0, 0.1),
                'workload_spike':     np.clip(1.0 + stress_drift * 0.5 + rng.normal(0, 0.2), 0.5, 3.0),
            }
            rows.append(row)

        agent_df = pd.DataFrame(rows)

        # Burnout label: high stress + sentiment decline in recent 7 days
        burnout_label = (
            (agent_df['sentiment_trend_7d'] < -0.15) &
            (agent_df['avg_stress_score'] > 0.50)
        ).astype(int)

        # Sliding windows
        feat = agent_df[FEATURE_COLS].values
        for i in range(len(feat) - SEQ_LEN):
            sequences.append(feat[i: i + SEQ_LEN])
            labels.append(float(burnout_label.iloc[i + SEQ_LEN - 1]))

    return sequences, labels


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_model(epochs=60, batch_size=32, lr=1e-3, use_real_data=True):
    """
    Train LSTM risk predictor.

    If agent_features.csv exists (real pipeline data), uses it.
    Otherwise falls back to synthetic data.

    Saves model to saved_models/marathon_risk_predictor.pth
    """
    sequences, labels = [], []

    # ── Try real data first ───────────────────────────────────────────────────
    real_data_path = Path("results/marathon/agent_features.csv")
    if use_real_data and real_data_path.exists():
        logger.info(f"Loading real agent features from {real_data_path}")
        df = pd.read_csv(real_data_path)

        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0

        df['burnout_label'] = (
            (df['sentiment_trend_7d'] < -0.15) &
            (df['avg_stress_score'] > 0.50)
        ).astype(int)

        for agent_id, group in df.groupby('agent_id'):
            group = group.sort_values('date')
            if len(group) < SEQ_LEN + 1:
                continue
            feat = group[FEATURE_COLS].values
            lbl = group['burnout_label'].values
            for i in range(len(feat) - SEQ_LEN):
                sequences.append(feat[i: i + SEQ_LEN])
                labels.append(float(lbl[i + SEQ_LEN - 1]))

    # ── Fallback: synthetic data ──────────────────────────────────────────────
    if len(sequences) < 50:
        logger.info("=" * 60)
        logger.info("DATA SOURCE: SYNTHETIC (real data has only %d sequences, need 50+)", len(sequences))
        logger.info("Re-run after 14+ days of real call history to train on real data.")
        logger.info("=" * 60)
        sequences, labels = generate_synthetic_data(n_agents=100, days_per_agent=60)
    else:
        logger.info("=" * 60)
        logger.info("DATA SOURCE: REAL (%d sequences from agent call history)", len(sequences))
        logger.info("=" * 60)

    logger.info(f"Dataset: {len(sequences)} sequences | Burnout rate: {np.mean(labels):.2%}")

    # ── Normalise features ────────────────────────────────────────────────────
    flat = np.stack(sequences).reshape(-1, INPUT_DIM)
    scaler = StandardScaler()
    scaler.fit(flat)
    sequences_scaled = [scaler.transform(s) for s in sequences]

    # ── Train / Val split ─────────────────────────────────────────────────────
    idx = list(range(len(sequences_scaled)))
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42,
                                          stratify=[int(l) for l in labels])

    train_ds = BurnoutDataset([sequences_scaled[i] for i in train_idx],
                              [labels[i] for i in train_idx])
    val_ds   = BurnoutDataset([sequences_scaled[i] for i in val_idx],
                              [labels[i] for i in val_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    # ── Model ─────────────────────────────────────────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LSTMRiskPredictor(input_dim=INPUT_DIM)
    model.to(device)

    # Pos-weight to handle class imbalance
    pos_weight = torch.tensor([(1 - np.mean(labels)) / (np.mean(labels) + 1e-6)]).to(device)
    criterion  = nn.BCELoss()   # model uses Sigmoid already
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')
    save_path = Path("saved_models/marathon_risk_predictor.pth")
    save_path.parent.mkdir(exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(X)
        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                val_loss += criterion(pred, y).item() * len(X)
                correct += ((pred > 0.5).float() == y).sum().item()
                total += len(y)
        val_loss /= len(val_ds)
        val_acc = correct / total

        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:3d}/{epochs} | "
                        f"Train Loss: {train_loss:.4f} | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"Val Acc: {val_acc:.2%}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

    logger.info(f"Training complete. Best model saved to {save_path}")
    logger.info(f"Best val loss: {best_val_loss:.4f}")
    return model


if __name__ == "__main__":
    train_model()
