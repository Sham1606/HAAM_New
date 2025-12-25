import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BurnoutDataset(Dataset):
    def __init__(self, features, labels, sequence_length=14):
        self.features = features
        self.labels = labels
        self.seq_len = sequence_length

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        # Return a window of seq_len days
        x = self.features[idx : idx + self.seq_len]
        y = self.labels[idx + self.seq_len - 1] # Target label for the end of sequence
        return torch.FloatTensor(x), torch.FloatTensor([y])

class LSTMRiskPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)
        # Take last time step
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)

def train_model():
    # 1. Load Data
    data_path = Path("results/marathon/agent_features.csv")
    if not data_path.exists():
        logger.error(f"Data not found at {data_path}. Run feature aggregation first.")
        return

    df = pd.read_csv(data_path)
    
    # Feature columns same as in aggregate_features.py
    feature_cols = [
        'total_calls', 'avg_sentiment', 'anger_pct', 'sadness_pct', 
        'fear_pct', 'joy_pct', 'avg_stress_score', 'engagement_score',
        'sentiment_trend_7d', 'anger_trend_7d', 'duration_trend_7d', 'workload_spike'
    ]
    
    # Simulate labels (burnout: 1 if sentiment decline + high stress)
    # In practice, this would be real historical data
    df['burnout_label'] = ((df['sentiment_trend_7d'] < -0.1) & (df['avg_stress_score'] > 0.5)).astype(int)
    
    # 2. Sequential Splitting per Agent
    all_x, all_y = [], []
    for agent_id, group in df.groupby('agent_id'):
        group = group.sort_values('date')
        if len(group) < 15: continue # Need at least 14 days + 1 label
        
        feats = group[feature_cols].values
        labels = group['burnout_label'].values
        
        all_x.append(feats)
        all_y.append(labels)

    # Simple flattening for demonstration (Dataset handles windowing)
    # Ideally we'd keep sequences grouped by agent
    
    # Prepare PyTorch objects
    # This is a simplified training loop for architectural demo
    input_dim = len(feature_cols)
    model = LSTMRiskPredictor(input_dim)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    logger.info(f"Model initialized with input dimension {input_dim}")
    
    # Placeholder for actual training loop
    # For now, we save the architecture and initialize weights
    save_path = Path("saved_models/marathon_risk_predictor.pth")
    save_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), save_path)
    logger.info(f"Burnout predictor initialized and saved to {save_path}")

if __name__ == "__main__":
    train_model()
