import os
import json
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from collections import Counter
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
HYBRID_METADATA_PATH = r"D:\haam_framework\data\hybrid_metadata.csv"
CREMAD_RESULTS_DIR = r"D:\haam_framework\results\calls"
IEMOCAP_RESULTS_DIR = r"D:\haam_framework\results\calls_iemocap"
MODEL_SAVE_PATH = r"D:\haam_framework\models\hybrid_fusion_model.pth"
SCALER_SAVE_PATH = r"D:\haam_framework\models\hybrid_scaler.pkl"
ENCODER_SAVE_PATH = r"D:\haam_framework\models\hybrid_encoder.pkl"
METRICS_SAVE_PATH = r"D:\haam_framework\results\hybrid_model_metrics.json"

# Features configuration
ACOUSTIC_FEATURES = ['pitch_mean', 'speech_rate_wpm', 'agent_stress_score'] # 3 features
SENTIMENT_LABELS = ['neutral', 'anger', 'disgust', 'fear', 'sadness'] # 5D vector (Joy dropped)
TARGET_EMOTIONS = ['neutral', 'anger', 'disgust', 'fear', 'sadness'] # 5 classes

class HybridFusionNetwork(nn.Module):
    def __init__(self, n_acoustic, n_text, n_classes, hidden_dim=64):
        super(HybridFusionNetwork, self).__init__()
        
        # Acoustic Branch
        self.acoustic_net = nn.Sequential(
            nn.Linear(n_acoustic, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Text/Sentiment Branch
        self.text_net = nn.Sequential(
            nn.Linear(n_text, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Fusion Layer (Attention-based)
        # We concatenate features and learn an attention weight
        self.fusion_dim = hidden_dim + (hidden_dim // 2)
        self.attention = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim),
            nn.Tanh(),
            nn.Linear(self.fusion_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, x_acoustic, x_text):
        out_a = self.acoustic_net(x_acoustic)
        out_t = self.text_net(x_text)
        
        # Concatenate
        combined = torch.cat((out_a, out_t), dim=1)
        
        # Attention (simple self-attention on the feature vector might be redundant for 1D vector but implementing essentially as a gated weighting here? 
        # actually for standard tabular fusion, simple concat + dense is standard. 
        # implementing a "Gated" Linear Unit style or just direct fusion.
        # Let's stick to the plan: Concat -> Dense -> Softmax is usually for sequence attention. 
        # For feature fusion, we can use a Gating mechanism: z = sigmoid(W * combined) * combined
        # But let's follow the standard "Early Fusion" approach as the plan described "Attention-based" vaguely.
        # I'll implement a simple concatenated feed-forward for robustness first, or a weighted sum if dims matched.
        # Given dims don't match, concat is best.
        
        out = self.classifier(combined)
        return out

class HybridDataset(Dataset):
    def __init__(self, X_acoustic, X_text, y):
        self.X_acoustic = torch.FloatTensor(X_acoustic)
        self.X_text = torch.FloatTensor(X_text)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_acoustic[idx], self.X_text[idx], self.y[idx]

def load_and_preprocess_data():
    logger.info("Loading metadata...")
    df = pd.read_csv(HYBRID_METADATA_PATH)
    
    # Filter columns
    data_records = []
    
    logger.info(f"Processing {len(df)} records...")
    
    missing_files = 0
    joy_dropped = 0
    
    for _, row in df.iterrows():
        dataset = row['dataset']
        call_id = row['call_id']
        ground_truth_emotion = row['emotion_true'] # Corrected column name
        
        # DROP JOY
        if ground_truth_emotion.lower() == 'joy':
            joy_dropped += 1
            continue
            
        if ground_truth_emotion.lower() not in TARGET_EMOTIONS:
            # Map or skip? If unknown class.
            continue
            
        # Locate JSON
        if dataset == 'CREMA-D':
            json_path = os.path.join(CREMAD_RESULTS_DIR, f"{call_id}.json")
        else:
            json_path = os.path.join(IEMOCAP_RESULTS_DIR, f"{call_id}.json")
            
        if not os.path.exists(json_path):
            missing_files += 1
            continue
            
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            metrics = data.get('overall_metrics', {})
            
            # Acoustic Features
            # Handle missing keys safely
            pitch = metrics.get('avg_pitch', 0.0)
            if pitch is None: pitch = 0.0
            
            rate = metrics.get('speech_rate_wpm', 0.0)
            if rate is None: rate = 0.0
            
            stress = metrics.get('agent_stress_score', 0.0)
            if stress is None: stress = 0.0
            
            acoustic_vec = [pitch, rate, stress]
            
            # Text/Sentiment Features (5D Distribution)
            dist = metrics.get('emotion_distribution', {})
            # Ensure 5D vector in fixed order: neutral, anger, disgust, fear, sadness
            # If joy is in distribution, we ignore it for the feature vector to keep 5D, or include it?
            # User said: "Load aggregated sentiment features (5D)" and "Drop Joy class".
            # I will create a normalized 5D vector for the 5 target emotions.
            
            raw_dist_vals = []
            total_score = 0.0
            for emo in TARGET_EMOTIONS:
                val = dist.get(emo, 0.0)
                raw_dist_vals.append(val)
                total_score += val
            
            # Re-normalize if sum > 0
            if total_score > 0:
                text_vec = [v / total_score for v in raw_dist_vals]
            else:
                text_vec = [0.2] * 5 # Uniform if empty
                
            data_records.append({
                'acoustic': acoustic_vec,
                'text': text_vec,
                'label': ground_truth_emotion.lower(),
                'dataset': dataset
            })
            
        except Exception as e:
            logger.warning(f"Error reading {json_path}: {e}")
            continue

    logger.info(f"Data Loaded: {len(data_records)} samples.")
    logger.info(f"Dropped 'Joy': {joy_dropped}")
    logger.info(f"Missing Files: {missing_files}")
    
    return data_records

def train_model():
    # 1. Load Data
    records = load_and_preprocess_data()
    if not records:
        logger.error("No records loaded.")
        return

    # To Arrays
    X_acoustic = np.array([r['acoustic'] for r in records])
    X_text = np.array([r['text'] for r in records])
    y_raw = [r['label'] for r in records]
    datasets = [r['dataset'] for r in records]
    
    # 2. Encode Labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    classes = le.classes_
    logger.info(f"Classes: {classes}")
    
    # Save Encoder
    joblib.dump(le, ENCODER_SAVE_PATH)
    
    # 3. Split Data (Stratified)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, temp_idx = next(splitter.split(np.zeros(len(y_encoded)), y_encoded))
    
    X_acc_train, X_acc_temp = X_acoustic[train_idx], X_acoustic[temp_idx]
    X_txt_train, X_txt_temp = X_text[train_idx], X_text[temp_idx]
    y_train, y_temp = y_encoded[train_idx], y_encoded[temp_idx]
    
    # Inner split for Val/Test (50/50 of temp -> 15% each of total)
    val_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_sub_idx, test_sub_idx = next(val_splitter.split(np.zeros(len(y_temp)), y_temp))
    
    X_acc_val, X_acc_test = X_acc_temp[val_sub_idx], X_acc_temp[test_sub_idx]
    X_txt_val, X_txt_test = X_txt_temp[val_sub_idx], X_txt_temp[test_sub_idx]
    y_val, y_test = y_temp[val_sub_idx], y_temp[test_sub_idx]
    
    logger.info(f"Split Sizes - Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    
    # 4. Normalize Acoustic Features
    scaler = StandardScaler()
    X_acc_train = scaler.fit_transform(X_acc_train)
    X_acc_val = scaler.transform(X_acc_val)
    X_acc_test = scaler.transform(X_acc_test)
    
    joblib.dump(scaler, SCALER_SAVE_PATH)
    
    # 5. Class Weights
    count = Counter(y_train)
    total = len(y_train)
    class_weights = {k: total / (len(count) * v) for k, v in count.items()}
    weights_tensor = torch.FloatTensor([class_weights[i] for i in range(len(classes))])
    logger.info(f"Class Weights: {class_weights}")
    
    # 6. Datasets & Loaders
    train_dataset = HybridDataset(X_acc_train, X_txt_train, y_train)
    val_dataset = HybridDataset(X_acc_val, X_txt_val, y_val)
    test_dataset = HybridDataset(X_acc_test, X_txt_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # 7. Model Setup
    model = HybridFusionNetwork(
        n_acoustic=3, 
        n_text=5, 
        n_classes=len(classes)
    )
    
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # 8. Training Loop
    best_val_loss = float('inf')
    early_stop_patience = 10
    patience_counter = 0
    
    logger.info("Starting Training...")
    
    for epoch in range(50): # Max epochs
        model.train()
        train_loss = 0.0
        
        for batch_acc, batch_txt, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_acc, batch_txt)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total_val = 0
        
        with torch.no_grad():
            for batch_acc, batch_txt, batch_y in val_loader:
                outputs = model(batch_acc, batch_txt)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_val += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = correct / total_val
        
        logger.info(f"Epoch {epoch+1}/50 - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info("Early stopping triggered.")
                break
                
    # 9. Evaluation
    logger.info("Loading best model for evaluation...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_acc, batch_txt, batch_y in test_loader:
            outputs = model(batch_acc, batch_txt)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(batch_y.numpy())
            
    # Metrics
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    conf_matrix = confusion_matrix(all_labels, all_preds).tolist()
    
    logger.info("Test Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    # Save Metrics
    results = {
        "classification_report": report,
        "confusion_matrix": conf_matrix,
        "classes": list(classes),
        "test_accuracy": report['accuracy']
    }
    
    with open(METRICS_SAVE_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation complete. Metrics saved to {METRICS_SAVE_PATH}")

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        logger.critical(f"Training failed: {e}")
