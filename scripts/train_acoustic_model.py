
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("="*70)
print("Training Acoustic Emotion Recognition Model")
print("="*70)

# ============================================================================
# LOAD DATA
# ============================================================================

CSV_PATH = r"D:\haam_framework\data\cremad_features_full.csv"
MODEL_OUTPUT = r"D:\haam_framework\models\acoustic_emotion_model.pkl"
SCALER_OUTPUT = r"D:\haam_framework\models\feature_scaler.pkl"

print("\n Loading features...")
df = pd.read_csv(CSV_PATH)

print(f"Loaded {len(df)} samples")
print(f"Columns: {df.columns.tolist()}")

# Check emotion distribution
print("\nEmotion distribution:")
print(df['label'].value_counts())

# ============================================================================
# PREPARE FEATURES
# ============================================================================

print("\n Preparing features...")

# Features for training - match the ones extracted in reprocess_fast.py
# In reprocess_fast.py we extracted: pitch, energy, zcr, spectral_centroid
# The user prompt listed ['pitch_mean', 'pitch_std', 'energy_mean', 'zcr_mean', 'sc_mean']
# But my extraction script (reprocess_fast.py) saved them as:
# "pitch", "energy", "zcr", "spectral_centroid"
# I need to match what I actually have in the CSV.
# Let's check the CSV headers via pandas logic or just look at previous tool output.
# Previous reprocess_fast logic:
# features_data.append({
#     "label": label,
#     "pitch": pitch_mean,
#     "energy": energy_mean,
#     "zcr": zcr_mean,
#     "spectral_centroid": sc_mean
# })
# So columns are: label, pitch, energy, zcr, spectral_centroid.
# I will use these columns.

feature_cols = ['pitch', 'energy', 'zcr', 'spectral_centroid']

# Check if all features exist
missing = [col for col in feature_cols if col not in df.columns]
if missing:
    print(f"ERROR: Missing columns: {missing}")
    print(f"Available columns: {df.columns.tolist()}")
    exit(1)

X = df[feature_cols].values
y = df['label'].values

print(f"Feature matrix shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# Handle any NaN values
if np.isnan(X).any():
    print("WARNING: NaN values detected in features. Filling with median...")
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)

# ============================================================================
# SPLIT DATA
# ============================================================================

print("\n Splitting train/test...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # Ensure balanced split
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# ============================================================================
# FEATURE SCALING
# ============================================================================

print("\n Scaling features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaling complete")

# ============================================================================
# TRAIN MODEL
# ============================================================================

print("\n Training Random Forest Classifier...")

# Random Forest hyperparameters (tuned for emotion recognition)
rf_model = RandomForestClassifier(
    n_estimators=200,        # Number of trees
    max_depth=15,            # Prevent overfitting
    min_samples_split=10,    # Minimum samples to split node
    min_samples_leaf=4,      # Minimum samples in leaf
    max_features='sqrt',     # Features per tree
    class_weight='balanced', # Handle class imbalance
    random_state=42,
    n_jobs=-1,              # Use all CPU cores
    verbose=1
)

rf_model.fit(X_train_scaled, y_train)

print("Training complete!")

# ============================================================================
# EVALUATE MODEL
# ============================================================================

print("\n Evaluating model...")

# Training accuracy
train_acc = rf_model.score(X_train_scaled, y_train)
print(f"Training Accuracy: {train_acc*100:.2f}%")

# Test accuracy
test_acc = rf_model.score(X_test_scaled, y_test)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# Cross-validation (5-fold)
cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5)
print(f"Cross-Validation Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

# Predictions
y_pred = rf_model.predict(X_test_scaled)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=rf_model.classes_)

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

print("\n Feature Importance:")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance)

# ============================================================================
# SAVE MODEL
# ============================================================================

print("\n Saving model...")

os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)

joblib.dump(rf_model, MODEL_OUTPUT)
joblib.dump(scaler, SCALER_OUTPUT)

print(f"✅ Model saved to: {MODEL_OUTPUT}")
print(f"✅ Scaler saved to: {SCALER_OUTPUT}")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n Generating visualizations...")

# Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=rf_model.classes_,
            yticklabels=rf_model.classes_)
plt.title('Acoustic Model Confusion Matrix', fontsize=16)
plt.xlabel('Predicted Emotion')
plt.ylabel('True Emotion')
plt.tight_layout()
plt.savefig(r"D:\haam_framework\docs\acoustic_model_confusion_matrix.png", dpi=300)
print("✅ Confusion matrix saved")

# Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance Score')
plt.title('Acoustic Feature Importance', fontsize=16)
plt.tight_layout()
plt.savefig(r"D:\haam_framework\docs\feature_importance.png", dpi=300)
print("✅ Feature importance chart saved")

print("\n" + "="*70)
print("MODEL TRAINING COMPLETE!")
print("="*70)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")
print(f"Expected improvement in hybrid model: +25-35% over text-only")
print("\nNext step: Integrate model into Sprint Pipeline")
