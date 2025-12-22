# HAAM Hybrid Emotion Model - Complete Guide

## Table of Contents
1. Overview
2. Model Architecture
3. Installation
4. Usage
5. API Reference
6. Performance Benchmarks
7. Troubleshooting
8. Citation

## 1. Overview

The HAAM Hybrid Fusion Network is a production-ready emotion recognition system that combines:
- **Acoustic analysis**: Pitch, speech rate, stress indicators
- **Text sentiment**: Natural language emotion distribution
- **Hybrid training**: Both acted (CREMA-D) and conversational (IEMOCAP) speech

### Key Features
- ✅ Real-time inference (<600ms per audio)
- ✅ REST API for easy integration
- ✅ 54.5% accuracy on 5-class problem
- ✅ Robust to dataset shift (works on both acted and spontaneous speech)

### Performance Summary
| Metric | Value |
|--------|-------|
| Test Accuracy | 54.5% |
| Weighted F1 | 0.544 |
| Macro F1 | 0.532 |
| Training Samples | 12,271 |
| Supported Emotions | 5 (neutral, anger, disgust, fear, sadness) |

## 2. Model Architecture

### Network Structure
```
Input Audio File
    ↓
Feature Extraction
    ├─→ Acoustic Branch (3 features)
    │   └─→ Linear(3 → 64) → BatchNorm → ReLU → Dropout(0.3)
    │
    └─→ Text Branch (5D sentiment)
        └─→ Linear(5 → 32) → BatchNorm → ReLU → Dropout(0.3)
            ↓
        Concatenation (96D representation)
            ↓
        Classifier
            └─→ Linear(96 → 64) → ReLU → Dropout(0.3) → Linear(64 → 5)
            ↓
        Output (5 emotions with probabilities)
```

### Feature Definitions
**Acoustic Features (3D):**
- `pitch_mean`: Average fundamental frequency (Hz)
- `speech_rate_wpm`: Speaking tempo (words per minute), derived from ASR transcript duration
- `agent_stress_score`: Heuristic score based on pitch > 250Hz and rate > 150 WPM

**Text Features (5D):**
- Normalized emotion distribution: [neutral, anger, disgust, fear, sadness]
- Extracted via: Whisper (tiny) ASR → DistilRoBERTa sentiment classifier

## 3. Installation

### Prerequisites
- Python 3.10+
- PyTorch 2.0+

### Step-by-Step Setup
```bash
# Clone repository
git clone https://github.com/Sham1606/HAAM_New.git
cd HAAM_New

# Install dependencies
pip install -r requirements.txt

# Verify installation
python scripts/validate_production_system.py
```

### Required Files
Ensure these model files exist in `models/`:
- `hybrid_fusion_model.pth`
- `hybrid_scaler.pkl`
- `hybrid_encoder.pkl`

## 4. Usage

### Command-Line Inference
```bash
python scripts/test_hybrid_inference.py path/to/audio.wav
```

**Output:**
```
Predicted Emotion: NEUTRAL
Confidence: 62.7%
Transcript: "I understand your concern"
```

### Python API
```python
from src.services.inference import HybridInference

engine = HybridInference()
result = engine.predict('audio.wav')

print(f"Emotion: {result['predicted_emotion']}")
print(f"Confidence: {result['confidence']:.1%}")
```

### REST API
Start the server:
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Make predictions:
```python
import requests

with open('audio.wav', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/predict/emotion',
        files={'audio': f}
    )

result = response.json()
print(result)
```

## 5. API Reference

### Endpoints

#### GET /health
Health check. Returns `{"status": "healthy"}`.

#### GET /api/model/info
Returns model metadata.

#### POST /api/predict/emotion
Predicts emotion from uploaded audio.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: audio file (WAV format, <10MB)

## 6. Performance Benchmarks

### Inference Speed
- Average: ~750ms (CPU, unoptimized) / <500ms (GPU)
- Memory: ~500MB loading

### Accuracy by Dataset
| Dataset | Accuracy | Notes |
|---------|----------|-------|
| CREMA-D (acted) | ~65% | Clean, single-speaker audio |
| IEMOCAP (conversational) | ~48% | Multi-speaker dialogues |
| Mixed test set | 54.5% | Balanced evaluation |

## 7. Troubleshooting

### Model files not found
**Error:** `FileNotFoundError: models/hybrid_fusion_model.pth`
**Solution:** Run `python scripts/train_hybrid_model.py`.

### Audio format errors
**Error:** `librosa.exceptions.AudioFormatError`
**Solution:** Ensure valid WAV/MP3.

## 8. Citation

If you use HAAM in your research, please cite:

```bibtex
@software{haam2025,
  title={HAAM: Hybrid Audio Analysis for Marathon Call Centers},
  year={2025},
  url={https://github.com/Sham1606/HAAM_New}
}
```
