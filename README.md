# HAAM Framework

**Human-Agent-Action-Management** (HAAM) is an agentic coding framework for call center audio analysis, feature aggregation, risk scoring, and visualization.

## Features
- **Sprint Layer**: Processes audio calls, transcribes text, and extracts sentiment/emotions.
- **Marathon Layer**: Aggregates data into time-series, calculates rolling trends, and scores burnout risk.
- **API**: FastAPI backend for real-time processing and data retrieval.
- **Dashboard**: React-based UI for monitoring calls and agent risk profiles.

---

## Installation & Quick Start

This repository is pre-configured with **fully trained models** and **pre-calculated call data**. There is no need to retrain models or run heavy data processing pipelines.

### Prerequisites
- Python 3.8+
- Node.js 14+
- (Optional) FFmpeg if you plan to process *new* audio live.

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd haam_framework
   ```

2. **Install Python backend dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install React dashboard dependencies**:
   ```bash
   cd src/dashboard
   npm install
   cd ../..
   ```

### Running the Application

To load the pre-processed dashboard with all historical calls and agent risk profiles, simply start the backend and frontend.

**1. Start the API (Terminal 1)**
```bash
uvicorn src.api.app:app --reload --port 8000
```

**2. Start the Dashboard (Terminal 2)**
```bash
cd src/dashboard
npm start
```
Access the dashboard at [http://localhost:3000](http://localhost:3000). All data will be perfectly loaded.

### Processing New Calls (Optional)
If you wish to test the live inference pipeline with a new `.wav` file:
```bash
curl -X POST http://localhost:8000/api/calls/process \
  -F "file=@sample.wav" \
  -F "agent_id=agent_01" \
  -F "call_id=call_001"
```

---

## Architecture

### Sprint Layer (`src/sprint_layer`)
- **Input**: Audio file (WAV/MP3)
- **Process**: 
  - `librosa` for acoustic features (pitch, speech rate).
  - `openai-whisper` for transcription.
  - Transformer models for Sentiment & Emotion analysis.
- **Output**: JSON file (`results/calls/call_{id}.json`).

### Marathon Layer (`src/marathon_layer`)
- **Input**: Collection of call JSONs.
- **Process**:
  - `aggregate_features.py`: Aggregates daily metrics and calculates rolling 7-day trends.
  - `risk_scoring.py`: Evaluates risk factors (Sentiment Decline, Stress, Anger, etc.).
- **Output**: Risk Scores CSV and actionable recommendations.

### Data Flow
`Audio` -> **Sprint Pipeline** -> `JSON` -> **Feature Aggregation** -> `Time-Series CSV` -> **Risk Scoring** -> `Risk Profiles` -> **API/Dashboard**

---

## API Documentation

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/calls/process` | POST | Upload and process audio (Async). |
| `/api/calls` | GET | List processed calls. |
| `/api/calls/{id}` | GET | Get detailed metrics & transcript. |
| `/api/agents` | GET | List agents stats. |
| `/api/agents/{id}/risk` | GET | Get specific agent risk profile. |
| `/api/marathon/aggregate` | POST | Trigger feature aggregation. |
| `/api/marathon/update-risk` | POST | Trigger risk scoring. |

---

## Dashboard

### Pages
- **Calls List**: Filterable table of all processed calls. Upload interface.
- **Call Detail**: Interactive transcript with sentiment analysis per segment. Charts for emotion timeline.
- **Agent Risk**: High-level view of agent burnout risk. 
- **Analytics**: System-wide overview of call volume, sentiment trends, and emotion distribution.

---

## Development

### Running Tests
Run the integration suite:
```bash
pytest tests/integration_test.py
```

### Directory Structure
```
src/
├── api/             # FastAPI App
├── dashboard/       # React App
├── feature_store/   # Aggregation Logic
├── marathon_layer/  # Risk Scoring
└── sprint_layer/    # Audio Processing
tests/               # Integration Tests
results/             # Output Directory
```

## Deployment with Docker

1. **Build and Run**:
   ```bash
   docker-compose up --build
   ```

2. **Access**:
   - API: `http://localhost:8000`
   - Dashboard: `http://localhost:3000` (mapped port)

---

## 🏗️ Core Architecture

HAAM employs a **Unified Fusion Model** combining audio + text at multiple stages:

1. **Intra-Call Analysis ("Sprint")**

   * Inputs: Raw audio (wav) + call transcript
   * Models:

     * Transformer NLP (BERT, RoBERTa) → intent & sentiment
     * CNN/LSTM acoustic models → pitch, tone, prosody
   * Fusion via attention to link **vocal cues** with specific words
   * Outputs: sentiment, emotion category, call reason, escalation flags
   * XAI: Highlights words + audio segments driving predictions

2. **Cross-Call Trend Analysis ("Marathon")**

   * Inputs: Aggregated features across multiple calls
   * Temporal models (LSTM, temporal CNNs, time-series analysis)
   * Detects **burnout risks, disengagement, or workload imbalances**
   * Explainable alerts (e.g., "increase in call duration by 25% over 2 weeks")

---

## ⚙️ End-to-End Workflow

1. Audio & transcript capture
2. Preprocessing (noise reduction, diarization, ASR)
3. Feature extraction (lexical + acoustic)
4. Real-time multimodal sentiment/emotion detection
5. Dashboard summarization
6. Cross-call trend & risk alerts with **XAI evidence**

---

## 🔬 Algorithms & Techniques

* **NLP**: Transformer-based contextual embeddings (BERT, RoBERTa)
* **Audio Models**: CNN/LSTM for acoustic features (pitch, MFCCs, prosody)
* **Fusion**: Attention-based multimodal feature linking
* **XAI**: Captum, SHAP, LIME for transparency
* **Time-Series**: Rolling statistics, temporal deep learning

---

## 📊 Training & Evaluation

* **Data Requirements**:

  * Raw call audio + transcripts + metadata (agent ID, time, call type)
* **Preprocessing**:

  * Speech enhancement, ASR (Wav2Vec 2.0, DeepSpeech), diarization
* **Metrics**:

  * Sentiment/emotion → Accuracy, F1, CCC
  * Burnout/trend → ROC-AUC, Precision\@K
  * Real-time latency → ms per call

---

## 🛠️ Tools & Frameworks

* **Deep Learning**: PyTorch, TensorFlow, HuggingFace Transformers
* **Speech Processing**: Librosa, Praat, OpenSMILE
* **ASR**: Wav2Vec 2.0, DeepSpeech
* **Visualization**: Dash, Plotly, React
* **Databases**: SQL/NoSQL
* **XAI**: Captum, SHAP, LIME

---

## 🚀 Deployment Strategy

* **Microservices** architecture for scalable, real-time ingestion
* **APIs** for integration with dashboards & analytics platforms
* **Cloud-based training** with privacy protection (data anonymization)
* **Multilingual & domain-adaptable** via transfer learning

---

## 📈 Comparison with Existing Approaches

| Aspect            | Legacy Approaches          | HAAM                          |
| ----------------- | -------------------------- | ----------------------------- |
| Input Modality    | Text-only                  | Audio + Text (fusion)         |
| Emotion Nuance    | Misses sarcasm/subtle cues | Captures nuance via attention |
| Coverage          | Manual, small %            | 100% of calls, real-time      |
| Trends            | Not tracked                | Longitudinal analytics        |
| Explainability    | Black-box scores           | Transparent with XAI          |
| Burnout Detection | Not supported              | Yes, trend-based              |
| Scalability       | Human-limited              | Automated, cloud-ready        |

---

## ✨ Key Contributions

* **Unified multimodal analysis** (text + audio)
* **Hierarchical attention** for both call-level & agent-level insights
* **Agent-centric monitoring** for burnout/disengagement detection
* **Explainable AI** for actionable supervisor feedback
* **Scalable & real-time performance** with multilingual adaptability
* **Ethical AI**: Transparency fosters fairness, trust & well-being

---

## 📌 Conclusion

The HAAM framework sets a **new benchmark for emotionally intelligent AI in call centers**. By combining **deep multimodal fusion, hierarchical attention, and explainable predictions**, it empowers organizations to:

✅ Improve customer satisfaction
✅ Reduce agent burnout
✅ Enable proactive, data-driven coaching
✅ Scale seamlessly across industries & languages

---

## 🖼️ Architecture Diagram

*(Insert architecture diagram here — from your `ARCHITECTURE DIAGRAM` section)*


---

## 📜 License

MIT License (or specify if different)

---

# HAAM
THIS REPOSITORY INVOLVES IN A BUILDING AN INNOVATIVE FRAMEWORK FOR AUDIO TRANSCRIPTION AND DETECTION OF CONTEXT AS WELL AS EMOTION IN THE CONVERSATIONT THROUGH AUDIO CALLS

## 🏃 Validation & Benchmarking

The framework includes a validation suite to benchmark emotion detection accuracy against the **CREMA-D** dataset.

### Steps to Run Validation

1. **Prepare Data**:
   This script scans your local CREMA-D dataset, selects balanced samples, and creates a ground truth CSV.
   ```bash
   python scripts/prepare_cremad_data.py
   ```

2. **Run Pipeline**:
   Processes the 90 selected calls through the Sprint and Marathon layers. This may take 15-20 minutes.
   ```bash
   python scripts/run_cremad_pipeline.py
   ```

3. **Generate Report**:
   Creates visualization charts and a detailed text report in the `docs/` directory.
   ```bash
   python scripts/validate_cremad_results.py
   ```

### Outputs
- **Confusion Matrix**: `docs/cremad_confusion_matrix.png`
- **Accuracy Chart**: `docs/cremad_accuracy_chart.png`
- **Detailed Report**: `docs/cremad_detailed_report.txt`


## 🧠 Hybrid Emotion Model Features (New!)

The core of HAAM is now a **Multi-modal Attention Fusion Network** trained on a hybrid dataset (CREMA-D + IEMOCAP).

- **Architecture**: Fuses acoustic features (Pitch, Rate, Stress) with textual sentiment (DistilRoBERTa).
- **Performance**: 78.0% Accuracy (5-class), robust to conversational speech.

- **Inference**: Real-time (<600ms), exposed via REST API.

### Quick Inference
```bash
python scripts/test_hybrid_inference.py data/sample.wav
```

### Documentation
See [HYBRID_MODEL_GUIDE.md](docs/HYBRID_MODEL_GUIDE.md) for full details.
