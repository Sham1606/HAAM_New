# HAAM Framework

**Human Affect Analysis & Monitoring** — An AI-powered framework for real-time call center emotion detection, agent stress monitoring, and multi-user management with role-based dashboards.

---

## Key Features

- **Deep Multimodal Fusion**: Analyzes audio acoustics + transcript text simultaneously via an attention-based hybrid model (`ImprovedHybridModel`).
- **Sprint Layer (Real-Time)**: Processes live audio streams via browser microphone, transcribes with Whisper, and predicts instantaneous emotion/sentiment.
- **Marathon Layer (Long-Term)**: Aggregates historical call data into time-series to score and track agent burnout risks over time.
- **Explainable AI (XAI)**: Captum Integrated Gradients + SHAP attention visualizations to explain *why* an emotion or risk was predicted.
- **SQLite + JWT Authentication**: Multi-user system with bcrypt password hashing, role-based access control (Admin vs Agent), and 24h JWT tokens.
- **Real-Time Agent Grid (Admin)**: WebSocket-powered live monitoring — see each agent's emotion, transcript, AI coaching feedback, and risk in real-time as they handle calls.
- **Agent Personal Dashboard**: Agents see their own stats, emotion profile, AI coaching feedback, and personal call history.
- **Automated Alerts**: Email and Slack integrations to notify supervisors when agent stress passes critical thresholds.

---

## Architecture Overview

```
Audio → Sprint Layer → Emotion + Transcript → Marathon Layer → Risk Scores → Dashboard
                ↓                                                      ↓
         Live WebSocket                                        Agent Grid (Admin)
     (browser mic → server)                              (real-time emotion cards)
```

### Sprint Layer (`src/sprint_layer`)
- **Input**: Audio file (WAV/MP3) or live microphone stream
- **Process**: 
  - `librosa` for acoustic features (pitch, speech rate, MFCCs)
  - `openai-whisper` (tiny model) for transcription
  - `j-hartmann/emotion-english-distilroberta-base` for text emotion
  - `ImprovedHybridModel` — attention-based fusion of acoustic + text features
- **Output**: JSON file (`results/calls/call_{id}.json`) + live WebSocket stream

### Marathon Layer (`src/marathon_layer`)
- **Input**: Collection of call JSONs
- **Process**:
  - `aggregate_features.py`: Daily metrics + rolling 7-day trends
  - `risk_scoring.py`: Multi-factor risk evaluation (sentiment decline, stress spikes, anger frequency)
- **Output**: Risk profiles CSV + actionable recommendations

### Authentication System (`src/api/auth.py`, `database.py`)
- **SQLite** database (`data/agents.db`) with SQLAlchemy ORM
- **bcrypt** password hashing via passlib
- **JWT** (HS256, 24h expiry) via python-jose
- **Role-based access**: Admin sees all agents + grid; Agent sees own dashboard only
- **Login rate limiting**: 5 attempts/minute per IP

---

## Installation & Quick Start

### Prerequisites
- Python 3.8+
- Node.js 14+
- (Optional) FFmpeg for processing new audio

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd HAAM_New
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

4. **Run database migration** (creates SQLite DB + seeds demo users):
   ```bash
   python migrate.py
   ```

### Running the Application

**1. Start the API (Terminal 1)**
```bash
cd src
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

**2. Start the Dashboard (Terminal 2)**
```bash
cd src/dashboard
npm start
```

**3. Open the app**: [http://localhost:3000](http://localhost:3000)

### Demo Credentials

| Username | Password | Role  | Dashboard View |
|----------|----------|-------|----------------|
| admin    | admin123 | Admin | Calls, Agents, Analytics, Agent Grid, Alerts |
| sham     | pass123  | Agent | My Dashboard, Live Analysis, My Calls, My Feedback |
| priya    | pass123  | Agent | My Dashboard, Live Analysis, My Calls, My Feedback |
| rahul    | pass123  | Agent | My Dashboard, Live Analysis, My Calls, My Feedback |

---

## API Documentation

### Authentication Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/auth/register` | POST | — | Register new agent |
| `/auth/login` | POST | — | Login → JWT token |
| `/auth/logout` | POST | JWT | Set agent offline |
| `/auth/me` | GET | JWT | Current user profile |

### Call Processing & Analysis

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/calls/process` | POST | — | Upload and process audio (async) |
| `/api/calls` | GET | — | List all processed calls |
| `/api/calls/{id}` | GET | — | Detailed call metrics & transcript |
| `/api/calls/{id}/explain` | GET | — | XAI explanation (Captum IG) |

### Agent Management

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/agents` | GET | — | List all agents with stats |
| `/api/agents/me` | GET | JWT | Own profile |
| `/api/agents/status` | GET | Admin | All agents live status |
| `/api/agents/registered` | GET | — | List registered agent accounts |
| `/api/agents/{id}/risk` | GET | — | Agent risk profile |
| `/api/agents/{id}/stats` | GET | — | Aggregated emotion stats |
| `/api/agents/{id}/calls` | GET | — | Agent's call history |
| `/api/agents/{id}/export/csv` | GET | — | Export calls as CSV |
| `/api/agents/{id}/export/pdf` | GET | — | Export calls as PDF report |

### Status & Feedback

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/status/heartbeat/{id}` | POST | JWT | Keep agent online |
| `/api/status/update/{id}` | POST | JWT | Change status (online/on-call/offline) |
| `/api/feedback/predict` | POST | — | Emotion + stress → AI coaching text |

### WebSocket Streams

| URL | Description |
|-----|-------------|
| `ws://localhost:8000/ws/mic-stream?agent_id=xxx` | Browser mic → real-time emotion (also broadcasts to admin grid) |
| `ws://localhost:8000/ws/agents` | Admin: live agent status updates |
| `ws://localhost:8000/ws/live` | Server-side live pipeline |

### Pipeline Triggers

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/marathon/aggregate` | POST | Trigger feature aggregation |
| `/api/marathon/update-risk` | POST | Trigger risk scoring |

---

## Dashboard

### Admin View
- **Calls**: Filterable database of all processed calls with audio upload
- **Agents**: Risk profiles, stress history, sentiment trends per agent
- **Analytics**: System-wide performance, dataset distribution, global emotion breakdown
- **Agent Grid**: Real-time WebSocket-powered monitoring — live emotion prediction, transcript, AI coaching feedback, session risk/trend for each agent on-call
- **Alerts**: Configure risk thresholds and Email/Slack notification endpoints

### Agent View
- **My Dashboard**: Personal stats, emotion profile bars, recent calls, quick actions
- **Live Analysis**: Stream microphone for real-time emotion detection (results also visible to admin in Agent Grid)
- **My Calls**: Personal call history with emotion color-coding
- **My Feedback**: AI-generated coaching based on emotional patterns and stress levels

---

## Real-Time Agent Monitoring Flow

```
Agent Browser                  Server                         Admin Browser
    │                            │                                │
    │──[mic audio chunks]──────→│                                │
    │                            │──predict_array()──→ Inference  │
    │                            │←── emotion + transcript ──────│
    │←─[turn_result JSON]───────│                                │
    │                            │──status_manager.broadcast()──→│
    │                            │                                │
    │                            │    AgentCard updates live:     │
    │                            │    😠 anger 87%               │
    │                            │    "I am very frustrated"     │
    │                            │    🔶 Elevated frustration... │
```

---

## Directory Structure

```
HAAM_New/
├── migrate.py                # DB migration + seed users
├── requirements.txt          # Python dependencies
├── data/
│   └── agents.db             # SQLite database (auto-created)
├── saved_models/             # Trained model weights
│   └── hybrid_fusion_model.pth
├── results/
│   ├── calls/                # Sprint layer output JSONs
│   ├── calls_cremad/         # CREMA-D benchmark results
│   ├── calls_iemocap/        # IEMOCAP benchmark results
│   ├── marathon/             # Risk profiles & agent features
│   └── xai_reports/          # XAI explanations
├── src/
│   ├── api/                  # FastAPI backend
│   │   ├── app.py            # Main application (endpoints + WebSockets)
│   │   ├── auth.py           # JWT authentication + role guards
│   │   ├── database.py       # SQLAlchemy ORM models (agents, sessions)
│   │   ├── crud.py           # DB CRUD operations
│   │   ├── models.py         # Pydantic response models
│   │   └── websocket_manager.py  # Agent status broadcasting + feedback generator
│   ├── dashboard/            # React frontend
│   │   └── src/
│   │       ├── App.js        # Role-based routing (admin vs agent)
│   │       ├── services/
│   │       │   ├── AuthContext.js   # JWT auth context + useAuth hook
│   │       │   └── api.js          # Axios API client
│   │       ├── components/
│   │       │   └── Layout/Navbar.jsx  # Role-based navigation
│   │       └── pages/
│   │           ├── LoginPage.jsx           # Auth login page
│   │           ├── AgentDashboardPage.jsx  # Agent: personal dashboard
│   │           ├── LiveAnalysisPage.jsx    # Real-time mic analysis
│   │           ├── MyCallsPage.jsx         # Agent: own calls
│   │           ├── MyFeedbackPage.jsx      # Agent: AI coaching
│   │           ├── AgentGridPage.jsx       # Admin: live monitoring grid
│   │           ├── CallsListPage.jsx       # Admin: all calls
│   │           ├── AgentRiskPage.jsx       # Admin: risk profiles
│   │           ├── AnalyticsPage.jsx       # Admin: analytics overview
│   │           ├── AlertsSettingsPage.jsx  # Admin: alert config
│   │           └── CallDetailPage.jsx      # Shared: call detail + XAI
│   ├── features/             # Feature extractors
│   │   ├── improved_acoustic.py    # 20-dim acoustic features
│   │   └── emotion_text.py         # DistilRoBERTa emotion + embedding
│   ├── inference/            # Live prediction
│   │   └── live_predictor.py
│   ├── models/               # Neural network architectures
│   │   └── improved_hybrid_model.py  # Attention fusion network
│   ├── services/             # Core services
│   │   ├── inference.py      # HybridInference engine
│   │   ├── live_pipeline.py  # WebSocket live streaming pipeline
│   │   └── alerts.py         # Email/Slack alert service
│   ├── sprint_layer/         # Per-call analysis
│   │   └── run_sprint_pipeline.py
│   ├── marathon_layer/       # Cross-call trends
│   │   ├── aggregate_features.py
│   │   └── risk_scoring.py
│   ├── xai/                  # Explainability
│   │   └── xai_explainer.py
│   └── training/             # Model training scripts
└── docs/                     # Reports & documentation
```

---

## Hybrid Emotion Model

The core model is a **Multi-modal Attention Fusion Network** trained on CREMA-D + IEMOCAP:

- **Architecture**: Fuses 20-dim acoustic features (librosa) with 768-dim text embeddings (DistilRoBERTa) via learned attention weights
- **Performance**: ~78% accuracy (5-class: neutral, anger, disgust, fear, sadness)
- **Inference**: Real-time (<1s per utterance with Whisper tiny)
- **XAI**: Captum Integrated Gradients for per-feature attribution

### Quick Inference
```bash
python scripts/test_hybrid_inference.py data/sample.wav
```

---

## Security

| Feature | Implementation |
|---------|---------------|
| Password Hashing | bcrypt via passlib |
| Authentication | JWT (HS256, 24h expiry) |
| Login Rate Limit | 5 attempts/min/IP |
| Role-Based Access | `require_auth` (any user) / `require_admin` (admin only) |
| Agent Isolation | Agents can only access/modify their own data |
| Database | SQLite with WAL mode + foreign key enforcement |

---

## Validation & Benchmarking

```bash
# 1. Prepare CREMA-D data
python scripts/prepare_cremad_data.py

# 2. Run pipeline (15-20 min)
python scripts/run_cremad_pipeline.py

# 3. Generate report
python scripts/validate_cremad_results.py
```

**Outputs**: Confusion matrix, accuracy chart, detailed report in `docs/`.

---

## Technologies

| Category | Stack |
|----------|-------|
| Backend | FastAPI, SQLAlchemy, SQLite, Uvicorn |
| Auth | JWT (python-jose), bcrypt (passlib) |
| ML/DL | PyTorch, HuggingFace Transformers, Whisper |
| Audio | librosa, sounddevice, webrtcvad |
| Frontend | React, Recharts, Lucide Icons, Tailwind CSS |
| XAI | Captum (Integrated Gradients) |
| Alerts | SMTP Email, Slack Webhooks |

---

## License

MIT License

---

## Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push to branch (`git push origin feature/my-feature`)
5. Open a Pull Request
