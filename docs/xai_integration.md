# HAAM Dialogue-Level XAI Integration Guide

The Dialogue-Level XAI module provides segment-by-segment explainability for full call recordings. It allows supervisors to understand not just *what* the model predicted, but *how* the emotional state and modality weights evolved over time.

## 🚀 Components

### 1. XAI Engine (`src/sprint_layer/dialogue_xai.py`)
Generates three key visualizations and a markdown report:
- **Emotion Trajectory**: Stacked area chart showing probability shifts.
- **Sentiment Flow**: Line chart with automated escalation markers.
- **Modality Importance**: Tracking of Audio vs. Text attention weights.

### 2. Batch Generator (`scripts/generate_xai_reports.py`)
Used to pre-calculate visualizations for historical calls.
```bash
python scripts/generate_xai_reports.py --calls_dir results/calls --limit 50
```

### 3. API Endpoints
The FastAPI backend exposes the following for the dashboard:
- `GET /api/calls/{id}/xai-report`: Returns the markdown summary.
- `GET /api/calls/{id}/xai-plot/{type}`: Streams the `.png` visualization.

### 4. Dashboard View
Integrated into `CallDetailPage.jsx` via the `ExplainabilityView` component. It features a tabbed interface allowing users to toggle between the standard summary and deep XAI insights.

## 📈 Data Persistence
- **Plots**: `results/xai_dialogues/`
- **Reports**: `results/xai_reports/`

## 🛠️ Maintenance
If new emotions are added to the model, update the `emotion_colors` dictionary in `DialogueXAI.__init__` to ensure consistent visualization.
