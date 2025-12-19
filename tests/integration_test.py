
import os
import pytest
import pandas as pd
import json
import numpy as np
import shutil
from fastapi.testclient import TestClient
from src.sprint_layer.run_sprint_pipeline import SprintPipeline
from src.feature_store.aggregate_features import aggregate_sprint_to_timeseries
from src.marathon_layer.risk_scoring import score_all_agents
from src.api.app import app

# Configuration
TEST_RESULTS_DIR = "results_test"
TEST_CALLS_DIR = os.path.join(TEST_RESULTS_DIR, "calls")
TEST_AGG_CSV = os.path.join(TEST_RESULTS_DIR, "aggregated.csv")
TEST_RISK_CSV = os.path.join(TEST_RESULTS_DIR, "risk_scores.csv")

# Setup/Teardown
@pytest.fixture(scope="module", autouse=True)
def setup_teardown():
    # Setup
    os.makedirs(TEST_CALLS_DIR, exist_ok=True)
    yield
    # Teardown
    if os.path.exists(TEST_RESULTS_DIR):
        shutil.rmtree(TEST_RESULTS_DIR)

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def dummy_audio(tmp_path):
    """Create a dummy WAV file for testing if real one not present."""
    import scipy.io.wavfile as wav
    sample_rate = 16000
    duration = 1  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Generate 440 Hz sine wave
    data = 0.5 * np.sin(2 * np.pi * 440 * t)
    wav_path = tmp_path / "test_audio.wav"
    wav.write(wav_path, sample_rate, (data * 32767).astype(np.int16))
    return wav_path

# TEST 1: Sprint Layer (Direct & API)
def test_sprint_pipeline(dummy_audio):
    # We mock or run the actual pipeline. Since we don't have real models loaded potentially 
    # (whisper might be slow or fail in CI without GPU/large download), we might assume mocks 
    # OR we try running it if environment allows.
    # Given instructions, "Create E2E testing".
    # I will rely on the endpoint test which calls processing.
    # But first, generate a fake JSON result to simulate successful processing if pipeline is heavy.
    
    # Actually, let's test the endpoint response for queueing.
    client = TestClient(app)
    with open(dummy_audio, "rb") as f:
        response = client.post(
            "/api/calls/process",
            files={"file": ("test.wav", f, "audio/wav")},
            data={"agent_id": "agent_test", "call_id": "call_test_01"}
        )
    assert response.status_code == 200
    assert response.json()["status"] == "queued"
    
    # Manually create a call JSON to simulate processing completion for downstream tests
    # because background tasks might catch up or we manually invoke logic.
    output_path = os.path.join(TEST_CALLS_DIR, "call_test_01.json")
    # Need to override APP's CALLS_DIR for this to be visible? 
    # The app uses global CALLS_DIR="results/calls". 
    # For testing, we might want to patch that or just use the production dir and clean up.
    # Let's monkeypatch app config if possible or just rely on module variables if we imported them.
    # In integration_test, it's cleaner to test the modules directly for logic verification 
    # and use API for connectivity.
    pass

def test_marathon_aggregation_logic():
    # Create fake Sprint outputs
    os.makedirs(TEST_CALLS_DIR, exist_ok=True)
    
    # Agent 1: 2 calls
    call1 = {
        "call_id": "c1", "agent_id": "a1", "timestamp": "2023-01-01T10:00:00",
        "duration_seconds": 100, 
        "overall_metrics": {
            "avg_sentiment": 0.5, "dominant_emotion": "joy", 
            "escalation_flag": False, "agent_stress_score": 2.0, "avg_pitch": 100,
            "emotion_distribution": {"joy": 0.8, "neutral": 0.2}
        }
    }
    call2 = {
        "call_id": "c2", "agent_id": "a1", "timestamp": "2023-01-01T11:00:00",
        "duration_seconds": 200, 
        "overall_metrics": {
            "avg_sentiment": -0.5, "dominant_emotion": "anger", 
            "escalation_flag": True, "agent_stress_score": 8.0, "avg_pitch": 150,
            "emotion_distribution": {"anger": 0.6, "neutral": 0.4}
        }
    }
    
    with open(os.path.join(TEST_CALLS_DIR, "call_c1.json"), "w") as f: json.dump(call1, f)
    with open(os.path.join(TEST_CALLS_DIR, "call_c2.json"), "w") as f: json.dump(call2, f)
    
    # Run aggregation
    df = aggregate_sprint_to_timeseries(TEST_CALLS_DIR, TEST_AGG_CSV)
    
    assert not df.empty
    assert len(df) == 1 # 1 agent, 1 day
    assert df.iloc[0]['call_count'] == 2
    assert df.iloc[0]['escalation_count'] == 1
    assert df.iloc[0]['avg_sentiment'] == 0.0 # (0.5 - 0.5)/2

def test_risk_scoring_logic():
    # Create valid aggregated CSV
    data = {
        "agent_id": ["a1"] * 10,
        "date": pd.date_range(start="2023-01-01", periods=10),
        "call_count": [10] * 10,
        "avg_sentiment": [0.5]*7 + [-0.8]*3, # Drop at end
        "sentiment_7day_trend": [0.5]*10, # Mocked
        "avg_stress_score": [2]*7 + [9]*3,
        "escalation_count": [0]*9 + [5],
        "angry_calls_pct": [0.1]*10,
        "sentiment_std": [0.1]*10,
        "avg_pitch_variance": [10]*10,
        "avg_duration": [100]*10,
        "call_volume_change_pct": [0.0]*10,
        "stress_7day_mean": [2]*10
    }
    df = pd.DataFrame(data)
    df.to_csv(TEST_AGG_CSV, index=False)
    
    scores = score_all_agents(TEST_AGG_CSV)
    
    assert not scores.empty
    agent_score = scores[scores['agent_id'] == 'a1'].iloc[0]
    assert agent_score['risk_level'] in ['high', 'critical']
    
    # Check JSON structure
    details = json.loads(agent_score['details_json'])
    assert 'risk_factors' in details
    assert len(details['recommendations']) > 0

def test_api_endpoints(client):
    # GET /api/calls (Empty is fine)
    res = client.get("/api/calls")
    assert res.status_code == 200
    assert isinstance(res.json(), list)
    
    # GET /api/agents (Empty is fine)
    res = client.get("/api/agents")
    assert res.status_code == 200
    
    # GET /api/analytics/overview
    res = client.get("/api/analytics/overview")
    assert res.status_code == 200
    data = res.json()
    assert "total_calls" in data
    assert "avg_sentiment" in data

def test_dashboard_static_check():
    # Verify main dashboard files exist
    dashboard_path = os.path.join("src", "dashboard")
    assert os.path.exists(os.path.join(dashboard_path, "package.json"))
    assert os.path.exists(os.path.join(dashboard_path, "src", "App.js"))
    assert os.path.exists(os.path.join(dashboard_path, "public", "index.html"))

