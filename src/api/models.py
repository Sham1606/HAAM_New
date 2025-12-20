
from pydantic import BaseModel
from typing import List, Dict, Optional, Any

class ProcessCallResponse(BaseModel):
    call_id: str
    message: str
    status: str

class SegmentModel(BaseModel):
    start_time: float
    end_time: float
    text: str
    emotion: str
    sentiment_score: float
    pitch_mean: float

class OverallMetricsModel(BaseModel):
    avg_sentiment: float
    dominant_emotion: str
    emotion_distribution: Dict[str, float]
    escalation_flag: bool
    agent_stress_score: float
    speech_rate_wpm: Optional[float] = 0.0
    avg_pitch: float

class CallDetailResponse(BaseModel):
    call_id: str
    agent_id: str
    timestamp: str
    duration_seconds: float
    transcript: str
    segments: List[SegmentModel]
    overall_metrics: OverallMetricsModel

class CallSummaryModel(BaseModel):
    call_id: str
    agent_id: str
    timestamp: str
    avg_sentiment: float
    dominant_emotion: str

class AgentStats(BaseModel):
    agent_id: str
    call_count: int
    avg_sentiment: float
    
class RiskFactor(BaseModel):
    factor: str
    severity: float
    description: str
    contribution: float

class KeyMetrics(BaseModel):
    current_sentiment: float
    baseline_sentiment: float
    current_stress: float

class RiskProfileResponse(BaseModel):
    agent_id: str
    risk_score: float
    risk_level: str
    trend_direction: str
    risk_factors: List[RiskFactor]
    recommendations: List[str]
    key_metrics: KeyMetrics

class AnalyticsOverview(BaseModel):
    total_calls: int
    total_agents: int
    avg_sentiment: float
    high_risk_agents: int
    emotion_distribution: Dict[str, float]

class OperationStatus(BaseModel):
    status: str
    details: str
