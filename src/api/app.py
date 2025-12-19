
import os
import glob
import json
import shutil
import logging
import asyncio
from typing import List, Optional
from datetime import datetime, timedelta
import pandas as pd

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Query, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Import core components
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sprint_layer.run_sprint_pipeline import SprintPipeline
from feature_store.aggregate_features import aggregate_sprint_to_timeseries, load_sprint_data
from marathon_layer.risk_scoring import score_all_agents, calculate_agent_risk

from api.models import (
    ProcessCallResponse, CallSummaryModel, CallDetailResponse, 
    AgentStats, RiskProfileResponse, AnalyticsOverview, OperationStatus
)

# Configuration
CALLS_DIR = "results/calls"
AGGREGATED_CSV = "results/aggregated/agent_features.csv"
RISK_SCORES_CSV = "results/risk_scores.csv"
UPLOAD_DIR = "data/uploads"

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HAAM_API")

# Rate Limiter
limiter = Limiter(key_func=get_remote_address)

# App Init
app = FastAPI(
    title="HAAM Framework API",
    description="Backend API for Call Center Audio Analysis & Risk Scoring",
    version="1.0.0"
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Pipeline Instance (Lazy Loading or Startup)
pipeline_instance: Optional[SprintPipeline] = None

def get_pipeline():
    global pipeline_instance
    if pipeline_instance is None:
        logger.info("Initializing Sprint Pipeline...")
        pipeline_instance = SprintPipeline()
    return pipeline_instance

# Helpers
def save_upload_file(upload_file: UploadFile, destination: str):
    try:
        with open(destination, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()

def process_audio_background(audio_path: str, agent_id: str, call_id: str):
    """
    Background task wrapper for pipeline processing.
    """
    try:
        pipeline = get_pipeline()
        result = pipeline.process_call(audio_path, agent_id, call_id)
        
        # Save result (Pipeline does return it, but also Main saves it? 
        # The pipeline.process_call returns the dict, but doesn't save it to disk in the class method 
        # heavily unless main() does it. Wait, checking run_sprint_pipeline.py...
        # It returns output_data. It does NOT save to disk inside process_call.
        # So we must save it here.
        
        output_dir = CALLS_DIR
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"call_{call_id}.json")
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
            
        logger.info(f"Background processing finished for call {call_id}")
        
        # Clean up upload? Maybe keep for record.
        # os.remove(audio_path) 
        
    except Exception as e:
        logger.error(f"Error in background processing for {call_id}: {e}")

# --- Endpoints ---

@app.post("/api/calls/process", response_model=ProcessCallResponse)
@limiter.limit("10/minute")
async def process_call(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    agent_id: str = Form(...),
    call_id: str = Form(...)
):
    """
    Upload and process an audio call.
    """
    # Validation
    if not file.filename.lower().endswith(('.wav', '.mp3')):
        raise HTTPException(status_code=400, detail="Invalid file type. Only WAV/MP3 supported.")
    
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, f"{call_id}_{file.filename}")
    
    save_upload_file(file, file_path)
    
    # Trigger background task
    background_tasks.add_task(process_audio_background, file_path, agent_id, call_id)
    
    return {
        "call_id": call_id,
        "message": "Call queued for processing",
        "status": "queued"
    }

@app.get("/api/calls", response_model=List[CallSummaryModel])
async def list_calls(
    agent_id: Optional[str] = None,
    # date_from: Optional[str] = None, # Simplified for now
    limit: int = 100
):
    """
    List processed calls with optional filtering.
    """
    files = glob.glob(os.path.join(CALLS_DIR, "call_*.json"))
    results = []
    
    for fpath in files: # This could be slow for many files, ideal: DB
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
                
            if agent_id and data.get('agent_id') != agent_id:
                continue
                
            metrics = data.get('overall_metrics', {})
            results.append({
                "call_id": data.get('call_id'),
                "agent_id": data.get('agent_id'),
                "timestamp": data.get('timestamp'),
                "avg_sentiment": metrics.get('avg_sentiment', 0.0),
                "dominant_emotion": metrics.get('dominant_emotion', 'neutral')
            })
        except:
            continue
            
    # Sort by timestamp desc
    results.sort(key=lambda x: x['timestamp'], reverse=True)
    return results[:limit]

@app.get("/api/calls/{call_id}", response_model=CallDetailResponse)
async def get_call_detail(call_id: str):
    """
    Get full details for a specific call.
    """
    fpath = os.path.join(CALLS_DIR, f"call_{call_id}.json")
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail="Call not found")
        
    with open(fpath, 'r') as f:
        data = json.load(f)
    return data

@app.get("/api/agents", response_model=List[AgentStats])
async def list_agents():
    """
    Get list of agents and summary stats.
    """
    # Load aggregated data if exists, else compute from raw calls (expensive)
    # Prefer raw calls for 'current' state? No, aggregation is better.
    # Let's verify if aggregated content is up to date?
    # For this endpoint, let's just aggregate on the fly from CSV if present
    if os.path.exists(AGGREGATED_CSV):
        df = pd.read_csv(AGGREGATED_CSV)
        # Group by agent
        summary = df.groupby('agent_id').agg({
            'call_count': 'sum',
            'avg_sentiment': 'mean'
        }).reset_index()
        
        return summary.to_dict('records')
    else:
        return []

@app.get("/api/agents/{agent_id}/risk", response_model=RiskProfileResponse)
async def get_agent_risk(agent_id: str):
    """
    Get latest risk profile for an agent.
    """
    # Check if we have pre-calculated scores
    if os.path.exists(RISK_SCORES_CSV):
        scores_df = pd.read_csv(RISK_SCORES_CSV)
        agent_score = scores_df[scores_df['agent_id'] == agent_id]
        
        if not agent_score.empty:
            # Parse the details_json
            details = json.loads(agent_score.iloc[0]['details_json'])
            return details
            
    # Fallback: Calculate on the fly if user requests and valid features exist
    if os.path.exists(AGGREGATED_CSV):
        df = pd.read_csv(AGGREGATED_CSV)
        agent_df = df[df['agent_id'] == agent_id]
        if not agent_df.empty:
            risk = calculate_agent_risk(agent_df)
            if risk:
                return risk
                
    raise HTTPException(status_code=404, detail="Risk profile not found (run aggregation first)")

@app.get("/api/analytics/overview", response_model=AnalyticsOverview)
async def get_analytics_overview():
    """
    Dashboard high-level metrics.
    """
    if not os.path.exists(AGGREGATED_CSV):
        return {
            "total_calls": 0,
            "total_agents": 0,
            "avg_sentiment": 0.0,
            "high_risk_agents": 0,
            "emotion_distribution": {}
        }
        
    df = pd.read_csv(AGGREGATED_CSV)
    
    total_calls = df['call_count'].sum()
    total_agents = df['agent_id'].nunique()
    avg_sent = df['avg_sentiment'].mean()
    
    # High risk count
    high_risk = 0
    if os.path.exists(RISK_SCORES_CSV):
        scores = pd.read_csv(RISK_SCORES_CSV)
        high_risk = len(scores[scores['risk_level'].isin(['high', 'critical'])])
        
    return {
        "total_calls": int(total_calls),
        "total_agents": int(total_agents),
        "avg_sentiment": round(float(avg_sent), 2) if not pd.isna(avg_sent) else 0.0,
        "high_risk_agents": high_risk,
        "emotion_distribution": {"neutral": 0.5, "anger": 0.2} # Placeholder or aggregate from JSONs
    }

@app.post("/api/marathon/aggregate", response_model=OperationStatus)
async def trigger_aggregation(background_tasks: BackgroundTasks):
    """
    Trigger feature aggregation.
    """
    def run_agg():
        logger.info("Starting aggregation...")
        aggregate_sprint_to_timeseries(CALLS_DIR, AGGREGATED_CSV)
        logger.info("Aggregation complete.")
        
    background_tasks.add_task(run_agg)
    return {"status": "processing", "details": "Aggregation started in background"}

@app.post("/api/marathon/update-risk", response_model=OperationStatus)
async def trigger_risk_scoring(background_tasks: BackgroundTasks):
    """
    Trigger risk scoring update.
    """
    def run_scoring():
        if not os.path.exists(AGGREGATED_CSV):
            logger.warning("Cannot score, aggregation missing.")
            return
        score_all_agents(AGGREGATED_CSV).to_csv(RISK_SCORES_CSV, index=False)
        logger.info("Risk scoring complete.")
        
    background_tasks.add_task(run_scoring)
    return {"status": "processing", "details": "Risk scoring started in background"}

# Init pipeline on import if desired, or let first request handle it.
# To ensure uvicorn startup doesn't hang, we leave it lazy.
