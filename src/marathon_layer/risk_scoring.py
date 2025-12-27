import pandas as pd
import argparse
import os
import json
import logging
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration & Thresholds ---
RISK_FACTORS = {
    'sentiment_decline': {
        'threshold': -0.15,  # 15% drop in avg sentiment over 7 days
        'weight': 0.25,
        'severity': 'high',
        'recommendation': "Agent showing declining satisfaction. Recommend wellness check-in."
    },
    'stress_spike': {
        'threshold': 0.40,   # >40% calls with negative sentiment (stress_indicator)
        'weight': 0.20,
        'severity': 'medium',
        'recommendation': "High stress levels detected in recent calls."
    },
    'anger_increase': {
        'threshold': 0.20,   # 20% increase in anger emotion
        'weight': 0.15,
        'severity': 'medium',
        'recommendation': "Increased anger exposure. Schedule coaching session."
    },
    'workload_overload': {
        'threshold': 1.5,    # 50% above historical average calls/day (workload_spike)
        'weight': 0.20,
        'severity': 'high',
        'recommendation': "Excessive call volume detected. Consider redistributing workload."
    },
    'disengagement': {
        'threshold': 0.30,   # engagement_score < 30%
        'weight': 0.20,
        'severity': 'medium',
        'recommendation': "Agent showing signs of disengagement from customer needs."
    }
}

def load_ml_model(model_path):
    """Loads the LSTM risk predictor if available."""
    if not os.path.exists(model_path):
        return None
    try:
        from src.marathon_layer.train_risk_predictor import LSTMRiskPredictor
        # Assuming 12 input features from aggregate_features.py
        model = LSTMRiskPredictor(input_dim=12) 
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    except Exception as e:
        logger.warning(f"Could not load ML model: {e}")
        return None

def calculate_agent_risk(agent_df, ml_model=None):
    """
    Calculate hybrid risk score using rules and optional ML metadata.
    """
    if agent_df.empty: return None
    
    # Latest day data
    current = agent_df.iloc[-1]
    
    risk_score = 0.0
    triggered_factors = []
    recommendations = []
    
    # 1. Rule-Based Scoring
    # a) Sentiment Decline (Using sentiment_trend_7d)
    if current['sentiment_trend_7d'] <= RISK_FACTORS['sentiment_decline']['threshold']:
        config = RISK_FACTORS['sentiment_decline']
        risk_score += config['weight']
        triggered_factors.append("sentiment_decline")
        recommendations.append(config['recommendation'])
        
    # b) Stress Spike (Using stress_indicator: % of negative calls)
    if current.get('stress_indicator', 0) >= RISK_FACTORS['stress_spike']['threshold']:
        config = RISK_FACTORS['stress_spike']
        risk_score += config['weight']
        triggered_factors.append("stress_spike")
        recommendations.append(config['recommendation'])
        
    # c) Anger Increase (Using anger_trend_7d)
    if current['anger_trend_7d'] >= RISK_FACTORS['anger_increase']['threshold']:
        config = RISK_FACTORS['anger_increase']
        risk_score += config['weight']
        triggered_factors.append("anger_increase")
        recommendations.append(config['recommendation'])
        
    # d) Workload Overload (Using workload_spike)
    if current['workload_spike'] >= RISK_FACTORS['workload_overload']['threshold']:
        config = RISK_FACTORS['workload_overload']
        risk_score += config['weight']
        triggered_factors.append("workload_overload")
        recommendations.append(config['recommendation'])
        
    # e) Disengagement (Using engagement_score)
    if current['engagement_score'] <= RISK_FACTORS['disengagement']['threshold']:
        config = RISK_FACTORS['disengagement']
        risk_score += config['weight']
        triggered_factors.append("disengagement")
        recommendations.append(config['recommendation'])

    # Special Combo Recommendation
    if "anger_increase" in triggered_factors and "stress_spike" in triggered_factors:
        recommendations.append("High emotional strain. Schedule coaching session.")

    # 2. ML Prediction (Optional)
    ml_prob = 0.0
    if ml_model and len(agent_df) >= 14:
        # Get last 14 days features
        feature_cols = [
            'total_calls', 'avg_sentiment', 'anger_pct', 'sadness_pct', 
            'fear_pct', 'joy_pct', 'avg_stress_score', 'engagement_score',
            'sentiment_trend_7d', 'anger_trend_7d', 'duration_trend_7d', 'workload_spike'
        ]
        seq = torch.FloatTensor(agent_df.tail(14)[feature_cols].values).unsqueeze(0)
        with torch.no_grad():
            ml_prob = ml_model(seq).item()
        
        # Hybrid combine: 70% Rules, 30% ML
        risk_score = (risk_score * 0.7) + (ml_prob * 0.3)
    
    # 3. Final Classification
    risk_score = min(risk_score, 1.0)
    if risk_score >= 0.6: risk_level = "Critical"
    elif risk_score >= 0.4: risk_level = "High"
    elif risk_score >= 0.2: risk_level = "Medium"
    else: risk_level = "Low"
    
    if not recommendations:
        recommendations.append("Maintain current performance.")

    # Structured factors for frontend
    risk_factors_detail = []
    for factor in triggered_factors:
        config = RISK_FACTORS.get(factor, {})
        risk_factors_detail.append({
            "factor": factor.replace('_', ' ').title(),
            "description": config.get('recommendation', ""),
            "contribution": config.get('weight', 0.2)
        })

    return {
        "agent_id": current['agent_id'],
        "risk_score": round(risk_score, 3),
        "risk_level": risk_level,
        "risk_factors": risk_factors_detail,
        "triggered_factors": [], # for BC
        "recommendations": list(set(recommendations)),
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def run_scoring(input_csv, output_csv, model_path=None):
    if not os.path.exists(input_csv):
        logger.error(f"Input file not found: {input_csv}")
        return
        
    df = pd.read_csv(input_csv)
    ml_model = load_ml_model(model_path) if model_path else None
    
    results = []
    for agent_id, group in df.groupby('agent_id'):
        group = group.sort_values('date')
        profile = calculate_agent_risk(group, ml_model)
        if profile:
            results.append(profile)
            
    res_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    res_df.to_csv(output_csv, index=False)
    logger.info(f"Generated risk profiles for {len(res_df)} agents in {output_csv}")
    
    # Summary stats
    logger.info(f"Risk Level Counts:\n{res_df['risk_level'].value_counts()}")

def main():
    parser = argparse.ArgumentParser(description="Marathon Layer Risk Scoring Engine")
    parser.add_argument("--input", default="results/marathon/agent_features.csv")
    parser.add_argument("--output", default="results/marathon/agent_risk_profiles.csv")
    parser.add_argument("--model", default="saved_models/marathon_risk_predictor.pth")
    args = parser.parse_args()
    
    run_scoring(args.input, args.output, args.model)

if __name__ == "__main__":
    main()
