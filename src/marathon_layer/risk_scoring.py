
import pandas as pd
import argparse
import os
import json
import logging
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def calculate_agent_risk(agent_timeseries_df, lookback_days=14):
    """
    Calculate risk score for a single agent based on their timeseries data.
    """
    if agent_timeseries_df.empty:
        return None

    # Sort by date
    df = agent_timeseries_df.sort_values('date').reset_index(drop=True)
    
    # Get recent data window
    recent_df = df.tail(lookback_days)
    if recent_df.empty:
        return None
        
    current_metrics = recent_df.iloc[-1]
    
    # Baseline: First 7 days of available data (or less if not enough history)
    baseline_window = df.head(7)
    
    # --- Metrics ---
    # Sentiment
    current_sentiment = recent_df['avg_sentiment'].mean() # Average over recent window or last day? "Recent 7 days" vs baseline.
    # Requirement: "Compare recent 7 days vs baseline (first 7 days)"
    # Let's take the last 7 days from recent_df
    recent_7d = df.tail(7)
    recent_sentiment_avg = recent_7d['avg_sentiment'].mean()
    baseline_sentiment_avg = baseline_window['avg_sentiment'].mean()
    
    # Stress
    current_stress = recent_7d['avg_stress_score'].mean()
    baseline_stress = baseline_window['avg_stress_score'].mean()
    
    # Negative Emotions (Anger)
    current_anger_pct = recent_7d['angry_calls_pct'].mean()
    baseline_anger_pct = baseline_window['angry_calls_pct'].mean()
    
    # Escalation
    current_escalation_rate = recent_7d['escalation_count'].sum() / recent_7d['call_count'].sum() if recent_7d['call_count'].sum() > 0 else 0.0
    
    # Workload
    # "If call_volume_change_pct > 30%". This is already a rolling feature for the last day.
    # Let's use the average of this metric over the last few days or just the latest value.
    # Using latest available value.
    current_vol_change = current_metrics['call_volume_change_pct']

    
    # --- Scoring Logic ---
    risk_score = 0.0
    risk_factors = []
    
    # a) Sentiment Decline (30% weight)
    # Decline calculation: (Baseline - Recent) / Baseline. 
    # Careful with signs. Sentiment is -1 to 1.
    # If baseline is 0.5 and recent is 0.4, decline is (0.5 - 0.4) / 0.5 = 0.2 (20%).
    # If baseline is negative and recent is more negative, it's worsening.
    # Absolute drop might be safer or check specific requirement logic.
    # Requirement: "If decline > 15%" implied percentage relative to baseline?
    # Or percentage points? "dropped 25%" usually means relative.
    # Handling divide by zero or small baseline:
    # Let's use raw difference if close to zero, or standard % change logic.
    # Shift to 0-2 scale for easier % calc? (x+1). 
    # Let's stick to simple algebraic decrease for now: if baseline > 0, (base-rec)/base.
    
    sentiment_decline_pct = 0.0
    if baseline_sentiment_avg > 0:
        sentiment_decline_pct = (baseline_sentiment_avg - recent_sentiment_avg) / baseline_sentiment_avg
    elif baseline_sentiment_avg == 0:
        if recent_sentiment_avg < 0: sentiment_decline_pct = 1.0 # arbitrary large
    else:
        # Baseline negative. Recent more negative?
        # e.g. -0.2 to -0.4. magnitude increased 100%. Worsened.
        if recent_sentiment_avg < baseline_sentiment_avg:
             sentiment_decline_pct = (recent_sentiment_avg - baseline_sentiment_avg) / baseline_sentiment_avg # Positive result
    
    if sentiment_decline_pct > 0.25:
        risk_score += 0.5
        risk_factors.append({
            "factor": "sentiment_decline",
            "severity": round(sentiment_decline_pct, 2),
            "description": f"Sentiment dropped {sentiment_decline_pct:.1%} vs baseline",
            "contribution": 0.5
        })
    elif sentiment_decline_pct > 0.15:
        risk_score += 0.3
        risk_factors.append({
            "factor": "sentiment_decline",
            "severity": round(sentiment_decline_pct, 2),
            "description": f"Sentiment dropped {sentiment_decline_pct:.1%} vs baseline",
            "contribution": 0.3
        })
        
    # b) Stress Level (25% weight)
    added_stress_risk = 0.0
    if current_stress > 0.6:
        added_stress_risk += 0.25
        risk_factors.append({
            "factor": "high_stress",
            "severity": round(current_stress, 2),
            "description": f"Average stress score {current_stress:.2f} > 0.6",
            "contribution": 0.25
        })
    
    # Stress increasing
    if current_stress > baseline_stress * 1.1: # >10% increase? Requirement: "If stress increasing". Let's say strictly greater.
        # Check simple increase
        if current_stress > baseline_stress:
             # But wait, logic says "add 0.15".
             # Is it cumulative with >0.6? Yes.
             added_stress_risk += 0.15
             risk_factors.append({
                "factor": "stress_increasing",
                "severity": round(current_stress - baseline_stress, 2),
                "description": "Stress level is trending upwards",
                "contribution": 0.15
             })
    risk_score += added_stress_risk
    
    # c) Negative Emotions (20% weight)
    added_anger_risk = 0.0
    if current_anger_pct > 0.15: # 15%
        added_anger_risk += 0.2
        risk_factors.append({
            "factor": "high_anger_exposure",
            "severity": round(current_anger_pct, 2),
            "description": f"{current_anger_pct:.1%} of calls have high anger",
            "contribution": 0.2
        })
    
    # Increasing > 5% vs baseline (Percentage points or relative? "increasing >5%". Usually points for percentages).
    # e.g. Baseline 5%, Current 11% (Diff 6%).
    if (current_anger_pct - baseline_anger_pct) > 0.05:
        added_anger_risk += 0.15
        risk_factors.append({
            "factor": "anger_increasing",
            "severity": round(current_anger_pct - baseline_anger_pct, 2),
            "description": "Anger exposure increased by >5%",
            "contribution": 0.15
        })
    risk_score += added_anger_risk
    
    # d) Escalation Rate (15% weight)
    if current_escalation_rate > 0.10:
        risk_score += 0.2
        risk_factors.append({
            "factor": "high_escalation",
            "severity": round(current_escalation_rate, 2),
            "description": f"Escalation rate {current_escalation_rate:.1%} > 10%",
            "contribution": 0.2
        })

    # e) Workload (10% weight)
    # Using 'call_volume_change_pct' from last record
    # This might be NaN for first week, assume 0
    vol_change = 0.0
    if not pd.isna(current_vol_change):
        vol_change = current_vol_change
        
    if vol_change > 0.30:
        risk_score += 0.1
        risk_factors.append({
            "factor": "workload_spike",
            "severity": round(vol_change, 2),
            "description": f"Call volume increased by {vol_change:.1%}",
            "contribution": 0.1
        })
        
    # Cap risk score at 1.0? Requirement doesn't say, but typical.
    risk_score = min(risk_score, 1.0)
    
    # Classify
    if risk_score >= 0.7:
        risk_level = "critical"
        trend_direction = "worsening" # simplistic assumption if high risk
    elif risk_score >= 0.5:
        risk_level = "high"
        trend_direction = "concerning" 
    elif risk_score >= 0.3:
        risk_level = "medium"
        trend_direction = "stable"
    else:
        risk_level = "low"
        trend_direction = "improving"

    # Recommendations
    recommendations = []
    if risk_score >= 0.5:
        recommendations.append("Schedule 1-on-1 check-in")
    if any(f['factor'] == 'high_anger_exposure' for f in risk_factors):
        recommendations.append("Review recent difficult calls for coaching")
    if any(f['factor'] == 'high_stress' for f in risk_factors):
        recommendations.append("Suggest short break or wellness check")
    if 'workload_spike' in [f['factor'] for f in risk_factors]:
        recommendations.append("Review scheduling/workload distribution")
        
    if not recommendations and risk_score < 0.3:
        recommendations.append("Maintain current performance")
        
    return {
        "agent_id": current_metrics['agent_id'],
        "risk_score": round(risk_score, 2),
        "risk_level": risk_level,
        "trend_direction": trend_direction,
        "risk_factors": risk_factors,
        "recommendations": list(set(recommendations)),
        "key_metrics": {
            "current_sentiment": round(recent_sentiment_avg, 2),
            "baseline_sentiment": round(baseline_sentiment_avg, 2),
            "current_stress": round(current_stress, 2)
        }
    }

def score_all_agents(features_csv):
    """
    Load features CSV and score all agents.
    Returns DataFrame.
    """
    if not os.path.exists(features_csv):
        logging.error(f"File not found: {features_csv}")
        return pd.DataFrame()
        
    df = pd.read_csv(features_csv)
    
    results = []
    agent_ids = df['agent_id'].unique()
    
    for agent in agent_ids:
        agent_df = df[df['agent_id'] == agent].copy()
        risk_profile = calculate_agent_risk(agent_df)
        
        if risk_profile:
            # Flatten for CSV output (simplified)
            row = {
                "agent_id": risk_profile['agent_id'],
                "risk_score": risk_profile['risk_score'],
                "risk_level": risk_profile['risk_level'],
                "top_factor": risk_profile['risk_factors'][0]['factor'] if risk_profile['risk_factors'] else "none",
                "recommendation_count": len(risk_profile['recommendations']),
                "details_json": json.dumps(risk_profile) # Embed full details
            }
            results.append(row)
            
    return pd.DataFrame(results)

def visualize_risk_factors(risk_details):
    """
    Generate a simple text-based visualization/report.
    """
    print(f"\n--- Risk Report for {risk_details['agent_id']} ---")
    print(f"Score: {risk_details['risk_score']} [{risk_details['risk_level'].upper()}]")
    print("Risk Factors:")
    if not risk_details['risk_factors']:
        print("  - None")
    else:
        for f in risk_details['risk_factors']:
            print(f"  [!] {f['factor']} (Contrib: {f['contribution']}): {f['description']}")
            
    print("Recommendations:")
    for r in risk_details['recommendations']:
        print(f"  -> {r}")
    print("-" * 40)

def main():
    parser = argparse.ArgumentParser(description="Marathon Layer Risk Scoring")
    parser.add_argument("--input", required=True, help="Aggregated features CSV")
    parser.add_argument("--output", required=True, help="Output CSV for risk scores")
    parser.add_argument("--visualize", action="store_true", help="Print detailed reports to console")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return

    scores_df = score_all_agents(args.input)
    
    if scores_df.empty:
        logger.warning("No scores generated.")
        return
        
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    scores_df.to_csv(args.output, index=False)
    logger.info(f"Risk scores saved to {args.output}")
    
    if args.visualize:
        # Visualize top risks
        # We need to parse the JSON back if we just used the dataframe
        for idx, row in scores_df.iterrows():
            details = json.loads(row['details_json'])
            visualize_risk_factors(details)

if __name__ == "__main__":
    main()
