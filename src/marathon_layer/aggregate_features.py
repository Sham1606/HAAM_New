import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import pickle
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_sprint_data(input_dir):
    """
    Loads all JSON files from the sprint results directory.
    """
    records = []
    input_path = Path(input_dir)
    
    if not input_path.exists():
        logger.error(f"Input directory {input_dir} does not exist.")
        return pd.DataFrame()

    json_files = list(input_path.glob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files in {input_dir}")

    for jf in json_files:
        try:
            with open(jf, 'r') as f:
                data = json.load(f)
            
            # Extract fields with fallbacks
            agent_id = data.get('agent_id', 'unknown')
            # Use timestamp or parse from filename if needed
            ts_str = data.get('timestamp')
            if ts_str:
                dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            else:
                # Fallback: parse date from filename call_YYYY-MM-DD...
                try:
                    parts = jf.name.split('_')
                    dt = datetime.strptime(parts[1], '%Y-%m-%d')
                except:
                    dt = datetime.now()

            metrics = data.get('overall_metrics', {})
            dist = metrics.get('emotion_distribution', {})
            
            # Extract text attention if available (Engagement score)
            # Default to 0.5 if not found
            engagement = data.get('attention_weights', {}).get('text_attention', 0.5)

            rec = {
                'agent_id': agent_id,
                'date': dt.date(),
                'datetime': dt,
                'call_id': data.get('call_id'),
                'sentiment': metrics.get('avg_sentiment', 0),
                'stress_score': metrics.get('agent_stress_score', 0),
                'duration': data.get('duration_seconds', 0),
                'anger': dist.get('anger', 0),
                'sadness': dist.get('sadness', 0),
                'fear': dist.get('fear', 0),
                'joy': dist.get('joy', 0),
                'neutral': dist.get('neutral', 0),
                'escalated': metrics.get('escalation_flag', False),
                'engagement': engagement
            }
            records.append(rec)
        except Exception as e:
            logger.warning(f"Failed to parse {jf.name}: {e}")
            
    return pd.DataFrame(records)

def aggregate_daily_metrics(df):
    """
    Aggregates call-level data into daily agent-level metrics.
    """
    if df.empty:
        return pd.DataFrame()

    # Group by agent and date
    daily = df.groupby(['agent_id', 'date']).agg(
        total_calls=('call_id', 'count'),
        avg_sentiment=('sentiment', 'mean'),
        avg_stress_score=('stress_score', 'mean'),
        avg_duration=('duration', 'mean'),
        anger_pct=('anger', 'mean'),
        sadness_pct=('sadness', 'mean'),
        fear_pct=('fear', 'mean'),
        joy_pct=('joy', 'mean'),
        escalation_count=('escalated', 'sum'),
        engagement_score=('engagement', 'mean')
    ).reset_index()

    # Calculate stress indicator: % of calls with negative sentiment
    # We need to compute this before grouping or use a custom agg
    neg_sent_counts = df[df['sentiment'] < -0.1].groupby(['agent_id', 'date'])['call_id'].count()
    daily = daily.merge(neg_sent_counts.rename('neg_sent_calls'), on=['agent_id', 'date'], how='left').fillna(0)
    daily['stress_indicator'] = daily['neg_sent_calls'] / daily['total_calls']
    
    return daily

def compute_rolling_trends(df):
    """
    Computes 7-day rolling trends for each agent.
    """
    df = df.sort_values(['agent_id', 'date'])
    
    results = []
    for agent_id, group in df.groupby('agent_id'):
        group = group.copy()
        
        # 7-day moving averages
        group['sentiment_ma7'] = group['avg_sentiment'].rolling(window=7, min_periods=1).mean()
        group['anger_ma7'] = group['anger_pct'].rolling(window=7, min_periods=1).mean()
        group['duration_ma7'] = group['avg_duration'].rolling(window=7, min_periods=1).mean()
        
        # Trends (current MA7 vs previous MA7)
        group['sentiment_trend_7d'] = group['sentiment_ma7'].diff(periods=7).fillna(0)
        
        # % Change for anger and duration (avoid division by zero)
        group['anger_trend_7d'] = group['anger_ma7'].pct_change(periods=7).replace([np.inf, -np.inf], 0).fillna(0)
        group['duration_trend_7d'] = group['duration_ma7'].pct_change(periods=7).replace([np.inf, -np.inf], 0).fillna(0)
        
        # Workload spike: calls per day / historical average
        historical_avg = group['total_calls'].expanding().mean()
        group['workload_spike'] = (group['total_calls'] / historical_avg).fillna(1.0)
        
        results.append(group)
        
    return pd.concat(results)

def main():
    input_dir = "results/calls"
    output_dir = Path("results/marathon")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Marathon Feature Aggregation...")
    
    # 1. Load data
    raw_df = load_sprint_data(input_dir)
    if raw_df.empty:
        logger.error("No data loaded. Exiting.")
        return

    # 2. Daily aggregation
    daily_df = aggregate_daily_metrics(raw_df)
    
    # 3. Rolling trends
    trend_df = compute_rolling_trends(daily_df)
    
    # 4. Normalize features for modeling
    feature_cols = [
        'total_calls', 'avg_sentiment', 'anger_pct', 'sadness_pct', 
        'fear_pct', 'joy_pct', 'avg_stress_score', 'engagement_score',
        'sentiment_trend_7d', 'anger_trend_7d', 'duration_trend_7d', 'workload_spike'
    ]
    
    # Ensure all columns exist
    for col in feature_cols:
        if col not in trend_df.columns:
            trend_df[col] = 0.0

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(trend_df[feature_cols])
    
    # Save results
    output_file = output_dir / "agent_features.csv"
    trend_df.to_csv(output_file, index=False)
    logger.info(f"Saved aggregated features to {output_file}")
    
    scaler_file = output_dir / "feature_scaler.pkl"
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Saved feature scaler to {scaler_file}")

    # Sample stats for report
    logger.info("\nAggregation Summary:")
    logger.info(f"Total Agents: {trend_df['agent_id'].nunique()}")
    logger.info(f"Total Days: {trend_df['date'].nunique()}")
    logger.info(f"Max Workload Spike: {trend_df['workload_spike'].max():.2f}")
    logger.info(f"Avg Sentiment: {trend_df['avg_sentiment'].mean():.3f}")

def run_aggregation(input_dir="results/calls", output_dir="results/marathon"):
    """
    Orchestration function for API/external usage.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Marathon Feature Aggregation...")
    raw_df = load_sprint_data(input_dir)
    if raw_df.empty:
        logger.error("No data loaded.")
        return False

    daily_df = aggregate_daily_metrics(raw_df)
    trend_df = compute_rolling_trends(daily_df)
    
    feature_cols = [
        'total_calls', 'avg_sentiment', 'anger_pct', 'sadness_pct', 
        'fear_pct', 'joy_pct', 'avg_stress_score', 'engagement_score',
        'sentiment_trend_7d', 'anger_trend_7d', 'duration_trend_7d', 'workload_spike'
    ]
    
    for col in feature_cols:
        if col not in trend_df.columns:
            trend_df[col] = 0.0

    scaler = StandardScaler()
    scaler.fit(trend_df[feature_cols])
    
    output_file = output_path / "agent_features.csv"
    trend_df.to_csv(output_file, index=False)
    
    scaler_file = output_path / "feature_scaler.pkl"
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
        
    logger.info("Aggregation complete.")
    return True

if __name__ == "__main__":
    main()
