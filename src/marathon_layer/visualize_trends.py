import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_agent_trends(agent_id, df, output_dir):
    """
    Generates three key plots for an agent and saves them.
    """
    agent_df = df[df['agent_id'] == agent_id].sort_values('date')
    if agent_df.empty:
        return
        
    agent_dir = Path(output_dir) / agent_id
    agent_dir.mkdir(parents=True, exist_ok=True)
    
    # Use a style
    plt.style.use('dark_background')
    sns.set_palette("viridis")

    # 1. Sentiment Trend
    plt.figure(figsize=(10, 5))
    plt.plot(agent_df['date'], agent_df['avg_sentiment'], alpha=0.3, label='Daily Sentiment', color='aqua')
    plt.plot(agent_df['date'], agent_df['sentiment_ma7'], label='7-Day Moving Avg', color='deepskyblue', linewidth=3)
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.title(f"Sentiment Trend over Time - {agent_id}", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Sentiment Score (-1 to 1)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(agent_dir / "sentiment_trend.png")
    plt.close()

    # 2. Emotion Distribution (Stacked Area)
    plt.figure(figsize=(10, 5))
    emotions = ['joy_pct', 'neutral_pct', 'sadness_pct', 'fear_pct', 'anger_pct']
    # Ensure they exist (renaming for plotting)
    plot_data = agent_df.set_index('date')
    available_emotions = [e for e in emotions if e in plot_data.columns]
    
    if available_emotions:
        plot_data[available_emotions].plot(kind='area', stacked=True, alpha=0.7, ax=plt.gca())
        plt.title(f"Emotion Distribution Evolution - {agent_id}", fontsize=14)
        plt.xlabel("Date")
        plt.ylabel("Composition (%)")
        plt.legend(loc='upper right')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(agent_dir / "emotion_distribution.png")
    plt.close()

    # 3. Workload Timeline
    plt.figure(figsize=(10, 5))
    sns.barplot(data=agent_df, x='date', y='total_calls', color='mediumpurple')
    plt.title(f"Daily Call Volume - {agent_id}", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Total Calls")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(agent_dir / "workload_volume.png")
    plt.close()

def main():
    features_file = "results/marathon/agent_features.csv"
    output_dir = "results/marathon/agent_trends"
    
    if not Path(features_file).exists():
        logger.error(f"Features file not found: {features_file}")
        return

    df = pd.read_csv(features_file)
    # Convert date back to datetime if needed for plotting
    df['date'] = pd.to_datetime(df['date'])
    
    # Handle missing neutral_pct if needed
    if 'neutral_pct' not in df.columns:
        # Compute as remainder of other emotions? Or just assume it missed
        pass

    agent_ids = df['agent_id'].unique()
    logger.info(f"Generating visualizations for {len(agent_ids)} agents...")

    for agent_id in agent_ids:
        try:
            plot_agent_trends(agent_id, df, output_dir)
            logger.info(f"Generated plots for {agent_id}")
        except Exception as e:
            logger.error(f"Failed to plot trends for {agent_id}: {e}")

    logger.info(f"All visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()
