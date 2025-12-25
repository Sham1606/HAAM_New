import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DialogueXAI:
    def __init__(self, output_dir="results/xai_dialogues", report_dir="results/xai_reports"):
        self.output_dir = Path(output_dir)
        self.report_dir = Path(report_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot styling
        plt.style.use('seaborn-v0_8-darkgrid')
        self.color_palette = sns.color_palette("muted")
        self.emotion_colors = {
            'neutral': 'gray',
            'anger': 'red',
            'disgust': 'green',
            'fear': 'purple',
            'sadness': 'blue',
            'joy': 'yellow'
        }

    def process_call(self, call_json_path):
        """
        Process a single call JSON and generate explainability outputs.
        """
        with open(call_json_path, 'r') as f:
            data = json.load(f)
            
        call_id = data.get('call_id', 'unknown_call')
        segments = data.get('segments', [])
        
        if not segments:
            logger.warning(f"No segments found for call {call_id}")
            return
            
        df = pd.DataFrame(segments)
        # Ensure turn index
        df['turn'] = range(1, len(df) + 1)
        
        # 1. Temporal Emotion Trajectory
        trajectory_path = self.plot_emotion_trajectory(df, call_id)
        
        # 2. Sentiment Flow
        sentiment_path = self.plot_sentiment_flow(df, call_id)
        
        # 3. Modality Importance Over Time
        modality_path = self.plot_modality_importance(df, call_id)
        
        # 4. Generate Report
        report_path = self.generate_report(data, df, {
            'trajectory': trajectory_path,
            'sentiment': sentiment_path,
            'modality': modality_path
        })
        
        return report_path

    def plot_emotion_trajectory(self, df, call_id):
        """
        Stacked area chart for emotion probabilities.
        Note: Needs segment-level distribution if available, 
        fallback to dominant emotion dummy encoding if not.
        """
        plt.figure(figsize=(12, 6))
        
        # If we have distribution in segments
        if 'emotion_distribution' in df.columns:
            # Expand distribution dicts to columns
            dist_df = pd.json_normalize(df['emotion_distribution']).fillna(0)
            available_emotions = [e for e in self.emotion_colors.keys() if e in dist_df.columns]
            
            dist_df[available_emotions].plot(kind='area', stacked=True, 
                                            color=[self.emotion_colors.get(e, 'gray') for e in available_emotions],
                                            alpha=0.6, ax=plt.gca())
        else:
            # Fallback based on dominant 'emotion' column
            emotions = df['emotion'].unique()
            for emo in emotions:
                subset = (df['emotion'] == emo).astype(float)
                plt.fill_between(df['turn'], subset, alpha=0.3, label=emo, color=self.emotion_colors.get(emo, 'gray'))

        plt.title(f"Temporal Emotion Trajectory - {call_id}", fontsize=14)
        plt.xlabel("Turn Number")
        plt.ylabel("Probability / Intensity")
        plt.legend(loc='upper right')
        
        save_path = self.output_dir / f"{call_id}_emotion_trajectory.png"
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return save_path

    def plot_sentiment_flow(self, df, call_id):
        """
        Timeline showing sentiment shift and escalation points.
        """
        plt.figure(figsize=(12, 4))
        
        # Smoothed sentiment
        plt.plot(df['turn'], df['sentiment_score'], marker='o', linestyle='-', color='teal', label='Utterance Sentiment')
        
        # Detect escalations (drop > 0.3)
        df['sentiment_shift'] = df['sentiment_score'].diff()
        escalations = df[df['sentiment_shift'] < -0.3]
        
        for _, row in escalations.iterrows():
            plt.annotate('ESCALATION', xy=(row['turn'], row['sentiment_score']), 
                         xytext=(row['turn'], row['sentiment_score'] - 0.2),
                         arrowprops=dict(facecolor='red', shrink=0.05),
                         fontsize=10, color='red', fontweight='bold')

        plt.axhline(0, color='black', alpha=0.3, linestyle='--')
        plt.title(f"Sentiment Flow & Escalation Detection - {call_id}", fontsize=14)
        plt.xlabel("Turn Number")
        plt.ylabel("Sentiment Score")
        plt.ylim(-1.1, 1.1)
        plt.legend()
        
        save_path = self.output_dir / f"{call_id}_sentiment_flow.png"
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return save_path

    def plot_modality_importance(self, df, call_id):
        """
        Audio vs Text attention weights across dialogue.
        """
        plt.figure(figsize=(12, 5))
        
        # Check if modality weights exist in segments
        if 'attention_weights' in df.columns:
            weights = pd.json_normalize(df['attention_weights']).fillna(0.5)
            # Some versions might call them audio/text or acoustic/text
            a_col = 'acoustic' if 'acoustic' in weights.columns else 'audio'
            t_col = 'text'
            
            if a_col in weights.columns:
                plt.plot(df['turn'], weights[a_col], label='Acoustic Importance', color='blue', linewidth=2)
                plt.plot(df['turn'], weights[t_col], label='Text Importance', color='purple', linewidth=2)
                
                # Highlight dominant turns
                audio_dom = df[weights[a_col] > 0.7]
                text_dom = df[weights[t_col] > 0.7]
                
                plt.scatter(audio_dom['turn'], [1.05]*len(audio_dom), marker='v', color='blue', label='Vocal Dominant')
                plt.scatter(text_dom['turn'], [1.05]*len(text_dom), marker='v', color='purple', label='Verbal Dominant')
        else:
            plt.text(0.5, 0.5, "Modality weights not available for this call version", ha='center')

        plt.title(f"Modality Importance over Time - {call_id}", fontsize=14)
        plt.xlabel("Turn Number")
        plt.ylabel("Attention Weight")
        plt.legend(loc='lower right')
        plt.ylim(0, 1.15)
        
        save_path = self.output_dir / f"{call_id}_modality_importance.png"
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return save_path

    def generate_report(self, data, df, plot_paths):
        """
        Generates markdown explainability report.
        """
        call_id = data.get('call_id')
        metrics = data.get('overall_metrics', {})
        
        # Analyze critical moments
        df['sentiment_shift'] = df['sentiment_score'].diff()
        escalations = df[df['sentiment_shift'] < -0.3]
        
        # Identify modality triggers
        # (Assuming attention_weights exists)
        modality_insights = []
        if 'attention_weights' in df.columns:
            weights = pd.json_normalize(df['attention_weights']).fillna(0.5)
            a_col = 'acoustic' if 'acoustic' in weights.columns else 'audio'
            for idx, row in df.iterrows():
                if weights.iloc[idx][a_col] > 0.75:
                    modality_insights.append(f"- Turn {idx+1}: Vocal cues dominant (Pitch/Tone shift likely)")
                elif weights.iloc[idx]['text'] > 0.75:
                    modality_insights.append(f"- Turn {idx+1}: Content dominant (Keyword/Language shift likely)")

        report_content = f"""# Call Explainability Report: {call_id}
- **Duration**: {data.get('duration_seconds', 0)}s ({len(df)} turns)
- **Overall Sentiment**: {metrics.get('avg_sentiment', 0):.2f}
- **Dominant Emotion**: {metrics.get('dominant_emotion', 'neutral')}

## 1. Emotion Evolution
![Emotion Trajectory]({plot_paths['trajectory'].absolute()})

## 2. Sentiment Flow
![Sentiment Flow]({plot_paths['sentiment'].absolute()})

## 3. Modality Importance
![Modality Importance]({plot_paths['modality'].absolute()})

## 📊 Critical Insights
### Escalation Points
"""
        if not escalations.empty:
            for _, row in escalations.iterrows():
                report_content += f"- **Turn {row['turn']}**: Significant sentiment drop ({row['sentiment_shift']:.2f}). Text: \"{row['text']}\"\n"
        else:
            report_content += "- No major sentiment drops detected.\n"

        report_content += "\n### Modality Triggers\n"
        report_content += "\n".join(modality_insights[:5]) # Top 5 insights
        
        report_content += f"""

## 💡 Recommendations
{"- Immediate de-escalation coaching needed for intensive sentiment drops." if not escalations.empty else "- Performance stable. Continue current de-escalation practices."}
{"- High vocal stress detected; suggest agent wellness check." if metrics.get('agent_stress_score', 0) > 0.6 else ""}
"""

        report_file = self.report_dir / f"{call_id}_xai_report.md"
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Generated report: {report_file}")
        return report_file

if __name__ == "__main__":
    # Test on one call if exists
    test_call = "results/calls/call_2024-12-10_agent_01_001.json"
    if os.path.exists(test_call):
        xai = DialogueXAI()
        xai.process_call(test_call)
    else:
        logger.info("Test call file not found for standalone run.")
