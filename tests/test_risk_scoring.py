
import unittest
import pandas as pd
import sys
import os
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from marathon_layer.risk_scoring import calculate_agent_risk

class TestRiskScoring(unittest.TestCase):
    
    def test_high_risk_agent(self):
        # Scenario: Sentiment dropping, High Stress, Anger
        # Baseline (First 7 days): Sentiment 0.5, Stress 0.2
        # Recent (Last 7 days): Sentiment 0.1, Stress 0.8
        
        dates = pd.date_range(start='2025-01-01', periods=14, freq='D')
        
        data = []
        for i, d in enumerate(dates):
            if i < 7:
                # Baseline
                data.append({
                    'agent_id': 'agent_high',
                    'date': d,
                    'avg_sentiment': 0.5,
                    'avg_stress_score': 0.2,
                    'angry_calls_pct': 0.05,
                    'escalation_count': 0,
                    'call_count': 10,
                    'call_volume_change_pct': 0.0
                })
            else:
                # Recent (Worsening)
                data.append({
                    'agent_id': 'agent_high',
                    'date': d,
                    'avg_sentiment': 0.1, # Drop of 0.4 (80%) -> Score +0.5
                    'avg_stress_score': 0.8, # >0.6 (+0.25) AND Increasing (+0.15) -> +0.4
                    'angry_calls_pct': 0.20, # >15% (+0.2) AND Increasing (+0.15) -> +0.35
                    'escalation_count': 2, # Rate 2/10 = 20% -> +0.2
                    'call_count': 10,
                    'call_volume_change_pct': 0.4 # >30% -> +0.1
                })
        
        df = pd.DataFrame(data)
        metrics = calculate_agent_risk(df)
        
        print("\nHigh Risk Metrics:")
        print(json.dumps(metrics, indent=2))
        
        # Expected score:
        # Sentiment: 0.5
        # Stress: 0.4
        # Anger: 0.35 (Actually 0.2 + 0.15)
        # Escalation: 0.2
        # Workload: 0.1
        # Total: > 1.0 (Capped at 1.0)
        
        self.assertEqual(metrics['risk_score'], 1.0)
        self.assertEqual(metrics['risk_level'], 'critical')
        self.assertTrue('high_anger_exposure' in [f['factor'] for f in metrics['risk_factors']])

    def test_low_risk_agent(self):
        dates = pd.date_range(start='2025-01-01', periods=14, freq='D')
        data = []
        for d in dates:
            data.append({
                'agent_id': 'agent_low',
                'date': d,
                'avg_sentiment': 0.5,
                'avg_stress_score': 0.2,
                'angry_calls_pct': 0.05,
                'escalation_count': 0,
                'call_count': 10,
                'call_volume_change_pct': 0.0
            })
            
        df = pd.DataFrame(data)
        metrics = calculate_agent_risk(df)
        
        print("\nLow Risk Metrics:")
        print(json.dumps(metrics, indent=2))
        
        self.assertEqual(metrics['risk_score'], 0.0)
        self.assertEqual(metrics['risk_level'], 'low')

if __name__ == '__main__':
    unittest.main()
