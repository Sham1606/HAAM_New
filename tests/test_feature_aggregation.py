
import unittest
import shutil
import tempfile
import json
import os
import sys
import pandas as pd
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from feature_store.aggregate_features import aggregate_sprint_to_timeseries

class TestFeatureAggregation(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.output_csv = os.path.join(self.test_dir, "output.csv")
        
        # Generate dummy data
        # Agent 1: 3 days continuous
        self.create_call(1, "agent_1", "2025-01-01T10:00:00", sentiment=0.5, stress=0.1)
        self.create_call(2, "agent_1", "2025-01-01T12:00:00", sentiment=0.1, stress=0.3)
        self.create_call(3, "agent_1", "2025-01-02T10:00:00", sentiment=-0.5, stress=0.8) # Bad day
        self.create_call(4, "agent_1", "2025-01-03T10:00:00", sentiment=0.2, stress=0.2)
        
        # Agent 2: Missing day in between
        self.create_call(5, "agent_2", "2025-01-01T10:00:00", sentiment=0.8)
        # Skip Jan 2
        self.create_call(6, "agent_2", "2025-01-03T10:00:00", sentiment=0.7)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def create_call(self, cid, agent, timestamp, sentiment=0.0, stress=0.0):
        data = {
            "call_id": f"call_{cid}",
            "agent_id": agent,
            "timestamp": timestamp,
            "duration_seconds": 100,
            "overall_metrics": {
                "avg_sentiment": sentiment,
                "emotion_distribution": {"neutral": 0.9, "anger": 0.1},
                "escalation_flag": False,
                "agent_stress_score": stress,
                "avg_pitch": 200.0
            }
        }
        with open(os.path.join(self.test_dir, f"call_{cid}.json"), 'w') as f:
            json.dump(data, f)

    def test_aggregation_logic(self):
        df = aggregate_sprint_to_timeseries(self.test_dir, self.output_csv)
        
        # Check Agent 1
        # Jan 1: 2 calls. Sentiment (0.5 + 0.1)/2 = 0.3
        a1_jan1 = df[(df['agent_id'] == 'agent_1') & (df['date'] == '2025-01-01')]
        self.assertEqual(len(a1_jan1), 1)
        self.assertEqual(a1_jan1.iloc[0]['call_count'], 2)
        self.assertAlmostEqual(a1_jan1.iloc[0]['avg_sentiment'], 0.3)
        
        # Check Agent 2 (Missing Day)
        a2 = df[df['agent_id'] == 'agent_2'].sort_values('date')
        # Should have Jan 1, Jan 2 (filled), Jan 3
        self.assertEqual(len(a2), 3)
        
        # Jan 2 should have call_count 0
        a2_jan2 = a2[a2['date'] == '2025-01-02']
        self.assertEqual(a2_jan2.iloc[0]['call_count'], 0)
        
    def test_rolling_features(self):
        # Create long history for rolling
        agent = "agent_rolling"
        start_date = datetime(2025, 1, 1)
        for i in range(10):
            date_str = (start_date + timedelta(days=i)).isoformat()
            self.create_call(100+i, agent, date_str, sentiment=float(i)) 
            # Sentiments: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        
        df = aggregate_sprint_to_timeseries(self.test_dir, self.output_csv)
        rdf = df[df['agent_id'] == agent].sort_values('date').reset_index(drop=True)
        
        # Day 7 (Index 6, Jan 7). Window: Jan 1..Jan 7 (Vals: 0,1,2,3,4,5,6)
        # Mean should be 3.0
        # Rolling implementation might include current day depending on 'right' alignment (default).
        # pandas rolling(window=7) includes current. 
        # So index 6 is mean of 0..6 -> 21/7 = 3.0
        self.assertAlmostEqual(rdf.iloc[6]['sentiment_7day_trend'], 3.0)
        
        # Day 8 (Index 7, Jan 8). Window: Jan 2..Jan 8 (Vals: 1..7) -> Mean 4.0
        self.assertAlmostEqual(rdf.iloc[7]['sentiment_7day_trend'], 4.0)

if __name__ == '__main__':
    unittest.main()
