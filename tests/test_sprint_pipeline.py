
import unittest
import sys
import os
import json
from unittest.mock import MagicMock, patch
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from sprint_layer.run_sprint_pipeline import SprintPipeline

class TestSprintPipeline(unittest.TestCase):
    
    @patch('sprint_layer.run_sprint_pipeline.whisper.load_model')
    @patch('sprint_layer.run_sprint_pipeline.pipeline')
    def setUp(self, mock_pipeline, mock_whisper):
        # Mock models
        self.mock_whisper_instance = MagicMock()
        mock_whisper.return_value = self.mock_whisper_instance
        
        self.mock_emotion = MagicMock()
        self.mock_sentiment = MagicMock()
        
        # Configure pipeline side_effect to return different mocks based on call
        def pipeline_side_effect(task, **kwargs):
            if task == "text-classification":
                return self.mock_emotion
            elif task == "sentiment-analysis":
                return self.mock_sentiment
        mock_pipeline.side_effect = pipeline_side_effect
        
        self.pipeline = SprintPipeline()

    @patch('sprint_layer.run_sprint_pipeline.librosa.load')
    @patch('sprint_layer.run_sprint_pipeline.librosa.piptrack')
    @patch('sprint_layer.run_sprint_pipeline.librosa.feature.mfcc')
    @patch('sprint_layer.run_sprint_pipeline.librosa.get_duration')
    def test_process_call_structure(self, mock_duration, mock_mfcc, mock_piptrack, mock_load):
        # Setup Mocks
        mock_load.return_value = (np.zeros(16000*5), 16000) # 5 seconds of silence
        mock_duration.return_value = 5.0
        mock_mfcc.return_value = np.zeros((13, 100))
        mock_piptrack.return_value = (np.zeros((10,10)), np.zeros((10,10)))
        
        # Mock Transcript
        self.mock_whisper_instance.transcribe.return_value = {
            "text": "Hello world. This is a test.",
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "Hello world."},
                {"start": 2.1, "end": 4.5, "text": "This is a test."}
            ]
        }
        
        # Mock Analysis
        # Emotion: [{'label': 'joy', 'score': 0.9}, ...]
        self.mock_emotion.return_value = [[{'label': 'joy', 'score': 0.9}, {'label': 'sadness', 'score': 0.1}]]
        
        # Sentiment: [{'label': 'positive', 'score': 0.8}, {'label': 'negative', 'score': 0.1}]
        self.mock_sentiment.return_value = [[{'label': 'positive', 'score': 0.8}, {'label': 'negative', 'score': 0.1}]]
        
        # Run
        # Create a dummy file check bypass or just rely on the fact that process_call checks existence.
        # We need to mock os.path.exists if we pass a fake path.
        with patch('os.path.exists', return_value=True):
            result = self.pipeline.process_call("dummy.wav", "agent_01", "call_test")
            
        # Verify Structure
        self.assertIn("call_id", result)
        self.assertEqual(result["call_id"], "call_test")
        self.assertIn("overall_metrics", result)
        self.assertIn("segments", result)
        self.assertEqual(len(result["segments"]), 2)
        
        # Check aggregations
        metrics = result["overall_metrics"]
        self.assertIn("avg_sentiment", metrics)
        self.assertIn("dominant_emotion", metrics)
        self.assertEqual(metrics["dominant_emotion"], "joy") # Since we mocked joy
        
        print("\nTest Result JSON:")
        print(json.dumps(result, indent=2))

if __name__ == '__main__':
    unittest.main()
