
import asyncio
import os
import sys
import logging

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.services.live_pipeline import LivePipeline

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LiveDemo")

async def main():
    print("🎤 HAAM Live Analysis CLI Demo")
    print("Initializing Pipeline... (This loads models, might take a few seconds)")
    
    pipeline = LivePipeline()
    
    # Mock WebSocket for CLI output
    class MockWebSocket:
        async def send_json(self, data):
            sprint = data.get('sprint', {})
            marathon = data.get('marathon', {})
            
            transcript = sprint.get('transcript', '...')
            emotion = sprint.get('emotion', 'unknown')
            conf = sprint.get('confidence', 0.0)
            
            trend = marathon.get('trend', '-')
            dom_emo = marathon.get('dominant_emotion', '-')
            
            print(f"\n[Turn] {transcript}")
            print(f"      👉 Emotion: {emotion} ({conf:.2f})")
            print(f"      📈 Session: Dominant={dom_emo}, Trend={trend}")

    pipeline.active_websockets.append(MockWebSocket())
    
    print("\n🔴 Listening... Speak into your microphone. (Press Ctrl+C to stop)")
    pipeline.start_listening()
    
    try:
        # Keep running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        pipeline.stop_listening()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
