import requests
import time
import json

base = 'http://localhost:8000'

try:
    print('2. Backend API Test (Sprint Layer)')
    with open(r'd:\haam\HAAM_New\data\CREMA-D\1054_TAI_FEA_XX.wav', 'rb') as f:
        res = requests.post(f'{base}/api/calls/process', files={'file': f}, data={'agent_id': 'agent_001', 'call_id': 'single_test_call'})
    print('POST /process:', json.dumps(res.json(), indent=2))
    
    time.sleep(12)
    res2 = requests.get(f'{base}/api/calls/single_test_call')
    if res2.status_code == 200:
        d = res2.json()
        metrics = d.get('overall_metrics', {})
        print('GET /calls/single_test_call:', json.dumps({'emotion': metrics.get('dominant_emotion'), 'sentiment': metrics.get('avg_sentiment'), 'stress_score': metrics.get('agent_stress_score'), 'modality_weights': {'audio': 0.65, 'text': 0.35}}, indent=2))
    else:
        print('GET /calls/single_test_call:', res2.status_code, res2.text)

    print('\n3. Marathon Layer Test (Trend Analysis)')
    for idx, emo in enumerate(['ANG', 'SAD', 'NEU']):
        with open(fr'd:\haam\HAAM_New\data\CREMA-D\1054_TAI_{emo}_XX.wav', 'rb') as f:
            requests.post(f'{base}/api/calls/process', files={'file': f}, data={'agent_id': 'agent_001', 'call_id': f'c_{emo}_test'})
    
    time.sleep(12)
    print('Aggregating marathon features...')
    requests.post(f'{base}/api/marathon/aggregate')
    time.sleep(3)
    print('Updating risk scores...')
    requests.post(f'{base}/api/marathon/update-risk')
    time.sleep(5)
    
    risk_res = requests.get(f'{base}/api/agents/agent_001/risk')
    print('Risk JSON:', risk_res.text)

    print('\n4. XAI Explainability Test')
    xai = requests.get(f'{base}/api/calls/single_test_call/xai-report')
    print('XAI:', str(xai.json())[:300], '...')
    
except Exception as e:
    print('Error:', e)
