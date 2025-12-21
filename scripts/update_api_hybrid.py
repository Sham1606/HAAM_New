
import os
import re

APP_PY_PATH = r"D:\haam_framework\src\api\app.py"

NEW_CONFIG = """
# Dataset Configuration
CALLS_DIR_CREMA = "results/calls"
CALLS_DIR_IEMOCAP = "results/calls_iemocap"
METRICS_FILE = "results/analysis/evaluation_metrics.json"
"""

NEW_ANALYTICS_ENDPOINT = """
@app.get("/api/analytics/overview", response_model=AnalyticsOverview)
async def get_analytics_overview():
    "Dashboard high-level metrics calculated from hybrid dataset."
    
    files_crema = glob.glob(os.path.join(CALLS_DIR_CREMA, "call_*.json"))
    files_iemocap = glob.glob(os.path.join(CALLS_DIR_IEMOCAP, "*.json"))
    all_files = files_crema + files_iemocap
    
    if not all_files:
        return {
            "total_calls": 0, "total_agents": 0, "avg_sentiment": 0.0,
            "high_risk_agents": 0, "emotion_distribution": {}
        }
        
    # Sampling for performance if too many files (e.g. read max 2000)
    # relevant_files = all_files[:2000] # Optional optimization
    
    # Quick aggregation
    emotions = {}
    sentiments = []
    agents = set()
    
    # We can use metadata CSV if available for speed, but fallback to file scan
    # For now, simplistic scan
    for fpath in all_files:
        try:
            with open(fpath, 'r') as f:
                d = json.load(f)
                agents.add(d.get('agent_id'))
                m = d.get('overall_metrics', {})
                sentiments.append(m.get('avg_sentiment', 0))
                emotions[m.get('dominant_emotion', 'neutral')] = emotions.get(m.get('dominant_emotion', 'neutral'), 0) + 1
        except:
            continue
            
    total = len(all_files)
    dist = {k: round(v/total, 3) for k,v in emotions.items()} if total else {}
    avg_s = sum(sentiments)/len(sentiments) if sentiments else 0
    
    # Validation Metrics (Async load)
    val_metrics = {}
    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, 'r') as f:
                m = json.load(f)
                val_metrics = {
                    "crema_d_accuracy": m.get('crema_d', {}).get('weighted_accuracy', 0)*100,
                    "iemocap_accuracy": m.get('iemocap', {}).get('weighted_accuracy', 0)*100,
                    "combined_accuracy": m.get('combined', {}).get('weighted_accuracy', 0)*100
                }
        except:
            pass

    return {
        "total_calls": total,
        "total_agents": len(agents),
        "avg_sentiment": round(avg_s, 2),
        "high_risk_agents": 0, # simplified
        "emotion_distribution": dist,
        "dataset_breakdown": {
            "crema_d": len(files_crema),
            "iemocap": len(files_iemocap)
        },
        "validation_metrics": val_metrics
    }
"""

NEW_DATASET_ENDPOINTS = """
@app.get("/api/datasets/metrics")
async def get_dataset_metrics():
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, 'r') as f:
            return json.load(f)
    return {}

@app.get("/api/datasets/comparison")
async def get_dataset_comparison():
    meta_path = "data/hybrid_metadata.csv"
    if not os.path.exists(meta_path):
        return {}
    
    try:
        df = pd.read_csv(meta_path)
        res = {}
        for ds in ['CREMA-D', 'IEMOCAP']:
            sub = df[df['dataset'] == ds]
            if not sub.empty:
                res[ds.lower().replace('-','_')] = {
                    "samples": len(sub),
                    "avg_duration": round(sub['duration'].mean(), 2),
                    "avg_confidence": round(sub['confidence'].mean(), 2)
                }
        return res
    except:
        return {}
"""

def update_app_py():
    with open(APP_PY_PATH, 'r') as f:
        content = f.read()
    
    # 1. Inject Config
    if "CALLS_DIR_IEMOCAP" not in content:
        content = content.replace('CALLS_DIR = "results/calls"', 'CALLS_DIR = "results/calls"\n' + NEW_CONFIG)
        
    # 2. Inject Analytics Output Model Update (This requires editing models.py ideally, but we can pass dict and FastAPI tolerates extra fields if not strict, or we update models later. 
    # For this script, we assume Pydantic model allows it or we return dict response directly bypassing strict model validation for new fields if we change response_model to dict temporarily or update model)
    # We will just append the new endpoints to the end of file for now
    
    if "/api/datasets/metrics" not in content:
        content += "\n" + NEW_DATASET_ENDPOINTS
    
    # 3. Replace Analytics Overview
    # This is tricky with regex. We'll simply append the new one and comment out the old one if possible, or just overwrite the function using regex.
    # A safer way allows duplication in python but last def wins.
    content += "\n" + NEW_ANALYTICS_ENDPOINT.replace("@app.get", "# Overriding\n@app.get")
    
    with open(APP_PY_PATH, 'w') as f:
        f.write(content)
    
    print(f"Updated {APP_PY_PATH}")

if __name__ == "__main__":
    update_app_py()
