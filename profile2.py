import time
import requests
import json
import sys

# Warm up
requests.get("http://localhost:8000/health")

lines = []
def time_request(url):
    t0 = time.time()
    r = requests.get(url)
    t1 = time.time()
    msg = f"URL: {url}\nTime: {(t1-t0)*1000:.2f}ms\n"
    if r.status_code == 200:
        data = r.json()
        if isinstance(data, list):
            msg += f"Length: {len(data)}\n"
        elif isinstance(data, dict):
            msg += f"Keys: {list(data.keys())[:3]}\n"
    else:
        msg += f"Error {r.status_code} on {url}\n"
    msg += "-" * 40 + "\n"
    lines.append(msg)

time_request("http://localhost:8000/api/agents/agent_001/calls?limit=10")
time_request("http://localhost:8000/api/agents/agent_001/stats")
time_request("http://localhost:8000/api/analytics/overview")
time_request("http://localhost:8000/api/calls?limit=50")

with open("profile_results.txt", "w", encoding="utf-8") as f:
    f.writelines(lines)
  