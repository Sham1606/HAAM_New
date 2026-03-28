"""Quick test of auth endpoints."""
import requests, json

BASE = "http://localhost:8000"

print("=" * 60)
print("TEST 1: Login as admin")
r = requests.post(f"{BASE}/auth/login", json={"username": "admin", "password": "admin123"})
print(f"  Status: {r.status_code}")
data = r.json()
print(f"  Response: {json.dumps(data, indent=2)}")
admin_token = data.get("access_token", "")

print()
print("TEST 2: Login as agent (sham)")
r2 = requests.post(f"{BASE}/auth/login", json={"username": "sham", "password": "pass123"})
print(f"  Status: {r2.status_code}")
data2 = r2.json()
print(f"  Response: {json.dumps(data2, indent=2)}")
agent_token = data2.get("access_token", "")

print()
print("TEST 3: GET /auth/me (admin)")
r3 = requests.get(f"{BASE}/auth/me", headers={"Authorization": f"Bearer {admin_token}"})
print(f"  Status: {r3.status_code}")
print(f"  Response: {json.dumps(r3.json(), indent=2)}")

print()
print("TEST 4: GET /api/agents/me (agent)")
r4 = requests.get(f"{BASE}/api/agents/me", headers={"Authorization": f"Bearer {agent_token}"})
print(f"  Status: {r4.status_code}")
print(f"  Response: {json.dumps(r4.json(), indent=2)}")

print()
print("TEST 5: GET /api/agents/status (admin-only)")
r5 = requests.get(f"{BASE}/api/agents/status", headers={"Authorization": f"Bearer {admin_token}"})
print(f"  Status: {r5.status_code}")
print(f"  Response: {json.dumps(r5.json(), indent=2)}")

print()
print("TEST 6: POST /api/status/heartbeat/agent_001 (agent)")
r6 = requests.post(f"{BASE}/api/status/heartbeat/agent_001", headers={"Authorization": f"Bearer {agent_token}"})
print(f"  Status: {r6.status_code}")
print(f"  Response: {json.dumps(r6.json(), indent=2)}")

print()
print("TEST 7: POST /api/feedback/predict")
r7 = requests.post(f"{BASE}/api/feedback/predict", json={"emotion": "anger", "stress_score": 0.7})
print(f"  Status: {r7.status_code}")
print(f"  Response: {json.dumps(r7.json(), indent=2)}")

print()
print("TEST 8: GET /api/agents/registered")
r8 = requests.get(f"{BASE}/api/agents/registered")
print(f"  Status: {r8.status_code}")
print(f"  Response: {json.dumps(r8.json(), indent=2)}")

print()
print("TEST 9: Invalid login (wrong password)")
r9 = requests.post(f"{BASE}/auth/login", json={"username": "admin", "password": "wrong"})
print(f"  Status: {r9.status_code}")
print(f"  Response: {json.dumps(r9.json(), indent=2)}")

print()
print("TEST 10: GET /api/agents/status (agent = forbidden)")
r10 = requests.get(f"{BASE}/api/agents/status", headers={"Authorization": f"Bearer {agent_token}"})
print(f"  Status: {r10.status_code}")
print(f"  Response: {json.dumps(r10.json(), indent=2)}")

print()
print("=" * 60)
print("ALL TESTS COMPLETED")
