
# HAAM Framework Demo Scenario

This guide walks through a full usage scenario of the HAAM framework.

## 1. Setup
Make sure services are running:
```bash
uvicorn src.api.app:app --port 8000
cd src/dashboard && npm start
```

## 2. Audio Processing (Sprint Layer)
We will simulate a customer call where the agent handles a difficult situation.

**Action**: Upload a sample audio file.
- Go to **Calls List** in Dashboard.
- Click **Upload Call**.
- Select `data/samples/complaint_call.wav` (if available) or any dummy wav.
- ID: `demo_call_01`, Agent: `agent_demo`.

**Expected Result**:
- Call appears in the list with "Processing" status (or completes instantly).
- Clicking the call shows the transcript and sentiment timeline.

## 3. Burnout Risk Analysis (Marathon Layer)
We will generate risk scores based on aggregated data.

**Action**: Trigger Aggregation.
- Use Curl or API docs manually: `POST /api/marathon/aggregate`.
- Then `POST /api/marathon/update-risk`.
 *Note: In a real scenario, this runs periodically.*

**Action**: Check Agent Risk.
- Go to **Agent Risk** page.
- Look for `agent_demo`.

**Expected Result**:
- If the call had negative sentiment/stress, risk score increases.
- Click `agent_demo` to see risk factors (e.g., "High Negative Emotion Detected").

## 4. Analytics
**Action**: View System Overview.
- Go to **Analytics** page.

**Expected Result**:
- Pie chart shows emotion distribution.
- Sentiment trend line updates.
