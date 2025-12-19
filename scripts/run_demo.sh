
#!/bin/bash
echo "Starting Demo..."
# Start API in background
uvicorn src.api.app:app --port 8000 &
API_PID=$!

# Wait for API
sleep 5

# Start Dashboard
cd src/dashboard
npm start

# Cleanup on exit
kill $API_PID
