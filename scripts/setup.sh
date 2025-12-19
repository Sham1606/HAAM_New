
#!/bin/bash
echo "Setting up HAAM Framework..."

# Python
echo "Installing Python dependencies..."
pip install -r requirements.txt

# React
echo "Installing Dashboard dependencies..."
cd src/dashboard
npm install
cd ../..

echo "Setup Complete!"
