
Write-Host "Setting up HAAM Framework..."

# Python
Write-Host "Installing Python dependencies..."
pip install -r requirements.txt

# React
Write-Host "Installing Dashboard dependencies..."
Set-Location src/dashboard
npm install
Set-Location ../..

Write-Host "Setup Complete!"
