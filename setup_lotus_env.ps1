# setup_lotus_env.ps1 - PowerShell script to set up Semantic Processing environment

Write-Host "Setting up Semantic Processing environment (Lotus alternative)..." -ForegroundColor Green

# Remove existing lotus environment if it exists
if (Test-Path ".lotus_env") {
    Write-Host "Removing existing .lotus_env directory..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force .lotus_env
}

# Create new virtual environment
Write-Host "Creating new virtual environment for semantic processing..." -ForegroundColor Green
python -m venv .lotus_env

# Check if environment was created successfully
if (-not (Test-Path ".lotus_env")) {
    Write-Host "Error: Failed to create virtual environment" -ForegroundColor Red
    exit 1
}

# Install requirements
Write-Host "Installing semantic processing requirements..." -ForegroundColor Green

# Upgrade pip first
.\.lotus_env\Scripts\python.exe -m pip install --upgrade pip

# Install core requirements (not actual lotus-ai)
.\.lotus_env\Scripts\pip.exe install pandas>=2.0.0 numpy>=1.24.0 requests>=2.31.0

# Install visualization libraries
Write-Host "Installing visualization libraries..." -ForegroundColor Green
.\.lotus_env\Scripts\pip.exe install matplotlib>=3.5.0 plotly>=5.0.0

# Install optional Google AI support
Write-Host "Installing optional Google AI support..." -ForegroundColor Green
.\.lotus_env\Scripts\pip.exe install google-generativeai>=0.5.4

# Verify installation
Write-Host "Verifying semantic processing installation..." -ForegroundColor Green
$verifyResult = .\.lotus_env\Scripts\python.exe -c "import pandas, numpy, requests; print('Semantic processing environment ready')" 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "SUCCESS: Semantic processing environment setup completed!" -ForegroundColor Green
    Write-Host "The enhanced semantic processing environment is now ready at .lotus_env/" -ForegroundColor Green
    Write-Host "INFO: This provides Lotus-like functionality without complex dependencies." -ForegroundColor Cyan
} else {
    Write-Host "ERROR: Installation verification failed" -ForegroundColor Red
    Write-Host $verifyResult -ForegroundColor Red
    exit 1
}