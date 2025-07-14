#!/bin/bash
# setup_lotus_env.sh - Script to set up Semantic Processing environment

echo "Setting up Semantic Processing environment (Lotus alternative)..."

# Check if we're on Windows (Git Bash) or Unix-like system
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    echo "Detected Windows environment"
    PYTHON_CMD="python"
    ACTIVATE_SCRIPT=".lotus_env/Scripts/activate"
    PIP_CMD=".lotus_env/Scripts/pip"
else
    echo "Detected Unix-like environment"
    PYTHON_CMD="python3"
    ACTIVATE_SCRIPT=".lotus_env/bin/activate"
    PIP_CMD=".lotus_env/bin/pip"
fi

# Remove existing lotus environment if it exists
if [ -d ".lotus_env" ]; then
    echo "Removing existing .lotus_env directory..."
    rm -rf .lotus_env
fi

# Create new virtual environment
echo "Creating new virtual environment for semantic processing..."
$PYTHON_CMD -m venv .lotus_env

# Check if environment was created successfully
if [ ! -d ".lotus_env" ]; then
    echo "Error: Failed to create virtual environment"
    exit 1
fi

# Install requirements
echo "Installing semantic processing requirements..."

# Install core requirements (not actual lotus-ai)
$PIP_CMD install --upgrade pip
$PIP_CMD install "pandas>=2.0.0" "numpy>=1.24.0" "requests>=2.31.0"

# Install visualization libraries
echo "Installing visualization libraries..."
$PIP_CMD install "matplotlib>=3.5.0" "plotly>=5.0.0"

# Install optional Google AI support
echo "Installing optional Google AI support..."
$PIP_CMD install "google-generativeai>=0.5.4"

# Verify installation
echo "Verifying semantic processing installation..."
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    .lotus_env/Scripts/python -c "import pandas, numpy, requests; print('Semantic processing environment ready')"
else
    .lotus_env/bin/python -c "import pandas, numpy, requests; print('Semantic processing environment ready')"
fi

if [ $? -eq 0 ]; then
    echo "SUCCESS: Semantic processing environment setup completed!"
    echo "The enhanced semantic processing environment is now ready at .lotus_env/"
    echo "INFO: This provides Lotus-like functionality without complex dependencies."
else
    echo "ERROR: Installation verification failed"
    exit 1
fi