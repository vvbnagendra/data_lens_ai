# File: setup_lotus_fix.py
# Run this to fix your Lotus environment immediately

import os
import subprocess
import sys
import platform

def fix_lotus_environment():
    """
    Complete fix for Lotus environment issues
    """
    print("🔧 Starting Lotus Environment Fix...")
    
    # Detect operating system
    is_windows = platform.system() == "Windows"
    python_cmd = "python" if is_windows else "python3"
    
    # Step 1: Clean up existing environment
    lotus_env_path = ".lotus_env"
    if os.path.exists(lotus_env_path):
        print("🗑️ Removing existing Lotus environment...")
        if is_windows:
            os.system(f'rmdir /s /q "{lotus_env_path}"')
        else:
            os.system(f'rm -rf "{lotus_env_path}"')
    
    # Step 2: Create new environment with specific Python version
    print("🆕 Creating new Lotus environment...")
    result = subprocess.run([python_cmd, "-m", "venv", lotus_env_path], 
                          capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Failed to create virtual environment: {result.stderr}")
        return False
    
    # Step 3: Install required packages
    print("📦 Installing required packages...")
    
    if is_windows:
        pip_cmd = f"{lotus_env_path}\\Scripts\\pip.exe"
        python_exe = f"{lotus_env_path}\\Scripts\\python.exe"
    else:
        pip_cmd = f"{lotus_env_path}/bin/pip"
        python_exe = f"{lotus_env_path}/bin/python"
    
    # Upgrade pip first
    subprocess.run([python_exe, "-m", "pip", "install", "--upgrade", "pip"], 
                  check=True)
    
    # Install core requirements
    requirements = [
        "pandas>=2.0.0",
        "numpy>=1.24.0", 
        "requests>=2.31.0",
        "matplotlib>=3.5.0",
        "plotly>=5.0.0",
        "scikit-learn>=1.3.0",  # For anomaly detection
        "google-generativeai>=0.5.4"
    ]
    
    for req in requirements:
        print(f"Installing {req}...")
        result = subprocess.run([pip_cmd, "install", req], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"⚠️ Warning: Failed to install {req}: {result.stderr}")
    
    # Step 4: Verify installation
    print("✅ Verifying installation...")
    verification_script = '''
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import IsolationForest
print("SUCCESS: All packages imported correctly")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
'''
    
    result = subprocess.run([python_exe, "-c", verification_script], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print("🎉 Lotus environment setup completed successfully!")
        print(result.stdout)
        return True
    else:
        print(f"❌ Verification failed: {result.stderr}")
        return False