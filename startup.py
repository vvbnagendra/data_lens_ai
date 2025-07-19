#!/usr/bin/env python3
# startup.py - Start Data Lens AI

import streamlit as st
import subprocess
import sys
import os

def main():
    """Start the application"""
    print("Starting Data Lens AI...")
    
    # Check if we're in the right directory
    if not os.path.exists("app/Home.py"):
        print("ERROR: app/Home.py not found!")
        print("Please run this script from the project root directory.")
        return
    
    try:
        # Start Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app/Home.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\nApplication stopped")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main()
