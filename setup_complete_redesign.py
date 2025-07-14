import os
import subprocess
import sys
import shutil
from pathlib import Path

def setup_complete_redesign():
    """
    Complete setup script for the Smart Data Profiler redesign
    """
    
    print("🚀 Setting up Smart Data Profiler Complete Redesign...")
    print("=" * 60)
    
    # Step 1: Fix Lotus Environment
    print("\n1. 🔧 Fixing Lotus Environment...")
    try:
        if fix_lotus_environment():
            print("✅ Lotus environment fixed successfully!")
        else:
            print("❌ Lotus environment fix failed!")
    except Exception as e:
        print(f"❌ Error fixing Lotus: {e}")
    
    # Step 2: Create Directory Structure
    print("\n2. 📁 Creating directory structure...")
    directories = [
        "app/assets",
        "app/database", 
        "app/pages",
        "app/core_logic",
        "app/data_quality",
        "app/outputs",
        "exports/charts",
        "exports/reports"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   Created: {directory}")
    
    # Step 3: Install Additional Requirements
    print("\n3. 📦 Installing additional requirements...")
    additional_packages = [
        "scikit-learn>=1.3.0",
        "plotly>=5.0.0",
        "ydata-profiling>=4.0.0"
    ]
    
    for package in additional_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"   ✅ Installed: {package}")
        except subprocess.CalledProcessError:
            print(f"   ⚠️  Failed to install: {package}")
    
    # Step 4: Create Configuration Files
    print("\n4. ⚙️ Creating configuration files...")
    
    # Create enhanced requirements.txt
    enhanced_requirements = """
streamlit==1.33.0
pandas==1.5.3
plotly==5.21.0
sqlalchemy==1.4.48
pymysql==1.1.0
ydata-profiling>=4.0.0
pandasai==1.5.15
psycopg2-binary==2.9.1
google-generativeai==0.5.4
scikit-learn>=1.3.0
numpy>=1.24.0
requests>=2.31.0
matplotlib>=3.5.0
"""
    
    with open("requirements_enhanced.txt", "w") as f:
        f.write(enhanced_requirements.strip())
    
    print("   ✅ Created enhanced requirements file")
    
    # Step 5: Initialize Database
    print("\n5. 🗄️ Initializing rule management database...")
    try:
        db = RuleManagementDB()
        print("   ✅ Database initialized successfully")
    except Exception as e:
        print(f"   ❌ Database initialization failed: {e}")
    
    # Step 6: Create Sample Data
    print("\n6. 📊 Creating sample data...")
    create_sample_data()
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run: streamlit run app/Home.py")
    print("2. Navigate to the new Anomaly Detection page")
    print("3. Try the enhanced Rule Management system") 
    print("4. Enjoy the professional new UI!")

def create_sample_data():
    """Create sample data for testing"""
    
    # Create sample CSV with anomalies
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    
    # Normal data
    normal_data = {
        'transaction_amount': np.random.normal(1000, 200, 950),
        'customer_age': np.random.normal(35, 10, 950),
        'account_balance': np.random.normal(5000, 1000, 950),
        'transaction_count': np.random.poisson(5, 950)
    }
    
    # Add some anomalies
    anomaly_data = {
        'transaction_amount': np.random.normal(10000, 1000, 50),  # High amounts
        'customer_age': np.random.choice([15, 95], 50),  # Unusual ages
        'account_balance': np.random.normal(-1000, 500, 50),  # Negative balances
        'transaction_count': np.random.poisson(50, 50)  # High frequency
    }
    
    # Combine data
    df_normal = pd.DataFrame(normal_data)
    df_anomaly = pd.DataFrame(anomaly_data)
    df_combined = pd.concat([df_normal, df_anomaly], ignore_index=True)
    
    # Shuffle the data
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Add some categorical data
    df_combined['customer_type'] = np.random.choice(['premium', 'standard', 'basic'], len(df_combined))
    df_combined['transaction_type'] = np.random.choice(['deposit', 'withdrawal', 'transfer'], len(df_combined))
    
    # Save sample data
    os.makedirs("sample_data", exist_ok=True)
    df_combined.to_csv("sample_data/transactions_with_anomalies.csv", index=False)
    
    print("   ✅ Created sample transaction data with anomalies")

if __name__ == "__main__":
    setup_complete_redesign()