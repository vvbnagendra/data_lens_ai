# test_lotus_setup.py - Test script to verify Lotus setup
import os
import sys
import subprocess
import json
import pandas as pd
from io import StringIO

def test_lotus_environment():
    """Test if Lotus environment is properly set up"""
    print("🔍 Testing Lotus Environment Setup...")
    
    # Check if lotus environment exists
    is_windows = sys.platform.startswith("win")
    if is_windows:
        lotus_python = ".lotus_env/Scripts/python.exe"
    else:
        lotus_python = ".lotus_env/bin/python"
    
    if not os.path.exists(lotus_python):
        print(f"❌ Lotus Python executable not found at: {lotus_python}")
        return False
    
    print(f"✅ Found Lotus Python at: {lotus_python}")
    
    # Test Lotus import
    try:
        result = subprocess.run(
            [lotus_python, "-c", "import lotus; print('SUCCESS: Lotus imported')"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ Lotus AI successfully imported")
        else:
            print(f"❌ Failed to import Lotus: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error testing Lotus import: {e}")
        return False
    
    return True

def test_lotus_runner():
    """Test the lotus runner script"""
    print("\n🔍 Testing Lotus Runner Script...")
    
    # Check if runner script exists
    runner_path = "app/data_quality/lotus_runner.py"
    if not os.path.exists(runner_path):
        print(f"❌ Lotus runner script not found at: {runner_path}")
        return False
    
    print(f"✅ Found lotus runner at: {runner_path}")
    
    # Create test data
    test_data = {
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35], 
        "city": ["New York", "Los Angeles", "Chicago"]
    }
    df = pd.DataFrame(test_data)
    
    # Prepare test input
    test_input = {
        "tables": [("test_table", df.to_csv(index=False))],
        "question": "Find all records",
        "model": "lotus-mixtral",
        "mode": "query",
        "api_key": None
    }
    
    # Test the runner
    is_windows = sys.platform.startswith("win")
    lotus_python = ".lotus_env/Scripts/python.exe" if is_windows else ".lotus_env/bin/python"
    
    try:
        result = subprocess.run(
            [lotus_python, runner_path],
            input=json.dumps(test_input).encode(),
            capture_output=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print("✅ Lotus runner executed successfully")
            try:
                response = json.loads(result.stdout.decode())
                print(f"✅ Response type: {response.get('type', 'unknown')}")
                return True
            except json.JSONDecodeError:
                print(f"⚠️  Runner executed but returned non-JSON: {result.stdout.decode()}")
                return True
        else:
            print(f"❌ Lotus runner failed: {result.stderr.decode()}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Lotus runner timed out")
        return False
    except Exception as e:
        print(f"❌ Error testing lotus runner: {e}")
        return False

def test_core_logic_imports():
    """Test if all core logic modules can be imported"""
    print("\n🔍 Testing Core Logic Imports...")
    
    try:
        sys.path.append('app')
        
        from core_logic.lotus_handler import handle_lotus_query, check_lotus_environment
        print("✅ lotus_handler imported successfully")
        
        from core_logic.chat_history_manager import add_to_chat_history, display_chat_history
        print("✅ chat_history_manager imported successfully")
        
        from data_quality.lotus_llm_adapter import LotusLLM
        print("✅ lotus_llm_adapter imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_file_structure():
    """Check if all required files exist"""
    print("\n🔍 Testing File Structure...")
    
    required_files = [
        "app/core_logic/lotus_handler.py",
        "app/data_quality/lotus_llm_adapter.py", 
        "app/data_quality/lotus_runner.py",
        "app/core_logic/chat_history_manager.py",
        "lotus_requirements.txt"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - NOT FOUND")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests"""
    print("🚀 Starting Lotus Setup Verification...\n")
    
    tests = [
        ("File Structure", test_file_structure),
        ("Core Logic Imports", test_core_logic_imports), 
        ("Lotus Environment", test_lotus_environment),
        ("Lotus Runner", test_lotus_runner)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*50)
    print("🏁 TEST SUMMARY")
    print("="*50)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:20} : {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Lotus setup is ready.")
        print("You can now use Lotus functionality in your Streamlit app.")
    else:
        print("⚠️  Some tests failed. Please review the errors above.")
        print("Run the setup scripts or follow the troubleshooting guide.")
    print("="*50)

if __name__ == "__main__":
    main()