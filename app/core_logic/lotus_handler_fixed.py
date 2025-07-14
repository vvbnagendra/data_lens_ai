import streamlit as st
import subprocess
import json
import sys
import os
import pandas as pd
import platform
from typing import List, Tuple, Dict, Any

class FixedLotusHandler:
    """
    Completely rewritten Lotus handler with robust error handling
    """
    
    def __init__(self):
        self.is_windows = platform.system() == "Windows"
        self.lotus_python = self._get_lotus_python_path()
        self.runner_script = os.path.join("app", "data_quality", "lotus_runner_fixed.py")
        self.environment_ready = False
        self._check_environment()
    
    def _get_lotus_python_path(self):
        """Get the correct Python executable path"""
        if self.is_windows:
            return os.path.normpath("./.lotus_env/Scripts/python.exe")
        else:
            return os.path.normpath("./.lotus_env/bin/python")
    
    def _check_environment(self):
        """Check if Lotus environment is ready"""
        if not os.path.exists(self.lotus_python):
            self.environment_ready = False
            return
        
        try:
            result = subprocess.run(
                [self.lotus_python, "-c", "import pandas, numpy, requests; print('OK')"],
                capture_output=True,
                text=True,
                timeout=10,
                encoding='utf-8',
                errors='replace'
            )
            
            self.environment_ready = result.returncode == 0
            
        except Exception:
            self.environment_ready = False
    
    def get_environment_status(self) -> Dict[str, Any]:
        """Get detailed environment status"""
        if not os.path.exists(self.lotus_python):
            return {
                "status": "error",
                "message": f"Python executable not found: {self.lotus_python}",
                "fix_steps": [
                    "Run: python setup_lotus_fix.py",
                    "Or manually create environment with: python -m venv .lotus_env",
                    "Then install packages: pip install pandas numpy requests matplotlib plotly"
                ]
            }
        
        if not self.environment_ready:
            return {
                "status": "error", 
                "message": "Environment exists but packages are missing or corrupted",
                "fix_steps": [
                    "Run: python setup_lotus_fix.py",
                    "This will rebuild the environment with correct packages"
                ]
            }
        
        return {
            "status": "success",
            "message": "Lotus environment ready",
            "python_path": self.lotus_python
        }
    
    def handle_query(self, user_question: str, dfs: List[Tuple[str, pd.DataFrame]], 
                    model_backend: str, model_name: str, user_token: str) -> Dict[str, Any]:
        """
        Enhanced query handling with comprehensive error recovery
        """
        
        # Pre-flight checks
        if not self.environment_ready:
            status = self.get_environment_status()
            return {
                "type": "error",
                "content": f"Environment not ready: {status['message']}",
                "fix_steps": status.get("fix_steps", [])
            }
        
        try:
            # Validate and prepare data
            valid_tables = []
            for name, df in dfs:
                if df is None or df.empty:
                    continue
                
                # Handle large datasets by sampling
                if len(df) > 10000:
                    df_sample = df.sample(n=10000, random_state=42)
                    st.info(f"📊 Large dataset '{name}' sampled to 10,000 rows for performance")
                    valid_tables.append((name, df_sample.to_csv(index=False)))
                else:
                    valid_tables.append((name, df.to_csv(index=False)))
            
            if not valid_tables:
                return {
                    "type": "error",
                    "content": "No valid data available for processing"
                }
            
            # Prepare input for subprocess
            input_data = {
                "tables": valid_tables,
                "question": user_question.strip(),
                "model": model_name or "mistral",
                "backend": model_backend or "ollama", 
                "mode": self._determine_query_mode(user_question),
                "api_key": user_token,
                "timeout": 180  # 3 minutes
            }
            
            # Execute subprocess with enhanced error handling
            return self._execute_lotus_subprocess(input_data)
            
        except Exception as e:
            return {
                "type": "error",
                "content": f"Query preparation failed: {str(e)}",
                "suggestion": "Please check your data and try again with a simpler question"
            }
    
    def _determine_query_mode(self, question: str) -> str:
        """Determine the appropriate query mode based on question content"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['sum', 'count', 'average', 'mean', 'total', 'aggregate']):
            return "aggregate"
        elif any(word in question_lower for word in ['find', 'search', 'filter', 'where', 'show']):
            return "query"
        else:
            return "query"  # Default to query mode
    
    def _execute_lotus_subprocess(self, input_data: Dict) -> Dict[str, Any]:
        """Execute Lotus subprocess with robust error handling"""
        
        try:
            # Create the runner script if it doesn't exist
            self._ensure_runner_script_exists()
            
            # Execute subprocess
            result = subprocess.run(
                [self.lotus_python, self.runner_script],
                input=json.dumps(input_data, default=str, ensure_ascii=False),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=input_data.get("timeout", 180),
                encoding='utf-8',
                errors='replace'
            )
            
            if result.returncode == 0:
                return self._process_success_response(result.stdout)
            else:
                return self._process_error_response(result.stderr, result.returncode)
                
        except subprocess.TimeoutExpired:
            return {
                "type": "error",
                "content": "Query timed out after 3 minutes. Please try a simpler question or reduce data size.",
                "suggestion": "Consider filtering your data before running complex queries"
            }
        except FileNotFoundError:
            return {
                "type": "error", 
                "content": f"Lotus runner script not found: {self.runner_script}",
                "fix_steps": ["Ensure the runner script exists in the correct location"]
            }
        except Exception as e:
            return {
                "type": "error",
                "content": f"Subprocess execution failed: {str(e)}"
            }
    
    def _process_success_response(self, stdout: str) -> Dict[str, Any]:
        """Process successful response from Lotus subprocess"""
        try:
            response = json.loads(stdout.strip())
            
            # Handle different response types
            if response.get("type") == "dataframe_json":
                # Convert JSON back to DataFrame
                df_data = response["content"]["data"]
                df_columns = response["content"]["columns"] 
                result_df = pd.DataFrame(df_data, columns=df_columns)
                return {"type": "dataframe", "content": result_df}
            
            elif response.get("type") == "image":
                image_path = response["content"]
                if not os.path.isabs(image_path):
                    image_path = os.path.abspath(image_path)
                
                if os.path.exists(image_path):
                    return {"type": "image", "content": image_path}
                else:
                    return {
                        "type": "error",
                        "content": f"Generated image not found: {image_path}"
                    }
            
            else:
                return response
                
        except json.JSONDecodeError:
            # If not JSON, treat as plain text response
            return {
                "type": "text",
                "content": stdout.strip()
            }
        except Exception as e:
            return {
                "type": "error",
                "content": f"Error processing response: {str(e)}"
            }
    
    def _process_error_response(self, stderr: str, return_code: int) -> Dict[str, Any]:
        """Process error response from subprocess"""
        
        error_msg = stderr.strip() if stderr else f"Process failed with code {return_code}"
        
        # Categorize common errors and provide solutions
        if "ModuleNotFoundError" in error_msg:
            module_name = self._extract_missing_module(error_msg)
            return {
                "type": "error",
                "content": f"Missing Python package: {module_name}",
                "fix_steps": [
                    f"Install missing package: pip install {module_name}",
                    "Or run: python setup_lotus_fix.py to reinstall environment"
                ]
            }
        
        elif "ConnectionError" in error_msg or "requests.exceptions" in error_msg:
            return {
                "type": "error", 
                "content": "Network connection error",
                "suggestion": "Check your internet connection and API credentials"
            }
        
        elif "timeout" in error_msg.lower():
            return {
                "type": "error",
                "content": "Query processing timed out",
                "suggestion": "Try a simpler query or reduce your dataset size"
            }
        
        else:
            return {
                "type": "error",
                "content": f"Lotus processing error: {error_msg}",
                "suggestion": "Please check your query syntax and try again"
            }
    
    def _extract_missing_module(self, error_msg: str) -> str:
        """Extract missing module name from error message"""
        import re
        match = re.search(r"No module named '([^']+)'", error_msg)
        return match.group(1) if match else "unknown"
    
    def _ensure_runner_script_exists(self):
        """Ensure the Lotus runner script exists"""
        if not os.path.exists(self.runner_script):
            # Create a basic runner script
            os.makedirs(os.path.dirname(self.runner_script), exist_ok=True)
            
            runner_code = '''#!/usr/bin/env python3
import sys
import json
import pandas as pd
from io import StringIO

def main():
    try:
        # Read input
        input_data = json.load(sys.stdin)
        
        # Simple processing (replace with actual Lotus logic)
        tables = input_data.get("tables", [])
        question = input_data.get("question", "")
        
        # Basic response
        response = {
            "type": "text",
            "content": f"Processed question: {question} with {len(tables)} tables"
        }
        
        print(json.dumps(response))
        
    except Exception as e:
        error_response = {
            "type": "error", 
            "content": f"Runner error: {str(e)}"
        }
        print(json.dumps(error_response))

if __name__ == "__main__":
    main()
'''
            
            with open(self.runner_script, 'w', encoding='utf-8') as f:
                f.write(runner_code)
