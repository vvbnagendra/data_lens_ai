# app/core_logic/lotus_handler.py
import streamlit as st
import subprocess
import json
import sys
import os
import pandas as pd
from typing import List, Tuple, Dict, Any

def handle_lotus_query(user_question: str, dfs: List[Tuple[str, pd.DataFrame]], model_backend: str, model_name: str, user_token: str) -> Dict[str, Any]:
    """
    Handles a query using the Lotus AI backend with standardized models.
    Returns a dictionary with 'type' and 'content' for the response.
    """
    
    try:
        # Prepare data for Lotus
        tables_for_lotus = []
        for name, df_item in dfs:
            # Convert DataFrame to CSV string
            csv_string = df_item.to_csv(index=False)
            tables_for_lotus.append((name, csv_string))

        # Determine query mode based on question content
        question_lower = user_question.lower()
        if any(word in question_lower for word in ['sum', 'average', 'count', 'max', 'min', 'mean', 'total', 'calculate']):
            mode = "aggregate"
        elif any(word in question_lower for word in ['search', 'find', 'look for', 'contains', 'filter', 'where']):
            mode = "query"
        else:
            mode = "query"  # Default to query mode

        input_data = {
            "tables": tables_for_lotus,
            "question": user_question,
            "model": model_name,
            "backend": model_backend,  # Pass the backend (ollama, huggingface, google)
            "mode": mode,
            "api_key": user_token
        }

        # Determine the correct Python executable path
        is_windows = sys.platform.startswith("win")
        
        # Try different possible paths for the lotus environment
        possible_paths = []
        if is_windows:
            possible_paths = [
                os.path.normpath("./.lotus_env/Scripts/python.exe"),
                os.path.normpath("./lotus_env/Scripts/python.exe"),
                os.path.normpath(".\\lotus_env\\Scripts\\python.exe"),
                "python"  # fallback to system python
            ]
        else:
            possible_paths = [
                os.path.normpath("./.lotus_env/bin/python"),
                os.path.normpath("./lotus_env/bin/python"),
                "python3",  # fallback to system python
                "python"
            ]
        
        lotus_python = None
        for path in possible_paths:
            if os.path.exists(path) or path in ["python", "python3"]:
                lotus_python = path
                break
        
        if not lotus_python:
            return {
                "type": "error",
                "content": "Lotus Python environment not found. Please ensure the .lotus_env directory exists with a proper Python installation."
            }

        # Path to the lotus runner script
        lotus_runner_script_path = os.path.join("app", "data_quality", "lotus_runner.py")
        
        if not os.path.exists(lotus_runner_script_path):
            return {
                "type": "error",
                "content": f"Lotus runner script not found at: {lotus_runner_script_path}"
            }
        
        # Run the lotus subprocess
        try:
            result = subprocess.run(
                [lotus_python, lotus_runner_script_path],
                input=json.dumps(input_data, default=str).encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=300,  # 5 minute timeout
                check=False
            )
            
            if result.returncode == 0:
                try:
                    lotus_raw_response = result.stdout.decode("utf-8").strip()
                    
                    # Parse the JSON response
                    parsed_lotus_response = json.loads(lotus_raw_response)
                    
                    if isinstance(parsed_lotus_response, dict) and "type" in parsed_lotus_response:
                        response_type = parsed_lotus_response["type"]
                        response_content = parsed_lotus_response["content"]
                        
                        # Handle different response types
                        if response_type == "dataframe_json":
                            # Convert back to DataFrame for display
                            df_data = response_content["data"]
                            df_columns = response_content["columns"]
                            result_df = pd.DataFrame(df_data, columns=df_columns)
                            return {"type": "dataframe", "content": result_df}
                        
                        elif response_type == "image":
                            # Handle image paths
                            if not os.path.isabs(response_content):
                                response_content = os.path.join(os.getcwd(), response_content)
                            if not os.path.exists(response_content):
                                return {
                                    "type": "error",
                                    "content": f"Lotus generated image path but file not found: {response_content}"
                                }
                            return {"type": "image", "content": response_content}
                        
                        else:
                            return {"type": response_type, "content": response_content}
                    
                    else:
                        return {
                            "type": "text",
                            "content": lotus_raw_response
                        }
                
                except json.JSONDecodeError:
                    return {
                        "type": "text",
                        "content": result.stdout.decode("utf-8").strip()
                    }
            
            else:
                stderr_output = result.stderr.decode('utf-8')
                return {
                    "type": "error",
                    "content": f"Lotus process failed (Code: {result.returncode}): {stderr_output}"
                }
        
        except subprocess.TimeoutExpired:
            return {
                "type": "error",
                "content": "Lotus query timed out after 5 minutes. Please try a simpler query."
            }

    except FileNotFoundError:
        return {
            "type": "error",
            "content": "Lotus environment or runner script not found. Please ensure Lotus is properly set up."
        }
    except Exception as e:
        return {
            "type": "error",
            "content": f"Lotus handler exception: {str(e)}"
        }

def check_lotus_environment() -> Dict[str, Any]:
    """
    Check if the Lotus environment is properly configured for our semantic alternative
    """
    is_windows = sys.platform.startswith("win")
    
    if is_windows:
        lotus_python = os.path.normpath("./.lotus_env/Scripts/python.exe")
    else:
        lotus_python = os.path.normpath("./.lotus_env/bin/python")
    
    if not os.path.exists(lotus_python):
        return {
            "status": "error",
            "message": f"Lotus Python executable not found at: {lotus_python}"
        }
    
    # Check if required packages are installed (not actual lotus, but our dependencies)
    try:
        result = subprocess.run(
            [lotus_python, "-c", "import pandas, requests, numpy; print('Semantic processing environment ready')"],
            capture_output=True,
            text=True,
            timeout=30,
            encoding='utf-8',
            errors='replace'  # Handle encoding issues gracefully
        )
        
        if result.returncode == 0:
            return {
                "status": "success",
                "message": "Semantic processing environment is properly configured (pandas + requests + numpy)"
            }
        else:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            return {
                "status": "error",
                "message": f"Required packages not installed: {error_msg}"
            }
    
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "message": "Environment check timed out"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error checking environment: {str(e)}"
        }