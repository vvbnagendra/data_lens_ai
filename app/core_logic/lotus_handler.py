# core_logic/lotus_handler.py
import streamlit as st
import subprocess
import json
import sys
import os # For os.getcwd, os.path.join

def handle_lotus_query(user_question, dfs, model_name, user_token):
    """
    Handles a query using the Lotus AI backend.
    Returns a dictionary with 'type' and 'content' for the response.
    """
    response_content = None
    response_type = "text"

    try:
        tables_for_lotus = [(name, df_item.to_csv(index=False)) for name, df_item in dfs]

        input_data = {
            "tables": tables_for_lotus,
            "question": user_question,
            "model": model_name,
            "mode": "query",
            "api_key": user_token
        }

        is_windows = sys.platform.startswith("win")
        # Ensure this path is correct relative to where Streamlit is run
        lotus_python = os.path.normpath(".\\.lotus_env\\Scripts\\python.exe") if is_windows else os.path.normpath("./.lotus_env/bin/python")

        # Path to your lotus_runner.py relative to the Streamlit app root
        lotus_runner_script_path = "app/data_quality/lotus_runner.py" 
        
        result = subprocess.run(
            [lotus_python, lotus_runner_script_path],
            input=json.dumps(input_data).encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False
        )
        
        if result.returncode == 0:
            lotus_raw_response = result.stdout.decode("utf-8").strip()
            try:
                parsed_lotus_response = json.loads(lotus_raw_response)
                if isinstance(parsed_lotus_response, dict) and "type" in parsed_lotus_response and "content" in parsed_lotus_response:
                    response_type = parsed_lotus_response["type"]
                    response_content = parsed_lotus_response["content"]
                    
                    if response_type == "image":
                        # If Lotus returns a relative path, make it absolute for Streamlit
                        if not os.path.isabs(response_content):
                            response_content = os.path.join(os.getcwd(), response_content)
                        if not os.path.exists(response_content):
                            st.warning(f"Lotus returned image path: {response_content}, but file not found. Check Lotus save location.")
                            response_type = "error"
                            response_content = f"Lotus generated image path but file not found: {response_content}"
                else:
                    response_content = f"Lotus returned unexpected JSON format: {lotus_raw_response}"
                    response_type = "error"
            except json.JSONDecodeError:
                response_content = lotus_raw_response
                response_type = "text"
        else:
            response_content = f"LOTUS Error (Code: {result.returncode}): {result.stderr.decode('utf-8')}"
            response_type = "error"

    except FileNotFoundError:
        response_content = "Error: Lotus environment or runner script not found. Please ensure it's set up correctly."
        response_type = "error"
    except Exception as e:
        response_content = f"LOTUS Exception: {e}"
        response_type = "error"
    
    return {"type": response_type, "content": response_content}