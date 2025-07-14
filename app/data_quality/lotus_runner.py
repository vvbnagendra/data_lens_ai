#!/usr/bin/env python3
# app/data_quality/lotus_runner.py
import sys
import os
import json
import pandas as pd
from io import StringIO
from typing import Dict, Any

# Add the app directory to Python path
app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, app_dir)

try:
    # Try to import actual Lotus first
    from data_quality.lotus_llm_adapter import LotusLLM
    USING_REAL_LOTUS = True
except ImportError:
    try:
        # Fallback to semantic alternative
        from data_quality.semantic_llm_adapter import SemanticLLM as LotusLLM
        USING_REAL_LOTUS = False
    except ImportError as e:
        print(json.dumps({
            "type": "error",
            "content": f"Failed to import any Lotus implementation: {e}"
        }))
        sys.exit(1)

def main():
    try:
        # Read input from stdin
        input_data = json.load(sys.stdin)
        
        # Extract data
        raw_tables = input_data.get("tables", [])
        question = input_data.get("question", "")
        model = input_data.get("model", "mistral")
        backend = input_data.get("backend", "ollama")
        mode = input_data.get("mode", "query")
        api_key = input_data.get("api_key")
        
        if not question:
            print(json.dumps({
                "type": "error",
                "content": "No question provided"
            }))
            return
        
        # Convert CSV strings back to DataFrames
        dfs = []
        for name, table_csv in raw_tables:
            try:
                df = pd.read_csv(StringIO(table_csv))
                dfs.append((name, df))
            except Exception as e:
                print(json.dumps({
                    "type": "error",
                    "content": f"Error parsing CSV for table {name}: {e}"
                }))
                return
        
        if not dfs:
            print(json.dumps({
                "type": "error", 
                "content": "No valid dataframes provided"
            }))
            return
        
        # Initialize Lotus LLM (real or semantic alternative)
        lotus = LotusLLM(dfs, model=model, backend=backend, mode=mode, api_key=api_key)
        
        # Run the query
        if USING_REAL_LOTUS:
            response = lotus.run_lotus(question)
        else:
            response = lotus.run_semantic_query(question)
        
        # Handle DataFrame responses
        if response.get("type") == "dataframe" and isinstance(response.get("content"), pd.DataFrame):
            df_result = response["content"]
            # Convert DataFrame to JSON for transmission
            response["content"] = {
                "data": df_result.to_dict('records'),
                "columns": df_result.columns.tolist(),
                "shape": df_result.shape
            }
            response["type"] = "dataframe_json"
        
        # Add info about which implementation was used
        if not USING_REAL_LOTUS:
            if response.get("type") != "error":
                original_content = response.get("content", "")
                if isinstance(original_content, str):
                    response["content"] = f"🔍 Semantic Analysis: {original_content}"
                elif isinstance(original_content, dict):
                    response["semantic_mode"] = True
        
        # Output the response as JSON
        print(json.dumps(response, indent=2, default=str))
        
    except json.JSONDecodeError as e:
        print(json.dumps({
            "type": "error",
            "content": f"Invalid JSON input: {e}"
        }))
    except Exception as e:
        print(json.dumps({
            "type": "error",
            "content": f"Unexpected error in lotus_runner: {e}"
        }))

if __name__ == "__main__":
    main()