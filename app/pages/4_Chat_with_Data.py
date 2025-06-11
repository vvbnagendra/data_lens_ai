# app/pages/4_Chat_with_Data.py

import streamlit as st
import pandas as pd
from sqlalchemy import inspect
import subprocess
import json
import sys

from data_quality.pandasai_chat import get_smart_chat
from data_quality.utils import get_ollama_models, get_huggingface_models

st.title("💬 Chat with Data")

# Load data sources from session_state
data_sources = {}

if "df" in st.session_state:
    data_sources["Uploaded CSV"] = st.session_state["df"]

if "engine" in st.session_state:
    inspector = inspect(st.session_state["engine"])
    for table in inspector.get_table_names():
        try:
            df = pd.read_sql(f"SELECT * FROM {table} LIMIT 1000", st.session_state["engine"])
            data_sources[f"DB Table: {table}"] = df
        except Exception as e:
            st.error(f"Failed to load {table}: {e}")

if not data_sources:
    st.warning("Upload a CSV or connect to a database first.")
    st.stop()

# Select tables to chat with
selected_keys = st.multiselect("Select table(s) or CSV(s)", list(data_sources.keys()), max_selections=3)
if not selected_keys:
    st.stop()

dfs = [(key.replace("DB Table: ", ""), data_sources[key]) for key in selected_keys]

# Show preview of selected data
if len(dfs) == 1:
    st.dataframe(dfs[0][1].head())
else:
    st.dataframe(pd.concat([df for _, df in dfs]).head())

# Choose LLM backend: pandasai or lotus
llm_backend = st.selectbox("LLM Backend", ["pandasai", "lotus"])

model_name = None
model_backend = None

if llm_backend == "pandasai":
    # Get Ollama + HuggingFace models and combine for dropdown
    ollama_models = get_ollama_models() or []
    hf_models = get_huggingface_models() or []

    combined_models = [f"ollama: {m}" for m in ollama_models] + [f"hf: {m}" for m in hf_models]

    if not combined_models:
        combined_models = ["hf: gpt2"]

    model_name_label = st.selectbox("Model Name", combined_models)

    if model_name_label.startswith("ollama: "):
        model_backend = "ollama"
        model_name = model_name_label.replace("ollama: ", "")
    elif model_name_label.startswith("hf: "):
        model_backend = "huggingface"
        model_name = model_name_label.replace("hf: ", "")
    else:
        model_backend = "huggingface"
        model_name = model_name_label

else:
    # For lotus backend, simple text input for model name
    model_name = st.text_input("Model Name", "lotus-mixtral")

# User question input
question = st.text_input("Ask a question")

if question:
    if llm_backend == "pandasai":
        try:
            print("pandas-buchiki")
            response = get_smart_chat(
                [df for _, df in dfs],
                backend=model_backend,
                model_name=model_name
            ).chat(question)
        except Exception as e:
            response = f"Error: {e}"

    elif llm_backend == "lotus":
        try:
            # Serialize tables as list of (name, CSV string)
            print("lotus-buchiki")
            tables_for_lotus = [(name, df.to_csv(index=False)) for name, df in dfs]

            input_data = {
                "tables": tables_for_lotus,
                "question": question,
                "model": model_name,
                "mode": "query"  # Keep it simple, no mode toggle here
            }

            # Path to python executable inside lotus virtual env
            is_windows = sys.platform.startswith("win")
            lotus_python = ".\\.lotus_env\\Scripts\\python.exe" if is_windows else "./.lotus_env/bin/python"

            result = subprocess.run(
                [lotus_python, "app/data_quality/lotus_runner.py"],
                input=json.dumps(input_data).encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if result.returncode == 0:
                response = result.stdout.decode("utf-8").strip()
            else:
                response = f"LOTUS Error: {result.stderr.decode('utf-8')}"

        except Exception as e:
            response = f"LOTUS Exception: {e}"

    st.markdown(f"🧠 **Response:** {response}")
