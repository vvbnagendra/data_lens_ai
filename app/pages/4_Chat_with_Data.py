import streamlit as st
import pandas as pd
from sqlalchemy import inspect
import subprocess
import json
import sys
import re

from data_quality.pandasai_chat import get_smart_chat
from data_quality.utils import get_ollama_models, get_huggingface_models, get_google_models

st.title("💬 Chat with Data")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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
    # Get Ollama + HuggingFace + Google models and combine for dropdown
    ollama_models = get_ollama_models() or []
    hf_models = get_huggingface_models() or []
    google_models = get_google_models() or []

    combined_models = (
        [f"ollama: {m}" for m in ollama_models] +
        [f"hf: {m}" for m in hf_models] +
        [f"google: {m}" for m in google_models]
    )

    if not combined_models:
        combined_models = ["hf: gpt2"]

    model_name_label = st.selectbox("Model Name", combined_models)

    if model_name_label.startswith("ollama: "):
        model_backend = "ollama"
        model_name = model_name_label.replace("ollama: ", "")
    elif model_name_label.startswith("hf: "):
        model_backend = "huggingface"
        model_name = model_name_label.replace("hf: ", "")
    elif model_name_label.startswith("google: "):
        model_backend = "google"
        model_name = model_name_label.replace("google: ", "")
    else:
        model_backend = "huggingface"
        model_name = model_name_label

else:
    model_name = st.text_input("Model Name", "lotus-mixtral")

# User question input
question = st.text_input("Ask a question")

# Handle question submission
if question:
    response = None

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
            print("lotus-buchiki")
            tables_for_lotus = [(name, df.to_csv(index=False)) for name, df in dfs]

            input_data = {
                "tables": tables_for_lotus,
                "question": question,
                "model": model_name,
                "mode": "query"
            }

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

    # Save interaction to chat history
    st.session_state.chat_history.append({
        "question": question,
        "response": response
    })

# Clear chat history
if st.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []
    st.experimental_rerun()

# Render chat history
# Render chat history
st.markdown("---")
st.markdown("### 🧠 Chat History")

for entry in reversed(st.session_state.chat_history):
    st.markdown(f"**🧑 You:** {entry['question']}")
    response = entry["response"]


    # Check for and render Plotly figure
    if isinstance(response, str) and "fig =" in response:
        try:
            local_env = {"pd": pd}
            if len(dfs) == 1:
                local_env["df"] = dfs[0][1]
            else:
                local_env["df"] = pd.concat([df for _, df in dfs])

            exec(response, {}, local_env)
            fig = local_env.get("fig")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                continue
        except Exception as e:
            st.error(f"⚠️ Plot rendering error: {e}")
            continue

    # Format as code if response looks like code
    if isinstance(response, str) and (
        "def " in response or "import " in response or "```" in response
    ):
        cleaned = re.sub(r"```(?:python)?", "", response).replace("```", "")
        st.code(cleaned.strip(), language="python")
    else:
    # If the response is a local image file path, show it as an image
        if isinstance(response, str) and re.match(r".*\.(png|jpg|jpeg)$", response, re.IGNORECASE):
            try:
                st.image(response, caption="📊 Generated Chart", use_column_width=True)
            except Exception as e:
                st.error(f"⚠️ Could not load image: {e}")
        else:
            st.markdown(f"**🤖 Response:**\n\n{response}", unsafe_allow_html=True)

