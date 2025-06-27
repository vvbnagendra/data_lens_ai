# Directory: app/pages/4_Chat with_data.py
# Chat with your CSV or DB Tables using pandasai or lotu-ai llms powered locally or over internet

import streamlit as st
import pandas as pd
from sqlalchemy import inspect
import subprocess
import json
import sys
import re
import os # Import os for path manipulation
import datetime # Import datetime for timestamps
import uuid # Import uuid for unique identifiers
import io # Import io for handling in-memory image buffers

# Assuming these are correctly imported from your project structure
# Make sure your get_smart_chat function in pandasai_chat.py returns an object 
# that can potentially expose the last generated chart's path, or you handle it here.
from data_quality.pandasai_chat import get_smart_chat 
from data_quality.utils import get_ollama_models, get_huggingface_models, get_google_models
import plotly.express as px # Import plotly.express for rendering plots
import matplotlib.pyplot as plt # Import matplotlib.pyplot for handling mpl plots

# --- Page Configuration ---
st.set_page_config(
    page_title="Chat with Data",
    page_icon="💬", # Icon for this page in the sidebar
    layout="wide"
)

# --- Header Section with Navigation ---
col_nav1, col_nav2, col_nav3 = st.columns([1, 4, 1])
with col_nav1:
    st.page_link("pages/3_Profile_Tables.py", label="⬅ Profile Tables", icon="📊")
with col_nav2:
    st.markdown("## 💬 Chat with Data")
with col_nav3:
    st.page_link("Home.py", label="Home 🏠", icon="🏠")

st.markdown("---")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Load data sources from session_state ---
data_sources = {}

if "csv_dataframes" in st.session_state:
    for name_key, df in st.session_state["csv_dataframes"].items():
        display_name = name_key.replace("csv_", "")
        data_sources[f"CSV: {display_name}"] = df

if "engine" in st.session_state:
    engine = st.session_state["engine"]
    try:
        inspector = inspect(engine)
        for table in inspector.get_table_names():
            df = pd.read_sql(f"SELECT * FROM {table} LIMIT 10000", engine)
            data_sources[f"DB Table: {table}"] = df
    except Exception as e:
        st.error(f"❌ Failed to load database tables for chat: {e}")

if not data_sources:
    st.warning("No data sources found. Please upload a CSV or connect to a database in the 'Load Data' page first.")
    st.stop()

st.subheader("Select Tables/CSVs to Chat With")
selected_keys = st.multiselect(
    "Select the tables or CSVs you want to chat with:",
    list(data_sources.keys()),
    default=list(data_sources.keys())[:1] if data_sources else [],
    max_selections=3,
    help="You can chat with up to 3 selected datasets simultaneously."
)

if not selected_keys:
    st.info("Please select at least one dataset to enable chat.")
    st.stop()

# Prepare dfs for PandasAI/Lotus (list of tuples: (name, dataframe))
dfs = [(key.replace("DB Table: ", "").replace("CSV: ", ""), data_sources[key]) for key in selected_keys]

# Get the base name for saving files (from the first selected key)
selected_base_name = ""
if selected_keys:
    # Extract the name without "CSV: " or "DB Table: " prefix
    selected_base_name = selected_keys[0].replace("DB Table: ", "").replace("CSV: ", "")
    # Remove file extension if it's a CSV (e.g., "data.csv" -> "data")
    selected_base_name = os.path.splitext(selected_base_name)[0]

# Display preview of selected dataframes
st.subheader("Selected Data Preview")
for name, df in dfs:
    with st.expander(f"Preview of *{name}*"):
        st.dataframe(df.head())

st.markdown("---")

# --- LLM Backend Configuration ---
st.subheader("Configure your AI Assistant")
llm_backend = st.selectbox(
    "Choose LLM Backend:",
    ["pandasai", "lotus"],
    key="llm_backend_select",
    help="*PandasAI* for more general data analysis questions. *Lotus* (if configured) for specific functionality."
)
user_token = st.text_input("🔑 Enter your API Token (optional, for some LLMs)", type="password", key="api_token_input")

model_name = None
model_backend = None

if llm_backend == "pandasai":
    ollama_models = get_ollama_models() or []
    hf_models = get_huggingface_models() or []
    google_models = get_google_models() or []

    combined_models = (
        [f"ollama: {m}" for m in ollama_models] +
        [f"hf: {m}" for m in hf_models] +
        [f"google: {m}" for m in google_models]
    )

    if not combined_models:
        combined_models = ["hf: gpt2"] # Fallback

    model_name_label = st.selectbox(
        "Select Model:",
        combined_models,
        key="model_name_select",
        help="Choose the specific language model to power your chat."
    )

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

    if model_backend == "google" and not user_token:
        st.warning("Please enter your Google API Key above if using Google models.")

else: # Lotus backend
    model_name = st.text_input("Lotus Model Name", "lotus-mixtral", key="lotus_model_name_input", help="Specify the model name for the Lotus backend.")
    if not user_token:
        st.warning("Please enter your Lotus API Token above.")

st.markdown("---")

# --- Chat Interface ---
st.subheader("Ask a Question About Your Data")

with st.form("chat_form", clear_on_submit=True):
    user_question = st.text_input(
        "Your Question:",
        placeholder="e.g., 'What is the average sales per product category?'",
        key="user_question_input_widget"
    )
    submit_button = st.form_submit_button("Ask")

# --- Memoize PandasAI SmartDataframe/Datalake initialization ---
@st.cache_resource(hash_funcs={pd.DataFrame: lambda _: None})
def get_pandasai_chat_instance(dataframes_list, backend_type, model_name_arg, api_key_arg):
    """
    Initializes and returns the PandasAI SmartDataframe/Datalake instance.
    This function will be memoized by Streamlit.
    """
    # Important: PandasAI configuration for chart saving needs to be here
    # or passed as a config dictionary to SmartDataframe/Datalake.
    # The get_smart_chat function in data_quality.pandasai_chat.py
    # should ideally handle this configuration. For now, assume it does.
    return get_smart_chat(
        dataframes_list,
        backend=backend_type,
        model_name=model_name_arg,
        api_key=api_key_arg
    )

# --- Function to save images ---
def save_chat_image(fig_obj, base_name, export_base_folder="exports/outputs"):
    """
    Saves a Plotly figure or Matplotlib figure object to a structured export folder
    with a unique filename.

    Args:
        fig_obj: The Plotly figure object (plotly.graph_objects.Figure) or Matplotlib figure object (matplotlib.figure.Figure).
        base_name (str): The base name for the subfolder (e.g., "data", "customers").
        export_base_folder (str): The main folder where exported content will reside (e.g., "exports/outputs").
    Returns:
        str: The full path to the saved image file, or None if saving failed.
    """
    if not base_name:
        st.warning("Cannot save image: No file/table selected to determine base name.")
        return None

    # Create the structured export directory: exports/outputs/{base_name}/
    specific_export_folder = os.path.join(export_base_folder, base_name)
    os.makedirs(specific_export_folder, exist_ok=True)

    # Generate a unique filename using a timestamp and UUID
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8] # Short UUID for uniqueness
    
    file_name = f"chart_{timestamp}_{unique_id}.png" # Renamed to chart_ for clarity
    file_path = os.path.join(specific_export_folder, file_name)

    try:
        if hasattr(fig_obj, 'write_image'):
            # It's a Plotly figure
            fig_obj.write_image(file_path)
            st.toast(f"📈 Chart saved to: {file_path}", icon="✅")
            return file_path
        elif isinstance(fig_obj, plt.Figure):
            # It's a Matplotlib figure
            fig_obj.savefig(file_path, bbox_inches='tight', dpi=300) # Increased DPI for better quality
            st.toast(f"📈 Chart saved to: {file_path}", icon="✅")
            plt.close(fig_obj) # Close the Matplotlib figure after saving to free up memory
            return file_path
        else:
            st.error("Unsupported plot type for saving. Please provide a Plotly figure or Matplotlib figure object.")
            return None

    except ImportError:
        st.error("Plotly image export requires `kaleido`. Please install it: `pip install kaleido`")
        return None
    except Exception as e:
        st.error(f"❌ Error saving chart to `{file_path}`: {e}")
        return None

# Handle question submission
if submit_button and user_question:
    response_content = None # Initialize content of the response
    response_type = "text" # Default response type

    with st.spinner("Thinking... Generating response..."):
        if llm_backend == "pandasai":
            try:
                smart_chat_instance = get_pandasai_chat_instance(
                    [df_item for _, df_item in dfs], # Pass the actual DataFrame objects
                    model_backend,
                    model_name,
                    user_token
                )
                
                # IMPORTANT: PandasAI's chat method *returns* the response.
                # If it generates a chart, it *might* return the path to the chart.
                # Or, it might store the chart in a configurable path.
                # The assumption here is that raw_pandasai_response will be the chart path if a chart is generated.
                raw_pandasai_response = smart_chat_instance.chat(user_question)

                # Check if the response is a recognized image path or raw plot code
                # PandasAI might return a path string directly, or code that you need to execute.
                # This logic tries to cover both.
                if isinstance(raw_pandasai_response, str) and (
                    raw_pandasai_response.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')) or "fig =" in raw_pandasai_response
                ):
                    # If PandasAI directly returns a path (e.g., from its save_charts_path)
                    if os.path.exists(raw_pandasai_response):
                        response_content = raw_pandasai_response
                        response_type = "image"
                    # If it returns Python code that generates a plot
                    elif "fig =" in raw_pandasai_response:
                        try:
                            # Context for plot execution
                            local_env = {"pd": pd, "px": px, "plt": plt}
                            if dfs:
                                if len(dfs) == 1:
                                    local_env["df"] = dfs[0][1]
                                else:
                                    local_env["df"] = pd.concat([df_item for _, df_item in dfs], ignore_index=True)

                            exec(raw_pandasai_response, {"_builtins_": None}, local_env)
                            fig = local_env.get("fig")

                            if fig:
                                saved_chart_path = save_chat_image(fig, selected_base_name)
                                if saved_chart_path:
                                    response_content = saved_chart_path
                                    response_type = "image"
                                else:
                                    response_content = f"Failed to save chart from generated code. Raw code:\n```python\n{raw_pandasai_response}\n```"
                                    response_type = "error" # Or "code" if you want to show the code
                            else:
                                response_content = f"Plot code executed, but no 'fig' object found. Raw code:\n```python\n{raw_pandasai_response}\n```"
                                response_type = "code" # Show code if no figure was made
                        except Exception as plot_e:
                            response_content = f"Error executing plot code: {plot_e}\nRaw code:\n```python\n{raw_pandasai_response}\n```"
                            response_type = "error"
                    else: # Fallback if it was a string but not a recognized image path or plot code
                        response_content = raw_pandasai_response
                        response_type = "text"
                else:
                    # Default: treat as text response
                    response_content = raw_pandasai_response
                    response_type = "text"

            except Exception as e:
                response_content = f"Error from PandasAI: {e}"
                response_type = "error"

        elif llm_backend == "lotus":
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
                lotus_python = ".\\.lotus_env\\Scripts\\python.exe" if is_windows else "./.lotus_env/bin/python"

                result = subprocess.run(
                    [lotus_python, "app/data_quality/lotus_runner.py"],
                    input=json.dumps(input_data).encode("utf-8"),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False
                )
                if result.returncode == 0:
                    lotus_raw_response = result.stdout.decode("utf-8").strip()
                    try:
                        # ASSUMPTION: Lotus returns JSON like {"type": "image", "content": "path/to/image.png"}
                        # or {"type": "text", "content": "Your text response"}
                        parsed_lotus_response = json.loads(lotus_raw_response)
                        if isinstance(parsed_lotus_response, dict) and "type" in parsed_lotus_response and "content" in parsed_lotus_response:
                            response_type = parsed_lotus_response["type"]
                            response_content = parsed_lotus_response["content"]
                            
                            # If Lotus returns an image and its path, ensure the path is correct
                            if response_type == "image":
                                # Lotus should return the absolute or relative path that Streamlit can find
                                # If lotus_runner.py saves it to exports/outputs/some_table/image.png,
                                # then this path must be accurate.
                                if not os.path.isabs(response_content):
                                    # If Lotus returns a relative path, make it relative to the app root
                                    # Adjust this if lotus_runner.py saves elsewhere relative to its own execution
                                    response_content = os.path.join(os.getcwd(), response_content)
                                if not os.path.exists(response_content):
                                    st.warning(f"Lotus returned image path: {response_content}, but file not found. Check Lotus save location.")
                                    response_type = "error" # Change type to error if image not found
                                    response_content = f"Lotus generated image path but file not found: {response_content}"
                        else:
                            response_content = f"Lotus returned unexpected JSON format: {lotus_raw_response}"
                            response_type = "error"
                    except json.JSONDecodeError:
                        response_content = lotus_raw_response # Fallback to raw if not JSON
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

    # Save interaction to chat history as a structured dictionary
    st.session_state.chat_history.append({
        "question": user_question,
        "response": {"type": response_type, "content": response_content}
    })
    # Form's clear_on_submit=True handles clearing the input, no need to manually set session_state

elif submit_button and not user_question:
    st.warning("Please enter a question before submitting.")


# Clear chat history
if st.button("🗑 Clear Chat History", help="Removes all past questions and answers from this session.", key="clear_chat_button"):
    st.session_state.chat_history = []
    st.rerun()

# --- Chat History Display ---
st.markdown("---")
st.subheader("🧠 Chat History")

if not st.session_state.chat_history:
    st.info("Your chat history will appear here. Ask a question to begin!")

with st.expander("View Full Chat History", expanded=True):
    for entry in reversed(st.session_state.chat_history):
        st.markdown(f"🧑 You:** {entry['question']}")
        
        # Ensure response_data is a dict with 'type' and 'content' for displaying
        response_data = entry["response"] 

        # Add a fallback for older history entries that might not be in the new dict format
        if not isinstance(response_data, dict) or "type" not in response_data or "content" not in response_data:
            response_data = {"type": "text", "content": str(response_data)} # Convert old entries to new format for display

        response_type = response_data["type"]
        response_content = response_data["content"]

        if response_type == "image":
            try:
                # --- DEBUGGING STEP 1 ---
                st.write(f"Attempting to load image from path: `{response_content}`")
                print(f"DEBUG: Image path received for display: {response_content}") # Check your terminal for this!
                # --- END DEBUGGING ---

                # --- DEBUGGING STEP 2 & Display ---
                if os.path.exists(response_content):
                    print(f"DEBUG: Image file EXISTS at {response_content}")
                    st.image(response_content, caption="📊 Generated Chart", use_column_width=True)
                else:
                    print(f"DEBUG: Image file DOES NOT EXIST at {response_content}")
                    st.error(f"❌ Image file not found at: `{response_content}`. Please check the path and generation process.")
                # --- END DEBUGGING ---

            except Exception as e:
                st.error(f"⚠️ Could not load image from path: {e}")
                st.markdown(f"**🤖 Response Path (Error):** {response_content}")

        elif response_type == "dataframe":
            st.markdown("🤖 Response (Data Table):")
            st.dataframe(response_content, use_container_width=True)

        elif response_type == "summary_text":
            st.markdown("🤖 Response (Data Summary):")
            st.info(response_content)

        elif response_type == "code":
            st.markdown("🤖 Response (Code Generated):")
            st.code(response_content.strip(), language="python")

        elif response_type == "error":
            st.error(f"🤖 Error: {response_content}")

        elif response_type == "text":
            st.markdown(f"🤖 Response:\n\n{response_content}")

        st.markdown("---")