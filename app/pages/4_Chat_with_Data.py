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
from data_quality.pandasai_chat import get_smart_chat # This function likely returns a SmartDataframe or SmartDatalake instance
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
    key="llm_backend_select", # Added a key to ensure consistent state
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
        combined_models = ["hf: gpt2"]

    model_name_label = st.selectbox(
        "Select Model:",
        combined_models,
        key="model_name_select", # Added a key
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

# Using a form to group text input and button for explicit submission
# This helps manage state changes and reruns more predictably
with st.form("chat_form", clear_on_submit=True):
    user_question = st.text_input(
        "Your Question:",
        placeholder="e.g., 'What is the average sales per product category?'",
        key="user_question_input_widget" # Use a distinct key for the widget itself
    )
    submit_button = st.form_submit_button("Ask")

# --- Memoize PandasAI SmartDataframe/Datalake initialization ---
@st.cache_resource(hash_funcs={pd.DataFrame: lambda _: None}) # Exclude DataFrame from hash to avoid hashing large data
def get_pandasai_chat_instance(dataframes_list, backend_type, model_name_arg, api_key_arg):
    """
    Initializes and returns the PandasAI SmartDataframe/Datalake instance.
    This function will be memoized by Streamlit.
    """
    return get_smart_chat(
        dataframes_list,
        backend=backend_type,
        model_name=model_name_arg,
        api_key=api_key_arg
    )

# --- Function to save images ---
def save_chat_image(fig_obj, base_name, export_base_folder="export"):
    """
    Saves a Plotly figure or Matplotlib figure object to a structured export folder
    with a unique filename.

    Args:
        fig_obj: The Plotly figure object (plotly.graph_objects.Figure) or Matplotlib figure object (matplotlib.figure.Figure).
        base_name (str): The base name for the subfolder and filename (e.g., "data").
        export_base_folder (str): The main folder where exported content will reside.
    """
    if not base_name:
        st.warning("Cannot save image: No file/table selected to determine base name.")
        return None

    # Create the export directory if it doesn't exist
    # E.g., export/data/
    specific_export_folder = os.path.join(export_base_folder, base_name)
    os.makedirs(specific_export_folder, exist_ok=True)

    # Generate a unique filename using a timestamp and UUID
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8] # Short UUID for uniqueness
    
    file_name = f"chat_generated_chart_{timestamp}_{unique_id}.png"
    file_path = os.path.join(specific_export_folder, file_name)

    try:
        if hasattr(fig_obj, 'write_image'):
            # It's a Plotly figure
            # Ensure kaleido is installed for image export: pip install kaleido
            fig_obj.write_image(file_path)
            st.toast(f"📈 Chart saved to: {file_path}", icon="✅")
            return file_path
        elif isinstance(fig_obj, plt.Figure):
            # It's a Matplotlib figure
            fig_obj.savefig(file_path, bbox_inches='tight') # bbox_inches='tight' often makes better crops
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
if submit_button and user_question: # Check if button was pressed AND question is not empty
    response = None
    with st.spinner("Thinking... Generating response..."):
        if llm_backend == "pandasai":
            try:
                # Get the memoized PandasAI chat instance
                smart_chat_instance = get_pandasai_chat_instance(
                    [df_item for _, df_item in dfs], # Pass the actual DataFrame objects
                    model_backend,
                    model_name,
                    user_token
                )
                
                # Now, call chat() on the instance
                response = smart_chat_instance.chat(user_question)
            except Exception as e:
                response = f"Error from PandasAI: {e}"

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
                    response = result.stdout.decode("utf-8").strip()
                else:
                    response = f"LOTUS Error (Code: {result.returncode}): {result.stderr.decode('utf-8')}"

            except FileNotFoundError:
                response = "Error: Lotus environment or runner script not found. Please ensure it's set up correctly."
            except Exception as e:
                response = f"LOTUS Exception: {e}"
    
    # Save interaction to chat history
    st.session_state.chat_history.append({
        "question": user_question,
        "response": response
    })
    # Form's clear_on_submit=True handles clearing the input, no need to manually set session_state

elif submit_button and not user_question: # If button was clicked but input was empty
    st.warning("Please enter a question before submitting.")


# Clear chat history
if st.button("🗑 Clear Chat History", help="Removes all past questions and answers from this session.", key="clear_chat_button"):
    st.session_state.chat_history = []
    st.rerun() # Rerun to update the display (clear chat history)

# --- Chat History Display ---
st.markdown("---")
st.subheader("🧠 Chat History")

if not st.session_state.chat_history:
    st.info("Your chat history will appear here. Ask a question to begin!")

with st.expander("View Full Chat History", expanded=True):
    for entry in reversed(st.session_state.chat_history):
        st.markdown(f"🧑 You:** {entry['question']}")
        response_text = entry["response"]

        # This block handles PandasAI generated plots (expected to be plotly or matplotlib objects)
        if isinstance(response_text, str) and "fig =" in response_text:
            try:
                local_env = {"pd": pd, "px": px, "plt": plt} # Add plt to local_env
                
                # Provide a DataFrame context for plotting
                if dfs:
                    if len(dfs) == 1:
                        local_env["df"] = dfs[0][1]
                    else:
                        # Simple concatenation for plotting context if multiple DFs are selected
                        local_env["df"] = pd.concat([df_item for _, df_item in dfs], ignore_index=True)

                # Execute the generated code
                exec(response_text, {"_builtins_": None}, local_env)
                
                fig = local_env.get("fig") # Get the figure object from the executed code

                if fig:
                    # Save the chart to the export folder
                    saved_path = save_chat_image(fig, selected_base_name)
                    
                    # Display the chart in Streamlit (Plotly or Matplotlib)
                    if hasattr(fig, 'to_json'): # Check if it's a Plotly figure
                        st.plotly_chart(fig, use_container_width=True)
                    elif isinstance(fig, plt.Figure): # Check if it's a Matplotlib figure
                        # To display Matplotlib figure in Streamlit without saving it again,
                        # we can save it to an in-memory buffer and then use st.image
                        img_buffer = io.BytesIO()
                        fig.savefig(img_buffer, format='png', bbox_inches='tight')
                        img_buffer.seek(0)
                        st.image(img_buffer, caption="📊 Generated Chart", use_column_width=True)
                        plt.close(fig) # Close the figure after displaying
                else:
                    st.markdown(f"🤖 Response (Code Generated):\n\n```python\n{response_text}\n```")
            except Exception as e:
                st.error(f"⚠ Plot rendering error: {e}. Raw response: {response_text}")
                st.code(response_text.strip(), language="python")
        
        # This block handles raw code responses
        elif isinstance(response_text, str) and (
            "def " in response_text or "import " in response_text or "" in response_text
        ):
            # Clean the code blocks of markdown backticks (and optional 'python')
            # This regex correctly targets blocks like ```python\n...\n``` or ```\n...\n```
            code_block_match = re.search(r"```(?:python)?\s*\n(.*?)\n```", response_text, re.DOTALL)
            if code_block_match:
                cleaned_code = code_block_match.group(1).strip()
                st.code(cleaned_code, language="python")
            else:
                # Fallback if no code block markers are found, assume it's just code
                st.code(response_text.strip(), language="python")
        
        # This block handles direct image paths (if the LLM were to return a path, less common)
        elif isinstance(response_text, str) and re.match(r".*\.(png|jpg|jpeg|gif)$", response_text, re.IGNORECASE):
            try:
                # If the LLM returns a direct image path, display it
                st.image(response_text, caption="📊 Generated Chart", use_column_width=True)
            except Exception as e:
                st.error(f"⚠ Could not load image from path: {e}")
                st.markdown(f"🤖 Response:** {response_text}")
        
        # This block handles plain text responses
        else:
            st.markdown(f"🤖 Response:\n\n{response_text}")

        st.markdown("---")