# app/pages/4_Chat_with_Data.py

import streamlit as st
import pandas as pd
import os

# Import functions from your core_logic modules
from core_logic.data_loader import load_all_data_sources, get_selected_dfs, get_base_name_from_selection
from core_logic.llm_config import configure_llm_backend
from core_logic.pandasai_handler import handle_pandasai_query
from core_logic.lotus_handler import handle_lotus_query, check_lotus_environment
from core_logic.chat_history_manager import add_to_chat_history, display_chat_history, clear_chat_history
from assets.streamlit_styles import apply_professional_styling, create_nav_header

# --- Page Configuration ---
st.set_page_config(
    page_title="Chat with Data",
    page_icon="💬",
    layout="wide"
)

apply_professional_styling()

# --- Navigation Header ---
create_nav_header("💬 Chat with Data", "Ask questions and get insights from your data using AI")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Load and Select Data ---
data_sources = load_all_data_sources()

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

dfs = get_selected_dfs(data_sources, selected_keys)
selected_base_name = get_base_name_from_selection(selected_keys)

# Display preview of selected dataframes
st.subheader("Selected Data Preview")
for name, df in dfs:
    with st.expander(f"Preview of **{name}**"):
        st.dataframe(df.head())

st.markdown("---")

# --- LLM Backend Configuration ---
llm_backend, model_backend, model_name, user_token = configure_llm_backend()

# Check Lotus environment if Lotus is selected
if llm_backend == "lotus":
    with st.expander("🔍 Semantic Processing Environment Status", expanded=True):
        lotus_status = check_lotus_environment()
        
        if lotus_status["status"] == "success":
            st.success(f"✅ {lotus_status['message']}")
            st.info("💡 Using enhanced semantic processing (no complex Lotus-AI installation required)")
        else:
            st.error(f"❌ {lotus_status['message']}")
            st.markdown("""
            **To fix this issue:**
            1. **Quick Fix** - Recreate the environment:
               ```bash
               # Remove old environment
               rm -rf .lotus_env
               
               # Create new simple environment
               python -m venv .lotus_env
               
               # Windows activation:
               .lotus_env\\Scripts\\activate
               pip install pandas numpy requests
               
               # Linux/Mac activation:
               source .lotus_env/bin/activate
               pip install pandas numpy requests
               ```
            2. **Or run the setup script**: `./setup_lotus_env.sh` (Linux/Mac) or `./setup_lotus_env.ps1` (Windows)
            """)
            
            # Add a quick fix button
            if st.button("🔧 Quick Environment Check", key="quick_env_check"):
                st.info("🔄 Checking environment status...")
                # Rerun the check
                st.rerun()

st.markdown("---")
with st.form("chat_form", clear_on_submit=True):
    user_question = st.text_input(
        "Your Question:",
        placeholder="e.g., 'What is the average sales per product category?'",
        key="user_question_input_widget"
    )
    
    # Additional options for advanced users
    with st.expander("🔧 Advanced Options", expanded=False):
        col_opt1, col_opt2 = st.columns(2)
        
        with col_opt1:
            if llm_backend == "lotus":
                query_mode = st.selectbox(
                    "Query Mode:",
                    ["auto", "query", "aggregate"],
                    help="Auto: system decides, Query: filter/search data, Aggregate: calculate statistics"
                )
            else:
                st.info("💡 PandasAI automatically determines the best approach for your question")
        
        with col_opt2:
            max_results = st.number_input(
                "Max Results:", 
                min_value=10, 
                max_value=1000, 
                value=100,
                help="Maximum number of rows to return"
            )
    
    submit_button = st.form_submit_button("Ask", type="primary")

# Handle question submission
if submit_button and user_question:
    response_dict = {}

    with st.spinner("🤔 Thinking... Generating response..."):
        try:
            if llm_backend == "pandasai":
                response_dict = handle_pandasai_query(
                    user_question, dfs, model_backend, model_name, user_token, selected_base_name
                )
            elif llm_backend == "lotus":
                # Check environment before proceeding
                lotus_status = check_lotus_environment()
                if lotus_status["status"] != "success":
                    response_dict = {
                        "type": "error",
                        "content": f"Lotus environment not ready: {lotus_status['message']}"
                    }
                else:
                    response_dict = handle_lotus_query(user_question, dfs, model_backend, model_name, user_token)
            else:
                response_dict = {
                    "type": "error", 
                    "content": f"Unknown LLM backend: {llm_backend}"
                }
        except Exception as e:
            response_dict = {
                "type": "error",
                "content": f"Error processing question: {str(e)}"
            }
    
    # Add to chat history
    add_to_chat_history(user_question, response_dict)
    
    # Display the current response immediately
    st.markdown("### 🤖 Latest Response:")
    response_type = response_dict.get("type", "text")
    response_content = response_dict.get("content", "No response")
    
    if response_type == "error":
        st.error(f"❌ {response_content}")
    elif response_type == "dataframe":
        st.dataframe(response_content, use_container_width=True)
        # Show basic stats for dataframe results
        if isinstance(response_content, pd.DataFrame) and not response_content.empty:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", len(response_content))
            with col2:
                st.metric("Columns", len(response_content.columns))
            with col3:
                st.metric("Memory Usage", f"{response_content.memory_usage(deep=True).sum() / 1024:.1f} KB")
    elif response_type == "image":
        if os.path.exists(response_content):
            st.image(response_content, caption="Generated Chart", use_column_width=True)
        else:
            st.error(f"Image file not found: {response_content}")
    elif response_type == "code":
        st.code(response_content, language="python")
    elif response_type == "summary_text":
        st.info(response_content)
    else:
        st.write(response_content)

elif submit_button and not user_question:
    st.warning("Please enter a question before submitting.")

# Clear chat history button and stats
st.markdown("---")

col_clear, col_stats = st.columns([1, 2])
with col_clear:
    clear_chat_history()
with col_stats:
    if hasattr(st.session_state, 'chat_history') and st.session_state.chat_history:
        st.metric("Chat History", f"{len(st.session_state.chat_history)} questions")

# --- Chat History Display ---
display_chat_history()

# --- Footer with helpful information ---
st.markdown("---")

# Performance warning for large datasets
total_rows = sum(len(df) for _, df in dfs)
if total_rows > 50000:
    st.warning(f"""
    ⚠️ **Large Dataset Notice**: You're working with {total_rows:,} rows. 
    For better performance:
    - Ask more specific questions to reduce processing time
    - Consider filtering data first before complex analysis
    - Lotus queries might be faster for simple filtering operations
    """)

# Debug information (only show in development)
if st.checkbox("🐛 Show Debug Info", key="debug_mode"):
    st.markdown("### Debug Information")
    st.json({
        "llm_backend": llm_backend,
        "model_backend": model_backend, 
        "model_name": model_name,
        "selected_keys": selected_keys,
        "total_dataframes": len(dfs),
        "total_rows": total_rows,
        "session_state_keys": list(st.session_state.keys())
    })