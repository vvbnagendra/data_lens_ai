# Directory: app/pages/4_Chat with_data.py

import streamlit as st
import pandas as pd
import os # For path operations

# Import functions from your new core_logic modules
from core_logic.data_loader import load_all_data_sources, get_selected_dfs, get_base_name_from_selection
from core_logic.llm_config import configure_llm_backend
from core_logic.pandasai_handler import handle_pandasai_query
from core_logic.lotus_handler import handle_lotus_query
from core_logic.chat_history_manager import add_to_chat_history, display_chat_history, clear_chat_history

# --- Page Configuration ---
st.set_page_config(
    page_title="Chat with Data",
    page_icon="💬",
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

# Initialize chat history (moved to chat_history_manager implicitly, but good to have a top-level check)
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
    with st.expander(f"Preview of *{name}*"):
        st.dataframe(df.head())

st.markdown("---")

# --- LLM Backend Configuration ---
llm_backend, model_backend, model_name, user_token = configure_llm_backend()
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

# Handle question submission
if submit_button and user_question:
    response_dict = {} # This will store {"type": ..., "content": ...}

    with st.spinner("Thinking... Generating response..."):
        if llm_backend == "pandasai":
            response_dict = handle_pandasai_query(user_question, dfs, model_backend, model_name, user_token, selected_base_name)
        elif llm_backend == "lotus":
            response_dict = handle_lotus_query(user_question, dfs, model_name, user_token)
    
    add_to_chat_history(user_question, response_dict)

elif submit_button and not user_question:
    st.warning("Please enter a question before submitting.")

# Clear chat history button
clear_chat_history()

# --- Chat History Display ---
display_chat_history()