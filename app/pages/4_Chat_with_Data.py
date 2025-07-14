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

# --- Chat Interface ---
st.subheader("Ask a Question About Your Data")

# Example questions based on backend
if llm_backend == "lotus":
    st.info("""
    **Lotus Examples:**
    - "Find all records where sales > 1000"
    - "Search for customers in California"
    - "Show me the top 5 products by revenue"
    - "Filter data where category contains 'electronics'"
    """)
else:
    st.info("""
    **PandasAI Examples:**
    - "What is the average sales per product category?"
    - "Show me a histogram of customer ages"
    - "Create a scatter plot of price vs quantity"
    - "What are the top 10 selling products?"
    - "Calculate the correlation between variables"
    """)

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

# --- Sidebar with helpful information ---
with st.sidebar:
    st.markdown("### 🔧 Current Configuration")
    st.info(f"""
    **Backend:** {llm_backend.title()}
    **Model:** {model_name or 'Default'}
    **Data Sources:** {len(selected_keys)}
    **Total Rows:** {sum(len(df) for _, df in dfs):,}
    """)
    
    if llm_backend == "lotus":
        st.markdown("### 🪷 Lotus Features (Same Models as PandasAI)")
        st.markdown("""
        - **Semantic Search**: Find records by meaning, not just exact matches
        - **Natural Filtering**: Use plain language conditions like "high sales" or "recent orders"
        - **Smart Aggregation**: Automatic grouping and calculations based on context
        - **Text Understanding**: Search within text columns intelligently
        - **Model Support**: Uses same Ollama, HuggingFace & Google models as PandasAI
        """)
    else:
        st.markdown("### 🐼 PandasAI Features")
        st.markdown("""
        - **Data Analysis**: Statistical calculations and insights
        - **Visualizations**: Automatic chart and plot generation
        - **Code Generation**: Python/Pandas code for your queries
        - **Multiple Models**: Various LLM backends (Ollama, HuggingFace, Google)
        - **Advanced Analytics**: Complex data transformations and analysis
        """)
    
    st.markdown("### 📊 Model Information")
    st.markdown(f"""
    **🔧 Current Configuration:**
    - **Backend**: {llm_backend.title()}
    - **Model Provider**: {model_backend.title()}
    - **Model**: {model_name}
    - **Mode**: {'Semantic Processing' if llm_backend == 'lotus' else 'Data Analysis'}
    """)
    
    st.markdown("### 📊 Quick Actions")
    if st.button("📈 Show Data Summary", key="quick_summary"):
        summary_data = []
        for name, df in dfs:
            summary_data.append({
                "Dataset": name,
                "Rows": len(df),
                "Columns": len(df.columns),
                "Memory (KB)": f"{df.memory_usage(deep=True).sum() / 1024:.1f}"
            })
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
    
    if st.button("🔍 Show Column Info", key="quick_columns"):
        for name, df in dfs:
            st.markdown(f"**{name}:**")
            col_info = pd.DataFrame({
                "Column": df.columns,
                "Type": df.dtypes.astype(str),
                "Non-Null": df.count(),
                "Null %": (df.isnull().sum() / len(df) * 100).round(1)
            })
            st.dataframe(col_info, use_container_width=True)

# --- Footer with helpful information ---
st.markdown("---")
st.markdown("### 💡 Tips for Better Results:")

if llm_backend == "lotus":
    st.markdown("""
    **🪷 Lotus Best Practices (Using Standard Models):**
    - **Natural Language Filtering**: "customers in California with high revenue"
    - **Semantic Search**: "find similar products" or "search for complaints"
    - **Smart Conditions**: "sales greater than average" or "recent transactions"
    - **Text Understanding**: "find positive reviews" or "locate error messages"
    - **Model Consistency**: Same Ollama/HuggingFace/Google models as PandasAI
    """)
else:
    st.markdown("""
    **🐼 PandasAI Best Practices:**
    - **Specific Calculations**: "calculate the mean, median, and mode of sales"
    - **Request Visualizations**: "create a bar chart of sales by category"
    - **Column References**: "plot price vs quantity with color by category"
    - **Statistical Analysis**: "show correlation matrix" or "detect outliers"
    - **Code Generation**: "generate code to clean this data"
    """)

st.markdown("""
**🎯 General Tips:**
- **Be Specific**: Mention exact column names when possible
- **One Question at a Time**: Break complex requests into smaller parts
- **Use Examples**: Reference specific values or ranges in your data
- **Try Variations**: If you get an error, rephrase your question
- **Check Data First**: Use the preview to understand your data structure
""")

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