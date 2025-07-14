# File: app/Home.py
# UPDATED WITH CLEAN, SIMPLE STYLING

import streamlit as st

# Try to import clean styling
try:
    from app.assets.clean_styles import apply_clean_styling, create_clean_header, create_simple_card
    STYLING_AVAILABLE = True
except ImportError:
    STYLING_AVAILABLE = False

# --- Page Configuration ---
st.set_page_config(
    page_title="Smart Data Profiler",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply clean styling if available
if STYLING_AVAILABLE:
    apply_clean_styling()

# --- Header Section ---
if STYLING_AVAILABLE:
    create_clean_header("🧠 Smart Data Profiler", "AI-powered data analysis and insights")
else:
    st.title("🧠 Smart Data Profiler")
    st.markdown("AI-powered data analysis and insights")

# --- Introduction ---
st.markdown("""
Transform your data exploration with our AI-powered platform. Upload CSV files or connect to databases, 
then get intelligent insights through natural language queries.
""")

# --- Quick Start Guide ---
st.subheader("🚀 Quick Start")

col1, col2, col3 = st.columns(3)

with col1:
    if STYLING_AVAILABLE:
        create_simple_card("1. 📂 Load Data", "Upload CSV files or connect to PostgreSQL/MySQL databases")
    else:
        st.markdown("### 1. 📂 Load Data")
        st.markdown("Upload CSV files or connect to PostgreSQL/MySQL databases")

with col2:
    if STYLING_AVAILABLE:
        create_simple_card("2. 📊 Profile Data", "Generate comprehensive reports and quality insights")
    else:
        st.markdown("### 2. 📊 Profile Data")
        st.markdown("Generate comprehensive reports and quality insights")

with col3:
    if STYLING_AVAILABLE:
        create_simple_card("3. 💬 Chat with Data", "Ask questions in natural language using AI")
    else:
        st.markdown("### 3. 💬 Chat with Data")
        st.markdown("Ask questions in natural language using AI")

# --- Key Features ---
st.subheader("✨ Key Features")

feature_col1, feature_col2 = st.columns(2)

with feature_col1:
    st.markdown("""
    **🐼 PandasAI Backend:**
    - Statistical analysis & calculations
    - Data visualizations & charts
    - Python code generation
    - Complex data transformations
    """)

with feature_col2:
    st.markdown("""
    **🪷 Lotus Backend:**
    - Semantic search & filtering
    - Natural language queries
    - Text understanding
    - Intelligent data discovery
    """)

# --- Additional Features ---
st.subheader("🔍 Advanced Features")

advanced_col1, advanced_col2 = st.columns(2)

with advanced_col1:
    if STYLING_AVAILABLE:
        create_simple_card("🔍 Anomaly Detection", "Find unusual patterns and outliers in your data using ML algorithms")
    else:
        st.markdown("**🔍 Anomaly Detection**")
        st.markdown("Find unusual patterns and outliers in your data using ML algorithms")

with advanced_col2:
    if STYLING_AVAILABLE:
        create_simple_card("📋 Rule Management", "Create and manage business rules using natural language")
    else:
        st.markdown("**📋 Rule Management**")
        st.markdown("Create and manage business rules using natural language")

# --- Platform Stats ---
st.subheader("📈 Platform Capabilities")

stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

with stat_col1:
    st.metric("AI Models", "15+", help="Multiple LLM backends")

with stat_col2:
    st.metric("Data Sources", "Multiple", help="CSV, PostgreSQL, MySQL")

with stat_col3:
    st.metric("Query Types", "Unlimited", help="Statistical, visual, semantic")

with stat_col4:
    st.metric("Export Formats", "3+", help="HTML, CSV, images")

# --- Navigation ---
st.subheader("🎯 Get Started")

nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    st.page_link(
        "pages/2_Load_Data_CSV_or_Database.py",
        label="📂 Load Data",
        help="Upload CSV or connect to database"
    )

with nav_col2:
    st.page_link(
        "pages/3_Profile_Tables.py",
        label="📊 Profile Data",
        help="Generate data quality reports"
    )

with nav_col3:
    st.page_link(
        "pages/4_Chat_with_Data.py",
        label="💬 Chat with Data",
        help="Ask questions in natural language"
    )

# --- Advanced Features Navigation ---
st.subheader("🔬 Advanced Analysis")

advanced_nav_col1, advanced_nav_col2 = st.columns(2)

with advanced_nav_col1:
    st.page_link(
        "pages/5_Anomaly_Detection.py",
        label="🔍 Anomaly Detection",
        help="Find unusual patterns in your data"
    )

with advanced_nav_col2:
    if st.button("📋 Rule Management (Coming Soon)", disabled=True):
        st.info("Rule management feature will be available in the next update!")

# --- What Makes Us Different ---
st.subheader("💡 What Makes Us Different")

diff_col1, diff_col2 = st.columns(2)

with diff_col1:
    st.markdown("""
    **🎯 Intelligent Analysis:**
    - Auto-detect data types and suggest analyses
    - Smart quality checks identify issues early
    - Context-aware AI responses
    - Multi-modal insights (text, numbers, visuals)
    """)

with diff_col2:
    st.markdown("""
    **🔧 Enterprise Ready:**
    - Secure local processing
    - Handles large datasets
    - Multiple AI backends
    - Professional reporting
    """)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 6px; margin-top: 2rem;">
    <p><strong>🎉 Ready to explore your data?</strong></p>
    <p style="color: #666; font-size: 0.9rem;">
        Built with ❤️ using Streamlit, PandasAI & Lotus
    </p>
</div>
""", unsafe_allow_html=True)