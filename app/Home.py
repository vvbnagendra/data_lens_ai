# Directory: app/Home.py
# this is landing page
import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="Smart Data Profiler",
    page_icon="🧠", # Icon for this page in the sidebar
    layout="wide",
    initial_sidebar_state="expanded" # Keep sidebar expanded by default
)

# --- Header Section ---
col1, col2 = st.columns([1, 5])
with col1:
    st.image("app/assets/logo.png", width=100)
with col2:
    st.title("🧠 Smart Data Profiler")
    st.markdown("Unlock insights from your data with ease and intelligence.")

st.markdown("---") # Add a horizontal line for visual separation

# --- Introduction & Benefits ---
st.header("Welcome to Your Data Exploration Hub!")
st.write(
    """
    The *Smart Data Profiler* is a powerful tool designed to help you quickly understand,
    analyze, and interact with your databases and CSV files. Whether you're a data analyst,
    scientist, or developer, our intuitive application streamlines data discovery
    and quality assessment.
    """
)

st.subheader("🚀 Key Features & Benefits:")
st.markdown(
    """
    -   *Rapid Data Ingestion:* Seamlessly connect to various SQL databases or upload CSV files.
    -   *Comprehensive Data Profiling:* Generate detailed statistics, distributions, and visual summaries for your datasets.
    -   *Intelligent Data Chat:* Interact with your data using natural language queries powered by cutting-edge LLMs.
    -   *Proactive Quality Checks:* Automatically identify common data quality issues like missing values, duplicates, and inconsistencies.
    -   *Time-Saving Automation:* Automate the tedious aspects of initial data exploration, freeing you to focus on insights.
    """
)

st.markdown("---")

# --- How to Get Started ---
st.header("Ready to Dive In?")
st.write(
    """
    Follow these simple steps to start your data profiling journey:
    """
)

st.markdown(
    """
    1.  *📂 Load Data:* Connect to your database or upload your CSV files.
    2.  *📊 Profile Tables:* Generate comprehensive reports and visualize your data.
    3.  *💬 Chat with Data:* Ask questions and get answers directly from your datasets.
    """
)

st.info("💡 *Tip:* Use the navigation links in the sidebar or the 'Get Started' button below to move through the application workflow.")

# --- Call to Action Button ---
st.markdown("<br>", unsafe_allow_html=True) # Add some space
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    st.page_link(
        "app/pages/2_Load_Data_CSV_or_Database.py",
        label="🚀 Get Started: Load Your Data",
        icon="👉",
        help="Click to start by loading your data from CSV or a database."
    )

st.markdown("---")

# --- Footer ---
st.caption("© 2023-2025 Smart Data Profiler. All rights reserved.")
st.markdown(
    """
    <style>
    .reportview-container .main .block-container{
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)