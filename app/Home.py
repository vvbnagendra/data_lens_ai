# File: app/Home.py
# REFACTORED WITH PROFESSIONAL COMPONENTS INSPIRED BY AZURE DEVOPS BOARDS

import streamlit as st
from assets.streamlit_styles import (
    apply_professional_styling,
    create_nav_header,
    create_pro_card,
    create_metric_card,
    create_status_badge,
)

# --- Page Configuration ---
st.set_page_config(
    page_title="Data Lens AI",
    page_icon="⚡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

apply_professional_styling()

# --- Navigation Header ---
create_nav_header("⚡️ Data Lens AI", "AI-powered Data Analysis and Anomaly Detection Tool")

image_html_path = "app/assets/datalensailogo.png" # This is the path Streamlit will understand


# --- Hero Section ---
st.markdown("""
<div style="display: flex; gap: 2rem; align-items: center; margin-bottom: 2rem;">
    <div style="flex: 2;">
        <h1 style="margin-bottom: 0.5rem; font-size: 2.5rem; font-weight: 700; color: #0078d4;">Welcome to Data Lens AI</h1>
        <p style="font-size: 1.2rem; color: #444; margin-bottom: 1.5rem;">
            <span style="background: #f3f9ff; padding: 0.5rem 1rem; border-radius: 8px; font-weight: 500; color: #0078d4;">AI-powered Data Analysis & Anomaly Detection</span>
        </p>
        <div style="margin-top: 1rem; display: flex; gap: 1rem; flex-wrap: wrap;">
            <a href='Load_Data_CSV_or_Database' style='background: #0078d4; color: white; padding: 0.85rem 2rem; border-radius: 8px; text-decoration: none; font-weight: 600; font-size: 1.1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08); transition: background 0.2s;'>Get Started</a>
            <a href='https://presight.ai/' target='_blank' style='background: #f3f9ff; color: #0078d4; padding: 0.85rem 2rem; border-radius: 8px; text-decoration: none; font-weight: 600; font-size: 1.1rem; border: 1px solid #0078d4; box-shadow: 0 2px 8px rgba(0,0,0,0.04); margin-left: 0.5rem;'>Learn More</a>
        </div>
    </div>
    <div style="flex: 1; text-align: center;">
        <img src="{image_html_path}" alt="Data Profiler" style="max-width: 220px; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,120,212,0.08);">
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# --- Feature Cards ---
st.markdown("## 🚀 Quick Start")
col1, col2, col3, col4 = st.columns(4)
with col1:
    create_pro_card("Load Data", "Upload CSV files or connect databases", icon="📂")
with col2:
    create_pro_card("Profile Data", "Generate comprehensive reports", icon="📊")
with col3:
    create_pro_card("Chat with Data", "Ask questions in natural language", icon="💬")
with col4:
    create_pro_card("Advanced Analysis", "Detect anomalies and patterns", icon="🔎" )

st.markdown("<hr>", unsafe_allow_html=True)

# --- Platform Stats ---
st.markdown("## 📈 Platform Capabilities")
stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
with stat_col1:
    create_metric_card("AI Models", "15+", icon="🤖")
with stat_col2:
    create_metric_card("Data Sources", "Multiple", icon="🗄️")
with stat_col3:
    create_metric_card("Query Types", "Unlimited", icon="🔎")
with stat_col4:
    create_metric_card("Export Formats", "3+", icon="📤")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 6px; margin-top: 2rem;">
    <p><strong>🎉 Ready to explore your data?</strong></p>
    <p style="color: #666; font-size: 0.9rem;">
        Built with ❤️ using Streamlit, PandasAI & LLMs powered by AI
    </p>
</div>
""", unsafe_allow_html=True)