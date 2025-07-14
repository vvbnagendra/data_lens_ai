# QUICK FIX: Create these files to resolve the import error

# 1. Create directory: app/assets/
# 2. Create file: app/assets/__init__.py (empty file)
# 3. Create file: app/assets/streamlit_styles.py (content below)

# File: app/assets/streamlit_styles.py

import streamlit as st

def get_professional_css():
    """Returns properly formatted CSS for Streamlit applications"""
    return """
    <style>
    /* Professional Theme */
    :root {
        --primary-blue: #0078d4;
        --primary-dark: #005a9e;
        --success-green: #107c10;
        --warning-orange: #ff8c00;
        --error-red: #d13438;
        --neutral-gray: #605e5c;
        --light-gray: #f3f2f1;
        --white: #ffffff;
        --card-shadow: 0 2px 8px rgba(0,0,0,0.1);
        --card-shadow-hover: 0 4px 16px rgba(0,0,0,0.15);
        --border-radius: 8px;
        --transition: all 0.3s ease;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Professional Navigation Header */
    .nav-header {
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--primary-dark) 100%);
        padding: 1.5rem 2rem;
        border-radius: var(--border-radius);
        margin-bottom: 2rem;
        color: var(--white);
        box-shadow: var(--card-shadow);
    }

    .nav-title {
        font-size: 2.5rem;
        font-weight: 600;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        color: white !important;
    }

    .nav-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0.5rem 0 0 0;
        color: white !important;
    }

    /* Professional Cards */
    .pro-card {
        background: var(--white);
        border-radius: var(--border-radius);
        padding: 1.5rem;
        box-shadow: var(--card-shadow);
        border: 1px solid #e1dfdd;
        transition: var(--transition);
        margin-bottom: 1rem;
    }

    .pro-card:hover {
        box-shadow: var(--card-shadow-hover);
        transform: translateY(-2px);
    }

    .pro-card-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--primary-blue);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .pro-card-content {
        color: var(--neutral-gray);
        line-height: 1.6;
    }

    /* Metric Cards */
    .metric-card {
        background: var(--white);
        border-radius: var(--border-radius);
        padding: 1.5rem;
        text-align: center;
        box-shadow: var(--card-shadow);
        border-left: 4px solid var(--primary-blue);
        transition: var(--transition);
        margin-bottom: 1rem;
    }

    .metric-card:hover {
        box-shadow: var(--card-shadow-hover);
        transform: translateY(-2px);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-blue);
        margin-bottom: 0.5rem;
        display: block;
    }

    .metric-label {
        font-size: 0.875rem;
        color: var(--neutral-gray);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        display: block;
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.875rem;
        font-weight: 500;
    }

    .status-success {
        background: #f3f9f1;
        color: var(--success-green);
        border: 1px solid #c4e3c0;
    }

    .status-warning {
        background: #fef7e6;
        color: var(--warning-orange);
        border: 1px solid #f4d091;
    }

    .status-error {
        background: #fdf3f4;
        color: var(--error-red);
        border: 1px solid #e8a4a6;
    }

    .status-info {
        background: #f3f9ff;
        color: var(--primary-blue);
        border: 1px solid #b3d6f7;
    }

    /* Enhanced Streamlit widgets */
    .stButton > button {
        background: var(--primary-blue);
        color: white;
        border: none;
        border-radius: var(--border-radius);
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: var(--transition);
    }

    .stButton > button:hover {
        background: var(--primary-dark);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,120,212,0.3);
    }

    /* Animation */
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """

def apply_professional_styling():
    """Apply professional styling to any Streamlit page"""
    st.markdown(get_professional_css(), unsafe_allow_html=True)

def create_nav_header(title: str, subtitle: str = ""):
    """Create a professional navigation header"""
    header_html = f"""
    <div class="nav-header fade-in">
        <h1 class="nav-title">{title}</h1>
        {f'<p class="nav-subtitle">{subtitle}</p>' if subtitle else ''}
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

def create_pro_card(header: str, content: str, icon: str = ""):
    """Create a professional card component"""
    card_html = f"""
    <div class="pro-card">
        <div class="pro-card-header">
            {f'<span>{icon}</span>' if icon else ''}{header}
        </div>
        <div class="pro-card-content">
            {content}
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def create_metric_card(label: str, value: str, icon: str = ""):
    """Create a professional metric card"""
    metric_html = f"""
    <div class="metric-card">
        {f'<div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>' if icon else ''}
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """
    st.markdown(metric_html, unsafe_allow_html=True)

def create_status_badge(text: str, status_type: str = "info"):
    """Create a status badge"""
    badge_html = f"""
    <span class="status-badge status-{status_type}">
        {text}
    </span>
    """
    st.markdown(badge_html, unsafe_allow_html=True)

# File: app/assets/__init__.py
# (This file can be empty - it just makes the directory a Python package)