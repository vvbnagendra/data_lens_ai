# File: app/assets/clean_styles.py
# MUCH SIMPLER AND CLEANER STYLES

import streamlit as st

def apply_clean_styling():
    """Apply clean, minimal styling that's not overwhelming"""
    st.markdown("""
    <style>
    /* Clean, minimal professional look */
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Simple clean header */
    .clean-header {
        background: #f8f9fa;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #0066cc;
        margin-bottom: 1.5rem;
    }
    
    .clean-title {
        color: #333;
        margin: 0;
        font-size: 1.8rem;
        font-weight: 500;
    }
    
    .clean-subtitle {
        color: #666;
        margin: 0.25rem 0 0 0;
        font-size: 0.95rem;
    }
    
    /* Simple cards */
    .simple-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .simple-card h4 {
        color: #0066cc;
        margin: 0 0 0.5rem 0;
        font-size: 1.1rem;
    }
    
    /* Clean metrics */
    .clean-metric {
        text-align: center;
        padding: 0.75rem;
        background: #f8f9fa;
        border-radius: 6px;
        border: 1px solid #e0e0e0;
    }
    
    .clean-metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #0066cc;
        margin: 0;
    }
    
    .clean-metric-label {
        font-size: 0.85rem;
        color: #666;
        margin: 0;
    }
    
    /* Status indicators */
    .status-simple {
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    .status-success { background: #d4edda; color: #155724; }
    .status-warning { background: #fff3cd; color: #856404; }
    .status-error { background: #f8d7da; color: #721c24; }
    .status-info { background: #d1ecf1; color: #0c5460; }
    
    /* Clean buttons */
    .stButton > button {
        background: #0066cc;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background: #0052a3;
    }
    
    /* Reduce spacing */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

def create_clean_header(title: str, subtitle: str = ""):
    """Create a simple, clean header"""
    header_html = f"""
    <div class="clean-header">
        <h1 class="clean-title">{title}</h1>
        {f'<p class="clean-subtitle">{subtitle}</p>' if subtitle else ''}
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

def create_simple_card(title: str, content: str):
    """Create a simple card"""
    card_html = f"""
    <div class="simple-card">
        <h4>{title}</h4>
        <p>{content}</p>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def create_clean_metric(label: str, value: str):
    """Create a clean metric display"""
    metric_html = f"""
    <div class="clean-metric">
        <div class="clean-metric-value">{value}</div>
        <div class="clean-metric-label">{label}</div>
    </div>
    """
    st.markdown(metric_html, unsafe_allow_html=True)

def create_status(text: str, status_type: str = "info"):
    """Create a simple status indicator"""
    status_html = f'<span class="status-simple status-{status_type}">{text}</span>'
    st.markdown(status_html, unsafe_allow_html=True)