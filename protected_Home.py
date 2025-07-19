# protected_Home.py - Example of protected home page
# Replace your app/Home.py with this version to add authentication

import streamlit as st
import base64
from assets.streamlit_styles import (
    apply_professional_styling,
    create_nav_header,
    create_pro_card,
    create_metric_card,
    create_status_badge,
)

# Import authentication
from auth.middleware import AuthenticationMiddleware

# --- Page Configuration ---
st.set_page_config(
    page_title="Data Lens AI",
    page_icon="⚡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

apply_professional_styling()

# Initialize authentication
auth = AuthenticationMiddleware()
is_authenticated = auth.validate_session()
current_user = auth.get_current_user()

# Show user info if authenticated
if is_authenticated:
    col1, col2 = st.columns([4, 1])
    with col1:
        st.info(f"Welcome, {current_user['full_name']} ({current_user['user_role']})")
    with col2:
        if st.button("Logout"):
            auth.logout_user()
            st.rerun()

# --- Navigation Header ---
create_nav_header("⚡️ Data Lens AI", "AI-powered Data Analysis and Anomaly Detection Tool")

# Load image
with open("app/assets/datalensai.png", "rb") as img_file:
    img_bytes = img_file.read()
    encoded = base64.b64encode(img_bytes).decode()

# Main content
if is_authenticated:
    # Authenticated user view
    st.markdown(f"""
    <div style="display: flex; gap: 2rem; align-items: center; margin-bottom: 2rem;">
        <div style="flex: 2;">
            <h1 style="margin-bottom: 0.5rem; font-size: 2.5rem; font-weight: 700; color: #0078d4;">
                Welcome back, {current_user['full_name']}!
            </h1>
            <p style="font-size: 1.2rem; color: #444; margin-bottom: 1.5rem;">
                <span style="background: #f3f9ff; padding: 0.5rem 1rem; border-radius: 8px; font-weight: 500; color: #0078d4;">
                    Role: {current_user['user_role']}
                </span>
            </p>
        </div>
        <div style="flex: 1; text-align: center;">
            <img src="data:image/png;base64,{encoded}" alt="Data Lens AI" style="max-width: 220px; border-radius: 12px;">
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## Your Available Features")
    
    # Navigation buttons for authenticated users
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Load Data", use_container_width=True):
            st.switch_page("pages/2_Load_Data_CSV_or_Database.py")
    
    with col2:
        if st.button("Profile Tables", use_container_width=True):
            st.switch_page("pages/3_Profile_Tables.py")
    
    with col3:
        if st.button("Chat with Data", use_container_width=True):
            st.switch_page("pages/4_Chat_with_Data.py")
    
    with col4:
        if st.button("Anomaly Detection", use_container_width=True):
            st.switch_page("pages/5_Anomaly_Detection.py")

else:
    # Non-authenticated user view
    st.markdown(f"""
    <div style="display: flex; gap: 2rem; align-items: center; margin-bottom: 2rem;">
        <div style="flex: 2;">
            <h1 style="margin-bottom: 0.5rem; font-size: 2.5rem; font-weight: 700; color: #0078d4;">Welcome to Data Lens AI</h1>
            <p style="font-size: 1.2rem; color: #444; margin-bottom: 1.5rem;">
                <span style="background: #f3f9ff; padding: 0.5rem 1rem; border-radius: 8px; font-weight: 500; color: #0078d4;">AI-powered Data Analysis & Anomaly Detection</span>
            </p>
            <div style="margin-top: 1rem;">
                Please log in to access all features.
            </div>
        </div>
        <div style="flex: 1; text-align: center;">
            <img src="data:image/png;base64,{encoded}" alt="Data Lens AI" style="max-width: 220px; border-radius: 12px;">
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Login button for non-authenticated users
    if st.button("Login to Data Lens AI", type="primary", use_container_width=True):
        st.switch_page("app/pages/Login.py")

# Rest of your existing home page content...
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("## Platform Features")

col1, col2, col3, col4 = st.columns(4)
with col1:
    create_pro_card("Load Data", "Upload CSV files or connect databases", icon="📂")
with col2:
    create_pro_card("Profile Data", "Generate comprehensive reports", icon="📊")
with col3:
    create_pro_card("Chat with Data", "Ask questions in natural language", icon="💬")
with col4:
    create_pro_card("Advanced Analysis", "Detect anomalies and patterns", icon="🔎")

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 6px; margin-top: 2rem;">
    <p><strong>{'Secure access enabled' if is_authenticated else 'Ready to explore your data?'}</strong></p>
    <p style="color: #666; font-size: 0.9rem;">
        Built with ❤️ using Streamlit, PandasAI & LLMs powered by AI
    </p>
</div>
""", unsafe_allow_html=True)
