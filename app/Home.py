# File: app/Home.py
# REFACTORED WITH PROFESSIONAL COMPONENTS AND AUTHENTICATION

import streamlit as st
import base64
from assets.streamlit_styles import (
    apply_professional_styling,
    create_nav_header,
    create_pro_card,
    create_metric_card,
    create_status_badge,
)

# Authentication imports
try:
    from auth.middleware import AuthenticationMiddleware
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False

# --- Page Configuration ---
st.set_page_config(
    page_title="Data Lens AI",
    page_icon="⚡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

apply_professional_styling()

# Initialize authentication if available
if AUTH_AVAILABLE:
    auth = AuthenticationMiddleware()
    is_authenticated = auth.validate_session()
    current_user = auth.get_current_user()
    
    # Show user info bar if authenticated
    if is_authenticated:
        st.markdown(
            """
            <style>
                /* Target the Streamlit info box content for vertical centering */
                .stAlert > div:first-child {
                    display: flex;
                    align-items: center; /* Vertically center icon and text */
                    height: 100%; /* Ensure it takes full height of its container */
                    padding-top: 0.5rem; /* Adjust padding for visual balance */
                    padding-bottom: 0.5rem;
                }

                /* Target the Streamlit button for fixed width and vertical centering */
                .stButton > button {
                    min-width: 60px; /* Make the button very small */
                    max-width: 300px; /* Set a max width as well */
                            align-items: center; /* Vertically center icon and text */
                            height: 100%; /* Ensure it takes full height of its container */
                            padding-top: 0.5rem; /* Adjust padding for visual balance */
                            padding-bottom: 0.5rem;
                        }

                        /* Target the Streamlit button for fixed width and vertical centering */
                        .stButton > button {
                            min-width: 60px; /* Make the button very small */
                            max-width: 300px; /* Set a max width as well */
                            width: 100%; /* Important for responsiveness within its column */
                            height: 38px; /* Fixed height for consistent vertical alignment with info box */
                            margin: 0; /* Remove default margins */
                            display: flex; /* Enable flexbox for icon/text alignment */
                            align-items: center; /* Vertically center icon/text */
                            justify-content: center; /* Horizontally center icon/text */
                            padding: 0.25rem; /* Reduce internal padding for a smaller look */
                        }

                        /* Ensure the columns container itself aligns its items (if default is off) */
                        .st-emotion-cache-row-kind > div { /* Targeting direct children of the row to align */
                            display: flex;
                            align-items: center; /* Vertically align content in columns */
                            height: 100%; /* Occupy full height */
                        }
                         /* Give a bit of margin below the entire header row */
                        .st-emotion-cache-row-kind {
                            margin-bottom: 1rem;
                        }
                    </style>
                    """, 
                    unsafe_allow_html=True
                )
        col_info, col_logout = st.columns([4, 0.5]) 
        
        with col_info:
            # Use st.info for the welcome message. The CSS above handles its alignment.
            st.info(f"👤 Welcome, {current_user['full_name']} ({current_user['user_role']})")
        with col_logout:
            # Use a standard Streamlit button with an icon and no text, and use_container_width=True.
            # The CSS targets it to make it small.
            if st.button("🚪 Logout", key="logout_button_decorator", help="Logout", use_container_width=True):
                auth.logout_user()
                st.rerun()

else:
    is_authenticated = False
    current_user = None

# --- Navigation Header ---
create_nav_header("⚡️ Data Lens AI", "AI-powered Data Analysis and Anomaly Detection Tool")

# Load image
with open("app/assets/datalensai.png", "rb") as img_file:
    img_bytes = img_file.read()
    encoded = base64.b64encode(img_bytes).decode()

# Main content based on authentication status
if AUTH_AVAILABLE and is_authenticated:
    # Authenticated user - show personalized dashboard
    st.markdown(f"""
    <div style="display: flex; gap: 2rem; align-items: center; margin-bottom: 2rem;">
        <div style="flex: 2;">
            <h1 style="margin-bottom: 0.5rem; font-size: 2.5rem; font-weight: 700; color: #0078d4;">
                Welcome back, {current_user['full_name']}! 👋
            </h1>
            <p style="font-size: 1.2rem; color: #444; margin-bottom: 1.5rem;">
                <span style="background: #f3f9ff; padding: 0.5rem 1rem; border-radius: 8px; font-weight: 500; color: #0078d4;">
                    {current_user['user_role']} • {current_user.get('organization_name', 'Data Lens AI')}
                </span>
            </p>
        </div>
        <div style="flex: 1; text-align: center;">
            <img src="data:image/png;base64,{encoded}" alt="Data Lens AI" style="max-width: 220px; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,120,212,0.08);">
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced Navigation based on user role and permissions
    st.markdown("## 🚀 Your Available Features")
    
    # Check user permissions for navigation
    user_role = current_user['user_role']
    
    # Core features available to all authenticated users
    col_nav1, col_nav2, col_nav3, col_nav4 = st.columns(4)
    
    with col_nav1:
        if st.button("📂 Load Data", use_container_width=True, type="primary"):
            st.switch_page("pages/2_Load_Data_CSV_or_Database.py")
    
    with col_nav2:
        if st.button("📊 Profile Tables", use_container_width=True):
            st.switch_page("pages/3_Profile_Tables.py")
    
    with col_nav3:
        if st.button("💬 Chat with Data", use_container_width=True):
            st.switch_page("pages/4_Chat_with_Data.py")
    
    with col_nav4:
        if st.button("🔍 Anomaly Detection", use_container_width=True):
            st.switch_page("pages/5_Anomaly_Detection.py")
    
    # Administrative features (only for Admin/SuperAdmin)
    if user_role in ['SuperAdmin', 'Admin']:
        st.markdown("---")
        st.markdown("## 🛡️ Administrative Features")
        
        admin_col1, admin_col2, admin_col3, admin_col4 = st.columns(4)
        
        with admin_col1:
            if st.button("👥 User Management", use_container_width=True, type="secondary"):
                st.switch_page("pages/Admin_User_Management.py")
        
        with admin_col2:
            if st.button("📊 Admin Dashboard", use_container_width=True, type="secondary"):
                # This would link to the admin dashboard from your existing auth system
                try:
                    # Store the target page and the desired tab index in session state
                    st.session_state["target_page_for_tab"] = "Admin_User_Management.py" # Or whatever your page file is named
                    st.session_state["target_tab_index"] = 1 # 0-indexed: User Management (0), Admin Dashboard (1), System Settings (2), Audit Logs (3), Emergency Tools (4)
                    # Use st.switch_page to navigate to the target page
                    # You need the full path relative to your project root if it's in a 'pages' folder
                    st.switch_page("pages/Admin_User_Management.py") # Adjust path if needed (e.g., just "Admin_Dashboard_Page.py")

                except ImportError:
                    st.info("📊 Admin Dashboard - Module not found")
        
        with admin_col3:
            if st.button("🔍 Audit Logs", use_container_width=True, type="secondary"):
                st.info("🔍 Audit Logs - Feature to be implemented")
        
        with admin_col4:
            if st.button("⚙️ System Settings", use_container_width=True, type="secondary"):
                st.info("⚙️ System Settings - Available in User Management page")
    
    # Role-specific features
    st.markdown("---")
    st.markdown(f"## 🎯 {user_role} Specific Features")
    
    role_col1, role_col2, role_col3, role_col4 = st.columns(4)
    
    if user_role == 'SuperAdmin':
        with role_col1:
            st.info("🔴 **SuperAdmin Access**\n\nYou have full system privileges including user management, system settings, and all data features.")
        with role_col2:
            st.success("✅ **All Features Unlocked**\n\nAccess to every feature in the application.")
        with role_col3:
            st.warning("⚠️ **High Privileges**\n\nUse administrative features responsibly.")
        with role_col4:
            if st.button("🚨 Emergency Tools", use_container_width=True, help="Emergency administrative tools"):
                st.info("🚨 Emergency tools available in User Management")
    
    elif user_role == 'Admin':
        with role_col1:
            st.info("🟠 **Admin Access**\n\nUser management and analytics access.")
        with role_col2:
            st.success("✅ **Management Features**\n\nCreate and manage user accounts.")
        with role_col3:
            st.info("📊 **Analytics Access**\n\nView system analytics and reports.")
        with role_col4:
            st.info("🔒 **Secure Operations**\n\nAll actions are logged and audited.")
    
    elif user_role == 'DataScientist':
        with role_col1:
            st.info("🔵 **Advanced Analytics**\n\nAccess to all data analysis features.")
        with role_col2:
            st.success("✅ **ML Capabilities**\n\nMachine learning and AI tools available.")
        with role_col3:
            st.info("📈 **Data Exploration**\n\nAdvanced data exploration and visualization.")
        with role_col4:
            if st.button("🧪 Advanced Tools", use_container_width=True):
                st.info("🧪 Advanced data science tools - Feature to be expanded")
    
    elif user_role == 'BusinessAnalyst':
        with role_col1:
            st.info("🟢 **Business Insights**\n\nBusiness-focused analysis tools.")
        with role_col2:
            st.success("✅ **Reporting**\n\nCreate business reports and dashboards.")
        with role_col3:
            st.info("📊 **KPI Tracking**\n\nMonitor key business metrics.")
        with role_col4:
            if st.button("📈 Business Tools", use_container_width=True):
                st.info("📈 Business analysis tools - Feature to be expanded")
    
    elif user_role == 'Developer':
        with role_col1:
            st.info("🟡 **Development Access**\n\nTechnical features and system logs.")
        with role_col2:
            st.success("✅ **API Access**\n\nProgrammatic access to data and features.")
        with role_col3:
            st.info("🔧 **Debug Tools**\n\nDebugging and development utilities.")
        with role_col4:
            if st.button("⚙️ Dev Tools", use_container_width=True):
                st.info("⚙️ Developer tools - Feature to be expanded")
    
    else:  # Viewer
        with role_col1:
            st.info("⚪ **Viewer Access**\n\nRead-only access to data and features.")
        with role_col2:
            st.success("✅ **Data Viewing**\n\nView and explore existing data.")
        with role_col3:
            st.info("📋 **Basic Reports**\n\nGenerate basic reports and exports.")
        with role_col4:
            st.warning("🔒 **Limited Access**\n\nContact admin for additional permissions.")


else:
    # Non-authenticated user - show login prompt
    st.markdown(f"""
    <div style="display: flex; gap: 2rem; align-items: center; margin-bottom: 2rem;">
        <div style="flex: 2;">
            <h1 style="margin-bottom: 0.5rem; font-size: 2.5rem; font-weight: 700; color: #0078d4;">Welcome to Data Lens AI</h1>
            <p style="font-size: 1.2rem; color: #444; margin-bottom: 1.5rem;">
                <span style="background: #f3f9ff; padding: 0.5rem 1rem; border-radius: 8px; font-weight: 500; color: #0078d4;">AI-powered Data Analysis & Anomaly Detection</span>
            </p>
            <div style="margin-top: 1rem; display: flex; gap: 1rem; flex-wrap: wrap;">
                {"" if not AUTH_AVAILABLE else "<!-- Authentication-aware Get Started button -->"}
            </div>
        </div>
        <div style="flex: 1; text-align: center;">
            <img src="data:image/png;base64,{encoded}" alt="Data Profiler" style="max-width: 220px; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,120,212,0.08);">
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Authentication-aware Get Started button
    if AUTH_AVAILABLE:
        # User not logged in - show login button
        col_login1, col_login2, col_login3 = st.columns([1, 2, 1])
        with col_login1:
            if st.button("🔑 Login to Get Started", type="primary", use_container_width=True):
                st.switch_page("pages/Login.py")
    else:
        # Fallback if auth not available
        st.info("Authentication system not available. Please check setup.")
        
        # Show original get started button
        col_orig1, col_orig2, col_orig3 = st.columns([1, 2, 1])
        with col_orig2:
            if st.button("🚀 Get Started", type="primary", use_container_width=True):
                st.info("Please set up authentication first!")

    # Learn More link (always available)
    col_learn1, col_learn2, col_learn3 = st.columns([1, 2, 1])
    with col_learn1:
        st.markdown("""
        <div style="text-align: left; margin-top: 1rem;">
            <a href='https://manthana.ai/' target='_blank' style='background: #f3f9ff; color: #0078d4; padding: 0.85rem 2rem; border-radius: 8px; text-decoration: none; font-weight: 600; font-size: 1.1rem; border: 1px solid #0078d4; box-shadow: 0 2px 8px rgba(0,0,0,0.04);'>Learn More</a>
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
authentication_status = ""
if AUTH_AVAILABLE:
    if is_authenticated:
        authentication_status = f" • Authenticated as {current_user['user_role']}"
    else:
        authentication_status = " • Please log in for full access"

st.markdown(f"""
<div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 6px; margin-top: 2rem;">
    <p><strong>🎉 {'Secure access enabled' if AUTH_AVAILABLE and is_authenticated else 'Ready to explore your data?'}</strong></p>
    <p style="color: #666; font-size: 0.9rem;">
        Built with ❤️ using Streamlit, PandasAI & LLMs powered by AI{authentication_status}
    </p>
</div>
""", unsafe_allow_html=True)