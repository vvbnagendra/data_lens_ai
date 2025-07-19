# app/auth/page_protection.py
import streamlit as st
from functools import wraps
from .middleware import AuthenticationMiddleware, show_login_page

def require_auth():
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            auth = AuthenticationMiddleware()
            
            if not auth.validate_session():
                show_login_page() 
                return 

            user = auth.get_current_user()
            if user:
                # --- Inject Minimal Custom CSS for Button Sizing and General Alignment ---
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
                
                # --- Streamlit Columns for the Header ---
                # A 1:6:1 ratio might work well: Small empty space, Info box, Small Logout button
                # Or 1:4:0.5 to make the button column even smaller
                col_info, col_logout = st.columns([4, 0.5]) 
                
                with col_info:
                    # Use st.info for the welcome message. The CSS above handles its alignment.
                    st.info(f"👤 Welcome, {user['full_name']} ({user['user_role']})")

                with col_logout:
                    # Use a standard Streamlit button with an icon and no text, and use_container_width=True.
                    # The CSS targets it to make it small.
                    if st.button("🚪 Logout", key="logout_button_decorator", help="Logout", use_container_width=True):
                        auth.logout_user()
                        st.rerun()
                
                # The `col_left_spacer` will naturally create a bit of space on the left.
                
            return func(*args, **kwargs)
        return wrapper
    return decorator