# app/pages/Login.py - Working login page
import streamlit as st
import sys
import os

# Add the app directory to Python path so we can import auth modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from auth.middleware import AuthenticationMiddleware
    AUTH_AVAILABLE = True
except ImportError as e:
    AUTH_AVAILABLE = False
    st.error(f"Authentication system not available: {e}")

def main():
    """Main login page function"""

    st.set_page_config(
        page_title="Login - Data Lens AI",
        page_icon="🔐",
        layout="centered"
    )

    if not AUTH_AVAILABLE:
        st.error("❌ Authentication system not available!")
        st.info("Please run the setup script to configure authentication.")
        return

    auth = AuthenticationMiddleware()

    # --- IMMEDIATE REDIRECTION FOR ALREADY AUTHENTICATED USERS ---
    if auth.validate_session():
        # Get user info for a brief message, though the redirect is fast
        current_user = auth.get_current_user()
        st.info(f"✅ Already logged in as {current_user['full_name']}. Redirecting to home page...")
        st.switch_page("Home.py")
        # IMPORTANT: Use st.stop() to immediately halt execution after switch_page
        # This prevents any further UI elements from rendering on this page before redirect.
        st.stop()


    # Login form
    st.title("🔐 Data Lens AI Login")
    st.markdown("Enter your credentials to access the platform")

    with st.form("login_form", clear_on_submit=False):
        st.markdown("### Login Credentials")

        email = st.text_input(
            "Email Address",
            placeholder="admin@datalensai.com",
            help="Enter your registered email address"
        )

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password",
            help="Enter your account password"
        )

        col1, col2 = st.columns(2)
        with col1:
            login_submitted = st.form_submit_button("🔑 Login", type="primary", use_container_width=True)
        with col2:
            clear_form = st.form_submit_button("🔄 Clear", use_container_width=True)

        if clear_form:
            st.rerun()

        if login_submitted:
            if not email or not password:
                st.error("❌ Please enter both email and password")
            else:
                with st.spinner("🔐 Authenticating..."):
                    result = auth.login_user(email, password)

                if result['success']:
                    st.success("✅ Login successful! Redirecting...")
                    # --- DIRECT REDIRECTION AFTER SUCCESSFUL LOGIN ---
                    st.switch_page("Home.py")
                    # IMPORTANT: Use st.stop() to immediately halt execution
                    # This prevents further UI elements from rendering after successful login.
                    st.stop()
                else:
                    st.error(f"❌ Login failed: {result['error']}")

    # Help section
    st.markdown("---")
    with st.expander("ℹ️ Login Help"):
        st.markdown("""
        **Default Credentials:**
        - Email: `admin@datalensai.com`
        - Password: `DataLens2024!`

        **Troubleshooting:**
        - Make sure you're using the correct email and password
        - Check that the authentication system is properly set up
        - Contact your administrator if you've forgotten your credentials

        **Security Note:**
        - Please change the default password after first login
        - Your session will expire after 24 hours of inactivity
        """)

if __name__ == "__main__":
    main()