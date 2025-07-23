# File: app/auth/middleware.py
# Enhanced Authentication Middleware with Password Reset Popup

import streamlit as st
import sqlite3
from .models import AuthenticationDB

class AuthenticationMiddleware:
    def __init__(self):
        self.db = AuthenticationDB()

        if 'auth_user' not in st.session_state:
            st.session_state.auth_user = None
        if 'auth_session_id' not in st.session_state:
            st.session_state.auth_session_id = None
        if 'password_reset_required' not in st.session_state:
            st.session_state.password_reset_required = False

    def login_user(self, email, password):
        user = self.db.verify_password(email, password)
        if user:
            session_id = self.db.create_session(user['user_id'])
            st.session_state.auth_user = user
            st.session_state.auth_session_id = session_id
            
            # Check if password reset is required
            self._check_password_reset_requirement(user['user_id'])
            
            return {'success': True, 'user': user}
        return {'success': False, 'error': 'Invalid credentials'}

    def logout_user(self):
        if st.session_state.auth_session_id:
            self.db.invalidate_session(st.session_state.auth_session_id)
        
        st.session_state.auth_user = None
        st.session_state.auth_session_id = None
        st.session_state.password_reset_required = False
        st.rerun()

    def validate_session(self):
        if not st.session_state.auth_session_id:
            return False

        session_data = self.db.validate_session(st.session_state.auth_session_id)
        if session_data:
            st.session_state.auth_user = dict(session_data)
            
            # Check password reset requirement
            self._check_password_reset_requirement(session_data['user_id'])
            
            return True
        else:
            st.session_state.auth_user = None
            st.session_state.auth_session_id = None
            st.session_state.password_reset_required = False
            return False

    def get_current_user(self):
        return st.session_state.auth_user

    def _check_password_reset_requirement(self, user_id):
        """Check if user needs to reset password"""
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM password_reset_requirements 
                    WHERE user_id = ? AND requires_reset = 1
                ''', (user_id,))
                
                result = cursor.fetchone()
                if result:
                    st.session_state.password_reset_required = True
                    st.session_state.reset_reason = result['reset_reason']
                    st.session_state.reset_by = result['reset_by']
                else:
                    st.session_state.password_reset_required = False
                    
        except Exception as e:
            print(f"Error checking password reset requirement: {e}")
            st.session_state.password_reset_required = False

    def change_password(self, current_password, new_password):
        """Change user password and clear reset requirement"""
        if not st.session_state.auth_user:
            return {'success': False, 'error': 'Not authenticated'}
        
        user = st.session_state.auth_user
        
        # Verify current password
        if not self.db.verify_password(user['email'], current_password):
            return {'success': False, 'error': 'Current password is incorrect'}
        
        # Validate new password
        if len(new_password) < 8:
            return {'success': False, 'error': 'Password must be at least 8 characters long'}
        
        try:
            import bcrypt
            
            # Hash new password
            salt = bcrypt.gensalt()
            password_hash = bcrypt.hashpw(new_password.encode('utf-8'), salt)
            
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                
                # Update password
                cursor.execute('''
                    UPDATE users 
                    SET password_hash = ?, salt = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                ''', (password_hash.decode('utf-8'), salt.decode('utf-8'), user['user_id']))
                
                # Clear password reset requirement
                cursor.execute('''
                    UPDATE password_reset_requirements 
                    SET requires_reset = 0 
                    WHERE user_id = ?
                ''', (user['user_id'],))
                
                conn.commit()
            
            # Clear session state
            st.session_state.password_reset_required = False
            
            # Log the password change
            self.db.log_activity(
                user_id=user['user_id'],
                action_type="PASSWORD_CHANGED",
                object_type="USER",
                object_id=str(user['user_id']),
                status="SUCCESS"
            )
            
            return {'success': True}
            
        except Exception as e:
            return {'success': False, 'error': f'Failed to update password: {str(e)}'}

def show_password_reset_popup():
    """Show password reset popup modal"""
    if not st.session_state.get('password_reset_required', False):
        return False
    
    # Create a modal-like container
    st.markdown("""
    <style>
    .password-reset-modal {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.5);
        z-index: 1000;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .password-reset-content {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        max-width: 500px;
        width: 90%;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Show the modal content
    st.error("🔐 **Password Reset Required**")
    st.warning(f"**Reason:** {st.session_state.get('reset_reason', 'Administrator required password change')}")
    st.info(f"**Reset by:** {st.session_state.get('reset_by', 'System Administrator')}")
    st.markdown("---")
    
    st.markdown("### 🔑 Change Your Password")
    st.markdown("You must change your password before you can continue using the system.")
    
    with st.form("password_change_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            current_password = st.text_input(
                "Current Password *", 
                type="password",
                help="Enter your current password"
            )
            new_password = st.text_input(
                "New Password *", 
                type="password",
                help="Must be at least 8 characters long"
            )
        
        with col2:
            confirm_password = st.text_input(
                "Confirm New Password *", 
                type="password",
                help="Re-enter your new password"
            )
            
            # Password strength indicator
            if new_password:
                strength_score = 0
                strength_messages = []
                
                if len(new_password) >= 8:
                    strength_score += 1
                    strength_messages.append("✅ At least 8 characters")
                else:
                    strength_messages.append("❌ At least 8 characters")
                
                if any(c.isupper() for c in new_password):
                    strength_score += 1
                    strength_messages.append("✅ Contains uppercase")
                else:
                    strength_messages.append("❌ Contains uppercase")
                
                if any(c.isdigit() for c in new_password):
                    strength_score += 1
                    strength_messages.append("✅ Contains numbers")
                else:
                    strength_messages.append("❌ Contains numbers")
                
                if any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in new_password):
                    strength_score += 1
                    strength_messages.append("✅ Contains symbols")
                else:
                    strength_messages.append("❌ Contains symbols")
                
                # Display strength
                if strength_score >= 3:
                    st.success("🔒 Strong password")
                elif strength_score >= 2:
                    st.warning("🔓 Moderate password")
                else:
                    st.error("🔓 Weak password")
                
                with st.expander("Password Requirements", expanded=False):
                    for msg in strength_messages:
                        st.markdown(f"- {msg}")
        
        st.markdown("---")
        
        col_change, col_logout = st.columns([2, 1])
        
        with col_change:
            if st.form_submit_button("🔑 Change Password", type="primary", use_container_width=True):
                if not all([current_password, new_password, confirm_password]):
                    st.error("❌ Please fill in all fields")
                elif new_password != confirm_password:
                    st.error("❌ New passwords don't match")
                elif len(new_password) < 8:
                    st.error("❌ Password must be at least 8 characters long")
                elif new_password == current_password:
                    st.error("❌ New password must be different from current password")
                else:
                    auth = AuthenticationMiddleware()
                    result = auth.change_password(current_password, new_password)
                    
                    if result['success']:
                        st.success("✅ Password changed successfully!")
                        st.balloons()
                        st.info("🔄 Redirecting to application...")
                        # Small delay before rerun to show success message
                        import time
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error(f"❌ {result['error']}")
        
        with col_logout:
            if st.form_submit_button("🚪 Logout", use_container_width=True):
                auth = AuthenticationMiddleware()
                auth.logout_user()
    
    # Show additional help
    with st.expander("❓ Need Help?", expanded=False):
        st.markdown("""
        **Why am I seeing this?**
        - An administrator has reset your password for security reasons
        - This is a security measure to ensure your account remains protected
        
        **Password Requirements:**
        - At least 8 characters long
        - Include uppercase and lowercase letters
        - Include at least one number
        - Symbols are recommended but not required
        
        **Having trouble?**
        - Contact your system administrator if you can't remember your current password
        - If you continue to have issues, use the logout button and contact support
        """)
    
    return True

def show_login_page():
    """Enhanced login page with password reset popup handling"""
    
    # Check if we need to show password reset popup first
    if show_password_reset_popup():
        return  # Password reset popup is shown, don't show login form
    
    st.title("🔐 Data Lens AI Login")
    st.markdown("Enter your credentials to access the platform")

    auth = AuthenticationMiddleware()

    # Check if already logged in
    if auth.validate_session():
        current_user = auth.get_current_user()
        if current_user:
            # Check if password reset is required
            if st.session_state.get('password_reset_required', False):
                # Password reset popup will be shown above
                return
            else:
                st.info(f"✅ Already logged in as {current_user['full_name']}. Redirecting...")
                st.switch_page("Home.py")
                return

    # Login form
    with st.form("login_form", clear_on_submit=False):
        st.markdown("### 🔑 Login Credentials")

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
                    st.success("✅ Login successful!")
                    
                    # Check if password reset is required
                    if st.session_state.get('password_reset_required', False):
                        st.warning("🔐 Password reset required. Please change your password.")
                        st.rerun()  # This will show the password reset popup
                    else:
                        st.info("🔄 Redirecting to dashboard...")
                        st.switch_page("Home.py")
                else:
                    st.error(f"❌ Login failed: {result['error']}")

    # Help section
    st.markdown("---")
    with st.expander("ℹ️ Login Help", expanded=False):
        st.markdown("""
        **Default Credentials:**
        - Email: `admin@datalensai.com`
        - Password: `DataLens2024!`

        **Troubleshooting:**
        - Make sure you're using the correct email and password
        - Check that the authentication system is properly set up
        - Contact your administrator if you've forgotten your credentials

        **Security Notes:**
        - Please change the default password after first login
        - Your session will expire after 24 hours of inactivity
        - You may be required to change your password if an administrator has reset it
        """)

    # Show login attempts and security info
    with st.expander("🔒 Security Information", expanded=False):
        st.info("""
        **Account Security:**
        - Accounts are locked after 5 failed login attempts
        - Locked accounts are automatically unlocked after 30 minutes
        - All login attempts are logged for security monitoring
        - Password changes are required periodically for security
        """)