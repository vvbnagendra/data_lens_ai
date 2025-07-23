# File: app/auth/page_protection.py
# Enhanced page protection with password reset popup handling

import streamlit as st
from functools import wraps
from .middleware import AuthenticationMiddleware, show_login_page, show_password_reset_popup

def require_auth():
    """Enhanced authentication decorator with password reset handling"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            auth = AuthenticationMiddleware()
            
            # First check if user is authenticated
            if not auth.validate_session():
                show_login_page()
                return
            
            # Get current user
            user = auth.get_current_user()
            if not user:
                show_login_page()
                return
            
            # Check if password reset is required
            if st.session_state.get('password_reset_required', False):
                # Show password reset popup - this will block the main content
                if show_password_reset_popup():
                    return  # Don't show main content until password is changed
            
            # If we get here, user is authenticated and doesn't need password reset
            # Show the user info bar
            show_user_info_bar(auth, user)
            
            # Call the original function
            return func(*args, **kwargs)
        return wrapper
    return decorator

def show_user_info_bar(auth, user):
    """Show user information bar with logout button"""
    
    # Enhanced CSS for better styling
    st.markdown("""
    <style>
        /* User info bar styling */
        .user-info-container {
            background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 0.75rem 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .user-info-text {
            color: #495057;
            font-weight: 500;
            margin: 0;
            display: flex;
            align-items: center;
        }
        
        .user-role-badge {
            background: #007bff;
            color: white;
            padding: 0.25rem 0.5rem;
            border-radius: 12px;
            font-size: 0.8rem;
            margin-left: 0.5rem;
        }
        
        .logout-button {
            background: #dc3545 !important;
            color: white !important;
            border: none !important;
            border-radius: 6px !important;
            padding: 0.5rem 1rem !important;
            font-size: 0.9rem !important;
            transition: all 0.2s ease !important;
        }
        
        .logout-button:hover {
            background: #c82333 !important;
            transform: translateY(-1px);
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        
        /* Password reset warning */
        .password-reset-warning {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 6px;
            padding: 0.75rem;
            margin-bottom: 1rem;
            color: #856404;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Create user info bar
    col_info, col_actions = st.columns([4, 1])
    
    with col_info:
        # Show user information
        role_color = get_role_color(user['user_role'])
        
        st.markdown(f"""
        <div class="user-info-container">
            <div class="user-info-text">
                👤 Welcome, <strong>{user['full_name']}</strong>
                <span class="user-role-badge" style="background-color: {role_color}">
                    {user['user_role']}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show additional info if SuperAdmin or Admin
        if user['user_role'] in ['SuperAdmin', 'Admin']:
            st.markdown("""
            <div style="background: #d1ecf1; border: 1px solid #bee5eb; border-radius: 4px; padding: 0.5rem; margin-bottom: 1rem; font-size: 0.9rem;">
                🛡️ <strong>Administrative Access:</strong> You have elevated privileges. Use responsibly.
            </div>
            """, unsafe_allow_html=True)
    
    with col_actions:
        # Action buttons
        action_col1, action_col2 = st.columns(2)
        
        with action_col1:
            if st.button("🔧 Profile", help="View and edit your profile", use_container_width=True):
                show_user_profile_popup(user)
        
        with action_col2:
            if st.button("🚪 Logout", help="Logout from the system", use_container_width=True, type="secondary"):
                auth.logout_user()

def get_role_color(role):
    """Get color for user role badge"""
    role_colors = {
        'SuperAdmin': '#dc3545',  # Red
        'Admin': '#fd7e14',       # Orange
        'DataScientist': '#6f42c1', # Purple
        'BusinessAnalyst': '#20c997', # Teal
        'Developer': '#6c757d',   # Gray
        'Viewer': '#28a745'       # Green
    }
    return role_colors.get(role, '#6c757d')

def show_user_profile_popup(user):
    """Show user profile popup"""
    
    if 'show_profile_popup' not in st.session_state:
        st.session_state.show_profile_popup = True
    
    if st.session_state.get('show_profile_popup', False):
        st.markdown("### 👤 User Profile")
        
        # Display user information in a nice format
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📋 Basic Information")
            st.text_input("Full Name", value=user['full_name'], disabled=True)
            st.text_input("Username", value=user['username'], disabled=True)
            st.text_input("Email", value=user['email'], disabled=True)
        
        with col2:
            st.markdown("#### 🏢 Organization Details")
            st.text_input("Role", value=user['user_role'], disabled=True)
            st.text_input("Organization", value=user.get('organization_name', 'N/A'), disabled=True)
            st.text_input("Department", value=user.get('department', 'N/A'), disabled=True)
        
        # Account status
        st.markdown("#### 📊 Account Status")
        status_col1, status_col2, status_col3 = st.columns(3)
        
        with status_col1:
            status = "✅ Active" if user['is_active'] else "❌ Inactive"
            st.metric("Account Status", status)
        
        with status_col2:
            verification = "✅ Verified" if user['email_verified'] else "❌ Not Verified"
            st.metric("Email Status", verification)
        
        with status_col3:
            last_login = user.get('last_login', 'Never')[:19] if user.get('last_login') else 'Never'
            st.metric("Last Login", last_login)
        
        # Password change section
        st.markdown("#### 🔑 Change Password")
        with st.expander("Change Your Password", expanded=False):
            with st.form("profile_password_change"):
                current_pwd = st.text_input("Current Password", type="password")
                new_pwd = st.text_input("New Password", type="password")
                confirm_pwd = st.text_input("Confirm New Password", type="password")
                
                if st.form_submit_button("🔑 Change Password", type="primary"):
                    if not all([current_pwd, new_pwd, confirm_pwd]):
                        st.error("Please fill in all password fields")
                    elif new_pwd != confirm_pwd:
                        st.error("New passwords don't match")
                    elif len(new_pwd) < 8:
                        st.error("Password must be at least 8 characters long")
                    else:
                        auth = AuthenticationMiddleware()
                        result = auth.change_password(current_pwd, new_pwd)
                        
                        if result['success']:
                            st.success("✅ Password changed successfully!")
                        else:
                            st.error(f"❌ {result['error']}")
        
        # Close button
        if st.button("❌ Close Profile", type="secondary"):
            st.session_state.show_profile_popup = False
            st.rerun()

def require_permission(permission_name):
    """Decorator to check specific permissions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            auth = AuthenticationMiddleware()
            
            if not auth.validate_session():
                st.error("❌ Authentication required")
                return
            
            user = auth.get_current_user()
            if not user:
                st.error("❌ User not found")
                return
            
            # Get user permissions (you'd need to implement this in your DB)
            user_permissions = auth.db.get_user_permissions(user['user_role'])
            
            if permission_name not in user_permissions:
                st.error(f"❌ Access denied. Required permission: {permission_name}")
                st.info(f"Your role ({user['user_role']}) does not have the required permission to access this feature.")
                return
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def require_role(allowed_roles):
    """Decorator to check specific roles"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            auth = AuthenticationMiddleware()
            
            if not auth.validate_session():
                st.error("❌ Authentication required")
                return
            
            user = auth.get_current_user()
            if not user:
                st.error("❌ User not found")
                return
            
            if user['user_role'] not in allowed_roles:
                st.error(f"❌ Access denied. Required roles: {', '.join(allowed_roles)}")
                st.info(f"Your role ({user['user_role']}) does not have access to this feature.")
                return
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Enhanced authentication checking functions
def check_admin_access():
    """Check if current user has admin access"""
    auth = AuthenticationMiddleware()
    
    if not auth.validate_session():
        return False
    
    user = auth.get_current_user()
    return user and user['user_role'] in ['SuperAdmin', 'Admin']

def check_superadmin_access():
    """Check if current user has superadmin access"""
    auth = AuthenticationMiddleware()
    
    if not auth.validate_session():
        return False
    
    user = auth.get_current_user()
    return user and user['user_role'] == 'SuperAdmin'

def get_user_permissions():
    """Get current user's permissions"""
    auth = AuthenticationMiddleware()
    
    if not auth.validate_session():
        return []
    
    user = auth.get_current_user()
    if not user:
        return []
    
    return auth.db.get_user_permissions(user['user_role'])

def has_permission(permission_name):
    """Check if current user has specific permission"""
    permissions = get_user_permissions()
    return permission_name in permissions

# Session management helpers
def extend_session():
    """Extend current user session"""
    auth = AuthenticationMiddleware()
    
    if st.session_state.auth_session_id:
        # Update session last activity
        import sqlite3
        from datetime import datetime, timedelta
        
        try:
            with sqlite3.connect(auth.db.db_path) as conn:
                cursor = conn.cursor()
                
                # Extend session by 24 hours
                new_expiry = datetime.utcnow() + timedelta(hours=24)
                
                cursor.execute('''
                    UPDATE user_sessions 
                    SET last_activity = CURRENT_TIMESTAMP, expires_at = ?
                    WHERE session_id = ?
                ''', (new_expiry, st.session_state.auth_session_id))
                
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            print(f"Error extending session: {e}")
            return False
    
    return False

def get_session_info():
    """Get current session information"""
    auth = AuthenticationMiddleware()
    
    if not st.session_state.auth_session_id:
        return None
    
    session_data = auth.db.validate_session(st.session_state.auth_session_id)
    if session_data:
        return {
            'session_id': session_data['session_id'],
            'created_at': session_data['created_at'],
            'last_activity': session_data['last_activity'],
            'expires_at': session_data['expires_at'],
            'ip_address': session_data.get('ip_address', 'Unknown'),
            'user_agent': session_data.get('user_agent', 'Unknown')
        }
    
    return None

# Utility functions for UI components
def show_access_denied(required_role=None, required_permission=None):
    """Show standardized access denied message"""
    st.error("🚫 **Access Denied**")
    
    auth = AuthenticationMiddleware()
    user = auth.get_current_user()
    
    if user:
        st.info(f"👤 **Current User:** {user['full_name']} ({user['user_role']})")
        
        if required_role:
            st.warning(f"🔐 **Required Role:** {required_role}")
        
        if required_permission:
            st.warning(f"🔑 **Required Permission:** {required_permission}")
        
        st.markdown("---")
        st.markdown("### 💡 What can you do?")
        st.markdown("""
        - **Contact your administrator** to request additional permissions
        - **Check if you're using the correct account** for this operation
        - **Review the navigation menu** for features available to your role
        - **Use the logout button** to switch to a different account
        """)
    else:
        st.warning("Please log in to access this feature.")

def show_permission_info():
    """Show current user's permissions"""
    permissions = get_user_permissions()
    
    if permissions:
        st.markdown("### 🔑 Your Permissions")
        
        # Group permissions by category
        permission_groups = {}
        for perm in permissions:
            category = perm.split('_')[1] if '_' in perm else 'general'
            if category not in permission_groups:
                permission_groups[category] = []
            permission_groups[category].append(perm)
        
        # Display permissions by group
        for category, perms in permission_groups.items():
            with st.expander(f"📋 {category.title()} Permissions", expanded=False):
                for perm in perms:
                    st.markdown(f"✅ {perm.replace('_', ' ').title()}")
    else:
        st.info("No specific permissions found for your role.")

# Security monitoring functions
def log_security_event(event_type, details=None):
    """Log security-related events"""
    auth = AuthenticationMiddleware()
    user = auth.get_current_user()
    
    if user:
        auth.db.log_activity(
            user_id=user['user_id'],
            action_type=f"SECURITY_{event_type}",
            object_type="SECURITY",
            status="SUCCESS",
            payload=details
        )

def check_suspicious_activity():
    """Check for suspicious activity patterns"""
    # This would contain logic to detect:
    # - Multiple failed login attempts
    # - Access from unusual locations
    # - Rapid permission escalation attempts
    # - Unusual access patterns
    
    # For now, just return False (no suspicious activity)
    return False

# Password policy enforcement
def validate_password_policy(password):
    """Validate password against policy"""
    errors = []
    
    if len(password) < 8:
        errors.append("Password must be at least 8 characters long")
    
    if not any(c.isupper() for c in password):
        errors.append("Password must contain at least one uppercase letter")
    
    if not any(c.islower() for c in password):
        errors.append("Password must contain at least one lowercase letter")
    
    if not any(c.isdigit() for c in password):
        errors.append("Password must contain at least one number")
    
    # Check for common passwords
    common_passwords = ['password', '123456', 'admin', 'login', 'welcome']
    if password.lower() in common_passwords:
        errors.append("Password is too common. Please choose a more unique password")
    
    return errors

def password_strength_indicator(password):
    """Show password strength indicator"""
    if not password:
        return
    
    strength = 0
    feedback = []
    
    # Length check
    if len(password) >= 8:
        strength += 1
        feedback.append("✅ Length (8+ characters)")
    else:
        feedback.append("❌ Length (needs 8+ characters)")
    
    # Uppercase check
    if any(c.isupper() for c in password):
        strength += 1
        feedback.append("✅ Uppercase letters")
    else:
        feedback.append("❌ Uppercase letters")
    
    # Lowercase check
    if any(c.islower() for c in password):
        strength += 1
        feedback.append("✅ Lowercase letters")
    else:
        feedback.append("❌ Lowercase letters")
    
    # Number check
    if any(c.isdigit() for c in password):
        strength += 1
        feedback.append("✅ Numbers")
    else:
        feedback.append("❌ Numbers")
    
    # Symbol check
    if any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
        strength += 1
        feedback.append("✅ Special characters")
    else:
        feedback.append("❌ Special characters")
    
    # Display strength
    if strength >= 4:
        st.success("🔒 Strong password")
    elif strength >= 3:
        st.warning("🔓 Moderate password")
    elif strength >= 2:
        st.warning("🔓 Weak password")
    else:
        st.error("🔓 Very weak password")
    
    # Show detailed feedback
    with st.expander("Password Requirements", expanded=False):
        for item in feedback:
            st.markdown(f"- {item}")
    
    return strength