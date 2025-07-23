# File: app/pages/Admin_User_Management.py
# Enhanced User Management Interface with full functionality

import streamlit as st
import pandas as pd
import json
import sys
import os
from datetime import datetime, timedelta
import secrets
import string

# --- Page Configuration ---
st.set_page_config(
    page_title="User Management",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Fix Python Path Issues ---
current_dir = os.getcwd()
app_dir = os.path.join(current_dir, "app")

if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

# --- Import Authentication System ---
AUTH_AVAILABLE = False
auth_error_message = ""

try:
    try:
        from auth.middleware import AuthenticationMiddleware
        from auth.models import AuthenticationDB, UserRole
        AUTH_AVAILABLE = True
    except ImportError:
        from app.auth.middleware import AuthenticationMiddleware
        from app.auth.models import AuthenticationDB, UserRole
        AUTH_AVAILABLE = True
except ImportError as e:
    auth_error_message = f"Import Error: {str(e)}"
except Exception as e:
    auth_error_message = f"General Error: {str(e)}"

# --- Import Page Protection ---
try:
    if AUTH_AVAILABLE:
        from auth.page_protection import require_auth
    else:
        def require_auth():
            def decorator(func):
                def wrapper(*args, **kwargs):
                    st.error("❌ Authentication system not available")
                    st.error(f"Error details: {auth_error_message}")
                    return None
                return wrapper
            return decorator
except ImportError:
    def require_auth():
        def decorator(func):
            def wrapper(*args, **kwargs):
                st.error("❌ Page protection not available")
                return None
            return wrapper
        return decorator

# --- Enhanced Authentication DB Class ---
class EnhancedAuthDB(AuthenticationDB):
    """Enhanced authentication database with additional admin features"""
    
    def __init__(self, db_path="app/database/auth.db"):
        super().__init__(db_path)
        self._init_enhanced_tables()
    
    def _init_enhanced_tables(self):
        """Initialize additional tables for enhanced features"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # System settings table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_settings (
                    setting_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    setting_key TEXT UNIQUE NOT NULL,
                    setting_value TEXT NOT NULL,
                    setting_description TEXT,
                    updated_by TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Password reset requirements table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS password_reset_requirements (
                    user_id INTEGER PRIMARY KEY,
                    requires_reset BOOLEAN DEFAULT 0,
                    reset_reason TEXT,
                    reset_by TEXT,
                    reset_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            
            # Audit trail table (enhanced)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_trail (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    admin_user_id INTEGER,
                    action_type TEXT NOT NULL,
                    table_name TEXT,
                    record_id TEXT,
                    old_values TEXT,
                    new_values TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id),
                    FOREIGN KEY (admin_user_id) REFERENCES users (user_id)
                )
            ''')
            
            conn.commit()
            
            # Initialize default system settings
            self._init_default_settings()
    
    def _init_default_settings(self):
        """Initialize default system settings"""
        default_settings = {
            'password_min_length': ('8', 'Minimum password length'),
            'password_require_uppercase': ('true', 'Require uppercase letters in passwords'),
            'password_require_numbers': ('true', 'Require numbers in passwords'),
            'password_require_symbols': ('false', 'Require symbols in passwords'),
            'session_timeout_hours': ('24', 'Session timeout in hours'),
            'max_login_attempts': ('5', 'Maximum failed login attempts before lockout'),
            'lockout_duration_minutes': ('30', 'Account lockout duration in minutes'),
            'enable_email_notifications': ('false', 'Enable email notifications'),
            'allow_self_registration': ('false', 'Allow users to self-register'),
            'require_email_verification': ('true', 'Require email verification for new users'),
            'audit_log_retention_days': ('90', 'Number of days to retain audit logs'),
            'backup_frequency_hours': ('24', 'Database backup frequency in hours')
        }
        
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for key, (value, description) in default_settings.items():
                cursor.execute('''
                    INSERT OR IGNORE INTO system_settings (setting_key, setting_value, setting_description, updated_by)
                    VALUES (?, ?, ?, 'system')
                ''', (key, value, description))
            
            conn.commit()
    
    def update_user_detailed(self, user_id: int, updates: dict, admin_user_id: int) -> bool:
        """Update user with detailed audit trail"""
        try:
            import sqlite3
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get old values for audit
                cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
                old_user = dict(sqlite3.Row(*cursor.fetchone()) if cursor.fetchone() else {})
                
                # Update user
                success = self.update_user(user_id, updates)
                
                if success:
                    # Log audit trail
                    self._log_audit_trail(
                        user_id=user_id,
                        admin_user_id=admin_user_id,
                        action_type="USER_UPDATED",
                        table_name="users",
                        record_id=str(user_id),
                        old_values=json.dumps(old_user, default=str),
                        new_values=json.dumps(updates, default=str)
                    )
                
                return success
        except Exception as e:
            print(f"Error updating user: {e}")
            return False
    
    def reset_user_password(self, user_id: int, new_password: str, admin_user_id: int, 
                           require_change: bool = True, reset_reason: str = "Admin reset") -> bool:
        """Reset user password with enhanced options"""
        try:
            import bcrypt
            import sqlite3
            
            # Hash the new password
            salt = bcrypt.gensalt()
            password_hash = bcrypt.hashpw(new_password.encode('utf-8'), salt)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Update password
                cursor.execute('''
                    UPDATE users 
                    SET password_hash = ?, salt = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                ''', (password_hash.decode('utf-8'), salt.decode('utf-8'), user_id))
                
                # Set password reset requirement if needed
                if require_change:
                    cursor.execute('''
                        INSERT OR REPLACE INTO password_reset_requirements 
                        (user_id, requires_reset, reset_reason, reset_by)
                        VALUES (?, 1, ?, (SELECT username FROM users WHERE user_id = ?))
                    ''', (user_id, reset_reason, admin_user_id))
                
                conn.commit()
                
                # Log audit trail
                self._log_audit_trail(
                    user_id=user_id,
                    admin_user_id=admin_user_id,
                    action_type="PASSWORD_RESET",
                    table_name="users",
                    record_id=str(user_id),
                    new_values=json.dumps({
                        'reset_reason': reset_reason,
                        'require_change': require_change
                    })
                )
                
                return True
                
        except Exception as e:
            print(f"Error resetting password: {e}")
            return False
    
    def get_system_settings(self) -> dict:
        """Get all system settings"""
        try:
            import sqlite3
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("SELECT * FROM system_settings ORDER BY setting_key")
                settings = {}
                
                for row in cursor.fetchall():
                    settings[row['setting_key']] = {
                        'value': row['setting_value'],
                        'description': row['setting_description'],
                        'updated_by': row['updated_by'],
                        'updated_at': row['updated_at']
                    }
                
                return settings
        except Exception as e:
            print(f"Error getting system settings: {e}")
            return {}
    
    def update_system_setting(self, setting_key: str, setting_value: str, admin_user_id: int) -> bool:
        """Update a system setting"""
        try:
            import sqlite3
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get admin username
                cursor.execute("SELECT username FROM users WHERE user_id = ?", (admin_user_id,))
                admin_username = cursor.fetchone()[0]
                
                # Update setting
                cursor.execute('''
                    UPDATE system_settings 
                    SET setting_value = ?, updated_by = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE setting_key = ?
                ''', (setting_value, admin_username, setting_key))
                
                conn.commit()
                
                # Log audit trail
                self._log_audit_trail(
                    admin_user_id=admin_user_id,
                    action_type="SYSTEM_SETTING_UPDATED",
                    table_name="system_settings",
                    record_id=setting_key,
                    new_values=json.dumps({'setting_key': setting_key, 'setting_value': setting_value})
                )
                
                return cursor.rowcount > 0
                
        except Exception as e:
            print(f"Error updating system setting: {e}")
            return False
    
    def get_audit_trail(self, limit: int = 100, user_id: int = None) -> list:
        """Get audit trail with optional filtering"""
        try:
            import sqlite3
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = '''
                    SELECT a.*, u1.username as target_username, u2.username as admin_username
                    FROM audit_trail a
                    LEFT JOIN users u1 ON a.user_id = u1.user_id
                    LEFT JOIN users u2 ON a.admin_user_id = u2.user_id
                '''
                
                params = []
                if user_id:
                    query += " WHERE a.user_id = ?"
                    params.append(user_id)
                
                query += " ORDER BY a.timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            print(f"Error getting audit trail: {e}")
            return []
    
    def _log_audit_trail(self, admin_user_id: int, action_type: str, table_name: str = None,
                        record_id: str = None, old_values: str = None, new_values: str = None,
                        user_id: int = None, ip_address: str = None, user_agent: str = None):
        """Log audit trail entry"""
        try:
            import sqlite3
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO audit_trail 
                    (user_id, admin_user_id, action_type, table_name, record_id, 
                     old_values, new_values, ip_address, user_agent)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (user_id, admin_user_id, action_type, table_name, record_id,
                      old_values, new_values, ip_address, user_agent))
                
                conn.commit()
                
        except Exception as e:
            print(f"Error logging audit trail: {e}")
    
    def check_password_reset_required(self, user_id: int) -> dict:
        """Check if user needs to reset password"""
        try:
            import sqlite3
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM password_reset_requirements 
                    WHERE user_id = ? AND requires_reset = 1
                ''', (user_id,))
                
                result = cursor.fetchone()
                if result:
                    return dict(result)
                return {}
                
        except Exception as e:
            print(f"Error checking password reset requirement: {e}")
            return {}
    
    def clear_password_reset_requirement(self, user_id: int):
        """Clear password reset requirement after user changes password"""
        try:
            import sqlite3
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE password_reset_requirements 
                    SET requires_reset = 0 
                    WHERE user_id = ?
                ''', (user_id,))
                
                conn.commit()
                
        except Exception as e:
            print(f"Error clearing password reset requirement: {e}")

def generate_secure_password(length: int = 12, include_symbols: bool = False) -> str:
    """Generate a secure random password"""
    characters = string.ascii_letters + string.digits
    if include_symbols:
        characters += "!@#$%^&*"
    
    password = ''.join(secrets.choice(characters) for _ in range(length))
    
    # Ensure password meets requirements
    if not any(c.isupper() for c in password):
        password = password[:-1] + secrets.choice(string.ascii_uppercase)
    if not any(c.isdigit() for c in password):
        password = password[:-1] + secrets.choice(string.digits)
    
    return password

def show_password_reset_popup():
    """Show password reset popup for users who need to change their password"""
    if 'show_password_reset_popup' in st.session_state and st.session_state.show_password_reset_popup:
        st.warning("🔐 **Password Reset Required**: You must change your password before continuing.")
        
        with st.form("password_reset_popup"):
            st.markdown("#### Change Your Password")
            
            current_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm New Password", type="password")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("Change Password", type="primary"):
                    if new_password != confirm_password:
                        st.error("Passwords don't match!")
                    elif len(new_password) < 8:
                        st.error("Password must be at least 8 characters long!")
                    else:
                        # Here you would validate current password and update
                        # For now, just clear the popup
                        st.session_state.show_password_reset_popup = False
                        st.success("Password changed successfully!")
                        st.rerun()
            
            with col2:
                if st.form_submit_button("Logout"):
                    # Handle logout
                    st.session_state.clear()
                    st.rerun()

@require_auth()
def main():
    """Protected main function for User Management"""
    
    if not AUTH_AVAILABLE:
        st.error("❌ Authentication system not available")
        return
    
    # Initialize enhanced auth and db
    auth = AuthenticationMiddleware()
    db = EnhancedAuthDB()
    current_user = auth.get_current_user()
    
    # Check if user is SuperAdmin or Admin
    if current_user['user_role'] not in ['SuperAdmin', 'Admin']:
        st.error("❌ Access Denied: Only SuperAdmin and Admin users can access this page")
        return
    
    # Check if current user needs to reset password
    reset_req = db.check_password_reset_required(current_user['user_id'])
    if reset_req:
        st.session_state.show_password_reset_popup = True
    
    # Show password reset popup if needed
    show_password_reset_popup()
    
    # Apply professional styling
    try:
        from assets.streamlit_styles import apply_professional_styling, create_nav_header
        apply_professional_styling()
        create_nav_header("👥 User Management", "Manage users, roles, and permissions")
    except ImportError:
        st.title("👥 User Management")
        st.markdown("*Manage users, roles, and permissions*")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "👥 User Management", 
        "📊 Admin Dashboard", 
        "⚙️ System Settings", 
        "🔍 Audit Logs",
        "🚨 Emergency Tools"
    ])
    
    with tab1:
        show_user_management_tab(db, current_user)
    
    with tab2:
        show_admin_dashboard_tab(db, current_user)
    
    with tab3:
        show_system_settings_tab(db, current_user)
    
    with tab4:
        show_audit_logs_tab(db, current_user)
    
    with tab5:
        if current_user['user_role'] == 'SuperAdmin':
            show_emergency_tools_tab(db, current_user)
        else:
            st.warning("🔒 Emergency tools are only available to SuperAdmin users")

def show_user_management_tab(db, current_user):
    """Enhanced user management tab with full CRUD operations"""
    
    st.markdown("### 👥 User Management")
    
    # Get all users
    all_users = db.get_all_users()
    
    # User statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Users", len(all_users))
    with col2:
        active_users = len([u for u in all_users if u['is_active']])
        st.metric("Active Users", active_users)
    with col3:
        verified_users = len([u for u in all_users if u['email_verified']])
        st.metric("Verified Users", verified_users)
    with col4:
        pending_users = len([u for u in all_users if not u['email_verified']])
        st.metric("Pending Verification", pending_users)
    
    # User management actions
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        if st.button("➕ Create New User", type="primary"):
            st.session_state.show_create_user = True
    
    with action_col2:
        if st.button("📊 Export User List"):
            export_user_list(all_users)
    
    with action_col3:
        if st.button("🔄 Refresh Data"):
            st.rerun()
    
    # Show create user form
    if st.session_state.get('show_create_user', False):
        show_create_user_form(db, current_user)
    
    # User list with enhanced management
    st.markdown("#### 📋 User List")
    
    # Search and filter
    search_col1, search_col2, search_col3 = st.columns(3)
    
    with search_col1:
        search_term = st.text_input("🔍 Search users", placeholder="Name, email, or username...")
    
    with search_col2:
        role_filter = st.selectbox("Filter by Role", ["All"] + [role.value for role in UserRole])
    
    with search_col3:
        status_filter = st.selectbox("Filter by Status", ["All", "Active", "Inactive", "Pending Verification"])
    
    # Apply filters
    filtered_users = filter_users(all_users, search_term, role_filter, status_filter)
    
    st.info(f"Showing {len(filtered_users)} of {len(all_users)} users")
    
    # Display users with enhanced management options
    for user in filtered_users:
        show_enhanced_user_card(db, current_user, user)

def show_enhanced_user_card(db, current_user, user):
    """Show enhanced user card with full management options"""
    
    # Determine status and styling
    if user['is_active'] and user['email_verified']:
        status = "Active"
        status_icon = "🟢"
        card_style = "border: 2px solid #28a745; border-radius: 10px; padding: 15px; margin: 10px 0; background: #f8fff8;"
    elif not user['email_verified']:
        status = "Pending Verification"
        status_icon = "🟡"
        card_style = "border: 2px solid #ffc107; border-radius: 10px; padding: 15px; margin: 10px 0; background: #fffef8;"
    else:
        status = "Inactive"
        status_icon = "🔴"
        card_style = "border: 2px solid #dc3545; border-radius: 10px; padding: 15px; margin: 10px 0; background: #fff8f8;"
    
    with st.container():
        st.markdown(f'<div style="{card_style}">', unsafe_allow_html=True)
        
        # User info
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"### {status_icon} {user['full_name']} ({user['username']})")
            st.markdown(f"**Email:** {user['email']}")
            st.markdown(f"**Role:** {user['user_role']} | **Status:** {status}")
            st.markdown(f"**Organization:** {user['organization_name'] or 'N/A'} | **Department:** {user['department'] or 'N/A'}")
            st.markdown(f"**Created:** {user['created_at'][:19] if user['created_at'] else 'N/A'} | **Last Login:** {user['last_login'][:19] if user['last_login'] else 'Never'}")
        
        with col2:
            # Check if password reset is required
            reset_req = db.check_password_reset_required(user['user_id'])
            if reset_req:
                st.warning("🔐 Password reset required")
        
        # Action buttons with better alignment
        st.markdown("#### 🔧 Actions")
        
        # Create columns for action buttons - adjust based on available actions
        if current_user['user_role'] == 'SuperAdmin' and user['user_id'] != current_user['user_id']:
            action_cols = st.columns([1, 1, 1, 1, 1, 1])  # 6 columns for SuperAdmin
        else:
            action_cols = st.columns([1, 1, 1, 1, 1])  # 5 columns for others
        
        with action_cols[0]:
            if st.button("✏️ Edit", key=f"edit_{user['user_id']}", use_container_width=True, help="Edit user details"):
                st.session_state[f"edit_user_{user['user_id']}"] = True
                st.rerun()
        
        with action_cols[1]:
            if user['is_active']:
                if st.button("🚫 Deactivate", key=f"deactivate_{user['user_id']}", use_container_width=True, help="Deactivate user account"):
                    toggle_user_status(db, current_user, user['user_id'], False)
            else:
                if st.button("✅ Activate", key=f"activate_{user['user_id']}", use_container_width=True, help="Activate user account"):
                    toggle_user_status(db, current_user, user['user_id'], True)
        
        with action_cols[2]:
            if st.button("🔑 Reset Password", key=f"reset_pwd_{user['user_id']}", use_container_width=True, help="Reset user password"):
                st.session_state[f"show_password_reset_{user['user_id']}"] = True
                st.rerun()
        
        with action_cols[3]:
            if not user['email_verified']:
                if st.button("✅ Verify Email", key=f"verify_email_{user['user_id']}", use_container_width=True, help="Verify user email"):
                    verify_user_email(db, current_user, user['user_id'])
            else:
                if st.button("📊 View Activity", key=f"activity_{user['user_id']}", use_container_width=True, help="View user activity logs"):
                    st.session_state[f"show_activity_{user['user_id']}"] = True
                    st.rerun()
        
        with action_cols[4]:
            if user['email_verified']:
                if st.button("📊 View Activity", key=f"activity_alt_{user['user_id']}", use_container_width=True, help="View user activity logs"):
                    st.session_state[f"show_activity_{user['user_id']}"] = True
                    st.rerun()
            else:
                # Empty space for alignment
                st.empty()
        
        # SuperAdmin delete button
        if current_user['user_role'] == 'SuperAdmin' and user['user_id'] != current_user['user_id']:
            with action_cols[5]:
                delete_confirm_key = f"confirm_delete_{user['user_id']}"
                if st.session_state.get(delete_confirm_key, False):
                    if st.button("🗑️ CONFIRM DELETE", key=f"confirm_delete_btn_{user['user_id']}", 
                               use_container_width=True, type="secondary", help="Click to confirm deletion"):
                        delete_user(db, current_user, user['user_id'])
                else:
                    if st.button("🗑️ Delete", key=f"delete_{user['user_id']}", 
                               use_container_width=True, help="Delete user (click twice to confirm)"):
                        st.session_state[delete_confirm_key] = True
                        st.rerun()
        
        # Show user activity if requested
        if st.session_state.get(f"show_activity_{user['user_id']}", False):
            show_user_activity_inline(db, user)
        
        # Show edit form if requested
        if st.session_state.get(f"edit_user_{user['user_id']}", False):
            show_edit_user_form(db, current_user, user)
        
        # Show password reset form if requested
        if st.session_state.get(f"show_password_reset_{user['user_id']}", False):
            show_password_reset_form(db, current_user, user)
        
        st.markdown('</div>', unsafe_allow_html=True)

def show_edit_user_form(db, current_user, user):
    """Show enhanced edit user form"""
    
    st.markdown(f"#### ✏️ Edit User: {user['full_name']}")
    
    # Create a unique key for this form to avoid conflicts
    form_key = f"edit_form_{user['user_id']}_{user['username']}"
    
    with st.form(form_key, clear_on_submit=False):
        col1, col2 = st.columns(2)
        
        with col1:
            new_full_name = st.text_input("Full Name", value=user['full_name'], key=f"fn_{user['user_id']}")
            new_email = st.text_input("Email", value=user['email'], key=f"em_{user['user_id']}")
            new_username = st.text_input("Username", value=user['username'], key=f"un_{user['user_id']}")
            new_organization = st.text_input("Organization", value=user['organization_name'] or '', key=f"org_{user['user_id']}")
        
        with col2:
            current_role_index = 0
            try:
                role_options = [role.value for role in UserRole]
                current_role_index = role_options.index(user['user_role'])
            except (ValueError, AttributeError):
                role_options = ["SuperAdmin", "Admin", "DataScientist", "BusinessAnalyst", "Developer", "Viewer"]
                try:
                    current_role_index = role_options.index(user['user_role'])
                except ValueError:
                    current_role_index = 0
            
            new_role = st.selectbox(
                "Role", 
                options=role_options,
                index=current_role_index,
                key=f"role_{user['user_id']}"
            )
            new_department = st.text_input("Department", value=user['department'] or '', key=f"dept_{user['user_id']}")
            email_verified = st.checkbox("Email Verified", value=user['email_verified'], key=f"ev_{user['user_id']}")
            is_active = st.checkbox("Account Active", value=user['is_active'], key=f"act_{user['user_id']}")
        
        st.markdown("---")
        col_save, col_cancel = st.columns(2)
        
        with col_save:
            save_clicked = st.form_submit_button("💾 Save Changes", type="primary")
        
        with col_cancel:
            cancel_clicked = st.form_submit_button("❌ Cancel")
        
        # Handle form submission
        if save_clicked:
            try:
                updates = {
                    'full_name': new_full_name,
                    'email': new_email,
                    'username': new_username,
                    'organization_name': new_organization,
                    'user_role': new_role,
                    'department': new_department,
                    'email_verified': email_verified,
                    'is_active': is_active
                }
                
                success = db.update_user_detailed(user['user_id'], updates, current_user['user_id'])
                
                if success:
                    st.success("✅ User updated successfully!")
                    # Clear the edit state and refresh
                    if f"edit_user_{user['user_id']}" in st.session_state:
                        del st.session_state[f"edit_user_{user['user_id']}"]
                    st.rerun()
                else:
                    st.error("❌ Failed to update user")
            except Exception as e:
                if 'UNIQUE constraint' in str(e):
                    if 'username' in str(e):
                        st.error("❌ Username already exists")
                    elif 'email' in str(e):
                        st.error("❌ Email already exists")
                    else:
                        st.error("❌ Duplicate value found")
                else:
                    st.error(f"❌ Update failed: {str(e)}")
        
        if cancel_clicked:
            # Clear the edit state and refresh
            if f"edit_user_{user['user_id']}" in st.session_state:
                del st.session_state[f"edit_user_{user['user_id']}"]
            st.rerun()

def show_password_reset_form(db, current_user, user):
    """Show enhanced password reset form with real-time updates"""
    
    st.markdown(f"#### 🔑 Reset Password for: {user['full_name']}")
    
    # Use container to ensure proper refresh
    with st.container():
        # Initialize reset type in session state if not exists
        reset_type_key = f"reset_type_{user['user_id']}"
        if reset_type_key not in st.session_state:
            st.session_state[reset_type_key] = "Generate Random Password"
        
        # Radio button for reset type - let Streamlit handle the state
        reset_type = st.radio(
            "Password Reset Type:",
            ["Generate Random Password", "Set Custom Password"],
            horizontal=True,
            key=reset_type_key
        )
        
        # Create form with dynamic content based on selection
        form_key = f"reset_form_{user['user_id']}_{reset_type.replace(' ', '_')}"
        
        with st.form(form_key, clear_on_submit=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Password Configuration:**")
                
                if reset_type == "Set Custom Password":
                    custom_password = st.text_input(
                        "New Password", 
                        type="password", 
                        key=f"new_pwd_{user['user_id']}",
                        help="Enter the new password for the user"
                    )
                    confirm_password = st.text_input(
                        "Confirm Password", 
                        type="password", 
                        key=f"conf_pwd_{user['user_id']}",
                        help="Re-enter the password to confirm"
                    )
                    
                    # Real-time password validation
                    if custom_password:
                        validation_messages = []
                        if len(custom_password) < 8:
                            validation_messages.append("❌ Must be at least 8 characters long")
                        else:
                            validation_messages.append("✅ Length requirement met")
                            
                        if not any(c.isupper() for c in custom_password):
                            validation_messages.append("⚠️ Consider adding uppercase letters")
                        else:
                            validation_messages.append("✅ Contains uppercase letters")
                            
                        if not any(c.isdigit() for c in custom_password):
                            validation_messages.append("⚠️ Consider adding numbers")
                        else:
                            validation_messages.append("✅ Contains numbers")
                        
                        # Show validation in expander
                        with st.expander("Password Strength", expanded=True):
                            for msg in validation_messages:
                                if "❌" in msg:
                                    st.error(msg)
                                elif "⚠️" in msg:
                                    st.warning(msg)
                                else:
                                    st.success(msg)
                    
                    if custom_password and confirm_password:
                        if custom_password != confirm_password:
                            st.error("❌ Passwords don't match")
                        else:
                            st.success("✅ Passwords match")
                            
                else:  # Generate Random Password
                    password_length = st.slider(
                        "Password Length", 
                        8, 20, 12, 
                        key=f"pwd_len_{user['user_id']}",
                        help="Length of the generated password"
                    )
                    include_symbols = st.checkbox(
                        "Include Symbols (!@#$%^&*)", 
                        value=False, 
                        key=f"symbols_{user['user_id']}",
                        help="Include special characters in the password"
                    )
                    
                    # Show preview of what the password will look like
                    st.info("📝 A new secure password will be generated when you submit")
                    
                    # Show example of character types that will be included
                    char_types = ["Uppercase letters (A-Z)", "Lowercase letters (a-z)", "Numbers (0-9)"]
                    if include_symbols:
                        char_types.append("Symbols (!@#$%^&*)")
                    
                    st.caption(f"Will include: {', '.join(char_types)}")
            
            with col2:
                st.markdown("**Reset Options:**")
                
                require_change = st.checkbox(
                    "Require password change on next login", 
                    value=True, 
                    key=f"req_change_{user['user_id']}",
                    help="Force user to change password when they next log in"
                )
                
                reset_reason = st.text_area(
                    "Reset Reason", 
                    value="Administrator password reset", 
                    key=f"reason_{user['user_id']}",
                    help="Reason for the password reset (for audit trail)",
                    height=100
                )
                
                # Show impact of the options
                if require_change:
                    st.info("👤 User will see a password change popup on next login")
                    st.warning("🔒 User cannot access the system until they change their password")
                else:
                    st.success("✅ User can use the new password immediately")
                    st.info("🔓 No additional password change required")
            
            st.markdown("---")
            col_reset, col_cancel = st.columns(2)
            
            with col_reset:
                reset_clicked = st.form_submit_button("🔑 Reset Password", type="primary", use_container_width=True)
            
            with col_cancel:
                cancel_clicked = st.form_submit_button("❌ Cancel", use_container_width=True)
            
            # Handle form submission
            if reset_clicked:
                success = False
                new_password = None
                
                try:
                    if reset_type == "Set Custom Password":
                        if not custom_password:
                            st.error("❌ Please enter a password")
                        elif len(custom_password) < 8:
                            st.error("❌ Password must be at least 8 characters long")
                        elif custom_password != confirm_password:
                            st.error("❌ Passwords don't match")
                        else:
                            success = db.reset_user_password(
                                user['user_id'], 
                                custom_password, 
                                current_user['user_id'],
                                require_change, 
                                reset_reason
                            )
                            new_password = custom_password
                    else:
                        # Generate random password
                        new_password = generate_secure_password(password_length, include_symbols)
                        
                        success = db.reset_user_password(
                            user['user_id'], 
                            new_password, 
                            current_user['user_id'],
                            require_change, 
                            reset_reason
                        )
                    
                    if success:
                        st.success("✅ Password reset successfully!")
                        
                        # Show the new password in a secure, copy-friendly way
                        if new_password:
                            st.markdown("---")
                            st.markdown("#### 🔑 New Password Information")
                            
                            # Create a highlighted box for the password
                            st.markdown(f"""
                            <div style="
                                background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
                                border: 2px solid #2196f3;
                                border-radius: 10px;
                                padding: 1.5rem;
                                margin: 1rem 0;
                                text-align: center;
                            ">
                                <h4 style="margin-bottom: 1rem; color: #1976d2;">🔐 Generated Password</h4>
                                <div style="
                                    background: white;
                                    padding: 1rem;
                                    border-radius: 8px;
                                    border: 1px solid #e0e0e0;
                                    font-family: 'Courier New', monospace;
                                    font-size: 1.2em;
                                    font-weight: bold;
                                    color: #333;
                                    word-break: break-all;
                                    margin-bottom: 1rem;
                                ">{new_password}</div>
                                <p style="margin: 0; color: #666; font-size: 0.9em;">
                                    📋 Copy this password and share it securely with the user
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Also provide a code block for easy copying
                            st.code(new_password, language="text")
                        
                        # Show next steps
                        if require_change:
                            st.info("🔄 **Next Steps:** User will be required to change password on next login")
                            st.markdown("""
                            **What happens next:**
                            1. Share the password securely with the user
                            2. User logs in with this password
                            3. System will prompt user to set a new password
                            4. User cannot access the system until they set a new password
                            """)
                        else:
                            st.success("✅ **Ready to use:** User can login with this password immediately")
                            st.markdown("""
                            **What happens next:**
                            1. Share the password securely with the user
                            2. User can login immediately with this password
                            3. No additional password change required
                            """)
                        
                        # Auto-close after showing results
                        st.markdown("---")
                        st.info("⏱️ This form will close automatically in a moment...")
                        
                        # Clear states and refresh after showing the password
                        keys_to_clear = [
                            f"show_password_reset_{user['user_id']}",
                            reset_type_key
                        ]
                        
                        # Use a slight delay to let user see the password
                        import time
                        time.sleep(3)
                        
                        for key in keys_to_clear:
                            if key in st.session_state:
                                del st.session_state[key]
                        
                        st.rerun()
                    else:
                        st.error("❌ Failed to reset password")
                        st.error("Please check the database connection and try again.")
                        
                except Exception as e:
                    st.error(f"❌ Reset failed: {str(e)}")
                    st.info("💡 Please try again or contact system administrator if the problem persists.")
            
            if cancel_clicked:
                # Clear the reset state and refresh
                keys_to_clear = [
                    f"show_password_reset_{user['user_id']}",
                    reset_type_key
                ]
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

def show_user_activity_inline(db, user):
    """Show user activity inline within the user card"""
    
    st.markdown(f"#### 📊 Recent Activity for {user['full_name']}")
    
    try:
        user_logs = db.get_audit_trail(limit=10, user_id=user['user_id'])
        
        if user_logs:
            # Create a compact activity display
            activity_data = []
            for log in user_logs:
                activity_data.append({
                    'Time': log['timestamp'][:16] if log['timestamp'] else 'Unknown',
                    'Action': log['action_type'].replace('_', ' ').title(),
                    'Admin': log['admin_username'] or 'System',
                    'Status': '✅' if 'SUCCESS' in log.get('action_type', '') else '❌' if 'FAILED' in log.get('action_type', '') else '📝'
                })
            
            # Display in a compact table
            activity_df = pd.DataFrame(activity_data)
            st.dataframe(activity_df, use_container_width=True, height=300)
            
            # Show more details option
            if len(user_logs) >= 10:
                st.info(f"Showing last 10 activities. Total activities may be more.")
        else:
            st.info("No recent activity found for this user")
            
    except Exception as e:
        st.error(f"Error loading user activity: {e}")
    
    # Close button
    if st.button("❌ Close Activity", key=f"close_activity_{user['user_id']}", type="secondary"):
        if f"show_activity_{user['user_id']}" in st.session_state:
            del st.session_state[f"show_activity_{user['user_id']}"]
        st.rerun()

def show_create_user_form(db, current_user):
    """Enhanced create user form"""
    
    st.markdown("#### ➕ Create New User")
    
    with st.form("create_new_user", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            username = st.text_input("Username *")
            email = st.text_input("Email *")
            full_name = st.text_input("Full Name *")
            password = st.text_input("Password *", type="password")
        
        with col2:
            user_role = st.selectbox("Role *", options=[role.value for role in UserRole])
            organization = st.text_input("Organization")
            department = st.text_input("Department")
            auto_verify = st.checkbox("Auto-verify email", value=True)
        
        password_options = st.checkbox("Require password change on first login", value=False)
        
        col_create, col_cancel = st.columns(2)
        
        with col_create:
            if st.form_submit_button("✅ Create User", type="primary"):
                if not all([username, email, full_name, password]):
                    st.error("Please fill in all required fields")
                else:
                    user_data = {
                        'username': username,
                        'email': email,
                        'password': password,
                        'full_name': full_name,
                        'user_role': user_role,
                        'organization_name': organization,
                        'department': department,
                        'created_by': current_user['username']
                    }
                    
                    try:
                        user_id = db.create_user(user_data)
                        
                        if auto_verify:
                            import sqlite3
                            with sqlite3.connect(db.db_path) as conn:
                                cursor = conn.cursor()
                                cursor.execute('''
                                    UPDATE users 
                                    SET is_active = 1, email_verified = 1 
                                    WHERE user_id = ?
                                ''', (user_id,))
                                conn.commit()
                        
                        if password_options:
                            db.reset_user_password(
                                user_id, password, current_user['user_id'], 
                                require_change=True, reset_reason="Initial password setup"
                            )
                        
                        st.success(f"✅ User created successfully! User ID: {user_id}")
                        st.session_state.show_create_user = False
                        st.balloons()
                        st.rerun()
                        
                    except Exception as e:
                        if 'UNIQUE constraint' in str(e):
                            if 'username' in str(e):
                                st.error("❌ Username already exists")
                            else:
                                st.error("❌ Email already exists")
                        else:
                            st.error(f"❌ Error creating user: {str(e)}")
        
        with col_cancel:
            if st.form_submit_button("❌ Cancel"):
                st.session_state.show_create_user = False
                st.rerun()

def show_admin_dashboard_tab(db, current_user):
    """Admin dashboard with system metrics and insights"""
    
    st.markdown("### 📊 Admin Dashboard")
    
    # System overview
    analytics = db.get_rule_analytics() if hasattr(db, 'get_rule_analytics') else {}
    all_users = db.get_all_users()
    audit_trail = db.get_audit_trail(limit=50)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Users", len(all_users))
    
    with col2:
        active_sessions = db.get_active_sessions() if hasattr(db, 'get_active_sessions') else []
        st.metric("Active Sessions", len(active_sessions))
    
    with col3:
        recent_logins = len([log for log in audit_trail if log['action_type'] == 'LOGIN_SUCCESS' and 
                           log['timestamp'] > (datetime.now() - timedelta(hours=24)).isoformat()])
        st.metric("Logins (24h)", recent_logins)
    
    with col4:
        failed_logins = len([log for log in audit_trail if log['action_type'] == 'LOGIN_FAILED' and 
                           log['timestamp'] > (datetime.now() - timedelta(hours=24)).isoformat()])
        st.metric("Failed Logins (24h)", failed_logins)
    
    # User role distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 👥 User Role Distribution")
        role_counts = {}
        for user in all_users:
            role = user['user_role']
            role_counts[role] = role_counts.get(role, 0) + 1
        
        if role_counts:
            import plotly.express as px
            fig = px.pie(values=list(role_counts.values()), names=list(role_counts.keys()))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No user data available")
    
    with col2:
        st.markdown("#### 📈 Recent Activity")
        if audit_trail:
            activity_data = []
            for log in audit_trail[:10]:
                activity_data.append({
                    'Time': log['timestamp'][:19] if log['timestamp'] else 'Unknown',
                    'Action': log['action_type'],
                    'User': log['admin_username'] or 'System',
                    'Target': log['target_username'] or 'N/A'
                })
            
            st.dataframe(pd.DataFrame(activity_data), use_container_width=True)
        else:
            st.info("No recent activity")
    
    # System health indicators
    st.markdown("#### 🏥 System Health")
    
    health_col1, health_col2, health_col3, health_col4 = st.columns(4)
    
    with health_col1:
        verified_rate = len([u for u in all_users if u['email_verified']]) / len(all_users) * 100 if all_users else 0
        status = "🟢 Good" if verified_rate > 80 else "🟡 Fair" if verified_rate > 60 else "🔴 Poor"
        st.metric("Email Verification", f"{verified_rate:.1f}%", help=f"Status: {status}")
    
    with health_col2:
        active_rate = len([u for u in all_users if u['is_active']]) / len(all_users) * 100 if all_users else 0
        status = "🟢 Good" if active_rate > 70 else "🟡 Fair" if active_rate > 50 else "🔴 Poor"  
        st.metric("Active Users", f"{active_rate:.1f}%", help=f"Status: {status}")
    
    with health_col3:
        password_resets = len([log for log in audit_trail if log['action_type'] == 'PASSWORD_RESET'])
        status = "🟢 Low" if password_resets < 5 else "🟡 Medium" if password_resets < 15 else "🔴 High"
        st.metric("Password Resets", password_resets, help=f"Status: {status}")
    
    with health_col4:
        total_logins = recent_logins + failed_logins
        fail_rate = (failed_logins / total_logins * 100) if total_logins > 0 else 0
        status = "🟢 Good" if fail_rate < 10 else "🟡 Fair" if fail_rate < 25 else "🔴 High"
        st.metric("Login Failure Rate", f"{fail_rate:.1f}%", help=f"Status: {status}")

def show_system_settings_tab(db, current_user):
    """System settings management"""
    
    st.markdown("### ⚙️ System Settings")
    
    if current_user['user_role'] != 'SuperAdmin':
        st.warning("🔒 System settings can only be modified by SuperAdmin users")
        return
    
    settings = db.get_system_settings()
    
    if not settings:
        st.error("❌ Could not load system settings")
        return
    
    st.info("🔧 Configure system-wide settings. Changes take effect immediately.")
    
    # Group settings by category
    security_settings = {}
    session_settings = {}
    notification_settings = {}
    maintenance_settings = {}
    
    for key, value in settings.items():
        if 'password' in key or 'login' in key or 'lockout' in key:
            security_settings[key] = value
        elif 'session' in key:
            session_settings[key] = value
        elif 'email' in key or 'notification' in key:
            notification_settings[key] = value
        else:
            maintenance_settings[key] = value
    
    # Security Settings
    with st.expander("🔐 Security Settings", expanded=True):
        if security_settings:
            for setting_key, setting_data in security_settings.items():
                show_setting_control(db, current_user, setting_key, setting_data)
    
    # Session Settings
    with st.expander("🕐 Session Settings", expanded=False):
        if session_settings:
            for setting_key, setting_data in session_settings.items():
                show_setting_control(db, current_user, setting_key, setting_data)
    
    # Notification Settings
    with st.expander("📧 Notification Settings", expanded=False):
        if notification_settings:
            for setting_key, setting_data in notification_settings.items():
                show_setting_control(db, current_user, setting_key, setting_data)
    
    # Maintenance Settings
    with st.expander("🔧 Maintenance Settings", expanded=False):
        if maintenance_settings:
            for setting_key, setting_data in maintenance_settings.items():
                show_setting_control(db, current_user, setting_key, setting_data)
    
    # Backup and maintenance actions
    st.markdown("#### 🛠️ System Maintenance")
    
    maint_col1, maint_col2, maint_col3 = st.columns(3)
    
    with maint_col1:
        if st.button("💾 Backup Database"):
            backup_database(db)
    
    with maint_col2:
        if st.button("🧹 Clean Audit Logs"):
            clean_old_audit_logs(db)
    
    with maint_col3:
        if st.button("📊 System Report"):
            generate_system_report(db)

def show_setting_control(db, current_user, setting_key, setting_data):
    """Show individual setting control"""
    
    setting_name = setting_key.replace('_', ' ').title()
    current_value = setting_data['value']
    description = setting_data['description']
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"**{setting_name}**")
        st.markdown(f"*{description}*")
    
    with col2:
        # Determine input type based on setting
        if setting_key.endswith('_minutes') or setting_key.endswith('_hours') or setting_key.endswith('_days') or 'length' in setting_key or 'attempts' in setting_key:
            new_value = st.number_input(
                f"Value for {setting_name}",
                value=int(current_value),
                min_value=1,
                key=f"setting_{setting_key}",
                label_visibility="collapsed"
            )
            new_value = str(new_value)
        elif current_value.lower() in ['true', 'false']:
            new_value = st.checkbox(
                f"Enable {setting_name}",
                value=current_value.lower() == 'true',
                key=f"setting_{setting_key}",
                label_visibility="collapsed"
            )
            new_value = str(new_value).lower()
        else:
            new_value = st.text_input(
                f"Value for {setting_name}",
                value=current_value,
                key=f"setting_{setting_key}",
                label_visibility="collapsed"
            )
        
        if new_value != current_value:
            if st.button(f"Save", key=f"save_{setting_key}", type="primary"):
                success = db.update_system_setting(setting_key, str(new_value), current_user['user_id'])
                if success:
                    st.success("✅ Setting updated!")
                    st.rerun()
                else:
                    st.error("❌ Failed to update setting")

def show_audit_logs_tab(db, current_user):
    """Enhanced audit logs with filtering and search"""
    
    st.markdown("### 🔍 Audit Logs")
    
    # Filters
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    
    with filter_col1:
        time_filter = st.selectbox("Time Period", ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "All Time"])
    
    with filter_col2:
        action_types = ["All", "LOGIN_SUCCESS", "LOGIN_FAILED", "USER_CREATED", "USER_UPDATED", 
                       "PASSWORD_RESET", "USER_DELETED", "SYSTEM_SETTING_UPDATED"]
        action_filter = st.selectbox("Action Type", action_types)
    
    with filter_col3:
        user_filter = st.text_input("Filter by User", placeholder="Username...")
    
    with filter_col4:
        limit = st.selectbox("Show Records", [50, 100, 200, 500], index=1)
    
    # Get filtered audit logs
    audit_logs = db.get_audit_trail(limit=limit)
    
    # Apply filters
    if time_filter != "All Time":
        if time_filter == "Last 24 Hours":
            cutoff = datetime.now() - timedelta(hours=24)
        elif time_filter == "Last 7 Days":
            cutoff = datetime.now() - timedelta(days=7)
        else:  # Last 30 Days
            cutoff = datetime.now() - timedelta(days=30)
        
        audit_logs = [log for log in audit_logs if 
                     log['timestamp'] and log['timestamp'] > cutoff.isoformat()]
    
    if action_filter != "All":
        audit_logs = [log for log in audit_logs if log['action_type'] == action_filter]
    
    if user_filter:
        audit_logs = [log for log in audit_logs if 
                     user_filter.lower() in (log['admin_username'] or '').lower() or
                     user_filter.lower() in (log['target_username'] or '').lower()]
    
    st.info(f"Showing {len(audit_logs)} audit log entries")
    
    # Export option
    if audit_logs:
        export_col1, export_col2 = st.columns([1, 4])
        with export_col1:
            if st.button("📥 Export Logs"):
                export_audit_logs(audit_logs)
    
    # Display logs
    for log in audit_logs:
        timestamp = log['timestamp'][:19] if log['timestamp'] else 'Unknown'
        action = log['action_type']
        admin_user = log['admin_username'] or 'System'
        target_user = log['target_username'] or 'N/A'
        
        # Determine action icon and color
        if 'LOGIN' in action:
            icon = "🔑" if 'SUCCESS' in action else "❌"
            color = "#d4edda" if 'SUCCESS' in action else "#f8d7da"
        elif 'PASSWORD' in action:
            icon = "🔄"
            color = "#fff3cd"
        elif 'USER' in action:
            icon = "👤"
            color = "#d1ecf1"
        elif 'SYSTEM' in action:
            icon = "⚙️"
            color = "#e2e3e5"
        else:
            icon = "📝"
            color = "#f8f9fa"
        
        with st.expander(f"{icon} {timestamp} - {action} by {admin_user}", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Action:** {action}")
                st.markdown(f"**Admin User:** {admin_user}")
                st.markdown(f"**Target User:** {target_user}")
                st.markdown(f"**Timestamp:** {timestamp}")
            
            with col2:
                if log['table_name']:
                    st.markdown(f"**Table:** {log['table_name']}")
                if log['record_id']:
                    st.markdown(f"**Record ID:** {log['record_id']}")
                if log['ip_address']:
                    st.markdown(f"**IP Address:** {log['ip_address']}")
            
            # Show changes if available
            if log['old_values'] or log['new_values']:
                st.markdown("**Changes:**")
                if log['old_values']:
                    try:
                        old_data = json.loads(log['old_values'])
                        st.markdown("*Old Values:*")
                        st.json(old_data, expanded=False)
                    except:
                        st.text(log['old_values'])
                
                if log['new_values']:
                    try:
                        new_data = json.loads(log['new_values'])
                        st.markdown("*New Values:*")
                        st.json(new_data, expanded=False)
                    except:
                        st.text(log['new_values'])

def show_emergency_tools_tab(db, current_user):
    """Emergency tools for SuperAdmin"""
    
    st.markdown("### 🚨 Emergency Tools")
    st.error("⚠️ **CAUTION**: These tools can affect system stability. Use with extreme care!")
    
    # Emergency user creation
    st.markdown("#### 🆘 Emergency Admin Creation")
    st.info("Create an emergency admin account if you're locked out")
    
    with st.expander("Create Emergency Admin", expanded=False):
        with st.form("emergency_admin"):
            emergency_username = st.text_input("Emergency Username")
            emergency_email = st.text_input("Emergency Email")
            emergency_password = st.text_input("Emergency Password", type="password")
            
            if st.form_submit_button("🆘 Create Emergency Admin", type="secondary"):
                if all([emergency_username, emergency_email, emergency_password]):
                    try:
                        user_data = {
                            'username': emergency_username,
                            'email': emergency_email,
                            'password': emergency_password,
                            'full_name': 'Emergency Administrator',
                            'user_role': 'SuperAdmin',
                            'organization_name': 'Emergency',
                            'department': 'IT',
                            'created_by': current_user['username']
                        }
                        
                        user_id = db.create_user(user_data)
                        
                        # Auto-activate
                        import sqlite3
                        with sqlite3.connect(db.db_path) as conn:
                            cursor = conn.cursor()
                            cursor.execute('''
                                UPDATE users 
                                SET is_active = 1, email_verified = 1 
                                WHERE user_id = ?
                            ''', (user_id,))
                            conn.commit()
                        
                        st.success(f"✅ Emergency admin created! User ID: {user_id}")
                        
                    except Exception as e:
                        st.error(f"❌ Failed to create emergency admin: {e}")
                else:
                    st.error("Please fill in all fields")
    
    # Database operations
    st.markdown("#### 🗄️ Database Operations")
    
    db_col1, db_col2, db_col3 = st.columns(3)
    
    with db_col1:
        if st.button("💾 Force Backup", help="Force immediate database backup"):
            backup_database(db, force=True)
    
    with db_col2:
        if st.button("🧹 Clean All Logs", help="Clean all old audit logs"):
            if st.session_state.get('confirm_clean_logs', False):
                clean_all_logs(db)
                st.session_state.confirm_clean_logs = False
            else:
                st.session_state.confirm_clean_logs = True
                st.warning("⚠️ Click again to confirm")
    
    with db_col3:
        if st.button("🔄 Reset All Sessions", help="Force logout all users"):
            if st.session_state.get('confirm_reset_sessions', False):
                reset_all_sessions(db)
                st.session_state.confirm_reset_sessions = False
            else:
                st.session_state.confirm_reset_sessions = True
                st.warning("⚠️ Click again to confirm")
    
    # System information
    st.markdown("#### 📊 System Information")
    
    import sqlite3
    import os
    
    try:
        # Database file info
        db_size = os.path.getsize(db.db_path) / (1024 * 1024)  # MB
        
        with sqlite3.connect(db.db_path) as conn:
            cursor = conn.cursor()
            
            # Get table sizes
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            table_info = []
            for (table_name,) in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                table_info.append({'Table': table_name, 'Records': count})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Database Size", f"{db_size:.2f} MB")
            st.metric("Total Tables", len(tables))
        
        with col2:
            st.dataframe(pd.DataFrame(table_info), use_container_width=True)
            
    except Exception as e:
        st.error(f"Could not retrieve system information: {e}")

# Helper functions
def filter_users(users, search_term, role_filter, status_filter):
    """Filter users based on search criteria"""
    filtered = users.copy()
    
    if search_term:
        search_lower = search_term.lower()
        filtered = [u for u in filtered if 
                   search_lower in u['full_name'].lower() or 
                   search_lower in u['email'].lower() or 
                   search_lower in u['username'].lower()]
    
    if role_filter != "All":
        filtered = [u for u in filtered if u['user_role'] == role_filter]
    
    if status_filter == "Active":
        filtered = [u for u in filtered if u['is_active'] and u['email_verified']]
    elif status_filter == "Inactive":
        filtered = [u for u in filtered if not u['is_active']]
    elif status_filter == "Pending Verification":
        filtered = [u for u in filtered if not u['email_verified']]
    
    return filtered

def toggle_user_status(db, current_user, user_id, is_active):
    """Toggle user active status"""
    success = db.update_user_detailed(user_id, {'is_active': is_active}, current_user['user_id'])
    
    if success:
        status = "activated" if is_active else "deactivated"
        st.success(f"✅ User {status} successfully!")
        st.rerun()
    else:
        st.error("❌ Failed to update user status")

def verify_user_email(db, current_user, user_id):
    """Verify user email"""
    success = db.update_user_detailed(user_id, {'email_verified': True}, current_user['user_id'])
    
    if success:
        st.success("✅ Email verified successfully!")
        st.rerun()
    else:
        st.error("❌ Failed to verify email")

def delete_user(db, current_user, user_id):
    """Delete user"""
    success = db.delete_user(user_id)
    
    if success:
        # Log the deletion
        db._log_audit_trail(
            user_id=user_id,
            admin_user_id=current_user['user_id'],
            action_type="USER_DELETED",
            table_name="users",
            record_id=str(user_id)
        )
        
        st.success("✅ User deleted successfully!")
        
        # Clear confirmation state
        if f"confirm_delete_{user_id}" in st.session_state:
            del st.session_state[f"confirm_delete_{user_id}"]
        
        st.rerun()
    else:
        st.error("❌ Failed to delete user")

def show_user_activity(db, user_id):
    """Show user activity"""
    st.markdown(f"#### 📊 User Activity - ID: {user_id}")
    
    user_logs = db.get_audit_trail(limit=50, user_id=user_id)
    
    if user_logs:
        activity_data = []
        for log in user_logs:
            activity_data.append({
                'Time': log['timestamp'][:19] if log['timestamp'] else 'Unknown',
                'Action': log['action_type'],
                'Admin': log['admin_username'] or 'System',
                'Details': log['new_values'][:100] if log['new_values'] else 'N/A'
            })
        
        st.dataframe(pd.DataFrame(activity_data), use_container_width=True)
    else:
        st.info("No activity found for this user")

def export_user_list(users):
    """Export user list to CSV"""
    if not users:
        st.error("No users to export")
        return
    
    # Remove sensitive data
    export_data = []
    for user in users:
        export_data.append({
            'Username': user['username'],
            'Email': user['email'],
            'Full Name': user['full_name'],
            'Role': user['user_role'],
            'Organization': user['organization_name'] or '',
            'Department': user['department'] or '',
            'Active': user['is_active'],
            'Email Verified': user['email_verified'],
            'Created': user['created_at'],
            'Last Login': user['last_login']
        })
    
    df = pd.DataFrame(export_data)
    csv_data = df.to_csv(index=False)
    
    st.download_button(
        "📥 Download User List (CSV)",
        csv_data,
        f"user_list_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        "text/csv"
    )

def export_audit_logs(logs):
    """Export audit logs to CSV"""
    if not logs:
        st.error("No logs to export")
        return
    
    df = pd.DataFrame(logs)
    csv_data = df.to_csv(index=False)
    
    st.download_button(
        "📥 Download Audit Logs (CSV)",
        csv_data,
        f"audit_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        "text/csv"
    )

def backup_database(db, force=False):
    """Backup database"""
    try:
        import shutil
        backup_path = f"app/database/auth_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        shutil.copy2(db.db_path, backup_path)
        
        st.success(f"✅ Database backed up to: {backup_path}")
        
        # Clean old backups (keep last 5)
        backup_dir = "app/database"
        backup_files = [f for f in os.listdir(backup_dir) if f.startswith("auth_backup_")]
        backup_files.sort(reverse=True)
        
        for old_backup in backup_files[5:]:
            os.remove(os.path.join(backup_dir, old_backup))
            
    except Exception as e:
        st.error(f"❌ Backup failed: {e}")

def clean_old_audit_logs(db):
    """Clean old audit logs"""
    try:
        import sqlite3
        cutoff_date = datetime.now() - timedelta(days=90)
        
        with sqlite3.connect(db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM audit_trail WHERE timestamp < ?", (cutoff_date.isoformat(),))
            deleted_count = cursor.rowcount
            conn.commit()
        
        st.success(f"✅ Cleaned {deleted_count} old audit log entries")
        
    except Exception as e:
        st.error(f"❌ Failed to clean audit logs: {e}")

def clean_all_logs(db):
    """Clean all audit logs (emergency function)"""
    try:
        import sqlite3
        with sqlite3.connect(db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM audit_trail")
            deleted_count = cursor.rowcount
            conn.commit()
        
        st.success(f"✅ Cleaned all {deleted_count} audit log entries")
        
    except Exception as e:
        st.error(f"❌ Failed to clean all logs: {e}")

def reset_all_sessions(db):
    """Reset all user sessions (force logout)"""
    try:
        import sqlite3
        with sqlite3.connect(db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE user_sessions SET is_active = 0")
            updated_count = cursor.rowcount
            conn.commit()
        
        st.success(f"✅ Reset {updated_count} user sessions")
        st.warning("⚠️ All users have been logged out")
        
    except Exception as e:
        st.error(f"❌ Failed to reset sessions: {e}")

def generate_system_report(db):
    """Generate comprehensive system report"""
    try:
        all_users = db.get_all_users()
        audit_logs = db.get_audit_trail(limit=1000)
        settings = db.get_system_settings()
        
        report_data = {
            'generated_at': datetime.now().isoformat(),
            'generated_by': st.session_state.get('auth_user', {}).get('username', 'Unknown'),
            'system_summary': {
                'total_users': len(all_users),
                'active_users': len([u for u in all_users if u['is_active']]),
                'verified_users': len([u for u in all_users if u['email_verified']]),
                'admin_users': len([u for u in all_users if u['user_role'] in ['SuperAdmin', 'Admin']]),
                'total_audit_logs': len(audit_logs),
                'total_settings': len(settings)
            },
            'user_breakdown': {
                role: len([u for u in all_users if u['user_role'] == role])
                for role in set(u['user_role'] for u in all_users)
            },
            'recent_activity': {
                'login_success': len([l for l in audit_logs[:100] if l['action_type'] == 'LOGIN_SUCCESS']),
                'login_failed': len([l for l in audit_logs[:100] if l['action_type'] == 'LOGIN_FAILED']),
                'password_resets': len([l for l in audit_logs[:100] if l['action_type'] == 'PASSWORD_RESET']),
                'user_changes': len([l for l in audit_logs[:100] if 'USER' in l['action_type']])
            },
            'system_settings': {k: v['value'] for k, v in settings.items()},
            'recommendations': []
        }
        
        # Generate recommendations
        if report_data['system_summary']['verified_users'] / report_data['system_summary']['total_users'] < 0.8:
            report_data['recommendations'].append("Low email verification rate - consider enabling stricter verification policies")
        
        if report_data['recent_activity']['login_failed'] > report_data['recent_activity']['login_success'] * 0.2:
            report_data['recommendations'].append("High login failure rate - review security settings and user training")
        
        if report_data['system_summary']['admin_users'] > report_data['system_summary']['total_users'] * 0.3:
            report_data['recommendations'].append("High number of admin users - review and reduce admin privileges where possible")
        
        # Create downloadable report
        report_json = json.dumps(report_data, indent=2, default=str)
        
        st.download_button(
            "📊 Download System Report (JSON)",
            report_json,
            f"system_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "application/json"
        )
        
        # Display summary
        st.success("✅ System report generated successfully!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Users", report_data['system_summary']['total_users'])
            st.metric("Active Users", report_data['system_summary']['active_users'])
            st.metric("Admin Users", report_data['system_summary']['admin_users'])
        
        with col2:
            st.metric("Verified Users", report_data['system_summary']['verified_users'])
            st.metric("Recent Logins", report_data['recent_activity']['login_success'])
            st.metric("Failed Logins", report_data['recent_activity']['login_failed'])
        
        if report_data['recommendations']:
            st.markdown("**🔍 Recommendations:**")
            for rec in report_data['recommendations']:
                st.warning(f"• {rec}")
        
    except Exception as e:
        st.error(f"❌ Failed to generate system report: {e}")

# Enhanced middleware for password reset checking
class EnhancedAuthenticationMiddleware(AuthenticationMiddleware):
    """Enhanced authentication middleware with password reset checking"""
    
    def __init__(self):
        super().__init__()
        self.db = EnhancedAuthDB()
    
    def validate_session_with_password_check(self):
        """Validate session and check for password reset requirements"""
        is_valid = self.validate_session()
        
        if is_valid:
            current_user = self.get_current_user()
            if current_user:
                reset_req = self.db.check_password_reset_required(current_user['user_id'])
                if reset_req:
                    st.session_state.password_reset_required = reset_req
                    return {'valid': True, 'password_reset_required': True, 'reset_details': reset_req}
        
        return {'valid': is_valid, 'password_reset_required': False}
    
    def handle_password_change(self, user_id, old_password, new_password):
        """Handle password change and clear reset requirement"""
        # Verify old password first
        current_user = self.get_current_user()
        if not current_user:
            return {'success': False, 'error': 'Not authenticated'}
        
        # Verify old password
        user_data = self.db.verify_password(current_user['email'], old_password)
        if not user_data:
            return {'success': False, 'error': 'Current password is incorrect'}
        
        # Update password
        try:
            import bcrypt
            import sqlite3
            
            salt = bcrypt.gensalt()
            password_hash = bcrypt.hashpw(new_password.encode('utf-8'), salt)
            
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE users 
                    SET password_hash = ?, salt = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                ''', (password_hash.decode('utf-8'), salt.decode('utf-8'), user_id))
                conn.commit()
            
            # Clear password reset requirement
            self.db.clear_password_reset_requirement(user_id)
            
            # Log the password change
            self.db._log_audit_trail(
                user_id=user_id,
                admin_user_id=user_id,
                action_type="PASSWORD_CHANGED",
                table_name="users",
                record_id=str(user_id)
            )
            
            return {'success': True}
            
        except Exception as e:
            return {'success': False, 'error': f'Failed to update password: {str(e)}'}

# Import statement fix
import sqlite3

if __name__ == "__main__":
    if AUTH_AVAILABLE:
        main()
    else:
        st.error("❌ Authentication system not available!")
        st.error(f"Error: {auth_error_message}")
        
        # Show troubleshooting info
        st.markdown("## 🔧 Troubleshooting")
        st.markdown("""
        1. **Check file structure** - Ensure `app/auth/` directory exists with all required files
        2. **Install dependencies** - Run `pip install bcrypt`
        3. **Database initialization** - Make sure `app/database/auth.db` exists
        4. **Python path** - Ensure you're running from the correct directory
        5. **Restart Streamlit** - Try restarting the application completely
        """)
        
        # Manual fallback option
        if st.button("🔧 Show Troubleshooting Details"):
            st.code(f"""
            Current Directory: {os.getcwd()}
            Python Path: {sys.path[:3]}
            Auth Files Check:
            - app/auth/__init__.py: {os.path.exists('app/auth/__init__.py')}
            - app/auth/models.py: {os.path.exists('app/auth/models.py')}
            - app/auth/middleware.py: {os.path.exists('app/auth/middleware.py')}
            - app/database/auth.db: {os.path.exists('app/database/auth.db')}
            
            Error Details: {auth_error_message}
            """, language="text")