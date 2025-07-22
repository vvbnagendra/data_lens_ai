# File: app/pages/Admin_User_Management.py
# Fixed User Management Interface with better error handling and imports

import streamlit as st
import pandas as pd
import json
import sys
import os
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="User Management",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Fix Python Path Issues ---
# Get the current directory and add app directory to Python path
current_dir = os.getcwd()
app_dir = os.path.join(current_dir, "app")

# Add both current directory and app directory to Python path
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

# Debug information (remove in production)
st.sidebar.write("**Debug Info:**")
st.sidebar.write(f"Current Dir: {current_dir}")
st.sidebar.write(f"App Dir: {app_dir}")
st.sidebar.write(f"Python Path: {sys.path[:2]}")

# --- Import Authentication System with Better Error Handling ---
AUTH_AVAILABLE = False
auth_error_message = ""

try:
    # Try different import patterns
    try:
        from auth.middleware import AuthenticationMiddleware
        from auth.models import AuthenticationDB, UserRole
        AUTH_AVAILABLE = True
        st.sidebar.success("✅ Direct auth import successful")
    except ImportError:
        # Try with app prefix
        from app.auth.middleware import AuthenticationMiddleware
        from app.auth.models import AuthenticationDB, UserRole
        AUTH_AVAILABLE = True
        st.sidebar.success("✅ App.auth import successful")
except ImportError as e:
    auth_error_message = f"Import Error: {str(e)}"
    st.sidebar.error(f"❌ Import failed: {str(e)}")
except Exception as e:
    auth_error_message = f"General Error: {str(e)}"
    st.sidebar.error(f"❌ General error: {str(e)}")

# --- Import Page Protection ---
try:
    if AUTH_AVAILABLE:
        from auth.page_protection import require_auth
        st.sidebar.success("✅ Page protection imported")
    else:
        # Create a dummy decorator if auth not available
        def require_auth():
            def decorator(func):
                def wrapper(*args, **kwargs):
                    st.error("❌ Authentication system not available")
                    st.error(f"Error details: {auth_error_message}")
                    show_troubleshooting_info()
                    return None
                return wrapper
            return decorator
        st.sidebar.warning("⚠️ Using dummy auth decorator")
except ImportError:
    # Create a dummy decorator
    def require_auth():
        def decorator(func):
            def wrapper(*args, **kwargs):
                st.error("❌ Page protection not available")
                show_troubleshooting_info()
                return None
            return wrapper
        return decorator
    st.sidebar.error("❌ Page protection import failed")

def show_troubleshooting_info():
    """Show troubleshooting information when auth system fails"""
    
    st.markdown("## 🔧 Authentication System Troubleshooting")
    
    with st.expander("🔍 Diagnostic Information", expanded=True):
        st.write("**Current working directory:**", os.getcwd())
        st.write("**Python path (first 3 entries):**", sys.path[:3])
        
        # Check for expected files
        st.write("**File existence check:**")
        expected_files = [
            "app/auth/__init__.py",
            "app/auth/models.py", 
            "app/auth/middleware.py",
            "app/database/auth.db"
        ]
        
        for file_path in expected_files:
            exists = os.path.exists(file_path)
            status = "✅" if exists else "❌"
            st.write(f"{status} {file_path}")
        
        st.write("**Error details:**", auth_error_message)
    
    with st.expander("🛠️ Quick Fixes", expanded=True):
        st.markdown("""
        ### Try these solutions:
        
        1. **Check if you're running from the correct directory:**
           ```bash
           # Make sure you're in the project root directory
           cd /path/to/your/project
           streamlit run app/Home.py
           ```
        
        2. **Verify file structure:**
           ```
           your-project/
           ├── app/
           │   ├── auth/
           │   │   ├── __init__.py
           │   │   ├── models.py
           │   │   ├── middleware.py
           │   │   └── page_protection.py
           │   └── pages/
           │       └── Admin_User_Management.py
           ```
        
        3. **Check if authentication database exists:**
           - Look for `app/database/auth.db` file
           - If missing, run your app's setup script
        
        4. **Install missing dependencies:**
           ```bash
           pip install bcrypt
           pip install sqlite3  # Usually included with Python
           ```
        
        5. **Restart Streamlit completely:**
           - Stop the app (Ctrl+C)
           - Start again: `streamlit run app/Home.py`
        """)
    
    # Manual login form as fallback
    with st.expander("🔑 Manual Login (Fallback)", expanded=False):
        st.warning("⚠️ This is a fallback login when the main auth system fails")
        
        with st.form("manual_login"):
            email = st.text_input("Email", value="admin@datalensai.com")
            password = st.text_input("Password", type="password", value="DataLens2024!")
            
            if st.form_submit_button("Try Manual Login"):
                if email == "admin@datalensai.com" and password == "DataLens2024!":
                    st.success("✅ Manual login successful!")
                    st.info("However, the full authentication system still needs to be fixed.")
                    
                    # Try to show minimal user management
                    show_minimal_user_management()
                else:
                    st.error("❌ Manual login failed")

def show_minimal_user_management():
    """Show minimal user management without full auth system"""
    
    st.markdown("## 👥 Minimal User Management (Fallback Mode)")
    st.warning("⚠️ Running in fallback mode - limited functionality available")
    
    # Try to access database directly
    try:
        import sqlite3
        
        db_path = "app/database/auth.db"
        if not os.path.exists(db_path):
            st.error(f"❌ Database not found: {db_path}")
            st.info("💡 The database needs to be created first. Try running your main app to initialize it.")
            return
        
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Get all users
            cursor.execute("SELECT user_id, username, email, full_name, user_role, is_active, email_verified, created_at FROM users ORDER BY created_at DESC")
            users = cursor.fetchall()
            
            if users:
                st.success(f"✅ Found {len(users)} users in database")
                
                # Display users in a table
                user_data = []
                for user in users:
                    user_data.append({
                        "ID": user[0],
                        "Username": user[1],
                        "Email": user[2], 
                        "Full Name": user[3],
                        "Role": user[4],
                        "Active": "✅ Yes" if user[5] else "❌ No",
                        "Email Verified": "✅ Yes" if user[6] else "❌ No",
                        "Created": user[7][:10] if user[7] else "Unknown"
                    })
                
                df = pd.DataFrame(user_data)
                st.dataframe(df, use_container_width=True)
                
                # Basic user creation form
                with st.expander("➕ Create New User (Basic)", expanded=False):
                    st.warning("⚠️ Basic user creation - limited validation")
                    
                    with st.form("basic_user_creation"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            new_username = st.text_input("Username")
                            new_email = st.text_input("Email")
                            new_password = st.text_input("Password", type="password")
                        
                        with col2:
                            new_full_name = st.text_input("Full Name")
                            new_role = st.selectbox("Role", ["SuperAdmin", "Admin", "DataScientist", "BusinessAnalyst", "Developer", "Viewer"])
                        
                        if st.form_submit_button("Create User"):
                            if all([new_username, new_email, new_password, new_full_name]):
                                try:
                                    # Hash password with bcrypt if available
                                    try:
                                        import bcrypt
                                        salt = bcrypt.gensalt()
                                        password_hash = bcrypt.hashpw(new_password.encode('utf-8'), salt).decode('utf-8')
                                        salt_str = salt.decode('utf-8')
                                    except ImportError:
                                        # Fallback to simple hashing (NOT SECURE - for demo only)
                                        import hashlib
                                        password_hash = hashlib.sha256(new_password.encode()).hexdigest()
                                        salt_str = "demo_salt"
                                        st.warning("⚠️ Using basic password hashing - not secure for production!")
                                    
                                    # Insert user
                                    cursor.execute('''
                                        INSERT INTO users (username, email, password_hash, salt, full_name, user_role, is_active, email_verified)
                                        VALUES (?, ?, ?, ?, ?, ?, 1, 1)
                                    ''', (new_username, new_email, password_hash, salt_str, new_full_name, new_role))
                                    conn.commit()
                                    
                                    st.success(f"✅ User {new_username} created successfully!")
                                    st.info("🔄 Refresh the page to see the new user")
                                    
                                except sqlite3.IntegrityError as e:
                                    if "username" in str(e).lower():
                                        st.error("❌ Username already exists")
                                    elif "email" in str(e).lower():
                                        st.error("❌ Email already exists")
                                    else:
                                        st.error(f"❌ Database error: {e}")
                                except Exception as e:
                                    st.error(f"❌ Error creating user: {e}")
                            else:
                                st.error("❌ Please fill in all required fields")
                
            else:
                st.info("📝 No users found in database")
                st.info("💡 This might mean the database needs to be initialized")
    
    except Exception as e:
        st.error(f"❌ Database access error: {e}")
        st.info("💡 Make sure the database file exists and is accessible")

@require_auth()
def main():
    """Protected main function for User Management"""
    
    if not AUTH_AVAILABLE:
        st.error("❌ Authentication system not available")
        st.error(f"Details: {auth_error_message}")
        show_troubleshooting_info()
        return
    
    # If we reach here, authentication is working
    # Initialize auth and db
    auth = AuthenticationMiddleware()
    db = AuthenticationDB()
    current_user = auth.get_current_user()
    
    # Check if user is SuperAdmin or Admin
    if current_user['user_role'] not in ['SuperAdmin', 'Admin']:
        st.error("❌ Access Denied: Only SuperAdmin and Admin users can access this page")
        st.info("👥 This page is for user management and requires administrative privileges")
        return
    
    # Apply professional styling
    try:
        from assets.streamlit_styles import apply_professional_styling, create_nav_header
        apply_professional_styling()
        create_nav_header("👥 User Management", "Manage users, roles, and permissions")
    except ImportError:
        st.title("👥 User Management")
        st.markdown("*Manage users, roles, and permissions*")
    
    # Success message
    st.success("✅ Authentication system working correctly!")
    st.info(f"👤 Logged in as: {current_user['full_name']} ({current_user['user_role']})")
    
    # Show full user management interface
    show_full_user_management(db, current_user)

def show_full_user_management(db, current_user):
    """Show the full user management interface"""
    
    st.markdown("### 🚀 Full User Management Interface")
    st.info("🎉 Your authentication system is working! Here you would see the full user management interface.")
    
    # Get basic user stats
    try:
        all_users = db.get_all_users()
        st.metric("Total Users", len(all_users))
        
        # Show user list
        if all_users:
            st.markdown("### 👥 Current Users")
            
            user_data = []
            for user in all_users:
                user_data.append({
                    "Username": user['username'],
                    "Full Name": user['full_name'],
                    "Email": user['email'],
                    "Role": user['user_role'],
                    "Status": "✅ Active" if user['is_active'] else "❌ Inactive",
                    "Email Verified": "✅ Yes" if user['email_verified'] else "❌ No"
                })
            
            df = pd.DataFrame(user_data)
            st.dataframe(df, use_container_width=True)
        
        # User creation form
        st.markdown("### ➕ Create New User")
        
        with st.form("create_user_full"):
            col1, col2 = st.columns(2)
            
            with col1:
                username = st.text_input("Username *")
                email = st.text_input("Email *") 
                full_name = st.text_input("Full Name *")
                password = st.text_input("Password *", type="password")
            
            with col2:
                user_role = st.selectbox("Role *", ["SuperAdmin", "Admin", "DataScientist", "BusinessAnalyst", "Developer", "Viewer"])
                organization = st.text_input("Organization")
                department = st.text_input("Department")
                auto_verify = st.checkbox("Auto-verify email", value=True)
            
            if st.form_submit_button("Create User", type="primary"):
                if all([username, email, full_name, password]):
                    try:
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
                        
                        user_id = db.create_user(user_data)
                        
                        if auto_verify:
                            # Auto-verify and activate
                            import sqlite3
                            with sqlite3.connect(db.db_path) as conn:
                                cursor = conn.cursor()
                                cursor.execute('''
                                    UPDATE users 
                                    SET is_active = 1, email_verified = 1 
                                    WHERE user_id = ?
                                ''', (user_id,))
                                conn.commit()
                        
                        st.success(f"✅ User {username} created successfully! (ID: {user_id})")
                        st.balloons()
                        
                    except Exception as e:
                        if 'UNIQUE constraint' in str(e):
                            if 'username' in str(e):
                                st.error("❌ Username already exists")
                            else:
                                st.error("❌ Email already exists") 
                        else:
                            st.error(f"❌ Error creating user: {e}")
                else:
                    st.error("❌ Please fill in all required fields")
    
    except Exception as e:
        st.error(f"❌ Error accessing database: {e}")

# Alternative entry point for debugging
if __name__ == "__main__":
    if AUTH_AVAILABLE:
        main()
    else:
        st.error("❌ Authentication system not available!")
        st.error(f"Error: {auth_error_message}")
        show_troubleshooting_info()