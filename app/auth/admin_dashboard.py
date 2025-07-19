# app/auth/admin_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

from .models import AuthenticationDB, UserRole
from .middleware import AuthenticationMiddleware, require_permission

class AdminDashboard:
    """
    Comprehensive admin dashboard for user management and analytics
    """
    
    def __init__(self):
        self.db = AuthenticationDB()
        self.auth = AuthenticationMiddleware()
    
    def show_dashboard(self):
        """Main admin dashboard interface"""
        
        # Check permissions
        if not self.auth.has_permission("can_access_admin_dashboard"):
            st.error("❌ Access denied. Admin privileges required.")
            return
        
        st.set_page_config(
            page_title="Admin Dashboard - Data Lens AI",
            page_icon="👑",
            layout="wide"
        )
        
        # Apply styling
        try:
            from assets.streamlit_styles import apply_professional_styling, create_nav_header
            apply_professional_styling()
            create_nav_header("👑 Admin Dashboard", "User management and system analytics")
        except ImportError:
            st.title("👑 Admin Dashboard")
        
        # Custom CSS for admin dashboard
        st.markdown("""
        <style>
            .admin-metric-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1.5rem;
                border-radius: 10px;
                color: white;
                text-align: center;
                margin-bottom: 1rem;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }
            
            .admin-section {
                background: white;
                padding: 2rem;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                margin-bottom: 2rem;
            }
            
            .user-card {
                border: 1px solid #e0e0e0;
                padding: 1rem;
                border-radius: 8px;
                margin: 0.5rem 0;
                background: #f9f9f9;
            }
            
            .active-user { border-left: 4px solid #28a745; }
            .inactive-user { border-left: 4px solid #dc3545; }
            .pending-user { border-left: 4px solid #ffc107; }
            
            .session-card {
                background: #f8f9fa;
                padding: 1rem;
                border-radius: 6px;
                margin: 0.5rem 0;
                border-left: 3px solid #007bff;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Main navigation
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", 
            "👥 User Management", 
            "🔑 Session Management", 
            "📈 Analytics", 
            "🔍 Activity Logs"
        ])
        
        with tab1:
            self.show_overview()
        
        with tab2:
            self.show_user_management()
        
        with tab3:
            self.show_session_management()
        
        with tab4:
            self.show_analytics()
        
        with tab5:
            self.show_activity_logs()
    
    def show_overview(self):
        """Show dashboard overview"""
        
        st.markdown("### 📊 System Overview")
        
        # Get statistics
        stats = self.db.get_login_statistics()
        all_users = self.db.get_all_users()
        active_sessions = self.db.get_active_sessions()
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="admin-metric-card">
                <h3>{stats['total_users']}</h3>
                <p>Total Users</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="admin-metric-card">
                <h3>{stats['active_users']}</h3>
                <p>Active Users</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="admin-metric-card">
                <h3>{len(active_sessions)}</h3>
                <p>Active Sessions</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            success_rate = 0
            if stats['successful_logins'] + stats['failed_logins'] > 0:
                success_rate = (stats['successful_logins'] / 
                              (stats['successful_logins'] + stats['failed_logins']) * 100)
            
            st.markdown(f"""
            <div class="admin-metric-card">
                <h3>{success_rate:.1f}%</h3>
                <p>Login Success Rate</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick stats section
        st.markdown("### 📋 Quick Statistics")
        
        stat_col1, stat_col2 = st.columns(2)
        
        with stat_col1:
            st.markdown('<div class="admin-section">', unsafe_allow_html=True)
            st.markdown("#### 👥 User Distribution by Role")
            
            if stats['users_by_role']:
                # Create role distribution chart
                roles = list(stats['users_by_role'].keys())
                counts = list(stats['users_by_role'].values())
                
                fig_roles = px.pie(
                    values=counts,
                    names=roles,
                    title="Users by Role",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_roles.update_layout(height=300)
                st.plotly_chart(fig_roles, use_container_width=True)
            else:
                st.info("No role distribution data available")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with stat_col2:
            st.markdown('<div class="admin-section">', unsafe_allow_html=True)
            st.markdown("#### 📊 Recent Activity")
            
            # Recent user registrations
            recent_users = [u for u in all_users if u['created_at']]
            recent_users.sort(key=lambda x: x['created_at'], reverse=True)
            
            st.markdown("**Recent Registrations:**")
            for user in recent_users[:5]:
                status_icon = "✅" if user['is_active'] else "⏳" if not user['email_verified'] else "❌"
                st.markdown(f"{status_icon} {user['full_name']} ({user['email']}) - {user['created_at'][:10]}")
            
            if not recent_users:
                st.info("No recent registrations")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # System health indicators
        st.markdown("### 🏥 System Health")
        
        health_col1, health_col2, health_col3 = st.columns(3)
        
        with health_col1:
            # Email verification rate
            verified_users = sum(1 for u in all_users if u['email_verified'])
            verification_rate = (verified_users / len(all_users) * 100) if all_users else 0
            
            status = "🟢 Good" if verification_rate > 80 else "🟡 Fair" if verification_rate > 60 else "🔴 Poor"
            st.metric("Email Verification Rate", f"{verification_rate:.1f}%", help=f"Status: {status}")
        
        with health_col2:
            # Failed login rate
            total_logins = stats['successful_logins'] + stats['failed_logins']
            failed_rate = (stats['failed_logins'] / total_logins * 100) if total_logins > 0 else 0
            
            status = "🟢 Good" if failed_rate < 10 else "🟡 Fair" if failed_rate < 25 else "🔴 High"
            st.metric("Failed Login Rate", f"{failed_rate:.1f}%", help=f"Status: {status}")
        
        with health_col3:
            # Active session ratio
            session_ratio = (len(active_sessions) / stats['active_users'] * 100) if stats['active_users'] > 0 else 0
            
            status = "🟢 Normal" if session_ratio < 150 else "🟡 High" if session_ratio < 300 else "🔴 Very High"
            st.metric("Session/User Ratio", f"{session_ratio:.1f}%", help=f"Status: {status}")
    
    def show_user_management(self):
        """Show user management interface"""
        
        if not self.auth.has_permission("can_manage_users"):
            st.error("❌ Access denied. User management privileges required.")
            return
        
        st.markdown("### 👥 User Management")
        
        # Get all users
        all_users = self.db.get_all_users()
        
        # User management actions
        action_col1, action_col2, action_col3 = st.columns(3)
        
        with action_col1:
            if st.button("➕ Create New User", type="primary"):
                st.session_state.show_create_user = True
        
        with action_col2:
            if st.button("📊 Export User List"):
                self.export_user_list(all_users)
        
        with action_col3:
            if st.button("🔄 Refresh Data"):
                st.rerun()
        
        # Show create user form if requested
        if st.session_state.get('show_create_user', False):
            self.show_create_user_form()
        
        # User filters
        st.markdown("#### 🔍 Filter Users")
        
        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
        
        with filter_col1:
            role_filter = st.selectbox(
                "Filter by Role",
                options=["All"] + [role.value for role in UserRole],
                key="user_role_filter"
            )
        
        with filter_col2:
            status_filter = st.selectbox(
                "Filter by Status",
                options=["All", "Active", "Inactive", "Pending Verification"],
                key="user_status_filter"
            )
        
        with filter_col3:
            search_term = st.text_input(
                "Search Users",
                placeholder="Name, email, or username...",
                key="user_search"
            )
        
        with filter_col4:
            sort_by = st.selectbox(
                "Sort by",
                options=["Created Date", "Last Login", "Name", "Email"],
                key="user_sort"
            )
        
        # Apply filters
        filtered_users = self.filter_users(all_users, role_filter, status_filter, search_term, sort_by)
        
        # Display user list
        st.markdown(f"#### 📋 User List ({len(filtered_users)} users)")
        
        if filtered_users:
            # Pagination
            users_per_page = 10
            total_pages = (len(filtered_users) + users_per_page - 1) // users_per_page
            
            if total_pages > 1:
                page = st.selectbox(
                    "Page",
                    options=list(range(1, total_pages + 1)),
                    key="user_page"
                )
                start_idx = (page - 1) * users_per_page
                end_idx = start_idx + users_per_page
                page_users = filtered_users[start_idx:end_idx]
            else:
                page_users = filtered_users
            
            # Display users
            for user in page_users:
                self.display_user_card(user)
        else:
            st.info("No users match your filter criteria.")
    
    def show_create_user_form(self):
        """Show create user form"""
        
        st.markdown("#### ➕ Create New User")
        
        with st.form("admin_create_user", clear_on_submit=True):
            col1, col2 = st.columns(2)
            
            with col1:
                username = st.text_input("Username *", help="Unique username for the user")
                email = st.text_input("Email *", help="User's email address")
                full_name = st.text_input("Full Name *", help="User's full name")
                password = st.text_input("Temporary Password *", type="password", 
                                       help="Temporary password (user should change on first login)")
            
            with col2:
                user_role = st.selectbox("Role *", options=[role.value for role in UserRole])
                organization = st.text_input("Organization", help="User's organization")
                department = st.text_input("Department", help="User's department")
                auto_verify = st.checkbox("Auto-verify email", value=True, 
                                        help="Automatically verify the user's email")
            
            col_submit, col_cancel = st.columns(2)
            
            with col_submit:
                submitted = st.form_submit_button("✅ Create User", type="primary")
            
            with col_cancel:
                cancelled = st.form_submit_button("❌ Cancel")
            
            if cancelled:
                st.session_state.show_create_user = False
                st.rerun()
            
            if submitted:
                # Validation
                if not all([username, email, full_name, password]):
                    st.error("Please fill in all required fields.")
                else:
                    user_data = {
                        'username': username,
                        'email': email,
                        'password': password,
                        'full_name': full_name,
                        'user_role': user_role,
                        'organization_name': organization,
                        'department': department,
                        'created_by': st.session_state.auth_user['username']
                    }
                    
                    try:
                        user_id = self.db.create_user(user_data)
                        
                        if auto_verify:
                            # Auto-verify email and activate account
                            with st.spinner("Activating account..."):
                                import sqlite3
                                with sqlite3.connect(self.db.db_path) as conn:
                                    cursor = conn.cursor()
                                    cursor.execute('''
                                        UPDATE users 
                                        SET is_active = 1, email_verified = 1 
                                        WHERE user_id = ?
                                    ''', (user_id,))
                                    conn.commit()
                        
                        st.success(f"✅ User created successfully! User ID: {user_id}")
                        st.session_state.show_create_user = False
                        
                        # Log admin action
                        self.db.log_activity(
                            user_id=st.session_state.auth_user['user_id'],
                            action_type="ADMIN_USER_CREATED",
                            object_type="USER",
                            object_id=str(user_id),
                            status="SUCCESS",
                            payload=json.dumps({
                                'created_user_id': user_id,
                                'username': username,
                                'role': user_role
                            })
                        )
                        
                        time.sleep(2)
                        st.rerun()
                        
                    except Exception as e:
                        if 'UNIQUE constraint failed' in str(e):
                            if 'username' in str(e):
                                st.error("❌ Username already exists.")
                            else:
                                st.error("❌ Email already exists.")
                        else:
                            st.error(f"❌ Error creating user: {str(e)}")
    
    def filter_users(self, users: List[Dict], role_filter: str, status_filter: str, 
                    search_term: str, sort_by: str) -> List[Dict]:
        """Filter and sort users based on criteria"""
        
        filtered = users.copy()
        
        # Role filter
        if role_filter != "All":
            filtered = [u for u in filtered if u['user_role'] == role_filter]
        
        # Status filter
        if status_filter == "Active":
            filtered = [u for u in filtered if u['is_active'] and u['email_verified']]
        elif status_filter == "Inactive":
            filtered = [u for u in filtered if not u['is_active']]
        elif status_filter == "Pending Verification":
            filtered = [u for u in filtered if not u['email_verified']]
        
        # Search filter
        if search_term:
            search_lower = search_term.lower()
            filtered = [
                u for u in filtered 
                if search_lower in u['full_name'].lower() or 
                   search_lower in u['email'].lower() or 
                   search_lower in u['username'].lower() or
                   search_lower in (u['organization_name'] or '').lower()
            ]
        
        # Sort
        if sort_by == "Created Date":
            filtered.sort(key=lambda x: x['created_at'] or '', reverse=True)
        elif sort_by == "Last Login":
            filtered.sort(key=lambda x: x['last_login'] or '', reverse=True)
        elif sort_by == "Name":
            filtered.sort(key=lambda x: x['full_name'].lower())
        elif sort_by == "Email":
            filtered.sort(key=lambda x: x['email'].lower())
        
        return filtered
    
    def display_user_card(self, user: Dict[str, Any]):
        """Display individual user card"""
        
        # Determine status and styling
        if user['is_active'] and user['email_verified']:
            status = "Active"
            card_class = "active-user"
            status_icon = "🟢"
        elif not user['email_verified']:
            status = "Pending Verification"
            card_class = "pending-user"
            status_icon = "🟡"
        else:
            status = "Inactive"
            card_class = "inactive-user"
            status_icon = "🔴"
        
        st.markdown(f"""
        <div class="user-card {card_class}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h4>{status_icon} {user['full_name']} ({user['username']})</h4>
                    <p><strong>Email:</strong> {user['email']}</p>
                    <p><strong>Role:</strong> {user['user_role']} | <strong>Status:</strong> {status}</p>
                    <p><strong>Organization:</strong> {user['organization_name'] or 'N/A'} | 
                       <strong>Department:</strong> {user['department'] or 'N/A'}</p>
                    <p><small>Created: {user['created_at'][:19] if user['created_at'] else 'N/A'} | 
                             Last Login: {user['last_login'][:19] if user['last_login'] else 'Never'}</small></p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Action buttons
        action_col1, action_col2, action_col3, action_col4, action_col5 = st.columns(5)
        
        with action_col1:
            if st.button("✏️ Edit", key=f"edit_user_{user['user_id']}"):
                st.session_state[f"edit_user_{user['user_id']}"] = True
        
        with action_col2:
            if user['is_active']:
                if st.button("🚫 Deactivate", key=f"deactivate_user_{user['user_id']}"):
                    self.toggle_user_status(user['user_id'], False)
            else:
                if st.button("✅ Activate", key=f"activate_user_{user['user_id']}"):
                    self.toggle_user_status(user['user_id'], True)
        
        with action_col3:
            if st.button("🔑 Reset Password", key=f"reset_pwd_{user['user_id']}"):
                self.reset_user_password(user['user_id'])
        
        with action_col4:
            if st.button("📊 View Activity", key=f"view_activity_{user['user_id']}"):
                self.show_user_activity(user['user_id'])
        
        with action_col5:
            if st.button("🗑️ Delete", key=f"delete_user_{user['user_id']}", 
                        help="Permanently delete user (use with caution)"):
                if st.session_state.get(f"confirm_delete_{user['user_id']}", False):
                    self.delete_user(user['user_id'])
                else:
                    st.session_state[f"confirm_delete_{user['user_id']}"] = True
                    st.warning("Click again to confirm deletion")
        
        # Show edit form if requested
        if st.session_state.get(f"edit_user_{user['user_id']}", False):
            self.show_edit_user_form(user)
        
        st.markdown("---")
    
    def show_edit_user_form(self, user: Dict[str, Any]):
        """Show edit user form"""
        
        st.markdown(f"#### ✏️ Edit User: {user['full_name']}")
        
        with st.form(f"edit_user_form_{user['user_id']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                new_full_name = st.text_input("Full Name", value=user['full_name'])
                new_email = st.text_input("Email", value=user['email'])
                new_organization = st.text_input("Organization", value=user['organization_name'] or '')
            
            with col2:
                new_role = st.selectbox(
                    "Role", 
                    options=[role.value for role in UserRole],
                    index=[role.value for role in UserRole].index(user['user_role'])
                )
                new_department = st.text_input("Department", value=user['department'] or '')
                email_verified = st.checkbox("Email Verified", value=user['email_verified'])
            
            col_save, col_cancel = st.columns(2)
            
            with col_save:
                if st.form_submit_button("💾 Save Changes", type="primary"):
                    try:
                        # Update user in database
                        with sqlite3.connect(self.db.db_path) as conn:
                            cursor = conn.cursor()
                            cursor.execute('''
                                UPDATE users 
                                SET full_name = ?, email = ?, organization_name = ?, 
                                    user_role = ?, department = ?, email_verified = ?,
                                    updated_at = CURRENT_TIMESTAMP
                                WHERE user_id = ?
                            ''', (
                                new_full_name, new_email, new_organization,
                                new_role, new_department, email_verified, user['user_id']
                            ))
                            conn.commit()
                        
                        # Log admin action
                        self.db.log_activity(
                            user_id=st.session_state.auth_user['user_id'],
                            action_type="ADMIN_USER_UPDATED",
                            object_type="USER",
                            object_id=str(user['user_id']),
                            status="SUCCESS",
                            payload=json.dumps({
                                'updated_user_id': user['user_id'],
                                'changes': {
                                    'full_name': new_full_name,
                                    'role': new_role,
                                    'email_verified': email_verified
                                }
                            })
                        )
                        
                        st.success("✅ User updated successfully!")
                        st.session_state[f"edit_user_{user['user_id']}"] = False
                        time.sleep(1)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error updating user: {str(e)}")
            
            with col_cancel:
                if st.form_submit_button("❌ Cancel"):
                    st.session_state[f"edit_user_{user['user_id']}"] = False
                    st.rerun()
    
    def toggle_user_status(self, user_id: int, is_active: bool):
        """Toggle user active status"""
        
        try:
            import sqlite3
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE users 
                    SET is_active = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                ''', (is_active, user_id))
                conn.commit()
            
            action = "ACTIVATED" if is_active else "DEACTIVATED"
            status_text = "activated" if is_active else "deactivated"
            
            # Log admin action
            self.db.log_activity(
                user_id=st.session_state.auth_user['user_id'],
                action_type=f"ADMIN_USER_{action}",
                object_type="USER",
                object_id=str(user_id),
                status="SUCCESS"
            )
            
            st.success(f"✅ User {status_text} successfully!")
            time.sleep(1)
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error updating user status: {str(e)}")
    
    def reset_user_password(self, user_id: int):
        """Reset user password (admin action)"""
        
        # Generate temporary password
        import secrets
        import string
        temp_password = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(12))
        
        try:
            import bcrypt
            import sqlite3
            
            # Hash the temporary password
            salt = bcrypt.gensalt()
            password_hash = bcrypt.hashpw(temp_password.encode('utf-8'), salt)
            
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE users 
                    SET password_hash = ?, salt = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                ''', (password_hash.decode('utf-8'), salt.decode('utf-8'), user_id))
                conn.commit()
            
            # Log admin action
            self.db.log_activity(
                user_id=st.session_state.auth_user['user_id'],
                action_type="ADMIN_PASSWORD_RESET",
                object_type="USER",
                object_id=str(user_id),
                status="SUCCESS"
            )
            
            st.success(f"✅ Password reset successfully!")
            st.info(f"🔑 Temporary password: `{temp_password}`")
            st.warning("⚠️ Please share this password securely with the user and ask them to change it on first login.")
            
        except Exception as e:
            st.error(f"❌ Error resetting password: {str(e)}")
    
    def delete_user(self, user_id: int):
        """Delete user (admin action)"""
        
        try:
            import sqlite3
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                # Soft delete - mark as inactive and scramble sensitive data
                cursor.execute('''
                    UPDATE users 
                    SET is_active = 0, 
                        email = 'deleted_' || user_id || '@deleted.local',
                        username = 'deleted_' || user_id,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                ''', (user_id,))
                conn.commit()
            
            # Log admin action
            self.db.log_activity(
                user_id=st.session_state.auth_user['user_id'],
                action_type="ADMIN_USER_DELETED",
                object_type="USER",
                object_id=str(user_id),
                status="SUCCESS"
            )
            
            st.success("✅ User deleted successfully!")
            
            # Clear confirmation state
            if f"confirm_delete_{user_id}" in st.session_state:
                del st.session_state[f"confirm_delete_{user_id}"]
            
            time.sleep(1)
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error deleting user: {str(e)}")
    
    def export_user_list(self, users: List[Dict]):
        """Export user list to CSV"""
        
        if not users:
            st.error("No users to export.")
            return
        
        # Create DataFrame
        df = pd.DataFrame(users)
        
        # Remove sensitive data
        if 'password_hash' in df.columns:
            df = df.drop(['password_hash', 'salt'], axis=1)
        
        # Convert to CSV
        csv_data = df.to_csv(index=False)
        
        # Download button
        st.download_button(
            "📥 Download User List (CSV)",
            csv_data,
            f"user_list_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            help="Download filtered user list as CSV"
        )
    
    def show_session_management(self):
        """Show active session management"""
        
        if not self.auth.has_permission("can_manage_users"):
            st.error("❌ Access denied. Session management privileges required.")
            return
        
        st.markdown("### 🔑 Active Session Management")
        
        # Get active sessions
        active_sessions = self.db.get_active_sessions()
        
        if not active_sessions:
            st.info("No active sessions found.")
            return
        
        st.success(f"Found {len(active_sessions)} active sessions")
        
        # Session management actions
        if st.button("🚫 Terminate All Sessions", type="secondary"):
            if st.session_state.get('confirm_terminate_all', False):
                self.terminate_all_sessions()
                st.session_state.confirm_terminate_all = False
            else:
                st.session_state.confirm_terminate_all = True
                st.warning("Click again to confirm terminating ALL sessions")
        
        # Display sessions
        for session in active_sessions:
            st.markdown(f"""
            <div class="session-card">
                <h4>👤 {session['full_name']} ({session['username']})</h4>
                <p><strong>Email:</strong> {session['email']}</p>
                <p><strong>IP Address:</strong> {session['ip_address']}</p>
                <p><strong>Session Started:</strong> {session['created_at']}</p>
                <p><strong>Last Activity:</strong> {session['last_activity']}</p>
                <p><strong>Session ID:</strong> <code>{session['session_id'][:16]}...</code></p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(
                f"🚫 Terminate Session", 
                key=f"terminate_{session['session_id'][:8]}",
                help="Terminate this specific session"
            ):
                self.terminate_session(session['session_id'])
            
            st.markdown("---")
    
    def terminate_session(self, session_id: str):
        """Terminate specific session"""
        
        try:
            self.db.invalidate_session(session_id)
            
            # Log admin action
            self.db.log_activity(
                user_id=st.session_state.auth_user['user_id'],
                action_type="ADMIN_SESSION_TERMINATED",
                object_type="SESSION",
                object_id=session_id,
                status="SUCCESS"
            )
            
            st.success("✅ Session terminated successfully!")
            time.sleep(1)
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error terminating session: {str(e)}")
    
    def terminate_all_sessions(self):
        """Terminate all active sessions except current"""
        
        try:
            import sqlite3
            current_session = st.session_state.auth_session_id
            
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE user_sessions 
                    SET is_active = 0 
                    WHERE is_active = 1 AND session_id != ?
                ''', (current_session,))
                terminated_count = cursor.rowcount
                conn.commit()
            
            # Log admin action
            self.db.log_activity(
                user_id=st.session_state.auth_user['user_id'],
                action_type="ADMIN_ALL_SESSIONS_TERMINATED",
                object_type="SESSION",
                status="SUCCESS",
                payload=json.dumps({'terminated_count': terminated_count})
            )
            
            st.success(f"✅ Terminated {terminated_count} sessions successfully!")
            time.sleep(1)
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error terminating sessions: {str(e)}")
    
    def show_analytics(self):
        """Show analytics dashboard"""
        
        if not self.auth.has_permission("can_view_user_analytics"):
            st.error("❌ Access denied. Analytics viewing privileges required.")
            return
        
        st.markdown("### 📈 User Analytics & Insights")
        
        # Get statistics for different time periods
        stats_7d = self.db.get_login_statistics(7)
        stats_30d = self.db.get_login_statistics(30)
        
        # Time period selector
        time_period = st.selectbox(
            "Select Time Period",
            options=["Last 7 Days", "Last 30 Days"],
            key="analytics_period"
        )
        
        current_stats = stats_7d if time_period == "Last 7 Days" else stats_30d
        
        # Analytics charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Login success vs failure
            login_data = {
                'Type': ['Successful Logins', 'Failed Logins'],
                'Count': [current_stats['successful_logins'], current_stats['failed_logins']]
            }
            
            fig_logins = px.bar(
                login_data,
                x='Type',
                y='Count',
                title=f"Login Attempts ({time_period})",
                color='Type',
                color_discrete_map={
                    'Successful Logins': '#28a745',
                    'Failed Logins': '#dc3545'
                }
            )
            st.plotly_chart(fig_logins, use_container_width=True)
        
        with chart_col2:
            # Daily login trend
            if current_stats['daily_logins']:
                daily_df = pd.DataFrame([
                    {'Date': date, 'Logins': count}
                    for date, count in current_stats['daily_logins'].items()
                ])
                daily_df['Date'] = pd.to_datetime(daily_df['Date'])
                
                fig_daily = px.line(
                    daily_df,
                    x='Date',
                    y='Logins',
                    title=f"Daily Login Trend ({time_period})",
                    markers=True
                )
                st.plotly_chart(fig_daily, use_container_width=True)
            else:
                st.info("No daily login data available for the selected period.")
        
        # Additional analytics
        st.markdown("#### 🔍 Detailed Analytics")
        
        detail_col1, detail_col2 = st.columns(2)
        
        with detail_col1:
            # User role distribution
            if current_stats.get('users_by_role'):
                role_df = pd.DataFrame([
                    {'Role': role, 'Count': count}
                    for role, count in current_stats['users_by_role'].items()
                ])
                
                fig_roles = px.pie(
                    role_df,
                    values='Count',
                    names='Role',
                    title="User Distribution by Role"
                )
                st.plotly_chart(fig_roles, use_container_width=True)
        
        with detail_col2:
            # Key metrics table
            metrics_data = {
                'Metric': [
                    'Total Users',
                    'Active Users', 
                    'Email Verified Users',
                    'Login Success Rate',
                    'Average Logins per Day'
                ],
                'Value': [
                    current_stats['total_users'],
                    current_stats['active_users'],
                    f"{len([u for u in self.db.get_all_users() if u['email_verified']])}",
                    f"{(current_stats['successful_logins'] / max(current_stats['successful_logins'] + current_stats['failed_logins'], 1) * 100):.1f}%",
                    f"{current_stats['successful_logins'] / max(len(current_stats['daily_logins']), 1):.1f}"
                ]
            }
            
            st.markdown("**Key Metrics:**")
            st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)
    
    def show_activity_logs(self):
        """Show user activity logs"""
        
        if not self.auth.has_permission("can_view_audit_logs"):
            st.error("❌ Access denied. Audit log viewing privileges required.")
            return
        
        st.markdown("### 🔍 User Activity Logs")
        
        # Filters for activity logs
        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
        
        with filter_col1:
            days_back = st.selectbox(
                "Time Period",
                options=[1, 7, 30, 90],
                index=1,
                format_func=lambda x: f"Last {x} day{'s' if x > 1 else ''}",
                key="logs_days"
            )
        
        with filter_col2:
            action_filter = st.selectbox(
                "Action Type",
                options=["All", "LOGIN_SUCCESS", "LOGIN_FAILED", "LOGOUT", 
                        "USER_CREATED", "PASSWORD_RESET", "ADMIN_ACTION"],
                key="logs_action"
            )
        
        with filter_col3:
            user_filter = st.text_input(
                "Filter by User",
                placeholder="Username or email...",
                key="logs_user"
            )
        
        with filter_col4:
            status_filter = st.selectbox(
                "Status",
                options=["All", "SUCCESS", "FAILED", "BLOCKED"],
                key="logs_status"
            )
        
        # Get activity logs
        logs = self.get_filtered_activity_logs(days_back, action_filter, user_filter, status_filter)
        
        if not logs:
            st.info("No activity logs found for the selected criteria.")
            return
        
        st.success(f"Found {len(logs)} activity log entries")
        
        # Export logs button
        if st.button("📥 Export Logs"):
            self.export_activity_logs(logs)
        
        # Display logs
        logs_per_page = 20
        total_pages = (len(logs) + logs_per_page - 1) // logs_per_page
        
        if total_pages > 1:
            page = st.selectbox(
                "Page",
                options=list(range(1, total_pages + 1)),
                key="logs_page"
            )
            start_idx = (page - 1) * logs_per_page
            end_idx = start_idx + logs_per_page
            page_logs = logs[start_idx:end_idx]
        else:
            page_logs = logs
        
        # Display log entries
        for log in page_logs:
            status_color = {
                'SUCCESS': '🟢',
                'FAILED': '🔴', 
                'BLOCKED': '🚫',
                'PENDING': '🟡'
            }.get(log.get('status', ''), '⚪')
            
            action_icon = {
                'LOGIN_SUCCESS': '🔑',
                'LOGIN_FAILED': '❌',
                'LOGOUT': '🚪',
                'USER_CREATED': '👤',
                'PASSWORD_RESET': '🔄',
                'ADMIN_ACTION': '👑'
            }.get(log.get('action_type', ''), '📝')
            
            # Parse payload if it exists
            payload_info = ""
            if log.get('payload'):
                try:
                    payload_data = json.loads(log['payload'])
                    payload_info = f" | Details: {', '.join([f'{k}: {v}' for k, v in payload_data.items() if k not in ['password', 'token']])}"
                except:
                    pass
            
            st.markdown(f"""
            <div style="border: 1px solid #e0e0e0; padding: 1rem; margin: 0.5rem 0; border-radius: 6px; background: #f9f9f9;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h5>{action_icon} {log.get('action_type', 'Unknown')} {status_color} {log.get('status', 'Unknown')}</h5>
                        <p><strong>User:</strong> {log.get('username', 'N/A')} ({log.get('email', 'N/A')})</p>
                        <p><strong>Time:</strong> {log.get('timestamp_utc', 'N/A')} | 
                           <strong>IP:</strong> {log.get('ip_address', 'N/A')} | 
                           <strong>Event ID:</strong> {log.get('event_id', 'N/A')}</p>
                        {f"<p><small>{payload_info}</small></p>" if payload_info else ""}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def get_filtered_activity_logs(self, days_back: int, action_filter: str, 
                                  user_filter: str, status_filter: str) -> List[Dict]:
        """Get filtered activity logs"""
        
        try:
            import sqlite3
            with sqlite3.connect(self.db.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = '''
                    SELECT * FROM user_activity_logs 
                    WHERE timestamp_utc > datetime('now', '-{} days')
                '''.format(days_back)
                
                params = []
                
                if action_filter != "All":
                    query += " AND action_type = ?"
                    params.append(action_filter)
                
                if user_filter:
                    query += " AND (username LIKE ? OR email LIKE ?)"
                    params.extend([f"%{user_filter}%", f"%{user_filter}%"])
                
                if status_filter != "All":
                    query += " AND status = ?"
                    params.append(status_filter)
                
                query += " ORDER BY timestamp_utc DESC LIMIT 1000"
                
                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            st.error(f"Error retrieving activity logs: {str(e)}")
            return []
    
    def export_activity_logs(self, logs: List[Dict]):
        """Export activity logs to CSV"""
        
        if not logs:
            st.error("No logs to export.")
            return
        
        # Create DataFrame
        df = pd.DataFrame(logs)
        
        # Convert to CSV
        csv_data = df.to_csv(index=False)
        
        # Download button
        st.download_button(
            "📥 Download Activity Logs (CSV)",
            csv_data,
            f"activity_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            help="Download filtered activity logs as CSV"
        )
    
    def show_user_activity(self, user_id: int):
        """Show specific user's activity"""
        
        st.markdown(f"#### 📊 User Activity - ID: {user_id}")
        
        try:
            import sqlite3
            with sqlite3.connect(self.db.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM user_activity_logs 
                    WHERE user_id = ? 
                    ORDER BY timestamp_utc DESC 
                    LIMIT 50
                ''', (user_id,))
                
                user_logs = [dict(row) for row in cursor.fetchall()]
                
                if user_logs:
                    for log in user_logs[:10]:  # Show last 10 activities
                        st.markdown(f"""
                        **{log['action_type']}** ({log['status']}) - {log['timestamp_utc']}  
                        IP: {log['ip_address']} | Event: {log['event_id']}
                        """)
                else:
                    st.info("No activity found for this user.")
                    
        except Exception as e:
            st.error(f"Error retrieving user activity: {str(e)}")

# Additional utility functions for admin dashboard

def create_admin_user():
    """Create initial admin user for the system"""
    
    db = AuthenticationDB()
    
    # Check if any admin users exist
    try:
        import sqlite3
        with sqlite3.connect(db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM users WHERE user_role = 'SuperAdmin'")
            admin_count = cursor.fetchone()[0]
            
            if admin_count == 0:
                st.info("🚀 No admin users found. Creating initial admin account...")
                
                # Create default admin
                admin_data = {
                    'username': 'admin',
                    'email': 'admin@datalensai.com',
                    'password': 'DataLens2024!',  # Should be changed immediately
                    'full_name': 'System Administrator',
                    'user_role': 'SuperAdmin',
                    'organization_name': 'Data Lens AI',
                    'department': 'IT Administration',
                    'created_by': 'system'
                }
                
                user_id = db.create_user(admin_data)
                
                # Auto-verify and activate
                with sqlite3.connect(db.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        UPDATE users 
                        SET is_active = 1, email_verified = 1 
                        WHERE user_id = ?
                    ''', (user_id,))
                    conn.commit()
                
                st.success("✅ Initial admin user created!")
                st.warning("🔐 Default credentials: admin@datalensai.com / DataLens2024!")
                st.error("⚠️ Please change the password immediately after first login!")
                
                return True
                
    except Exception as e:
        st.error(f"Error checking/creating admin user: {str(e)}")
        return False
    
    return False