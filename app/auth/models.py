# File: app/auth/models.py - Fixed version with UserRole enum and missing methods

import sqlite3
import hashlib
import secrets
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from enum import Enum
import bcrypt

class UserRole(Enum):
    """User roles enum - this was missing!"""
    SUPER_ADMIN = "SuperAdmin"
    ADMIN = "Admin"
    DATA_SCIENTIST = "DataScientist"
    BUSINESS_ANALYST = "BusinessAnalyst"
    DEVELOPER = "Developer"
    VIEWER = "Viewer"

class AuthenticationDB:
    """Enhanced authentication database with all required methods"""
    
    def __init__(self, db_path="app/database/auth.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT,
                    full_name TEXT NOT NULL,
                    user_role TEXT DEFAULT 'Viewer',
                    organization_name TEXT,
                    department TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    email_verified BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    created_by TEXT DEFAULT 'system'
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    ip_address TEXT,
                    user_agent TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            
            # Activity logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_activity_logs (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    user_id INTEGER,
                    username TEXT,
                    email TEXT,
                    timestamp_utc TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    action_type TEXT NOT NULL,
                    object_type TEXT,
                    object_id TEXT,
                    status TEXT NOT NULL,
                    ip_address TEXT,
                    user_agent TEXT,
                    payload TEXT,
                    session_id TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            
            # Role permissions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS role_permissions (
                    permission_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role_name TEXT NOT NULL,
                    permission_name TEXT NOT NULL,
                    permission_description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(role_name, permission_name)
                )
            ''')
            
            conn.commit()
            
            # Create default admin if none exists
            cursor.execute("SELECT COUNT(*) FROM users WHERE user_role = 'SuperAdmin'")
            if cursor.fetchone()[0] == 0:
                self._create_default_admin()
                
            # Initialize default permissions
            self._init_default_permissions()
    
    def _create_default_admin(self):
        """Create default admin user"""
        password = "DataLens2024!"
        salt = bcrypt.gensalt()
        password_hash = bcrypt.hashpw(password.encode('utf-8'), salt)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, salt, full_name, user_role, is_active, email_verified)
                VALUES (?, ?, ?, ?, ?, ?, 1, 1)
            ''', ('admin', 'admin@datalensai.com', password_hash.decode('utf-8'), 
                   salt.decode('utf-8'), 'System Administrator', 'SuperAdmin'))
            conn.commit()
            print("[INFO] Created default admin: admin@datalensai.com / DataLens2024!")
    
    def _init_default_permissions(self):
        """Initialize default role-based permissions"""
        default_permissions = {
            "SuperAdmin": [
                "can_manage_users",
                "can_view_user_analytics", 
                "can_access_admin_dashboard",
                "can_manage_system_settings",
                "can_view_audit_logs",
                "can_manage_roles",
                "can_export_data",
                "can_import_data",
                "can_chat_with_data",
                "can_create_rules",
                "can_manage_database_connections",
                "can_profile_data"
            ],
            "Admin": [
                "can_manage_users",
                "can_view_user_analytics",
                "can_access_admin_dashboard",
                "can_view_audit_logs",
                "can_export_data",
                "can_import_data",
                "can_chat_with_data",
                "can_create_rules",
                "can_manage_database_connections",
                "can_profile_data"
            ],
            "DataScientist": [
                "can_export_data",
                "can_import_data",
                "can_chat_with_data",
                "can_create_rules",
                "can_manage_database_connections",
                "can_profile_data",
                "can_view_advanced_analytics"
            ],
            "BusinessAnalyst": [
                "can_export_data",
                "can_import_data",
                "can_chat_with_data",
                "can_create_rules",
                "can_profile_data",
                "can_view_basic_analytics"
            ],
            "Developer": [
                "can_export_data",
                "can_import_data",
                "can_chat_with_data",
                "can_create_rules",
                "can_manage_database_connections",
                "can_profile_data",
                "can_view_system_logs"
            ],
            "Viewer": [
                "can_view_data",
                "can_export_limited_data"
            ]
        }
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for role, permissions in default_permissions.items():
                for permission in permissions:
                    cursor.execute('''
                        INSERT OR IGNORE INTO role_permissions (role_name, permission_name)
                        VALUES (?, ?)
                    ''', (role, permission))
            
            conn.commit()
    
    def create_user(self, user_data: Dict[str, Any]) -> int:
        """Create a new user with secure password hashing"""
        
        # Generate salt and hash password
        salt = bcrypt.gensalt()
        password_hash = bcrypt.hashpw(user_data['password'].encode('utf-8'), salt)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO users (
                    username, email, password_hash, salt, full_name,
                    organization_name, user_role, department, created_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_data['username'],
                user_data['email'],
                password_hash.decode('utf-8'),
                salt.decode('utf-8'),
                user_data['full_name'],
                user_data.get('organization_name', ''),
                user_data.get('user_role', 'Viewer'),
                user_data.get('department', ''),
                user_data.get('created_by', 'system')
            ))
            
            user_id = cursor.lastrowid
            conn.commit()
            
            # Log user creation
            self.log_activity(
                user_id=None,
                action_type="USER_CREATED",
                object_type="USER",
                object_id=str(user_id),
                status="SUCCESS"
            )
            
            return user_id
    
    def verify_password(self, email, password):
        """Verify user password and return user data if valid"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE email = ? AND is_active = 1", (email,))
            user = cursor.fetchone()
            
            if user and bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
                # Update last login
                cursor.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE user_id = ?", (user['user_id'],))
                conn.commit()
                return dict(user)
        return None
    
    def create_session(self, user_id, ip_address="127.0.0.1", user_agent="Streamlit App"):
        """Create a new user session"""
        session_id = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(hours=24)  # 24-hour session
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO user_sessions (session_id, user_id, ip_address, user_agent, expires_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (session_id, user_id, ip_address, user_agent, expires_at))
            conn.commit()
        return session_id
    
    def validate_session(self, session_id):
        """Validate and return session data if valid"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT s.*, u.* FROM user_sessions s
                JOIN users u ON s.user_id = u.user_id
                WHERE s.session_id = ? AND s.is_active = 1 
                AND s.expires_at > CURRENT_TIMESTAMP
                AND u.is_active = 1
            ''', (session_id,))
            session = cursor.fetchone()
            
            if session:
                # Update last activity
                cursor.execute('''
                    UPDATE user_sessions 
                    SET last_activity = CURRENT_TIMESTAMP 
                    WHERE session_id = ?
                ''', (session_id,))
                conn.commit()
                return dict(session)
        return None
    
    def invalidate_session(self, session_id):
        """Invalidate a user session"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE user_sessions 
                SET is_active = 0 
                WHERE session_id = ?
            ''', (session_id,))
            conn.commit()
    
    def get_all_users(self) -> List[Dict[str, Any]]:
        """Get all users for admin interface - THIS WAS MISSING!"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT user_id, username, email, full_name, organization_name,
                       user_role, department, is_active, email_verified,
                       created_at, last_login, created_by
                FROM users
                ORDER BY created_at DESC
            ''')
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get all active sessions for admin interface"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT s.session_id, s.ip_address, s.created_at, s.last_activity,
                       u.username, u.email, u.full_name
                FROM user_sessions s
                JOIN users u ON s.user_id = u.user_id
                WHERE s.is_active = 1 AND s.expires_at > CURRENT_TIMESTAMP
                ORDER BY s.last_activity DESC
            ''')
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_login_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get login statistics for dashboard"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Total users
            cursor.execute('SELECT COUNT(*) FROM users')
            stats['total_users'] = cursor.fetchone()[0]
            
            # Active users
            cursor.execute('SELECT COUNT(*) FROM users WHERE is_active = 1')
            stats['active_users'] = cursor.fetchone()[0]
            
            # Users by role
            cursor.execute('''
                SELECT user_role, COUNT(*) FROM users 
                WHERE is_active = 1 GROUP BY user_role
            ''')
            stats['users_by_role'] = dict(cursor.fetchall())
            
            # Login attempts in last N days
            cursor.execute('''
                SELECT COUNT(*) FROM user_activity_logs 
                WHERE action_type = 'LOGIN_SUCCESS' 
                AND timestamp_utc > datetime('now', '-{} days')
            '''.format(days))
            successful_logins = cursor.fetchone()[0] or 0
            stats['successful_logins'] = successful_logins
            
            cursor.execute('''
                SELECT COUNT(*) FROM user_activity_logs 
                WHERE action_type = 'LOGIN_FAILED' 
                AND timestamp_utc > datetime('now', '-{} days')
            '''.format(days))
            failed_logins = cursor.fetchone()[0] or 0
            stats['failed_logins'] = failed_logins
            
            # Calculate success rate
            total_logins = successful_logins + failed_logins
            stats['success_rate'] = (successful_logins / max(total_logins, 1)) * 100
            stats['total_executions'] = total_logins
            
            # Daily login trends
            cursor.execute('''
                SELECT DATE(timestamp_utc) as date, COUNT(*) as count
                FROM user_activity_logs 
                WHERE action_type = 'LOGIN_SUCCESS' 
                AND timestamp_utc > datetime('now', '-{} days')
                GROUP BY DATE(timestamp_utc)
                ORDER BY date
            '''.format(days))
            stats['daily_executions'] = dict(cursor.fetchall())
            
            return stats
    
    def get_user_permissions(self, user_role: str) -> List[str]:
        """Get permissions for a user role"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT permission_name FROM role_permissions 
                WHERE role_name = ?
            ''', (user_role,))
            return [row[0] for row in cursor.fetchall()]
    
    def log_activity(self, user_id: Optional[int], action_type: str, 
                    object_type: str = None, object_id: str = None,
                    status: str = "SUCCESS", ip_address: str = None,
                    user_agent: str = None, payload: str = None,
                    session_id: str = None):
        """Log user activity"""
        
        event_id = secrets.token_hex(16)
        
        # Get user info if user_id provided
        username = None
        email = None
        if user_id:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT username, email FROM users WHERE user_id = ?', (user_id,))
                result = cursor.fetchone()
                if result:
                    username, email = result
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO user_activity_logs (
                    event_id, user_id, username, email, action_type,
                    object_type, object_id, status, ip_address,
                    user_agent, payload, session_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event_id, user_id, username, email, action_type,
                object_type, object_id, status, ip_address,
                user_agent, payload, session_id
            ))
            conn.commit()
    
    def update_user(self, user_id: int, updates: Dict[str, Any]) -> bool:
        """Update an existing user"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build UPDATE query dynamically
                set_clauses = []
                values = []
                
                allowed_fields = ['full_name', 'email', 'organization_name', 
                                'user_role', 'department', 'is_active', 'email_verified']
                
                for key, value in updates.items():
                    if key in allowed_fields:
                        set_clauses.append(f"{key} = ?")
                        values.append(value)
                
                if set_clauses:
                    set_clauses.append("updated_at = CURRENT_TIMESTAMP")
                    values.append(user_id)
                    
                    query = f"UPDATE users SET {', '.join(set_clauses)} WHERE user_id = ?"
                    cursor.execute(query, values)
                    conn.commit()
                    return cursor.rowcount > 0
                return False
        except Exception:
            return False
    
    def delete_user(self, user_id: int) -> bool:
        """Delete a user (soft delete)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE users SET is_active = 0, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?",
                    (user_id,)
                )
                conn.commit()
                return cursor.rowcount > 0
        except Exception:
            return False