# app/auth/__init__.py
# Authentication system package initialization

# app/auth/models.py
import sqlite3
import hashlib
import secrets
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from enum import Enum
import bcrypt

class UserRole(Enum):
    SUPER_ADMIN = "SuperAdmin"
    ADMIN = "Admin"
    DATA_SCIENTIST = "DataScientist"
    BUSINESS_ANALYST = "BusinessAnalyst"
    DEVELOPER = "Developer"
    VIEWER = "Viewer"

class AuthenticationDB:
    """
    Enterprise-grade authentication database with comprehensive security features
    """
    
    def __init__(self, db_path: str = "app/database/auth.db"):
        self.db_path = db_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize all authentication-related tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    full_name TEXT NOT NULL,
                    organization_name TEXT,
                    user_role TEXT NOT NULL DEFAULT 'Viewer',
                    department TEXT,
                    is_active BOOLEAN DEFAULT 0,
                    email_verified BOOLEAN DEFAULT 0,
                    mfa_enabled BOOLEAN DEFAULT 0,
                    mfa_secret TEXT,
                    failed_login_attempts INTEGER DEFAULT 0,
                    account_locked_until TIMESTAMP,
                    last_login TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_by TEXT DEFAULT 'system'
                )
            ''')
            
            # Email verification tokens
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS email_verification_tokens (
                    token_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    token TEXT UNIQUE NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    used BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            
            # Password reset tokens
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS password_reset_tokens (
                    token_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    token TEXT UNIQUE NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    used BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            
            # User sessions
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id INTEGER,
                    ip_address TEXT,
                    user_agent TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            
            # Activity logging
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
            
            # OAuth providers (for future SSO integration)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS oauth_providers (
                    provider_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider_name TEXT UNIQUE NOT NULL,
                    client_id TEXT NOT NULL,
                    client_secret TEXT NOT NULL,
                    authorization_url TEXT NOT NULL,
                    token_url TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # OAuth user links
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS oauth_user_links (
                    link_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    provider_id INTEGER,
                    provider_user_id TEXT NOT NULL,
                    provider_email TEXT,
                    linked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id),
                    FOREIGN KEY (provider_id) REFERENCES oauth_providers (provider_id),
                    UNIQUE(provider_id, provider_user_id)
                )
            ''')
            
            # Role permissions
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
            
            # Initialize default permissions
            self._init_default_permissions()
    
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
                status="SUCCESS",
                ip_address=user_data.get('ip_address'),
                user_agent=user_data.get('user_agent'),
                payload=json.dumps({
                    'created_user_id': user_id,
                    'username': user_data['username'],
                    'email': user_data['email'],
                    'role': user_data.get('user_role', 'Viewer')
                })
            )
            
            return user_id
    
    def verify_password(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        """Verify user password and return user data if valid"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM users WHERE email = ? AND is_active = 1
            ''', (email,))
            
            user = cursor.fetchone()
            
            if not user:
                return None
            
            # Check if account is locked
            if user['account_locked_until']:
                lock_time = datetime.fromisoformat(user['account_locked_until'])
                if datetime.utcnow() < lock_time:
                    return None
            
            # Verify password
            stored_hash = user['password_hash'].encode('utf-8')
            if bcrypt.checkpw(password.encode('utf-8'), stored_hash):
                # Reset failed login attempts on successful login
                cursor.execute('''
                    UPDATE users 
                    SET failed_login_attempts = 0, last_login = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                ''', (user['user_id'],))
                conn.commit()
                
                return dict(user)
            else:
                # Increment failed login attempts
                new_attempts = user['failed_login_attempts'] + 1
                lock_until = None
                
                # Lock account after 5 failed attempts for 30 minutes
                if new_attempts >= 5:
                    lock_until = datetime.utcnow() + timedelta(minutes=30)
                
                cursor.execute('''
                    UPDATE users 
                    SET failed_login_attempts = ?, account_locked_until = ?
                    WHERE user_id = ?
                ''', (new_attempts, lock_until, user['user_id']))
                conn.commit()
                
                return None
    
    def create_session(self, user_id: int, ip_address: str, user_agent: str) -> str:
        """Create a new user session"""
        
        session_id = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(hours=24)  # 24-hour session
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO user_sessions (
                    session_id, user_id, ip_address, user_agent, expires_at
                ) VALUES (?, ?, ?, ?, ?)
            ''', (session_id, user_id, ip_address, user_agent, expires_at))
            
            conn.commit()
        
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Validate and return session data if valid"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT s.*, u.user_id, u.username, u.email, u.full_name, 
                       u.user_role, u.is_active, u.email_verified
                FROM user_sessions s
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
    
    def invalidate_session(self, session_id: str):
        """Invalidate a user session"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE user_sessions 
                SET is_active = 0 
                WHERE session_id = ?
            ''', (session_id,))
            
            conn.commit()
    
    def create_email_verification_token(self, user_id: int) -> str:
        """Create email verification token"""
        
        token = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(hours=24)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO email_verification_tokens (
                    user_id, token, expires_at
                ) VALUES (?, ?, ?)
            ''', (user_id, token, expires_at))
            
            conn.commit()
        
        return token
    
    def verify_email_token(self, token: str) -> bool:
        """Verify email verification token"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT user_id FROM email_verification_tokens 
                WHERE token = ? AND used = 0 AND expires_at > CURRENT_TIMESTAMP
            ''', (token,))
            
            result = cursor.fetchone()
            
            if result:
                user_id = result[0]
                
                # Mark token as used
                cursor.execute('''
                    UPDATE email_verification_tokens 
                    SET used = 1 WHERE token = ?
                ''', (token,))
                
                # Activate user account
                cursor.execute('''
                    UPDATE users 
                    SET is_active = 1, email_verified = 1 
                    WHERE user_id = ?
                ''', (user_id,))
                
                conn.commit()
                return True
        
        return False
    
    def create_password_reset_token(self, email: str) -> Optional[str]:
        """Create password reset token for user"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT user_id FROM users WHERE email = ?', (email,))
            result = cursor.fetchone()
            
            if result:
                user_id = result[0]
                token = secrets.token_urlsafe(32)
                expires_at = datetime.utcnow() + timedelta(hours=1)  # 1-hour expiry
                
                cursor.execute('''
                    INSERT INTO password_reset_tokens (
                        user_id, token, expires_at
                    ) VALUES (?, ?, ?)
                ''', (user_id, token, expires_at))
                
                conn.commit()
                return token
        
        return None
    
    def reset_password_with_token(self, token: str, new_password: str) -> bool:
        """Reset password using reset token"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT user_id FROM password_reset_tokens 
                WHERE token = ? AND used = 0 AND expires_at > CURRENT_TIMESTAMP
            ''', (token,))
            
            result = cursor.fetchone()
            
            if result:
                user_id = result[0]
                
                # Generate new salt and hash
                salt = bcrypt.gensalt()
                password_hash = bcrypt.hashpw(new_password.encode('utf-8'), salt)
                
                # Update password
                cursor.execute('''
                    UPDATE users 
                    SET password_hash = ?, salt = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                ''', (password_hash.decode('utf-8'), salt.decode('utf-8'), user_id))
                
                # Mark token as used
                cursor.execute('''
                    UPDATE password_reset_tokens 
                    SET used = 1 WHERE token = ?
                ''', (token,))
                
                conn.commit()
                return True
        
        return False
    
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
    
    def get_user_permissions(self, user_role: str) -> List[str]:
        """Get permissions for a user role"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT permission_name FROM role_permissions 
                WHERE role_name = ?
            ''', (user_role,))
            
            return [row[0] for row in cursor.fetchall()]
    
    def get_all_users(self) -> List[Dict[str, Any]]:
        """Get all users for admin interface"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT user_id, username, email, full_name, organization_name,
                       user_role, department, is_active, email_verified,
                       created_at, last_login
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
            stats['successful_logins'] = cursor.fetchone()[0]
            
            cursor.execute('''
                SELECT COUNT(*) FROM user_activity_logs 
                WHERE action_type = 'LOGIN_FAILED' 
                AND timestamp_utc > datetime('now', '-{} days')
            '''.format(days))
            stats['failed_logins'] = cursor.fetchone()[0]
            
            # Daily login trends
            cursor.execute('''
                SELECT DATE(timestamp_utc) as date, COUNT(*) as count
                FROM user_activity_logs 
                WHERE action_type = 'LOGIN_SUCCESS' 
                AND timestamp_utc > datetime('now', '-{} days')
                GROUP BY DATE(timestamp_utc)
                ORDER BY date
            '''.format(days))
            stats['daily_logins'] = dict(cursor.fetchall())
            
            return stats