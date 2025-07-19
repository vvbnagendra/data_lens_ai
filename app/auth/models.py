# app/auth/models.py - Basic authentication models
import sqlite3
import hashlib
import secrets
import os
from datetime import datetime
import bcrypt

class AuthenticationDB:
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
                    full_name TEXT NOT NULL,
                    user_role TEXT DEFAULT 'Viewer',
                    is_active BOOLEAN DEFAULT 1,
                    email_verified BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            conn.commit()
            
            # Create default admin if none exists
            cursor.execute("SELECT COUNT(*) FROM users WHERE user_role = 'SuperAdmin'")
            if cursor.fetchone()[0] == 0:
                self._create_default_admin()
    
    def _create_default_admin(self):
        password = "DataLens2024!"
        salt = bcrypt.gensalt()
        password_hash = bcrypt.hashpw(password.encode('utf-8'), salt)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, full_name, user_role)
                VALUES (?, ?, ?, ?, ?)
            ''', ('admin', 'admin@datalensai.com', password_hash.decode('utf-8'), 
                   'System Administrator', 'SuperAdmin'))
            conn.commit()
            print("[OK] Created default admin: admin@datalensai.com / DataLens2024!")
    
    def verify_password(self, email, password):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE email = ? AND is_active = 1", (email,))
            user = cursor.fetchone()
            
            if user and bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
                return dict(user)
        return None
    
    def create_session(self, user_id):
        session_id = secrets.token_urlsafe(32)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO user_sessions (session_id, user_id)
                VALUES (?, ?)
            ''', (session_id, user_id))
            conn.commit()
        return session_id
    
    def validate_session(self, session_id):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT s.*, u.* FROM user_sessions s
                JOIN users u ON s.user_id = u.user_id
                WHERE s.session_id = ? AND s.is_active = 1 AND u.is_active = 1
            ''', (session_id,))
            return cursor.fetchone()
