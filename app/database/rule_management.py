import sqlite3
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
import os

class RuleManagementDB:
    """
    SQLite-based rule management system for storing and executing natural language rules
    """
    
    def __init__(self, db_path: str = "app/database/rules.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._initialize_database()
    
    def _initialize_database(self):
        """Create database tables if they don't exist"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Rules table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rules (
                    rule_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_name TEXT NOT NULL UNIQUE,
                    description TEXT,
                    natural_language_text TEXT NOT NULL,
                    created_by TEXT DEFAULT 'system',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE,
                    rule_type TEXT DEFAULT 'validation',
                    priority INTEGER DEFAULT 1,
                    parsed_conditions TEXT,  -- JSON string
                    generated_code TEXT
                )
            """)
            
            # Rule executions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rule_executions (
                    execution_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_id INTEGER REFERENCES rules(rule_id) ON DELETE CASCADE,
                    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    dataset_name TEXT,
                    records_processed INTEGER,
                    records_affected INTEGER,
                    execution_status TEXT DEFAULT 'success',
                    error_message TEXT,
                    execution_time_ms INTEGER,
                    violation_details TEXT  -- JSON string
                )
            """)
            
            # Rule performance metrics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rule_metrics (
                    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_id INTEGER REFERENCES rules(rule_id) ON DELETE CASCADE,
                    metric_date DATE DEFAULT CURRENT_DATE,
                    total_executions INTEGER DEFAULT 0,
                    total_violations INTEGER DEFAULT 0,
                    avg_execution_time_ms REAL DEFAULT 0,
                    success_rate REAL DEFAULT 100.0
                )
            """)
            
            conn.commit()
    
    def create_rule(self, rule_data: Dict[str, Any]) -> int:
        """Create a new rule and return its ID"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO rules (
                    rule_name, description, natural_language_text, rule_type, 
                    priority, is_active, parsed_conditions, generated_code
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rule_data['rule_name'],
                rule_data.get('description', ''),
                rule_data['natural_language_text'],
                rule_data.get('rule_type', 'validation'),
                rule_data.get('priority', 1),
                rule_data.get('is_active', True),
                json.dumps(rule_data.get('parsed_conditions', {})),
                rule_data.get('generated_code', '')
            ))
            
            rule_id = cursor.lastrowid
            conn.commit()
            return rule_id
    
    def get_rules(self, active_only: bool = False) -> List[Dict[str, Any]]:
        """Get all rules or only active ones"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = "SELECT * FROM rules"
            if active_only:
                query += " WHERE is_active = TRUE"
            query += " ORDER BY created_at DESC"
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            rules = []
            for row in rows:
                rule_dict = dict(row)
                if rule_dict['parsed_conditions']:
                    rule_dict['parsed_conditions'] = json.loads(rule_dict['parsed_conditions'])
                rules.append(rule_dict)
            
            return rules
    
    def update_rule(self, rule_id: int, updates: Dict[str, Any]) -> bool:
        """Update an existing rule"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Build dynamic update query
            set_clauses = []
            values = []
            
            for key, value in updates.items():
                if key in ['rule_name', 'description', 'natural_language_text', 
                          'rule_type', 'priority', 'is_active', 'generated_code']:
                    set_clauses.append(f"{key} = ?")
                    values.append(value)
                elif key == 'parsed_conditions':
                    set_clauses.append("parsed_conditions = ?")
                    values.append(json.dumps(value))
            
            if not set_clauses:
                return False
            
            set_clauses.append("updated_at = CURRENT_TIMESTAMP")
            values.append(rule_id)
            
            query = f"UPDATE rules SET {', '.join(set_clauses)} WHERE rule_id = ?"
            cursor.execute(query, values)
            
            conn.commit()
            return cursor.rowcount > 0
    
    def delete_rule(self, rule_id: int) -> bool:
        """Delete a rule and its execution history"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM rules WHERE rule_id = ?", (rule_id,))
            conn.commit()
            
            return cursor.rowcount > 0
    
    def log_execution(self, execution_data: Dict[str, Any]) -> int:
        """Log a rule execution"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO rule_executions (
                    rule_id, dataset_name, records_processed, records_affected,
                    execution_status, error_message, execution_time_ms, violation_details
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                execution_data['rule_id'],
                execution_data.get('dataset_name', ''),
                execution_data.get('records_processed', 0),
                execution_data.get('records_affected', 0),
                execution_data.get('execution_status', 'success'),
                execution_data.get('error_message', ''),
                execution_data.get('execution_time_ms', 0),
                json.dumps(execution_data.get('violation_details', {}))
            ))
            
            execution_id = cursor.lastrowid
            conn.commit()
            return execution_id
    
    def get_execution_history(self, rule_id: Optional[int] = None, 
                            limit: int = 100) -> List[Dict[str, Any]]:
        """Get rule execution history"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = """
                SELECT re.*, r.rule_name, r.rule_type
                FROM rule_executions re
                JOIN rules r ON re.rule_id = r.rule_id
            """
            params = []
            
            if rule_id:
                query += " WHERE re.rule_id = ?"
                params.append(rule_id)
            
            query += " ORDER BY re.executed_at DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            executions = []
            for row in rows:
                execution_dict = dict(row)
                if execution_dict['violation_details']:
                    execution_dict['violation_details'] = json.loads(execution_dict['violation_details'])
                executions.append(execution_dict)
            
            return executions
    
    def get_rule_analytics(self) -> Dict[str, Any]:
        """Get comprehensive rule analytics"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            analytics = {}
            
            # Basic counts
            cursor.execute("SELECT COUNT(*) FROM rules")
            analytics['total_rules'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM rules WHERE is_active = TRUE")
            analytics['active_rules'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM rule_executions")
            analytics['total_executions'] = cursor.fetchone()[0]
            
            # Success rate
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN execution_status = 'success' THEN 1 ELSE 0 END) as successful
                FROM rule_executions
            """)
            row = cursor.fetchone()
            if row[0] > 0:
                analytics['success_rate'] = (row[1] / row[0]) * 100
            else:
                analytics['success_rate'] = 100.0
            
            # Average execution time
            cursor.execute("SELECT AVG(execution_time_ms) FROM rule_executions WHERE execution_time_ms > 0")
            avg_time = cursor.fetchone()[0]
            analytics['avg_execution_time_ms'] = avg_time or 0
            
            # Rule type distribution
            cursor.execute("SELECT rule_type, COUNT(*) FROM rules GROUP BY rule_type")
            analytics['rule_types'] = dict(cursor.fetchall())
            
            # Recent activity (last 30 days)
            cursor.execute("""
                SELECT DATE(executed_at) as exec_date, COUNT(*) as executions
                FROM rule_executions 
                WHERE executed_at >= datetime('now', '-30 days')
                GROUP BY DATE(executed_at)
                ORDER BY exec_date
            """)
            analytics['daily_executions'] = dict(cursor.fetchall())
            
            return analytics
