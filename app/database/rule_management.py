# File: app/database/rule_management.py
# Fixed rule management database implementation

import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import hashlib

class RuleManagementDB:
    """
    Simple rule management database using SQLite
    """
    
    def __init__(self, db_path: str = "app/database/rules.db"):
        """Initialize the rule management database"""
        self.db_path = db_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Rules table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rules (
                    rule_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_name TEXT NOT NULL,
                    description TEXT,
                    natural_language_text TEXT NOT NULL,
                    rule_type TEXT DEFAULT 'anomaly_detection',
                    priority INTEGER DEFAULT 1,
                    is_active BOOLEAN DEFAULT 1,
                    dataset_signature TEXT,
                    parsed_conditions TEXT,
                    generated_code TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_by TEXT DEFAULT 'system',
                    auto_execute BOOLEAN DEFAULT 0
                )
            ''')
            
            # Rule executions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rule_executions (
                    execution_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_id INTEGER,
                    dataset_name TEXT,
                    records_processed INTEGER,
                    records_affected INTEGER,
                    execution_status TEXT,
                    error_message TEXT,
                    execution_time_ms INTEGER,
                    violation_details TEXT,
                    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (rule_id) REFERENCES rules (rule_id)
                )
            ''')
            
            # Rule performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rule_performance (
                    performance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_id INTEGER,
                    avg_execution_time_ms REAL,
                    success_rate REAL,
                    avg_anomaly_rate REAL,
                    last_calculated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (rule_id) REFERENCES rules (rule_id)
                )
            ''')
            
            conn.commit()
    
    def create_rule(self, rule_data: Dict[str, Any]) -> int:
        """Create a new rule"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO rules (
                    rule_name, description, natural_language_text, rule_type,
                    priority, dataset_signature, parsed_conditions, generated_code,
                    auto_execute
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                rule_data.get('rule_name'),
                rule_data.get('description', ''),
                rule_data.get('natural_language_text'),
                rule_data.get('rule_type', 'anomaly_detection'),
                rule_data.get('priority', 1),
                rule_data.get('dataset_signature'),
                json.dumps(rule_data.get('parsed_conditions', {})),
                rule_data.get('generated_code', ''),
                rule_data.get('auto_execute', False)
            ))
            
            rule_id = cursor.lastrowid
            conn.commit()
            return rule_id
    
    def get_rules(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """Get all rules"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = "SELECT * FROM rules"
            if active_only:
                query += " WHERE is_active = 1"
            query += " ORDER BY created_at DESC"
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            rules = []
            for row in rows:
                rule = dict(row)
                # Parse JSON fields
                try:
                    rule['parsed_conditions'] = json.loads(rule['parsed_conditions'] or '{}')
                except:
                    rule['parsed_conditions'] = {}
                rules.append(rule)
            
            return rules
    
    def get_rule_by_id(self, rule_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific rule by ID"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM rules WHERE rule_id = ?", (rule_id,))
            row = cursor.fetchone()
            
            if row:
                rule = dict(row)
                try:
                    rule['parsed_conditions'] = json.loads(rule['parsed_conditions'] or '{}')
                except:
                    rule['parsed_conditions'] = {}
                return rule
            return None
    
    def update_rule(self, rule_id: int, updates: Dict[str, Any]) -> bool:
        """Update an existing rule"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build UPDATE query dynamically
                set_clauses = []
                values = []
                
                for key, value in updates.items():
                    if key in ['rule_name', 'description', 'natural_language_text', 
                              'rule_type', 'priority', 'dataset_signature', 
                              'generated_code', 'auto_execute']:
                        set_clauses.append(f"{key} = ?")
                        values.append(value)
                    elif key == 'parsed_conditions':
                        set_clauses.append("parsed_conditions = ?")
                        values.append(json.dumps(value))
                
                if set_clauses:
                    set_clauses.append("updated_at = CURRENT_TIMESTAMP")
                    values.append(rule_id)
                    
                    query = f"UPDATE rules SET {', '.join(set_clauses)} WHERE rule_id = ?"
                    cursor.execute(query, values)
                    conn.commit()
                    return cursor.rowcount > 0
                return False
        except Exception:
            return False
    
    def delete_rule(self, rule_id: int) -> bool:
        """Delete a rule (soft delete)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE rules SET is_active = 0, updated_at = CURRENT_TIMESTAMP WHERE rule_id = ?",
                    (rule_id,)
                )
                conn.commit()
                return cursor.rowcount > 0
        except Exception:
            return False
    
    def log_execution(self, execution_data: Dict[str, Any]) -> int:
        """Log rule execution"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO rule_executions (
                    rule_id, dataset_name, records_processed, records_affected,
                    execution_status, error_message, execution_time_ms, violation_details
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                execution_data.get('rule_id'),
                execution_data.get('dataset_name'),
                execution_data.get('records_processed', 0),
                execution_data.get('records_affected', 0),
                execution_data.get('execution_status', 'unknown'),
                execution_data.get('error_message'),
                execution_data.get('execution_time_ms', 0),
                json.dumps(execution_data.get('violation_details', {}))
            ))
            
            execution_id = cursor.lastrowid
            conn.commit()
            return execution_id
    
    def get_execution_history(self, rule_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        """Get execution history for a rule"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM rule_executions 
                WHERE rule_id = ? 
                ORDER BY executed_at DESC 
                LIMIT ?
            ''', (rule_id, limit))
            
            rows = cursor.fetchall()
            
            executions = []
            for row in rows:
                execution = dict(row)
                try:
                    execution['violation_details'] = json.loads(execution['violation_details'] or '{}')
                except:
                    execution['violation_details'] = {}
                executions.append(execution)
            
            return executions
    
    def get_rule_analytics(self) -> Dict[str, Any]:
        """Get comprehensive rule analytics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total rules
            cursor.execute("SELECT COUNT(*) FROM rules WHERE is_active = 1")
            total_rules = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM rules")
            all_rules = cursor.fetchone()[0]
            
            # Total executions
            cursor.execute("SELECT COUNT(*) FROM rule_executions")
            total_executions = cursor.fetchone()[0]
            
            # Success rate
            cursor.execute("SELECT COUNT(*) FROM rule_executions WHERE execution_status = 'success'")
            successful_executions = cursor.fetchone()[0]
            success_rate = (successful_executions / max(total_executions, 1)) * 100
            
            # Rule types
            cursor.execute("SELECT rule_type, COUNT(*) FROM rules WHERE is_active = 1 GROUP BY rule_type")
            rule_types = dict(cursor.fetchall())
            
            # Daily executions (last 30 days)
            cursor.execute('''
                SELECT DATE(executed_at) as date, COUNT(*) as count
                FROM rule_executions 
                WHERE executed_at >= datetime('now', '-30 days')
                GROUP BY DATE(executed_at)
                ORDER BY date
            ''')
            daily_executions = dict(cursor.fetchall())
            
            return {
                'total_rules': total_rules,
                'active_rules': total_rules,
                'inactive_rules': all_rules - total_rules,
                'total_executions': total_executions,
                'success_rate': success_rate,
                'rule_types': rule_types,
                'daily_executions': daily_executions
            }
    
    def get_rules_for_dataset(self, dataset_signature: str) -> List[Dict[str, Any]]:
        """Get all rules for a specific dataset"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM rules 
                WHERE dataset_signature = ? AND is_active = 1
                ORDER BY priority DESC, created_at DESC
            ''', (dataset_signature,))
            
            rows = cursor.fetchall()
            
            rules = []
            for row in rows:
                rule = dict(row)
                try:
                    rule['parsed_conditions'] = json.loads(rule['parsed_conditions'] or '{}')
                except:
                    rule['parsed_conditions'] = {}
                rules.append(rule)
            
            return rules


# Enhanced Rule Management DB with additional features
class EnhancedRuleManagementDB(RuleManagementDB):
    """Enhanced version with additional features"""
    
    def __init__(self, db_path: str = "app/database/enhanced_rules.db"):
        super().__init__(db_path)
        self._init_enhanced_tables()
    
    def _init_enhanced_tables(self):
        """Initialize additional tables for enhanced features"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Rule tags table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rule_tags (
                    tag_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_id INTEGER,
                    tag_name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (rule_id) REFERENCES rules (rule_id)
                )
            ''')
            
            # Rule audit trail
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rule_audit_trail (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_id INTEGER,
                    action TEXT NOT NULL,
                    old_values TEXT,
                    new_values TEXT,
                    changed_by TEXT,
                    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (rule_id) REFERENCES rules (rule_id)
                )
            ''')
            
            # Dataset registry
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_name TEXT NOT NULL,
                    dataset_signature TEXT UNIQUE NOT NULL,
                    column_info TEXT,
                    row_count INTEGER,
                    last_analyzed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    def register_dataset(self, dataset_name: str, dataset_signature: str, 
                        column_info: Dict[str, Any], row_count: int) -> int:
        """Register a dataset for rule management"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO datasets (
                    dataset_name, dataset_signature, column_info, row_count, last_analyzed
                ) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                dataset_name,
                dataset_signature,
                json.dumps(column_info),
                row_count
            ))
            
            dataset_id = cursor.lastrowid
            conn.commit()
            return dataset_id
    
    def add_rule_tags(self, rule_id: int, tags: List[str]):
        """Add tags to a rule"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Remove existing tags
            cursor.execute("DELETE FROM rule_tags WHERE rule_id = ?", (rule_id,))
            
            # Add new tags
            for tag in tags:
                cursor.execute(
                    "INSERT INTO rule_tags (rule_id, tag_name) VALUES (?, ?)",
                    (rule_id, tag.strip().lower())
                )
            
            conn.commit()
    
    def get_rules_by_tag(self, tag_name: str) -> List[Dict[str, Any]]:
        """Get rules by tag"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT r.* FROM rules r
                JOIN rule_tags rt ON r.rule_id = rt.rule_id
                WHERE rt.tag_name = ? AND r.is_active = 1
                ORDER BY r.priority DESC
            ''', (tag_name.lower(),))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def audit_rule_change(self, rule_id: int, action: str, old_values: Dict = None, 
                         new_values: Dict = None, changed_by: str = "system"):
        """Log rule changes for audit trail"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO rule_audit_trail (
                    rule_id, action, old_values, new_values, changed_by
                ) VALUES (?, ?, ?, ?, ?)
            ''', (
                rule_id,
                action,
                json.dumps(old_values or {}),
                json.dumps(new_values or {}),
                changed_by
            ))
            
            conn.commit()