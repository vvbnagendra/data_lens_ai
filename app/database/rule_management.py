# File: app/database/rule_management.py
# ENHANCED VERSION WITH DATASET ASSOCIATION AND ADVANCED FEATURES

import sqlite3
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import os
import hashlib

class EnhancedRuleManagementDB:
    """
    Enhanced SQLite-based rule management system with dataset association,
    advanced analytics, and audit trail capabilities
    """
    
    def __init__(self, db_path: str = "app/database/rules_enhanced.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._initialize_database()
    
    def _initialize_database(self):
        """Create enhanced database tables with new features"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Enhanced rules table with dataset association
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rules (
                    rule_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_name TEXT NOT NULL,
                    description TEXT,
                    natural_language_text TEXT NOT NULL,
                    created_by TEXT DEFAULT 'system',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE,
                    rule_type TEXT DEFAULT 'anomaly_detection',
                    priority INTEGER DEFAULT 1,
                    parsed_conditions TEXT,  -- JSON string
                    generated_code TEXT,
                    dataset_signature TEXT,  -- NEW: Links rule to specific dataset
                    auto_execute BOOLEAN DEFAULT FALSE,  -- NEW: Auto-run on data load
                    execution_count INTEGER DEFAULT 0,  -- NEW: Track usage
                    last_executed TIMESTAMP,  -- NEW: Last execution time
                    avg_execution_time_ms REAL DEFAULT 0,  -- NEW: Performance tracking
                    UNIQUE(rule_name, dataset_signature)  -- Unique rule per dataset
                )
            """)
            
            # Enhanced rule executions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rule_executions (
                    execution_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_id INTEGER REFERENCES rules(rule_id) ON DELETE CASCADE,
                    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    dataset_name TEXT,
                    dataset_signature TEXT,  -- NEW: Track dataset version
                    records_processed INTEGER,
                    records_affected INTEGER,
                    execution_status TEXT DEFAULT 'success',
                    error_message TEXT,
                    execution_time_ms INTEGER,
                    violation_details TEXT,  -- JSON string
                    anomaly_percentage REAL DEFAULT 0,  -- NEW: Track severity
                    execution_context TEXT,  -- NEW: How was rule triggered
                    user_session TEXT  -- NEW: Track user sessions
                )
            """)
            
            # NEW: Dataset registry table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_name TEXT NOT NULL,
                    dataset_signature TEXT UNIQUE NOT NULL,
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    columns_info TEXT,  -- JSON: column names and types
                    shape_info TEXT,    -- JSON: rows, columns count
                    data_quality_score REAL DEFAULT 0,  -- Overall quality score
                    total_rules INTEGER DEFAULT 0,  -- Number of rules for this dataset
                    total_executions INTEGER DEFAULT 0,
                    last_anomaly_count INTEGER DEFAULT 0
                )
            """)
            
            # NEW: Rule performance metrics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rule_performance (
                    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_id INTEGER REFERENCES rules(rule_id) ON DELETE CASCADE,
                    metric_date DATE DEFAULT CURRENT_DATE,
                    executions_count INTEGER DEFAULT 0,
                    avg_anomalies_found REAL DEFAULT 0,
                    avg_execution_time_ms REAL DEFAULT 0,
                    success_rate REAL DEFAULT 100.0,
                    false_positive_rate REAL DEFAULT 0,  -- If feedback available
                    user_rating REAL DEFAULT 0  -- User feedback on rule quality
                )
            """)
            
            # NEW: Rule categories and tags
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rule_tags (
                    tag_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_id INTEGER REFERENCES rules(rule_id) ON DELETE CASCADE,
                    tag_name TEXT NOT NULL,
                    tag_category TEXT DEFAULT 'general',  -- audit, compliance, quality, etc.
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # NEW: Audit trail for rule changes
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rule_audit_trail (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_id INTEGER REFERENCES rules(rule_id) ON DELETE CASCADE,
                    action TEXT NOT NULL,  -- created, updated, deleted, executed
                    old_values TEXT,  -- JSON of old values
                    new_values TEXT,  -- JSON of new values
                    changed_by TEXT DEFAULT 'system',
                    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    session_id TEXT,
                    change_reason TEXT
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rules_dataset ON rules(dataset_signature)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_executions_rule ON rule_executions(rule_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_executions_date ON rule_executions(executed_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_datasets_signature ON datasets(dataset_signature)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_rule ON rule_performance(rule_id)")
            
            conn.commit()
    
    def register_dataset(self, dataset_name: str, dataset_signature: str, 
                        columns_info: Dict, shape_info: Dict) -> int:
        """Register or update a dataset in the registry"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if dataset exists
            cursor.execute("SELECT dataset_id FROM datasets WHERE dataset_signature = ?", 
                         (dataset_signature,))
            existing = cursor.fetchone()
            
            if existing:
                # Update last seen
                cursor.execute("""
                    UPDATE datasets 
                    SET last_seen = CURRENT_TIMESTAMP, dataset_name = ?
                    WHERE dataset_signature = ?
                """, (dataset_name, dataset_signature))
                dataset_id = existing[0]
            else:
                # Insert new dataset
                cursor.execute("""
                    INSERT INTO datasets (
                        dataset_name, dataset_signature, columns_info, shape_info
                    ) VALUES (?, ?, ?, ?)
                """, (
                    dataset_name, dataset_signature,
                    json.dumps(columns_info), json.dumps(shape_info)
                ))
                dataset_id = cursor.lastrowid
            
            conn.commit()
            return dataset_id
    
    def create_rule(self, rule_data: Dict[str, Any], dataset_signature: str = None) -> int:
        """Create a new rule with enhanced features"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            try:
                cursor.execute("""
                    INSERT INTO rules (
                        rule_name, description, natural_language_text, rule_type, 
                        priority, is_active, parsed_conditions, generated_code,
                        dataset_signature, auto_execute
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    rule_data['rule_name'],
                    rule_data.get('description', ''),
                    rule_data['natural_language_text'],
                    rule_data.get('rule_type', 'anomaly_detection'),
                    rule_data.get('priority', 1),
                    rule_data.get('is_active', True),
                    json.dumps(rule_data.get('parsed_conditions', {})),
                    rule_data.get('generated_code', ''),
                    dataset_signature,
                    rule_data.get('auto_execute', False)
                ))
                
                rule_id = cursor.lastrowid
                
                # Log audit trail
                self._log_audit_trail(cursor, rule_id, 'created', {}, rule_data, 'system')
                
                # Update dataset rule count
                if dataset_signature:
                    cursor.execute("""
                        UPDATE datasets 
                        SET total_rules = total_rules + 1 
                        WHERE dataset_signature = ?
                    """, (dataset_signature,))
                
                conn.commit()
                return rule_id
                
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" in str(e):
                    raise ValueError(f"Rule '{rule_data['rule_name']}' already exists for this dataset")
                raise e
    
    def get_rules(self, active_only: bool = False, dataset_signature: str = None) -> List[Dict[str, Any]]:
        """Get rules with optional filtering by dataset and active status"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = "SELECT * FROM rules WHERE 1=1"
            params = []
            
            if active_only:
                query += " AND is_active = TRUE"
            
            if dataset_signature:
                query += " AND dataset_signature = ?"
                params.append(dataset_signature)
            
            query += " ORDER BY priority DESC, created_at DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            rules = []
            for row in rows:
                rule_dict = dict(row)
                if rule_dict['parsed_conditions']:
                    try:
                        rule_dict['parsed_conditions'] = json.loads(rule_dict['parsed_conditions'])
                    except:
                        rule_dict['parsed_conditions'] = {}
                rules.append(rule_dict)
            
            return rules
    
    def update_rule(self, rule_id: int, updates: Dict[str, Any], changed_by: str = 'system') -> bool:
        """Update an existing rule with audit trail"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get old values for audit
            cursor.execute("SELECT * FROM rules WHERE rule_id = ?", (rule_id,))
            old_rule = cursor.fetchone()
            if not old_rule:
                return False
            
            old_values = dict(old_rule)
            
            # Build dynamic update query
            set_clauses = []
            values = []
            
            for key, value in updates.items():
                if key in ['rule_name', 'description', 'natural_language_text', 
                          'rule_type', 'priority', 'is_active', 'generated_code', 'auto_execute']:
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
            
            # Log audit trail
            self._log_audit_trail(cursor, rule_id, 'updated', old_values, updates, changed_by)
            
            conn.commit()
            return cursor.rowcount > 0
    
    def delete_rule(self, rule_id: int, deleted_by: str = 'system') -> bool:
        """Delete a rule with audit trail"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get rule data for audit
            cursor.execute("SELECT * FROM rules WHERE rule_id = ?", (rule_id,))
            rule_data = cursor.fetchone()
            if not rule_data:
                return False
            
            rule_dict = dict(rule_data)
            
            # Log audit trail before deletion
            self._log_audit_trail(cursor, rule_id, 'deleted', rule_dict, {}, deleted_by)
            
            # Update dataset rule count
            if rule_dict.get('dataset_signature'):
                cursor.execute("""
                    UPDATE datasets 
                    SET total_rules = total_rules - 1 
                    WHERE dataset_signature = ?
                """, (rule_dict['dataset_signature'],))
            
            # Delete the rule (cascades to related tables)
            cursor.execute("DELETE FROM rules WHERE rule_id = ?", (rule_id,))
            
            conn.commit()
            return cursor.rowcount > 0
    
    def log_execution(self, execution_data: Dict[str, Any]) -> int:
        """Log a rule execution with enhanced metrics"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Calculate anomaly percentage
            anomaly_percentage = 0
            if execution_data.get('records_processed', 0) > 0:
                anomaly_percentage = (execution_data.get('records_affected', 0) / 
                                    execution_data['records_processed']) * 100
            
            cursor.execute("""
                INSERT INTO rule_executions (
                    rule_id, dataset_name, dataset_signature, records_processed, records_affected,
                    execution_status, error_message, execution_time_ms, violation_details,
                    anomaly_percentage, execution_context, user_session
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                execution_data['rule_id'],
                execution_data.get('dataset_name', ''),
                execution_data.get('dataset_signature', ''),
                execution_data.get('records_processed', 0),
                execution_data.get('records_affected', 0),
                execution_data.get('execution_status', 'success'),
                execution_data.get('error_message', ''),
                execution_data.get('execution_time_ms', 0),
                json.dumps(execution_data.get('violation_details', {})),
                anomaly_percentage,
                execution_data.get('execution_context', 'manual'),
                execution_data.get('user_session', '')
            ))
            
            execution_id = cursor.lastrowid
            
            # Update rule statistics
            cursor.execute("""
                UPDATE rules 
                SET execution_count = execution_count + 1,
                    last_executed = CURRENT_TIMESTAMP
                WHERE rule_id = ?
            """, (execution_data['rule_id'],))
            
            # Update dataset statistics
            if execution_data.get('dataset_signature'):
                cursor.execute("""
                    UPDATE datasets 
                    SET total_executions = total_executions + 1,
                        last_anomaly_count = ?
                    WHERE dataset_signature = ?
                """, (execution_data.get('records_affected', 0), execution_data['dataset_signature']))
            
            # Log audit trail
            self._log_audit_trail(cursor, execution_data['rule_id'], 'executed', {}, execution_data, 'system')
            
            conn.commit()
            return execution_id
    
    def get_execution_history(self, rule_id: Optional[int] = None, 
                            dataset_signature: Optional[str] = None,
                            limit: int = 100) -> List[Dict[str, Any]]:
        """Get rule execution history with filtering options"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = """
                SELECT re.*, r.rule_name, r.rule_type, d.dataset_name
                FROM rule_executions re
                JOIN rules r ON re.rule_id = r.rule_id
                LEFT JOIN datasets d ON re.dataset_signature = d.dataset_signature
                WHERE 1=1
            """
            params = []
            
            if rule_id:
                query += " AND re.rule_id = ?"
                params.append(rule_id)
            
            if dataset_signature:
                query += " AND re.dataset_signature = ?"
                params.append(dataset_signature)
            
            query += " ORDER BY re.executed_at DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            executions = []
            for row in rows:
                execution_dict = dict(row)
                if execution_dict['violation_details']:
                    try:
                        execution_dict['violation_details'] = json.loads(execution_dict['violation_details'])
                    except:
                        execution_dict['violation_details'] = {}
                executions.append(execution_dict)
            
            return executions
    
    def get_rule_analytics(self, dataset_signature: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive rule analytics"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            analytics = {}
            
            # Basic rule counts
            if dataset_signature:
                cursor.execute("SELECT COUNT(*) FROM rules WHERE dataset_signature = ?", (dataset_signature,))
                analytics['total_rules'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM rules WHERE dataset_signature = ? AND is_active = TRUE", (dataset_signature, True))
                analytics['active_rules'] = cursor.fetchone()[0]
            else:
                cursor.execute("SELECT COUNT(*) FROM rules")
                analytics['total_rules'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM rules WHERE is_active = TRUE")
                analytics['active_rules'] = cursor.fetchone()[0]
            analytics['active_rules'] = cursor.fetchone()[0]
            
            # Execution statistics
            if dataset_signature:
                cursor.execute("SELECT COUNT(*) FROM rule_executions WHERE dataset_signature = ?", (dataset_signature,))
                analytics['total_executions'] = cursor.fetchone()[0]
                
                # Success rate
                cursor.execute("SELECT COUNT(*) FROM rule_executions WHERE dataset_signature = ? AND execution_status = 'success'", (dataset_signature,))
                successful_executions = cursor.fetchone()[0]
            else:
                cursor.execute("SELECT COUNT(*) FROM rule_executions")
                analytics['total_executions'] = cursor.fetchone()[0]
                
                # Success rate
                cursor.execute("SELECT COUNT(*) FROM rule_executions WHERE execution_status = 'success'")
                successful_executions = cursor.fetchone()[0]
            
            if analytics['total_executions'] > 0:
                analytics['success_rate'] = (successful_executions / analytics['total_executions']) * 100
            else:
                analytics['success_rate'] = 100.0
            
            # Average execution time
            if dataset_signature:
                cursor.execute("SELECT AVG(execution_time_ms) FROM rule_executions WHERE dataset_signature = ?", (dataset_signature,))
            else:
                cursor.execute("SELECT AVG(execution_time_ms) FROM rule_executions")
            
            avg_time = cursor.fetchone()[0]
            analytics['avg_execution_time_ms'] = avg_time or 0
            
            # Rule type distribution
            if dataset_signature:
                cursor.execute("SELECT rule_type, COUNT(*) FROM rules WHERE dataset_signature = ? GROUP BY rule_type", (dataset_signature,))
            else:
                cursor.execute("SELECT rule_type, COUNT(*) FROM rules GROUP BY rule_type")
            
            analytics['rule_types'] = dict(cursor.fetchall())
            
            # Recent activity (last 30 days)
            if dataset_signature:
                cursor.execute("""
                    SELECT DATE(executed_at) as exec_date, COUNT(*) as executions
                    FROM rule_executions 
                    WHERE executed_at >= datetime('now', '-30 days') AND dataset_signature = ?
                    GROUP BY DATE(executed_at) ORDER BY exec_date
                """, (dataset_signature,))
            else:
                cursor.execute("""
                    SELECT DATE(executed_at) as exec_date, COUNT(*) as executions
                    FROM rule_executions 
                    WHERE executed_at >= datetime('now', '-30 days')
                    GROUP BY DATE(executed_at) ORDER BY exec_date
                """)
            
            analytics['daily_executions'] = dict(cursor.fetchall())
            
            # Top performing rules
            if dataset_signature:
                cursor.execute("""
                    SELECT r.rule_name, COUNT(re.execution_id) as exec_count,
                           AVG(re.records_affected) as avg_anomalies,
                           AVG(re.execution_time_ms) as avg_time
                    FROM rules r
                    LEFT JOIN rule_executions re ON r.rule_id = re.rule_id
                    WHERE r.dataset_signature = ?
                    GROUP BY r.rule_id ORDER BY exec_count DESC LIMIT 10
                """, (dataset_signature,))
            else:
                cursor.execute("""
                    SELECT r.rule_name, COUNT(re.execution_id) as exec_count,
                           AVG(re.records_affected) as avg_anomalies,
                           AVG(re.execution_time_ms) as avg_time
                    FROM rules r
                    LEFT JOIN rule_executions re ON r.rule_id = re.rule_id
                    GROUP BY r.rule_id ORDER BY exec_count DESC LIMIT 10
                """)
            
            analytics['top_rules'] = [dict(zip(['rule_name', 'exec_count', 'avg_anomalies', 'avg_time'], row)) 
                                    for row in cursor.fetchall()]
            
            # Dataset-specific metrics if signature provided
            if dataset_signature:
                cursor.execute("SELECT * FROM datasets WHERE dataset_signature = ?", (dataset_signature,))
                dataset_info = cursor.fetchone()
                if dataset_info:
                    analytics['dataset_info'] = dict(dataset_info)
            
            return analytics
    
    def get_datasets(self) -> List[Dict[str, Any]]:
        """Get all registered datasets"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT d.*, 
                       COUNT(DISTINCT r.rule_id) as active_rules,
                       COUNT(DISTINCT re.execution_id) as total_executions,
                       MAX(re.executed_at) as last_execution
                FROM datasets d
                LEFT JOIN rules r ON d.dataset_signature = r.dataset_signature AND r.is_active = TRUE
                LEFT JOIN rule_executions re ON d.dataset_signature = re.dataset_signature
                GROUP BY d.dataset_id
                ORDER BY d.last_seen DESC
            """)
            
            datasets = []
            for row in cursor.fetchall():
                dataset_dict = dict(row)
                if dataset_dict['columns_info']:
                    try:
                        dataset_dict['columns_info'] = json.loads(dataset_dict['columns_info'])
                    except:
                        dataset_dict['columns_info'] = {}
                if dataset_dict['shape_info']:
                    try:
                        dataset_dict['shape_info'] = json.loads(dataset_dict['shape_info'])
                    except:
                        dataset_dict['shape_info'] = {}
                datasets.append(dataset_dict)
            
            return datasets
    
    def add_rule_tags(self, rule_id: int, tags: List[str], category: str = 'general') -> bool:
        """Add tags to a rule for better organization"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            try:
                for tag in tags:
                    cursor.execute("""
                        INSERT OR IGNORE INTO rule_tags (rule_id, tag_name, tag_category)
                        VALUES (?, ?, ?)
                    """, (rule_id, tag.lower().strip(), category))
                
                conn.commit()
                return True
            except Exception:
                return False
    
    def get_rules_by_tags(self, tags: List[str], dataset_signature: str = None) -> List[Dict[str, Any]]:
        """Get rules filtered by tags"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            placeholders = ','.join(['?' for _ in tags])
            query = f"""
                SELECT DISTINCT r.* 
                FROM rules r
                JOIN rule_tags rt ON r.rule_id = rt.rule_id
                WHERE rt.tag_name IN ({placeholders})
            """
            params = [tag.lower().strip() for tag in tags]
            
            if dataset_signature:
                query += " AND r.dataset_signature = ?"
                params.append(dataset_signature)
            
            query += " AND r.is_active = TRUE ORDER BY r.priority DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            rules = []
            for row in rows:
                rule_dict = dict(row)
                if rule_dict['parsed_conditions']:
                    try:
                        rule_dict['parsed_conditions'] = json.loads(rule_dict['parsed_conditions'])
                    except:
                        rule_dict['parsed_conditions'] = {}
                rules.append(rule_dict)
            
            return rules
    
    def cleanup_old_executions(self, days_to_keep: int = 90) -> int:
        """Clean up old execution records"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            cursor.execute("""
                DELETE FROM rule_executions 
                WHERE executed_at < ?
            """, (cutoff_date,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            return deleted_count
    
    def _log_audit_trail(self, cursor, rule_id: int, action: str, old_values: Dict, 
                        new_values: Dict, changed_by: str, session_id: str = None):
        """Log changes to the audit trail"""
        
        cursor.execute("""
            INSERT INTO rule_audit_trail (
                rule_id, action, old_values, new_values, changed_by, session_id
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            rule_id, action,
            json.dumps(old_values, default=str),
            json.dumps(new_values, default=str),
            changed_by, session_id
        ))
    
    def get_audit_trail(self, rule_id: Optional[int] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get audit trail for rules"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = """
                SELECT at.*, r.rule_name
                FROM rule_audit_trail at
                JOIN rules r ON at.rule_id = r.rule_id
            """
            params = []
            
            if rule_id:
                query += " WHERE at.rule_id = ?"
                params.append(rule_id)
            
            query += " ORDER BY at.changed_at DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            audit_records = []
            for row in rows:
                record = dict(row)
                try:
                    if record['old_values']:
                        record['old_values'] = json.loads(record['old_values'])
                    if record['new_values']:
                        record['new_values'] = json.loads(record['new_values'])
                except:
                    pass
                audit_records.append(record)
            
            return audit_records

# Create an alias for backward compatibility
RuleManagementDB = EnhancedRuleManagementDB