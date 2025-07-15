# File: app/database/debug_helper.py
# Database debugging and troubleshooting utilities

import sqlite3
import json
import os
from typing import Dict, Any, List

def debug_database_schema(db_path: str = "app/database/rules_enhanced.db"):
    """Debug and display database schema information"""
    
    print(f"🔍 Debugging database: {db_path}")
    
    if not os.path.exists(db_path):
        print(f"❌ Database file not found: {db_path}")
        return
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            print(f"\n📊 Found {len(tables)} tables:")
            
            for table_name, in tables:
                print(f"\n🔧 Table: {table_name}")
                
                # Get table schema
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                
                print(f"   Columns ({len(columns)}):")
                for col in columns:
                    print(f"   - {col[1]} ({col[2]}) {'PRIMARY KEY' if col[5] else ''} {'NOT NULL' if col[3] else ''}")
                
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
                print(f"   Rows: {row_count}")
                
                # Show sample data if exists
                if row_count > 0:
                    cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
                    sample_rows = cursor.fetchall()
                    print(f"   Sample data:")
                    for i, row in enumerate(sample_rows, 1):
                        print(f"     Row {i}: {row}")
            
            print("\n✅ Database schema analysis complete")
            
    except Exception as e:
        print(f"❌ Error analyzing database: {e}")

def test_database_operations(db_path: str = "app/database/rules_enhanced.db"):
    """Test basic database operations"""
    
    print(f"\n🧪 Testing database operations: {db_path}")
    
    try:
        from database.rule_management import EnhancedRuleManagementDB
        
        # Initialize database
        db = EnhancedRuleManagementDB(db_path)
        print("✅ Database initialized successfully")
        
        # Test rule creation
        test_rule = {
            'rule_name': 'Debug Test Rule',
            'description': 'Test rule for debugging',
            'natural_language_text': 'amount is greater than 1000',
            'rule_type': 'test',
            'priority': 1
        }
        
        try:
            rule_id = db.create_rule(test_rule, 'debug_dataset_signature')
            print(f"✅ Rule created successfully with ID: {rule_id}")
        except Exception as e:
            print(f"❌ Rule creation failed: {e}")
            return
        
        # Test rule retrieval
        try:
            rules = db.get_rules(dataset_signature='debug_dataset_signature')
            print(f"✅ Retrieved {len(rules)} rules")
        except Exception as e:
            print(f"❌ Rule retrieval failed: {e}")
        
        # Test execution logging
        try:
            execution_data = {
                'rule_id': rule_id,
                'dataset_name': 'debug_dataset',
                'dataset_signature': 'debug_dataset_signature',
                'records_processed': 100,
                'records_affected': 5,
                'execution_status': 'success'
            }
            
            execution_id = db.log_execution(execution_data)
            print(f"✅ Execution logged successfully with ID: {execution_id}")
        except Exception as e:
            print(f"❌ Execution logging failed: {e}")
        
        # Test analytics
        try:
            analytics = db.get_rule_analytics('debug_dataset_signature')
            print(f"✅ Analytics retrieved: {analytics.get('total_rules', 0)} rules")
        except Exception as e:
            print(f"❌ Analytics retrieval failed: {e}")
        
        # Cleanup test data
        try:
            db.delete_rule(rule_id)
            print("✅ Test rule cleaned up successfully")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")
        
        print("✅ All database operations tested successfully")
        
    except Exception as e:
        print(f"❌ Database operations test failed: {e}")

def fix_common_database_issues(db_path: str = "app/database/rules_enhanced.db"):
    """Fix common database issues"""
    
    print(f"\n🔧 Fixing common database issues: {db_path}")
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Check for foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")
            print("✅ Foreign key constraints enabled")
            
            # Check database integrity
            cursor.execute("PRAGMA integrity_check")
            integrity_result = cursor.fetchone()[0]
            if integrity_result == "ok":
                print("✅ Database integrity check passed")
            else:
                print(f"⚠️ Database integrity issues: {integrity_result}")
            
            # Analyze database for optimization
            cursor.execute("ANALYZE")
            print("✅ Database analyzed for optimization")
            
            # Vacuum database to reclaim space
            cursor.execute("VACUUM")
            print("✅ Database vacuumed")
            
    except Exception as e:
        print(f"❌ Error fixing database issues: {e}")

def create_sample_rules_data(db_path: str = "app/database/rules_enhanced.db"):
    """Create sample rules data for testing"""
    
    print(f"\n📝 Creating sample rules data: {db_path}")
    
    try:
        from database.rule_management import EnhancedRuleManagementDB
        
        db = EnhancedRuleManagementDB(db_path)
        
        # Sample dataset signature
        dataset_sig = "sample_financial_data_12345"
        
        # Register sample dataset
        dataset_info = {
            'columns': ['amount', 'customer_age', 'account_balance', 'transaction_type'],
            'dtypes': {'amount': 'float64', 'customer_age': 'int64', 'account_balance': 'float64', 'transaction_type': 'object'}
        }
        shape_info = {'rows': 1000, 'columns': 4}
        
        db.register_dataset("Sample Financial Data", dataset_sig, dataset_info, shape_info)
        print("✅ Sample dataset registered")
        
        # Sample rules
        sample_rules = [
            {
                'rule_name': 'High Value Transactions',
                'description': 'Detect transactions above $10,000',
                'natural_language_text': 'amount is greater than 10000',
                'rule_type': 'financial_monitoring',
                'priority': 3
            },
            {
                'rule_name': 'Suspicious Age',
                'description': 'Detect customers with unrealistic ages',
                'natural_language_text': 'customer_age is less than 18 or customer_age is greater than 100',
                'rule_type': 'data_quality',
                'priority': 2
            },
            {
                'rule_name': 'Negative Balance',
                'description': 'Detect accounts with negative balances',
                'natural_language_text': 'account_balance is less than 0',
                'rule_type': 'compliance',
                'priority': 4
            },
            {
                'rule_name': 'Rare Transaction Types',
                'description': 'Detect uncommon transaction types',
                'natural_language_text': 'transaction_type is rare',
                'rule_type': 'fraud_detection',
                'priority': 1
            }
        ]
        
        created_rule_ids = []
        for rule_data in sample_rules:
            try:
                rule_id = db.create_rule(rule_data, dataset_sig)
                created_rule_ids.append(rule_id)
                print(f"✅ Created rule: {rule_data['rule_name']} (ID: {rule_id})")
            except Exception as e:
                print(f"⚠️ Failed to create rule {rule_data['rule_name']}: {e}")
        
        # Create sample execution data
        import random
        from datetime import datetime, timedelta
        
        for rule_id in created_rule_ids:
            # Create 3-5 sample executions per rule
            for i in range(random.randint(3, 5)):
                execution_data = {
                    'rule_id': rule_id,
                    'dataset_name': 'Sample Financial Data',
                    'dataset_signature': dataset_sig,
                    'records_processed': random.randint(500, 1500),
                    'records_affected': random.randint(0, 50),
                    'execution_status': random.choice(['success', 'success', 'success', 'error']),
                    'execution_time_ms': random.randint(100, 2000),
                    'execution_context': random.choice(['manual', 'auto', 'scheduled'])
                }
                
                try:
                    db.log_execution(execution_data)
                except Exception as e:
                    print(f"⚠️ Failed to log execution for rule {rule_id}: {e}")
        
        print(f"✅ Created {len(created_rule_ids)} sample rules with execution history")
        
        # Test analytics
        analytics = db.get_rule_analytics(dataset_sig)
        print(f"📊 Sample analytics: {analytics['total_rules']} rules, {analytics['total_executions']} executions")
        
    except Exception as e:
        print(f"❌ Error creating sample data: {e}")

def main():
    """Main debugging function"""
    
    print("🔍 Enhanced Rule Management Database Debugger")
    print("=" * 50)
    
    db_path = "app/database/rules_enhanced.db"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Debug schema
    debug_database_schema(db_path)
    
    # Test operations
    test_database_operations(db_path)
    
    # Fix common issues
    fix_common_database_issues(db_path)
    
    # Create sample data if database is empty
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM rules")
            rule_count = cursor.fetchone()[0]
            
            if rule_count == 0:
                print(f"\n📝 Database is empty, creating sample data...")
                create_sample_rules_data(db_path)
            else:
                print(f"\n📊 Database contains {rule_count} existing rules")
    
    except Exception as e:
        print(f"❌ Error checking database contents: {e}")
    
    print("\n🎉 Database debugging complete!")

if __name__ == "__main__":
    main()