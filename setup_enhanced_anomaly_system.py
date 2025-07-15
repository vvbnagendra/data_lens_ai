# File: setup_enhanced_anomaly_system.py
# Quick setup script for the enhanced anomaly detection with rule management

import os
import sys
import shutil
from pathlib import Path

def setup_enhanced_anomaly_system():
    """
    Setup script for the enhanced anomaly detection system with rule management
    """
    
    print("🚀 Setting up Enhanced Anomaly Detection with Rule Management...")
    print("=" * 70)
    
    # Step 1: Create required directories
    print("\n1. 📁 Creating directory structure...")
    directories = [
        "app/database",
        "app/pages", 
        "app/assets",
        "app/core_logic",
        "app/data_quality",
        "app/outputs",
        "exports/anomaly_reports"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   Created: {directory}")
    
    # Step 2: Create __init__.py files for Python packages
    print("\n2. 🐍 Creating Python package files...")
    init_files = [
        "app/__init__.py",
        "app/database/__init__.py",
        "app/core_logic/__init__.py",
        "app/data_quality/__init__.py",
        "app/assets/__init__.py"
    ]
    
    for init_file in init_files:
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write("# Package initialization file\n")
            print(f"   Created: {init_file}")
    
    # Step 3: Test database creation
    print("\n3. 🗄️ Testing enhanced rule management database...")
    try:
        # Import and test the enhanced database
        sys.path.append('app')
        from database.rule_management import EnhancedRuleManagementDB
        
        db = EnhancedRuleManagementDB("app/database/test_rules.db")
        print("   ✅ Database initialized successfully!")
        
        # Test basic operations
        test_rule = {
            'rule_name': 'Test Setup Rule',
            'description': 'Test rule created during setup',
            'natural_language_text': 'amount is greater than 1000',
            'rule_type': 'test',
            'priority': 1
        }
        
        rule_id = db.create_rule(test_rule, 'test_dataset_signature')
        analytics = db.get_rule_analytics()
        
        print(f"   ✅ Test rule created with ID: {rule_id}")
        print(f"   ✅ Analytics retrieved: {analytics['total_rules']} rules")
        
        # Clean up test rule
        db.delete_rule(rule_id)
        print("   ✅ Test cleanup completed")
        
        # Remove test database
        os.remove("app/database/test_rules.db")
        
    except Exception as e:
        print(f"   ❌ Database test failed: {e}")
        return False
    
    # Step 4: Check required Python packages
    print("\n4. 📦 Checking required packages...")
    required_packages = [
        'streamlit',
        'pandas', 
        'numpy',
        'plotly',
        'sqlite3'  # Built-in with Python
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'sqlite3':
                import sqlite3
            else:
                __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n   🔧 To install missing packages:")
        print(f"   pip install {' '.join(missing_packages)}")
    
    # Step 5: Create sample data for testing
    print("\n5. 📊 Creating sample data for testing...")
    create_sample_anomaly_data()
    
    # Step 6: Verify file structure
    print("\n6. 🔍 Verifying file structure...")
    required_files = [
        "app/pages/5_Anomaly_Detection.py",
        "app/database/rule_management.py"
    ]
    
    all_files_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - MISSING")
            all_files_exist = False
    
    # Step 7: Create quick start guide
    print("\n7. 📋 Creating quick start guide...")
    create_quick_start_guide()
    
    # Final summary
    print("\n" + "="*70)
    print("🎉 SETUP COMPLETED!")
    print("="*70)
    
    if all_files_exist and not missing_packages:
        print("✅ All components are ready!")
        print("\n🚀 Next steps:")
        print("1. Run: streamlit run app/Home.py")
        print("2. Navigate to 'Anomaly Detection' page")
        print("3. Upload your data and create your first rule!")
        print("4. Check out the Quick Start Guide: QUICK_START_ANOMALY_RULES.md")
    else:
        print("⚠️  Some issues need to be resolved:")
        if missing_packages:
            print(f"   - Install missing packages: {', '.join(missing_packages)}")
        if not all_files_exist:
            print("   - Ensure all required files are in place")
    
    print("\n📖 Documentation:")
    print("   - Rule syntax examples in the app interface")
    print("   - Database schema in app/database/rule_management.py")
    print("   - Sample data in sample_data/financial_transactions.csv")
    
    return all_files_exist and not missing_packages

def create_sample_anomaly_data():
    """Create comprehensive sample data for testing anomaly detection"""
    
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Create realistic financial transaction data with built-in anomalies
    n_normal = 2000
    n_anomalies = 200
    
    # Normal transactions
    normal_data = {
        'transaction_id': [f"TXN_{i:06d}" for i in range(1, n_normal + 1)],
        'amount': np.random.lognormal(mean=6, sigma=1, size=n_normal),  # Log-normal for realistic amounts
        'customer_age': np.random.normal(35, 12, n_normal).astype(int),
        'account_balance': np.random.normal(15000, 8000, n_normal),
        'transaction_count_daily': np.random.poisson(3, n_normal),
        'customer_type': np.random.choice(['bronze', 'silver', 'gold', 'platinum'], n_normal, p=[0.4, 0.3, 0.2, 0.1]),
        'transaction_type': np.random.choice(['purchase', 'withdrawal', 'transfer', 'payment'], n_normal, p=[0.4, 0.2, 0.3, 0.1]),
        'merchant_category': np.random.choice(['grocery', 'gas', 'restaurant', 'retail', 'online', 'other'], n_normal),
        'hour_of_day': np.random.choice(range(6, 23), n_normal, p=[0.05]*4 + [0.15]*8 + [0.08]*5),
        'is_weekend': np.random.choice([True, False], n_normal, p=[0.2, 0.8])
    }
    
    # Create transaction dates (last 90 days)
    base_date = datetime.now() - timedelta(days=90)
    normal_data['transaction_date'] = [
        base_date + timedelta(days=np.random.randint(0, 90), 
                             hours=np.random.randint(0, 24), 
                             minutes=np.random.randint(0, 60))
        for _ in range(n_normal)
    ]
    
    normal_df = pd.DataFrame(normal_data)
    
    # Create anomalous transactions
    anomaly_data = {
        'transaction_id': [f"TXN_{i:06d}" for i in range(n_normal + 1, n_normal + n_anomalies + 1)],
        'amount': [],
        'customer_age': [],
        'account_balance': [],
        'transaction_count_daily': [],
        'customer_type': [],
        'transaction_type': [],
        'merchant_category': [],
        'hour_of_day': [],
        'is_weekend': [],
        'transaction_date': []
    }
    
    # Different types of anomalies
    anomaly_types = {
        'high_amount': int(n_anomalies * 0.3),      # Unusually high amounts
        'suspicious_age': int(n_anomalies * 0.2),   # Suspicious ages (too young/old)
        'negative_balance': int(n_anomalies * 0.15), # Negative account balances
        'high_frequency': int(n_anomalies * 0.15),   # High transaction frequency
        'off_hours': int(n_anomalies * 0.1),         # Transactions at unusual hours
        'future_dates': int(n_anomalies * 0.1)       # Future transaction dates
    }
    
    current_idx = 0
    
    # High amount anomalies
    for _ in range(anomaly_types['high_amount']):
        anomaly_data['amount'].append(np.random.uniform(50000, 200000))
        anomaly_data['customer_age'].append(np.random.randint(25, 65))
        anomaly_data['account_balance'].append(np.random.normal(15000, 8000))
        anomaly_data['transaction_count_daily'].append(np.random.poisson(3))
        anomaly_data['customer_type'].append(np.random.choice(['bronze', 'silver', 'gold', 'platinum']))
        anomaly_data['transaction_type'].append(np.random.choice(['purchase', 'withdrawal', 'transfer']))
        anomaly_data['merchant_category'].append(np.random.choice(['luxury', 'casino', 'investment', 'crypto']))
        anomaly_data['hour_of_day'].append(np.random.randint(6, 23))
        anomaly_data['is_weekend'].append(np.random.choice([True, False]))
        anomaly_data['transaction_date'].append(
            base_date + timedelta(days=np.random.randint(0, 90), hours=np.random.randint(0, 24))
        )
    
    # Suspicious age anomalies
    for _ in range(anomaly_types['suspicious_age']):
        anomaly_data['amount'].append(np.random.lognormal(6, 1))
        anomaly_data['customer_age'].append(np.random.choice([15, 16, 17, 95, 96, 97, 98, 99, 100]))
        anomaly_data['account_balance'].append(np.random.normal(15000, 8000))
        anomaly_data['transaction_count_daily'].append(np.random.poisson(3))
        anomaly_data['customer_type'].append(np.random.choice(['bronze', 'silver']))
        anomaly_data['transaction_type'].append(np.random.choice(['purchase', 'withdrawal']))
        anomaly_data['merchant_category'].append(np.random.choice(['grocery', 'gas', 'restaurant']))
        anomaly_data['hour_of_day'].append(np.random.randint(6, 23))
        anomaly_data['is_weekend'].append(np.random.choice([True, False]))
        anomaly_data['transaction_date'].append(
            base_date + timedelta(days=np.random.randint(0, 90), hours=np.random.randint(0, 24))
        )
    
    # Negative balance anomalies
    for _ in range(anomaly_types['negative_balance']):
        anomaly_data['amount'].append(np.random.lognormal(6, 1))
        anomaly_data['customer_age'].append(np.random.randint(18, 75))
        anomaly_data['account_balance'].append(np.random.uniform(-10000, -100))
        anomaly_data['transaction_count_daily'].append(np.random.poisson(3))
        anomaly_data['customer_type'].append(np.random.choice(['bronze', 'silver']))
        anomaly_data['transaction_type'].append(np.random.choice(['withdrawal', 'purchase']))
        anomaly_data['merchant_category'].append(np.random.choice(['grocery', 'gas', 'restaurant', 'retail']))
        anomaly_data['hour_of_day'].append(np.random.randint(6, 23))
        anomaly_data['is_weekend'].append(np.random.choice([True, False]))
        anomaly_data['transaction_date'].append(
            base_date + timedelta(days=np.random.randint(0, 90), hours=np.random.randint(0, 24))
        )
    
    # High frequency anomalies
    for _ in range(anomaly_types['high_frequency']):
        anomaly_data['amount'].append(np.random.lognormal(6, 1))
        anomaly_data['customer_age'].append(np.random.randint(25, 55))
        anomaly_data['account_balance'].append(np.random.normal(15000, 8000))
        anomaly_data['transaction_count_daily'].append(np.random.randint(25, 100))
        anomaly_data['customer_type'].append(np.random.choice(['bronze', 'silver', 'gold']))
        anomaly_data['transaction_type'].append(np.random.choice(['transfer', 'purchase']))
        anomaly_data['merchant_category'].append(np.random.choice(['online', 'retail', 'other']))
        anomaly_data['hour_of_day'].append(np.random.randint(0, 24))
        anomaly_data['is_weekend'].append(np.random.choice([True, False]))
        anomaly_data['transaction_date'].append(
            base_date + timedelta(days=np.random.randint(0, 90), hours=np.random.randint(0, 24))
        )
    
    # Off-hours anomalies
    for _ in range(anomaly_types['off_hours']):
        anomaly_data['amount'].append(np.random.lognormal(6, 1))
        anomaly_data['customer_age'].append(np.random.randint(18, 75))
        anomaly_data['account_balance'].append(np.random.normal(15000, 8000))
        anomaly_data['transaction_count_daily'].append(np.random.poisson(3))
        anomaly_data['customer_type'].append(np.random.choice(['bronze', 'silver', 'gold']))
        anomaly_data['transaction_type'].append(np.random.choice(['withdrawal', 'transfer']))
        anomaly_data['merchant_category'].append(np.random.choice(['gas', 'online', 'other']))
        anomaly_data['hour_of_day'].append(np.random.choice([1, 2, 3, 4, 5, 23]))
        anomaly_data['is_weekend'].append(np.random.choice([True, False]))
        anomaly_data['transaction_date'].append(
            base_date + timedelta(days=np.random.randint(0, 90), hours=np.random.randint(0, 24))
        )
    
    # Future date anomalies
    for _ in range(anomaly_types['future_dates']):
        anomaly_data['amount'].append(np.random.lognormal(6, 1))
        anomaly_data['customer_age'].append(np.random.randint(18, 75))
        anomaly_data['account_balance'].append(np.random.normal(15000, 8000))
        anomaly_data['transaction_count_daily'].append(np.random.poisson(3))
        anomaly_data['customer_type'].append(np.random.choice(['bronze', 'silver', 'gold']))
        anomaly_data['transaction_type'].append(np.random.choice(['purchase', 'transfer']))
        anomaly_data['merchant_category'].append(np.random.choice(['grocery', 'retail', 'online']))
        anomaly_data['hour_of_day'].append(np.random.randint(6, 23))
        anomaly_data['is_weekend'].append(np.random.choice([True, False]))
        # Future dates (1-30 days in the future)
        anomaly_data['transaction_date'].append(
            datetime.now() + timedelta(days=np.random.randint(1, 30), hours=np.random.randint(0, 24))
        )
    
    # Create anomaly DataFrame
    anomaly_df = pd.DataFrame(anomaly_data)
    
    # Combine normal and anomalous data
    combined_df = pd.concat([normal_df, anomaly_df], ignore_index=True)
    
    # Shuffle the data
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Add some derived fields that might be useful for rules
    combined_df['amount_zscore'] = (combined_df['amount'] - combined_df['amount'].mean()) / combined_df['amount'].std()
    combined_df['is_high_amount'] = combined_df['amount'] > combined_df['amount'].quantile(0.95)
    combined_df['is_vip_customer'] = combined_df['customer_type'].isin(['gold', 'platinum'])
    
    # Round numeric columns
    combined_df['amount'] = combined_df['amount'].round(2)
    combined_df['account_balance'] = combined_df['account_balance'].round(2)
    combined_df['amount_zscore'] = combined_df['amount_zscore'].round(3)
    
    # Ensure age is positive
    combined_df['customer_age'] = combined_df['customer_age'].clip(lower=0)
    
    # Create sample_data directory
    os.makedirs("sample_data", exist_ok=True)
    
    # Save the comprehensive dataset
    combined_df.to_csv("sample_data/financial_transactions.csv", index=False)
    print(f"   ✅ Created financial_transactions.csv with {len(combined_df)} records")
    print(f"       - {n_normal} normal transactions")
    print(f"       - {n_anomalies} anomalous transactions")
    
    # Create a smaller test dataset
    test_df = combined_df.sample(n=500, random_state=42)
    test_df.to_csv("sample_data/test_transactions.csv", index=False)
    print(f"   ✅ Created test_transactions.csv with {len(test_df)} records")
    
    # Create a simple employee dataset for HR rules
    employee_data = {
        'employee_id': [f"EMP_{i:04d}" for i in range(1, 101)],
        'first_name': [f"Employee_{i}" for i in range(1, 101)],
        'last_name': [f"LastName_{i}" for i in range(1, 101)],
        'age': np.random.randint(22, 65, 100),
        'salary': np.random.normal(75000, 25000, 100).round(0),
        'department': np.random.choice(['IT', 'HR', 'Finance', 'Marketing', 'Operations'], 100),
        'years_experience': np.random.randint(0, 40, 100),
        'performance_score': np.random.uniform(1, 5, 100).round(2),
        'hire_date': [datetime.now() - timedelta(days=np.random.randint(30, 3650)) for _ in range(100)]
    }
    
    # Add some anomalies to employee data
    # Unrealistic salaries
    employee_data['salary'][0:5] = [200000, 300000, 500000, 1000000, 50]
    # Unrealistic experience vs age
    employee_data['years_experience'][5:10] = [50, 60, 45, 55, 40]
    employee_data['age'][5:10] = [25, 30, 28, 32, 35]
    
    employee_df = pd.DataFrame(employee_data)
    employee_df.to_csv("sample_data/employee_data.csv", index=False)
    print(f"   ✅ Created employee_data.csv with {len(employee_df)} records")

def create_quick_start_guide():
    """Create a comprehensive quick start guide"""
    
    guide_content = """# 🚀 Quick Start Guide: Enhanced Anomaly Detection with Rule Management

## Overview
This enhanced anomaly detection system allows you to create, test, and manage audit rules using natural language. Rules are automatically saved to a database and can be reused across sessions.

## 🎯 Key Features
- **Natural Language Rules**: Write rules in plain English
- **Persistent Storage**: Rules are saved to SQLite database
- **Dataset Association**: Rules are linked to specific datasets
- **Execution History**: Track when and how rules were executed
- **Performance Analytics**: Monitor rule effectiveness over time
- **Audit Trail**: Complete history of rule changes

## 📊 Sample Datasets
The setup created three sample datasets:

1. **financial_transactions.csv** (2,200 records)
   - Normal and anomalous financial transactions
   - Built-in anomalies: high amounts, suspicious ages, negative balances, etc.

2. **test_transactions.csv** (500 records)
   - Smaller subset for quick testing

3. **employee_data.csv** (100 records)
   - HR data with salary and experience anomalies

## 🧠 Rule Examples

### Financial Transaction Rules
```
amount is greater than 50000
customer_age is less than 18 or customer_age is greater than 90
account_balance is less than 0
transaction_count_daily is greater than 20
hour_of_day is between 1 and 5
transaction_date is future
amount is outlier
customer_type is rare
```

### Employee Data Rules
```
salary is greater than 200000
years_experience is greater than age
performance_score is less than 2
age is less than 22 or age is greater than 65
hire_date is future
department is not in ['IT', 'HR', 'Finance', 'Marketing', 'Operations']
```

## 🔧 Advanced Rule Syntax

### Operators
- **Comparison**: `greater than`, `less than`, `equals`, `between`
- **Text**: `contains`, `starts with`, `ends with`
- **Lists**: `in`, `not in`
- **Special**: `is outlier`, `is rare`, `is missing`, `is duplicate`
- **Dates**: `is future`, `is past`, `is weekend`
- **Logic**: `and`, `or`

### Complex Examples
```
amount is greater than 10000 and customer_type is not 'platinum'
salary is outlier or years_experience is greater than age
transaction_date is weekend and amount is greater than 5000
department is 'IT' and salary is less than 50000
```

## 📈 Using the System

### 1. Create New Rules
1. Go to Anomaly Detection page
2. Select "Smart Rule Management"
3. Choose your dataset
4. Click "Create New Rule" tab
5. Fill in rule details and test

### 2. Manage Existing Rules
1. View all rules for selected dataset
2. Execute, edit, or delete rules
3. View execution history
4. Monitor performance

### 3. Quick Execute
1. Select multiple rules
2. Run them all at once
3. Get consolidated results
4. Export findings

### 4. Analytics
1. View rule performance metrics
2. Track execution trends
3. Identify top-performing rules
4. Database maintenance

## 🎨 Best Practices

### Rule Naming
- Use descriptive names: "High Value Transactions"
- Include context: "Weekend Large Transfers"
- Be specific: "Salary Outliers Above 200K"

### Rule Description
- Explain the business purpose
- Document why this rule matters
- Include expected frequency

### Testing Strategy
1. Start with simple rules
2. Test on small datasets first
3. Verify results manually
4. Gradually add complexity

### Performance
- Monitor execution times
- Review false positive rates
- Update rules based on feedback
- Archive unused rules

## 🔍 Troubleshooting

### Common Issues
1. **Rule parsing fails**
   - Check column names match exactly
   - Use supported operators
   - Verify value formats

2. **No anomalies found**
   - Rule might be too restrictive
   - Check data types
   - Verify column contents

3. **Slow execution**
   - Complex rules take longer
   - Large datasets require time
   - Consider sampling

### Debug Tips
- Use the rule structure viewer
- Check execution details
- Review audit trail
- Test on smaller datasets

## 📚 Database Schema

### Tables
- **rules**: Rule definitions and metadata
- **rule_executions**: Execution history and results
- **datasets**: Dataset registry and statistics
- **rule_performance**: Performance metrics
- **rule_tags**: Rule categorization
- **rule_audit_trail**: Change history

### Key Fields
- `dataset_signature`: Links rules to specific datasets
- `auto_execute`: Rules run automatically on data load
- `execution_count`: How many times rule has run
- `anomaly_percentage`: Severity of findings

## 🚀 Next Steps

1. **Explore Sample Data**: Load the sample datasets and examine their structure
2. **Create First Rule**: Start with a simple rule like "amount is greater than 50000"
3. **Review Results**: Examine the anomalous records found
4. **Build Rule Library**: Create rules for different types of anomalies
5. **Monitor Performance**: Use analytics to track rule effectiveness
6. **Iterate and Improve**: Refine rules based on results

## 💡 Advanced Features

### Auto-Execute Rules
- Rules marked as auto-execute run when dataset is loaded
- Useful for continuous monitoring
- Results available immediately

### Rule Tags
- Categorize rules by type (audit, compliance, quality)
- Filter rules by tags
- Organize large rule libraries

### Audit Trail
- Track all rule changes
- See who modified what and when
- Maintain compliance records

### Performance Metrics
- Success rates
- Average execution times
- Anomaly detection rates
- Historical trends

## 🎯 Success Metrics

Track these KPIs to measure system effectiveness:
- Number of active rules per dataset
- Rule execution frequency
- Anomaly detection rate
- False positive rate (if feedback available)
- Time to detect issues
- User adoption rate

---

**Happy Rule Building! 🎉**

For more help, check the in-app examples and documentation.
"""
    
    with open("QUICK_START_ANOMALY_RULES.md", "w") as f:
        f.write(guide_content)
    
    print("   ✅ Created comprehensive quick start guide: QUICK_START_ANOMALY_RULES.md")

if __name__ == "__main__":
    success = setup_enhanced_anomaly_system()
    
    if success:
        print("\n🎉 Setup completed successfully!")
        print("   You can now run: streamlit run app/Home.py")
    else:
        print("\n⚠️  Setup completed with issues. Please resolve them before proceeding.")
    
    input("\nPress Enter to exit...")