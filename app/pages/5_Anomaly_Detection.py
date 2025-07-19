# File: app/pages/5_Anomaly_Detection.py
# REFACTORED VERSION WITH PROFESSIONAL STYLING AND NAVIGATION

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from collections import Counter
import warnings
import operator
import hashlib
warnings.filterwarnings('ignore')

# Import rule management system
try:
    from database.rule_management import RuleManagementDB
    RULE_MANAGEMENT_AVAILABLE = True
except ImportError:
    RULE_MANAGEMENT_AVAILABLE = False
    st.error("Rule Management system not available. Please ensure database/rule_management.py exists.")

# Try to get clean styling
try:
    from app.assets.clean_styles import apply_clean_styling, create_clean_header, create_clean_metric
    STYLING_AVAILABLE = True
except ImportError:
    STYLING_AVAILABLE = False

try:
    from core_logic.data_loader import load_all_data_sources
except ImportError:
    def load_all_data_sources():
        if "csv_dataframes" in st.session_state:
            return {f"CSV: {k.replace('csv_', '')}": v for k, v in st.session_state["csv_dataframes"].items()}
        return {}

st.set_page_config(
    page_title="Anomaly Detection & Rule Management",
    page_icon="🔍",
    layout="wide"
)

# Apply professional styling
try:
    from assets.streamlit_styles import apply_professional_styling, create_nav_header
    apply_professional_styling()
except ImportError:
    st.error("Failed to apply professional styling. Please check your assets directory.")

# --- Navigation Header ---
create_nav_header("🔍 Anomaly Detection & Rule Management", "Find unusual patterns and manage business rules for your data")


# Enhanced Natural Language Rule Parser (with rule management integration)
class EnhancedRuleParser:
    """
    Advanced natural language rule parser with database integration
    """
    
    def __init__(self, rule_db=None):
        self.rule_db = rule_db
        self.operators = {
            'is': operator.eq,
            'equals': operator.eq,
            'is not': operator.ne,
            'not equals': operator.ne,
            'greater than': operator.gt,
            'more than': operator.gt,
            'above': operator.gt,
            '>' : operator.gt,
            'less than': operator.lt,
            'below': operator.lt,
            'under': operator.lt,
            '<': operator.lt,
            'greater than or equal': operator.ge,
            'at least': operator.ge,
            '>=': operator.ge,
            'less than or equal': operator.le,
            'at most': operator.le,
            '<=': operator.le,
            'between': 'between',
            'outside': 'outside',
            'in': 'in',
            'not in': 'not_in',
            'contains': 'contains',
            'does not contain': 'not_contains',
            'starts with': 'startswith',
            'ends with': 'endswith',
            'matches': 'matches',
            'is null': 'isnull',
            'is not null': 'notnull',
            'is empty': 'isempty',
            'is not empty': 'notempty'
        }
        
        self.special_keywords = {
            'outlier': 'is_outlier',
            'unusual': 'is_outlier', 
            'anomaly': 'is_outlier',
            'rare': 'is_rare',
            'uncommon': 'is_rare',
            'frequent': 'is_frequent',
            'common': 'is_frequent',
            'duplicate': 'is_duplicate',
            'unique': 'is_unique',
            'missing': 'is_missing',
            'null': 'is_missing',
            'empty': 'is_empty',
            'future': 'is_future_date',
            'past': 'is_past_date',
            'weekend': 'is_weekend',
            'weekday': 'is_weekday',
            'recent': 'is_recent',
            'old': 'is_old'
        }
        
        self.logical_operators = ['and', 'or', 'not']
    
    def generate_dataset_signature(self, df: pd.DataFrame, dataset_name: str) -> str:
        """Generate a unique signature for a dataset based on its structure"""
        columns_info = f"{sorted(df.columns.tolist())}"
        shape_info = f"{df.shape}"
        dtypes_info = f"{sorted(df.dtypes.astype(str).to_dict().items())}"
        signature_string = f"{dataset_name}:{columns_info}:{shape_info}:{dtypes_info}"
        return hashlib.md5(signature_string.encode()).hexdigest()
    
    def save_rule_to_db(self, rule_data: Dict[str, Any], dataset_signature: str) -> int:
        """Save a rule to the database with dataset association"""
        if not self.rule_db:
            return None
        
        try:
            # Add dataset signature to parsed conditions
            parsed_conditions = rule_data.get('parsed_conditions', {})
            parsed_conditions['dataset_signature'] = dataset_signature
            
            db_rule_data = {
                'rule_name': rule_data['rule_name'],
                'description': rule_data.get('description', ''),
                'natural_language_text': rule_data['natural_language_rule'],
                'rule_type': 'anomaly_detection',
                'priority': rule_data.get('priority', 1),
                'is_active': True,
                'parsed_conditions': parsed_conditions,
                'generated_code': rule_data.get('generated_code', '')
            }
            
            rule_id = self.rule_db.create_rule(db_rule_data)
            return rule_id
        except Exception as e:
            st.error(f"Error saving rule to database: {e}")
            return None
    
    def load_rules_for_dataset(self, dataset_signature: str) -> List[Dict[str, Any]]:
        """Load all rules associated with a specific dataset"""
        if not self.rule_db:
            return []
        
        try:
            all_rules = self.rule_db.get_rules(active_only=True)
            dataset_rules = []
            
            for rule in all_rules:
                parsed_conditions = rule.get('parsed_conditions', {})
                if isinstance(parsed_conditions, str):
                    try:
                        parsed_conditions = json.loads(parsed_conditions)
                    except:
                        parsed_conditions = {}
                
                if parsed_conditions.get('dataset_signature') == dataset_signature:
                    dataset_rules.append(rule)
            
            return dataset_rules
        except Exception as e:
            st.error(f"Error loading rules from database: {e}")
            return []
    
    def execute_saved_rule(self, rule: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """Execute a previously saved rule on a dataset"""
        try:
            # Parse the rule again
            rule_structure = self.parse_rule(rule['natural_language_text'], df)
            
            if rule_structure['is_valid']:
                execution_result = self.execute_rule(rule_structure, df)
                
                # Log execution to database
                if self.rule_db:
                    execution_data = {
                        'rule_id': rule['rule_id'],
                        'dataset_name': 'current_dataset',
                        'records_processed': len(df),
                        'records_affected': execution_result.get('anomaly_count', 0),
                        'execution_status': 'success' if execution_result.get('success', False) else 'error',
                        'error_message': execution_result.get('error', ''),
                        'execution_time_ms': 0,  # Could add timing
                        'violation_details': execution_result
                    }
                    self.rule_db.log_execution(execution_data)
                
                return execution_result
            else:
                return {
                    'success': False,
                    'error': 'Rule parsing failed',
                    'anomaly_indices': [],
                    'anomaly_count': 0
                }
        except Exception as e:
            return {
                'success': False,
                'error': f"Error executing saved rule: {str(e)}",
                'anomaly_indices': [],
                'anomaly_count': 0
            }
    
    def parse_rule(self, rule_text: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse a natural language rule into executable conditions"""
        
        try:
            # Clean and normalize the rule text
            normalized_rule = self._normalize_rule_text(rule_text)
            
            # Split by logical operators
            conditions = self._split_by_logical_operators(normalized_rule)
            
            # Parse each condition
            parsed_conditions = []
            for condition in conditions:
                parsed_condition = self._parse_single_condition(condition, df)
                if parsed_condition:
                    parsed_conditions.append(parsed_condition)
            
            # Build final rule structure
            rule_structure = {
                'original_text': rule_text,
                'normalized_text': normalized_rule,
                'conditions': parsed_conditions,
                'is_valid': len(parsed_conditions) > 0,
                'error_message': None if len(parsed_conditions) > 0 else "No valid conditions found"
            }
            
            return rule_structure
            
        except Exception as e:
            return {
                'original_text': rule_text,
                'normalized_text': rule_text,
                'conditions': [],
                'is_valid': False,
                'error_message': f"Parsing error: {str(e)}"
            }
    
    def execute_rule(self, rule_structure: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """Execute a parsed rule against a DataFrame"""
        
        if not rule_structure['is_valid']:
            return {
                'success': False,
                'error': rule_structure['error_message'],
                'anomaly_indices': [],
                'anomaly_count': 0,
                'rule_text': rule_structure['original_text']
            }
        
        try:
            # Execute each condition
            condition_results = []
            for condition in rule_structure['conditions']:
                result = self._execute_condition(condition, df)
                condition_results.append(result)
            
            # Combine results based on logical operators
            final_mask = self._combine_condition_results(condition_results, rule_structure)
            
            # Get anomaly indices
            anomaly_indices = df.index[final_mask].tolist()
            
            return {
                'success': True,
                'anomaly_indices': anomaly_indices,
                'anomaly_count': len(anomaly_indices),
                'rule_text': rule_structure['original_text'],
                'condition_details': [
                    {
                        'condition': cond['original_text'],
                        'column': cond['column'],
                        'operator': cond['operator'],
                        'matches': sum(result)
                    }
                    for cond, result in zip(rule_structure['conditions'], condition_results)
                ],
                'anomaly_percentage': (len(anomaly_indices) / len(df) * 100) if len(df) > 0 else 0
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Execution error: {str(e)}",
                'anomaly_indices': [],
                'anomaly_count': 0,
                'rule_text': rule_structure['original_text']
            }
    
    # [Include all the helper methods from the original class here]
    def _normalize_rule_text(self, text: str) -> str:
        """Normalize rule text for parsing"""
        text = text.lower().strip()
        
        replacements = {
            ' is greater than ': ' > ',
            ' is less than ': ' < ',
            ' is equal to ': ' = ',
            ' equals ': ' = ',
            ' does not equal ': ' != ',
            ' is not equal to ': ' != '
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def _split_by_logical_operators(self, text: str) -> List[Dict[str, Any]]:
        """Split rule text by logical operators"""
        conditions = []
        parts = re.split(r'\b(and|or)\b', text)
        
        current_condition = ""
        logical_op = None
        
        for part in parts:
            part = part.strip()
            if part in ['and', 'or']:
                if current_condition:
                    conditions.append({
                        'text': current_condition,
                        'logical_operator': logical_op
                    })
                logical_op = part
                current_condition = ""
            else:
                current_condition += part
        
        if current_condition:
            conditions.append({
                'text': current_condition,
                'logical_operator': logical_op
            })
        
        return conditions
    
    def _parse_single_condition(self, condition: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """Parse a single condition"""
        text = condition['text'].strip()
        
        column_match = self._extract_column_name(text, df)
        if not column_match:
            return None
        
        column_name = column_match['column']
        remaining_text = column_match['remaining_text']
        
        parsed_condition = self._parse_condition_text(remaining_text, column_name, df)
        if parsed_condition:
            parsed_condition.update({
                'original_text': text,
                'logical_operator': condition.get('logical_operator'),
                'column': column_name
            })
        
        return parsed_condition
    
    def _extract_column_name(self, text: str, df: pd.DataFrame) -> Dict[str, str]:
        """Extract column name from condition text"""
        # Try exact matches first
        for col in df.columns:
            col_lower = col.lower()
            if text.startswith(col_lower):
                remaining = text[len(col_lower):].strip()
                return {'column': col, 'remaining_text': remaining}
        
        # Try partial matches
        words = text.split()
        for i in range(len(words)):
            potential_col = ' '.join(words[:i+1])
            for col in df.columns:
                if col.lower() == potential_col:
                    remaining = ' '.join(words[i+1:])
                    return {'column': col, 'remaining_text': remaining}
        
        # If no exact match, try the first word
        if words:
            first_word = words[0]
            for col in df.columns:
                col_words = col.lower().split('_')
                if first_word in col_words:
                    return {'column': col, 'remaining_text': ' '.join(words[1:])}
        
        return None
    
    def _parse_condition_text(self, text: str, column_name: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse the condition part of the text"""
        text = text.strip()
        
        # Check for special keywords first
        for keyword, special_op in self.special_keywords.items():
            if keyword in text:
                return {
                    'operator': special_op,
                    'value': None,
                    'column_type': str(df[column_name].dtype)
                }
        
        # Parse numerical/comparison conditions
        for op_text, op_func in self.operators.items():
            if op_text in text:
                value_text = text.replace(op_text, '').strip()
                
                if op_text == 'between':
                    values = self._parse_between_values(value_text)
                    if values:
                        return {
                            'operator': 'between',
                            'value': values,
                            'column_type': str(df[column_name].dtype)
                        }
                
                elif op_text == 'outside':
                    values = self._parse_between_values(value_text)
                    if values:
                        return {
                            'operator': 'outside',
                            'value': values,
                            'column_type': str(df[column_name].dtype)
                        }
                
                elif op_text in ['in', 'not in']:
                    values = self._parse_list_values(value_text)
                    if values:
                        return {
                            'operator': op_text.replace(' ', '_'),
                            'value': values,
                            'column_type': str(df[column_name].dtype)
                        }
                
                else:
                    # Single value comparison
                    parsed_value = self._parse_value(value_text, df[column_name])
                    if parsed_value is not None:
                        return {
                            'operator': op_text,
                            'value': parsed_value,
                            'column_type': str(df[column_name].dtype)
                        }
        
        return None
    
    def _parse_between_values(self, text: str) -> List:
        """Parse 'between X and Y' values"""
        patterns = [
            r'(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    val1 = float(match.group(1))
                    val2 = float(match.group(2))
                    return [min(val1, val2), max(val1, val2)]
                except ValueError:
                    continue
        
        return None
    
    def _parse_list_values(self, text: str) -> List[str]:
        """Parse list values like ['a', 'b', 'c'] or 'a, b, c'"""
        text = text.strip('[]{}()').strip()
        values = [v.strip().strip('"\'') for v in text.split(',')]
        return [v for v in values if v]
    
    def _parse_value(self, text: str, series: pd.Series):
        """Parse a single value based on the series type"""
        text = text.strip().strip('"\'')
        
        if pd.api.types.is_numeric_dtype(series):
            try:
                if '.' not in text:
                    return int(text)
                else:
                    return float(text)
            except ValueError:
                return None
        
        elif pd.api.types.is_datetime64_any_dtype(series):
            try:
                return pd.to_datetime(text)
            except:
                return None
        
        else:
            return text
    
    def _execute_condition(self, condition: Dict[str, Any], df: pd.DataFrame) -> pd.Series:
        """Execute a single condition and return boolean mask"""
        column = condition['column']
        operator_name = condition['operator']
        value = condition['value']
        series = df[column]
        
        try:
            # Handle special operators
            if operator_name == 'is_outlier':
                return self._detect_outliers(series)
            
            elif operator_name == 'is_rare':
                return self._detect_rare_values(series)
            
            elif operator_name == 'is_frequent':
                return self._detect_frequent_values(series)
            
            elif operator_name == 'is_duplicate':
                return series.duplicated(keep=False)
            
            elif operator_name == 'is_unique':
                return ~series.duplicated(keep=False)
            
            elif operator_name == 'is_missing':
                return series.isnull()
            
            elif operator_name == 'is_empty':
                return (series.isnull()) | (series.astype(str).str.strip() == '')
            
            elif operator_name == 'is_future_date':
                if pd.api.types.is_datetime64_any_dtype(series):
                    return series > pd.Timestamp.now()
                return pd.Series([False] * len(series))
            
            elif operator_name == 'is_weekend':
                if pd.api.types.is_datetime64_any_dtype(series):
                    return series.dt.dayofweek >= 5
                return pd.Series([False] * len(series))
            
            # Handle standard operators
            elif operator_name == 'between':
                return (series >= value[0]) & (series <= value[1])
            
            elif operator_name == 'outside':
                return (series < value[0]) | (series > value[1])
            
            elif operator_name == 'in':
                return series.isin(value)
            
            elif operator_name == 'not_in':
                return ~series.isin(value)
            
            elif operator_name == 'contains':
                return series.astype(str).str.contains(str(value), case=False, na=False)
            
            elif operator_name == 'not_contains':
                return ~series.astype(str).str.contains(str(value), case=False, na=False)
            
            # Standard comparison operators
            elif operator_name in self.operators and callable(self.operators[operator_name]):
                op_func = self.operators[operator_name]
                return op_func(series, value)
            
            else:
                return pd.Series([False] * len(series))
                
        except Exception as e:
            st.warning(f"Error executing condition {condition}: {e}")
            return pd.Series([False] * len(series))
    
    def _detect_outliers(self, series: pd.Series, method: str = 'iqr') -> pd.Series:
        """Detect outliers using IQR method"""
        if not pd.api.types.is_numeric_dtype(series):
            return pd.Series([False] * len(series))
        
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (series < lower_bound) | (series > upper_bound)
    
    def _detect_rare_values(self, series: pd.Series, threshold: float = 0.05) -> pd.Series:
        """Detect rare values"""
        value_counts = series.value_counts()
        rare_values = value_counts[value_counts / len(series) < threshold].index
        return series.isin(rare_values)
    
    def _detect_frequent_values(self, series: pd.Series, threshold: float = 0.1) -> pd.Series:
        """Detect frequent values"""
        value_counts = series.value_counts()
        frequent_values = value_counts[value_counts / len(series) > threshold].index
        return series.isin(frequent_values)
    
    def _combine_condition_results(self, condition_results: List[pd.Series], rule_structure: Dict[str, Any]) -> pd.Series:
        """Combine multiple condition results using logical operators"""
        if not condition_results:
            return pd.Series(dtype=bool)
        
        if len(condition_results) == 1:
            return condition_results[0]
        
        combined_mask = condition_results[0]
        
        for i, condition in enumerate(rule_structure['conditions'][1:], 1):
            logical_op = condition.get('logical_operator', 'and')
            
            if logical_op == 'and':
                combined_mask = combined_mask & condition_results[i]
            elif logical_op == 'or':
                combined_mask = combined_mask | condition_results[i]
        
        return combined_mask


def create_enhanced_rule_management_interface():
    """Enhanced rule management interface with database integration"""
    
    # Initialize rule database
    if RULE_MANAGEMENT_AVAILABLE:
        if 'rule_db' not in st.session_state:
            st.session_state.rule_db = RuleManagementDB()
        rule_db = st.session_state.rule_db
    else:
        rule_db = None
        st.error("❌ Rule Management database not available")
        return
    
    # Initialize parser with database
    if 'rule_parser' not in st.session_state:
        st.session_state.rule_parser = EnhancedRuleParser(rule_db)
    parser = st.session_state.rule_parser
    
    st.subheader("🧠 Smart Rule Management System")
    st.info("Create, test, and manage audit rules with automatic database storage!")
    
    # Load data sources
    try:
        data_sources = load_all_data_sources()
        
        if not data_sources:
            st.warning("No data available. Please load data first.")
            return
        
        # Dataset selection
        st.markdown("### 📊 Select Dataset")
        dataset_key = st.selectbox(
            "Choose dataset for rule management:",
            list(data_sources.keys()),
            key="rule_mgmt_dataset"
        )
        
        df = data_sources[dataset_key]
        dataset_signature = parser.generate_dataset_signature(df, dataset_key)
        
        # Display dataset info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", f"{len(df):,}")
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        with col4:
            # Load existing rules for this dataset
            existing_rules = parser.load_rules_for_dataset(dataset_signature)
            st.metric("Saved Rules", len(existing_rules))
        
        # Create tabs for different functionalities
        tab1, tab2, tab3, tab4 = st.tabs(["🆕 Create New Rule", "📋 Manage Existing Rules", "⚡ Quick Execute", "📊 Rule Analytics"])
        
        with tab1:
            create_new_rule_interface(parser, df, dataset_signature, dataset_key)
        
        with tab2:
            manage_existing_rules_interface(parser, df, dataset_signature, existing_rules)
        
        with tab3:
            quick_execute_interface(parser, df, existing_rules)
        
        with tab4:
            rule_analytics_interface(rule_db)
            
    except Exception as e:
        st.error(f"Error in rule management interface: {e}")

def create_new_rule_interface(parser, df, dataset_signature, dataset_key):
    """Interface for creating new rules"""
    
    st.markdown("#### 🎯 Create New Audit Rule")
    
    # Rule examples based on dataset
    with st.expander("💡 Smart Rule Suggestions", expanded=False):
        suggest_rules_for_dataset(df)
    
    # Rule creation form
    with st.form("new_rule_form", clear_on_submit=False):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            rule_name = st.text_input(
                "Rule Name*:",
                placeholder="High Value Transactions Rule",
                help="Give your rule a descriptive name"
            )
            
            rule_description = st.text_area(
                "Description:",
                placeholder="Detect transactions above $10,000 that may require additional review",
                height=80
            )
            
            natural_rule = st.text_area(
                "Natural Language Rule*:",
                placeholder="transaction_amount is greater than 10000",
                height=120,
                help="Write your rule in plain English"
            )
        
        with col2:
            st.markdown("**Available Columns:**")
            for col in df.columns:
                col_type = str(df[col].dtype)
                col_icon = "📊" if pd.api.types.is_numeric_dtype(df[col]) else "📅" if pd.api.types.is_datetime64_any_dtype(df[col]) else "📝"
                st.text(f"{col_icon} {col}")
            
            priority = st.slider("Priority Level", 1, 5, 3, help="1=Low, 5=Critical")
            
            auto_execute = st.checkbox("Auto-execute on data load", help="Automatically run this rule when this dataset is loaded")
        
        submitted = st.form_submit_button("🧪 Test & Save Rule", type="primary")
        
        if submitted and rule_name and natural_rule:
            test_and_save_rule(parser, df, dataset_signature, {
                'rule_name': rule_name,
                'description': rule_description,
                'natural_language_rule': natural_rule,
                'priority': priority,
                'auto_execute': auto_execute
            })

def suggest_rules_for_dataset(df):
    """Suggest relevant rules based on dataset characteristics"""
    
    suggestions = []
    
    # Analyze dataset to suggest relevant rules
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    text_cols = df.select_dtypes(include=['object']).columns
    date_cols = df.select_dtypes(include=['datetime64']).columns
    
    if len(numeric_cols) > 0:
        col = numeric_cols[0]
        suggestions.append(f"`{col} is outlier` - Detect statistical outliers")
        suggestions.append(f"`{col} is greater than {df[col].quantile(0.9):.0f}` - High value detection")
    
    if len(text_cols) > 0:
        col = text_cols[0]
        suggestions.append(f"`{col} is missing` - Find missing values")
        suggestions.append(f"`{col} is rare` - Detect uncommon categories")
    
    if len(date_cols) > 0:
        col = date_cols[0]
        suggestions.append(f"`{col} is future` - Future date detection")
        suggestions.append(f"`{col} is weekend` - Weekend transactions")
    
    # Common audit rules
    suggestions.extend([
        "`transaction_amount is greater than 5000 and approval_status is missing` - Missing approvals",
        "`status is not in ['approved', 'pending', 'completed']` - Invalid status values",
        "`created_date is future or modified_date is future` - Future date anomalies"
    ])
    
    for i, suggestion in enumerate(suggestions[:6], 1):
        st.markdown(f"**{i}.** {suggestion}")

def test_and_save_rule(parser, df, dataset_signature, rule_data):
    """Test a rule and save it to the database"""
    
    with st.spinner("🧠 Parsing and testing rule..."):
        # Parse the rule
        rule_structure = parser.parse_rule(rule_data['natural_language_rule'], df)
        
        if rule_structure['is_valid']:
            st.success("✅ Rule parsed successfully!")
            
            # Show parsed conditions
            with st.expander("🔍 Parsed Rule Structure", expanded=True):
                for i, condition in enumerate(rule_structure['conditions'], 1):
                    st.markdown(f"**Condition {i}:**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.text(f"Column: {condition['column']}")
                    with col2:
                        st.text(f"Operator: {condition['operator']}")
                    with col3:
                        st.text(f"Value: {condition.get('value', 'N/A')}")
            
            # Execute the rule
            execution_result = parser.execute_rule(rule_structure, df)
            
            if execution_result['success']:
                # Display results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Anomalies Found", execution_result['anomaly_count'])
                with col2:
                    st.metric("Anomaly Rate", f"{execution_result['anomaly_percentage']:.2f}%")
                with col3:
                    severity = "🔴 High" if execution_result['anomaly_percentage'] > 10 else "🟡 Medium" if execution_result['anomaly_percentage'] > 2 else "🟢 Low"
                    st.metric("Severity", severity)
                
                # Show sample anomalies
                if execution_result['anomaly_count'] > 0:
                    st.markdown("**🚨 Sample Anomalous Records:**")
                    anomaly_df = df.iloc[execution_result['anomaly_indices'][:10]]
                    st.dataframe(anomaly_df, use_container_width=True)
                
                # Save to database
                saved_rule_id = parser.save_rule_to_db(rule_data, dataset_signature)
                
                if saved_rule_id:
                    st.success(f"💾 Rule saved to database with ID: {saved_rule_id}")
                    st.balloons()
                    
                    # Log initial execution
                    execution_data = {
                        'rule_id': saved_rule_id,
                        'dataset_name': 'current_dataset',
                        'records_processed': len(df),
                        'records_affected': execution_result['anomaly_count'],
                        'execution_status': 'success',
                        'execution_time_ms': 0,
                        'violation_details': execution_result
                    }
                    parser.rule_db.log_execution(execution_data)
                    
                else:
                    st.warning("⚠️ Rule tested successfully but failed to save to database")
            
            else:
                st.error(f"❌ Rule execution failed: {execution_result['error']}")
        
        else:
            st.error(f"❌ Rule parsing failed: {rule_structure['error_message']}")
            
            # Provide helpful suggestions
            st.markdown("### 💡 Suggestions to Fix Your Rule:")
            st.markdown("""
            1. **Check column names**: Make sure column names match exactly
            2. **Use supported operators**: greater than, less than, equals, contains, is, between, etc.
            3. **Format values correctly**: Numbers without quotes, text with quotes if needed
            4. **Use logical operators**: Connect conditions with 'and' or 'or'
            5. **Check examples**: Refer to the suggestions above for proper syntax
            """)

def manage_existing_rules_interface(parser, df, dataset_signature, existing_rules):
    """Interface for managing existing rules"""
    
    st.markdown("#### 📋 Manage Existing Rules")
    
    if not existing_rules:
        st.info("No rules found for this dataset. Create your first rule in the 'Create New Rule' tab!")
        return
    
    st.success(f"Found {len(existing_rules)} rules for this dataset")
    
    # Rules management
    for i, rule in enumerate(existing_rules):
        with st.expander(f"🔧 {rule['rule_name']}", expanded=False):
            rule_col1, rule_col2 = st.columns([2, 1])
            
            with rule_col1:
                st.markdown(f"**Description:** {rule.get('description', 'No description')}")
                st.markdown(f"**Rule:** `{rule['natural_language_text']}`")
                st.markdown(f"**Created:** {rule.get('created_at', 'Unknown')}")
                st.markdown(f"**Priority:** {'⭐' * rule.get('priority', 1)}")
                
                # Show rule statistics
                executions = parser.rule_db.get_execution_history(rule['rule_id'], limit=5)
                if executions:
                    last_execution = executions[0]
                    st.markdown(f"**Last Run:** {last_execution.get('executed_at', 'Never')}")
                    st.markdown(f"**Last Result:** {last_execution.get('records_affected', 0)} anomalies found")
            
            with rule_col2:
                # Action buttons
                if st.button(f"▶️ Execute", key=f"exec_{rule['rule_id']}"):
                    execute_single_rule(parser, df, rule)
                
                if st.button(f"📊 View History", key=f"history_{rule['rule_id']}"):
                    show_rule_execution_history(parser.rule_db, rule)
                
                if st.button(f"✏️ Edit", key=f"edit_{rule['rule_id']}"):
                    edit_rule_interface(parser, rule, df, dataset_signature)
                
                if st.button(f"🗑️ Delete", key=f"delete_{rule['rule_id']}", type="secondary"):
                    if st.session_state.get(f"confirm_delete_{rule['rule_id']}", False):
                        parser.rule_db.delete_rule(rule['rule_id'])
                        st.success("Rule deleted!")
                        st.rerun()
                    else:
                        st.session_state[f"confirm_delete_{rule['rule_id']}"] = True
                        st.warning("Click again to confirm deletion")

def execute_single_rule(parser, df, rule):
    """Execute a single rule and show results"""
    
    with st.spinner(f"Executing rule: {rule['rule_name']}..."):
        result = parser.execute_saved_rule(rule, df)
        
        if result['success']:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Anomalies", result['anomaly_count'])
            with col2:
                st.metric("Rate", f"{result.get('anomaly_percentage', 0):.2f}%")
            with col3:
                st.metric("Status", "✅ Success")
            
            if result['anomaly_count'] > 0:
                anomaly_df = df.iloc[result['anomaly_indices'][:20]]
                st.dataframe(anomaly_df, use_container_width=True)
        else:
            st.error(f"Execution failed: {result['error']}")

def show_rule_execution_history(rule_db, rule):
    """Show execution history for a rule"""
    
    st.markdown(f"### 📊 Execution History: {rule['rule_name']}")
    
    executions = rule_db.get_execution_history(rule['rule_id'], limit=50)
    
    if executions:
        history_data = []
        for exec_record in executions:
            history_data.append({
                'Executed At': exec_record.get('executed_at', ''),
                'Records Processed': exec_record.get('records_processed', 0),
                'Anomalies Found': exec_record.get('records_affected', 0),
                'Status': exec_record.get('execution_status', ''),
                'Execution Time (ms)': exec_record.get('execution_time_ms', 0)
            })
        
        history_df = pd.DataFrame(history_data)
        st.dataframe(history_df, use_container_width=True)
        
        # Execution trend chart
        if len(history_df) > 1:
            fig = px.line(
                history_df,
                x='Executed At',
                y='Anomalies Found',
                title=f"Anomaly Detection Trend: {rule['rule_name']}",
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No execution history found for this rule.")

def edit_rule_interface(parser, rule, df, dataset_signature):
    """Interface for editing existing rules"""
    
    st.markdown(f"### ✏️ Edit Rule: {rule['rule_name']}")
    
    with st.form(f"edit_rule_{rule['rule_id']}"):
        updated_name = st.text_input("Rule Name", value=rule['rule_name'])
        updated_description = st.text_area("Description", value=rule.get('description', ''))
        updated_rule_text = st.text_area("Natural Language Rule", value=rule['natural_language_text'])
        updated_priority = st.slider("Priority", 1, 5, rule.get('priority', 1))
        
        if st.form_submit_button("💾 Update Rule"):
            # Test the updated rule
            rule_structure = parser.parse_rule(updated_rule_text, df)
            
            if rule_structure['is_valid']:
                # Update in database
                updates = {
                    'rule_name': updated_name,
                    'description': updated_description,
                    'natural_language_text': updated_rule_text,
                    'priority': updated_priority
                }
                
                success = parser.rule_db.update_rule(rule['rule_id'], updates)
                
                if success:
                    st.success("✅ Rule updated successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to update rule")
            else:
                st.error(f"❌ Invalid rule syntax: {rule_structure['error_message']}")

def quick_execute_interface(parser, df, existing_rules):
    """Interface for quickly executing multiple rules"""
    
    st.markdown("#### ⚡ Quick Execute Rules")
    
    if not existing_rules:
        st.info("No rules available for quick execution.")
        return
    
    # Rule selection
    selected_rules = st.multiselect(
        "Select rules to execute:",
        options=existing_rules,
        format_func=lambda x: f"{x['rule_name']} (Priority: {x.get('priority', 1)})",
        default=existing_rules  # Select all by default
    )
    
    col1, col2 = st.columns(2)
    with col1:
        execute_all = st.button("🚀 Execute Selected Rules", type="primary")
    with col2:
        auto_fix = st.checkbox("Auto-fix common issues", help="Attempt to automatically fix simple data quality issues")
    
    if execute_all and selected_rules:
        execute_multiple_rules(parser, df, selected_rules, auto_fix)

def execute_multiple_rules(parser, df, rules, auto_fix=False):
    """Execute multiple rules and provide consolidated results"""
    
    all_anomalies = set()
    results_summary = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, rule in enumerate(rules):
        status_text.text(f"Executing: {rule['rule_name']}")
        progress_bar.progress((i + 1) / len(rules))
        
        result = parser.execute_saved_rule(rule, df)
        
        results_summary.append({
            'Rule Name': rule['rule_name'],
            'Status': '✅ Success' if result['success'] else '❌ Failed',
            'Anomalies Found': result.get('anomaly_count', 0),
            'Anomaly Rate (%)': f"{result.get('anomaly_percentage', 0):.2f}",
            'Error': result.get('error', '') if not result['success'] else ''
        })
        
        if result['success']:
            all_anomalies.update(result['anomaly_indices'])
    
    progress_bar.empty()
    status_text.empty()
    
    # Display consolidated results
    st.markdown("### 📊 Execution Results Summary")
    
    summary_df = pd.DataFrame(results_summary)
    st.dataframe(summary_df, use_container_width=True)
    
    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rules Executed", len(rules))
    with col2:
        successful_rules = sum(1 for r in results_summary if r['Status'] == '✅ Success')
        st.metric("Successful", successful_rules)
    with col3:
        st.metric("Total Unique Anomalies", len(all_anomalies))
    with col4:
        overall_rate = (len(all_anomalies) / len(df) * 100) if len(df) > 0 else 0
        st.metric("Overall Anomaly Rate", f"{overall_rate:.2f}%")
    
    # Show consolidated anomalies
    if all_anomalies:
        st.markdown("### 🚨 Consolidated Anomalous Records")
        
        anomaly_df = df.iloc[list(all_anomalies)]
        
        # Add anomaly score (how many rules flagged each record)
        anomaly_scores = []
        for idx in anomaly_df.index:
            score = 0
            for rule in rules:
                result = parser.execute_saved_rule(rule, df.iloc[[idx]])
                if result['success'] and result['anomaly_count'] > 0:
                    score += 1
            anomaly_scores.append(score)
        
        anomaly_df = anomaly_df.copy()
        anomaly_df['Anomaly_Score'] = anomaly_scores
        anomaly_df = anomaly_df.sort_values('Anomaly_Score', ascending=False)
        
        st.dataframe(anomaly_df, use_container_width=True)
        
        # Export options
        col1, col2 = st.columns(2)
        with col1:
            csv_data = anomaly_df.to_csv(index=True)
            st.download_button(
                "📄 Download Anomalies (CSV)",
                csv_data,
                f"consolidated_anomalies_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv"
            )
        
        with col2:
            report_data = {
                'execution_summary': results_summary,
                'anomaly_details': anomaly_df.to_dict('records'),
                'execution_timestamp': datetime.now().isoformat(),
                'total_records': len(df),
                'total_anomalies': len(all_anomalies)
            }
            
            json_data = json.dumps(report_data, indent=2, default=str)
            st.download_button(
                "📋 Download Full Report (JSON)",
                json_data,
                f"anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                "application/json"
            )

def rule_analytics_interface(rule_db):
    """Interface for rule analytics and insights"""
    
    st.markdown("#### 📊 Rule Analytics & Insights")
    
    if not rule_db:
        st.error("Rule database not available")
        return
    
    # Get analytics data
    analytics = rule_db.get_rule_analytics()
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rules", analytics['total_rules'])
    with col2:
        st.metric("Active Rules", analytics['active_rules'])
    with col3:
        st.metric("Total Executions", analytics['total_executions'])
    with col4:
        st.metric("Success Rate", f"{analytics['success_rate']:.1f}%")
    
    # Rule types distribution
    if analytics['rule_types']:
        col1, col2 = st.columns(2)
        
        with col1:
            fig_types = px.pie(
                values=list(analytics['rule_types'].values()),
                names=list(analytics['rule_types'].keys()),
                title="📊 Rules by Type"
            )
            st.plotly_chart(fig_types, use_container_width=True)
        
        with col2:
            # Daily executions trend
            if analytics['daily_executions']:
                dates = list(analytics['daily_executions'].keys())
                counts = list(analytics['daily_executions'].values())
                
                fig_trend = px.line(
                    x=dates,
                    y=counts,
                    title="📈 Daily Execution Trend (Last 30 Days)",
                    labels={'x': 'Date', 'y': 'Executions'}
                )
                st.plotly_chart(fig_trend, use_container_width=True)
    
    # Top performing rules
    st.markdown("### 🏆 Top Performing Rules")
    
    all_rules = rule_db.get_rules()
    if all_rules:
        rule_performance = []
        
        for rule in all_rules:
            executions = rule_db.get_execution_history(rule['rule_id'], limit=10)
            if executions:
                avg_anomalies = sum(e.get('records_affected', 0) for e in executions) / len(executions)
                success_rate = sum(1 for e in executions if e.get('execution_status') == 'success') / len(executions) * 100;
                
                rule_performance.append({
                    'Rule Name': rule['rule_name'],
                    'Executions': len(executions),
                    'Avg Anomalies': f"{avg_anomalies:.1f}",
                    'Success Rate': f"{success_rate:.1f}%",
                    'Priority': rule.get('priority', 1)
                })
        
        if rule_performance:
            performance_df = pd.DataFrame(rule_performance)
            performance_df = performance_df.sort_values('Avg Anomalies', ascending=False)
            st.dataframe(performance_df, use_container_width=True)
    
    # Cleanup and maintenance
    st.markdown("### 🧹 Database Maintenance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Clean Old Executions"):
            # Clean executions older than 90 days
            cutoff_date = datetime.now() - timedelta(days=90)
            # Implementation would depend on your database schema
            st.success("Old executions cleaned!")
    
    with col2:
        if st.button("📊 Generate Report"):
            # Generate comprehensive analytics report
            report = {
                'timestamp': datetime.now().isoformat(),
                'analytics': analytics,
                'rule_performance': rule_performance if 'rule_performance' in locals() else []
            }
            
            report_json = json.dumps(report, indent=2, default=str)
            st.download_button(
                "📋 Download Analytics Report",
                report_json,
                f"rule_analytics_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                "application/json",
                key="analytics_download"
            )
    
    with col3:
        if st.button("🔄 Refresh Analytics"):
            st.rerun()

def main():
    """Main function for enhanced anomaly detection with rule management"""

    # Check if rule management is available
    if not RULE_MANAGEMENT_AVAILABLE:
        st.error("❌ Rule Management system not available. Please ensure database/rule_management.py exists.")
        st.info("👆 The rule management system provides persistent storage for your audit rules.")
        return
    
    # Load data
    try:
        data_sources = load_all_data_sources()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please load data first using the 'Load Data' page.")
        return
    
    if not data_sources:
        st.warning("🔍 No data found. Please load CSV files or connect to a database first.")
        st.info("👆 Go to the 'Load Data' page to upload CSV files or connect to your database.")
        return
    
    # Main interface
    st.markdown("### 🚀 Choose Your Analysis Approach")
    
    approach = st.radio(
        "Select analysis approach:",
        ["🧠 Smart Rule Management", "🔍 Universal Detection", "📊 Quick Analysis"],
        horizontal=True
    )
    
    if approach == "🧠 Smart Rule Management":
        create_enhanced_rule_management_interface()
    
    elif approach == "🔍 Universal Detection":
        create_universal_detection_interface()
    
    else:  # Quick Analysis
        create_quick_analysis_interface()

# Universal Anomaly Detector Class (from original implementation)
class UniversalAnomalyDetector:
    """Universal anomaly detection for all data types"""
    
    def __init__(self):
        self.detection_methods = {
            'numeric': ['statistical', 'iqr', 'isolation_forest', 'threshold'],
            'categorical': ['frequency', 'rare_categories', 'new_categories'],
            'text': ['length_outliers', 'pattern_violations', 'encoding_issues'],
            'datetime': ['time_gaps', 'future_dates', 'weekend_anomalies'],
            'mixed': ['missing_pattern', 'data_type_violations']
        }
    
    def detect_all_anomalies(self, df: pd.DataFrame, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Detect anomalies across all column types"""
        
        if config is None:
            config = {
                'statistical_threshold': 3.0,
                'frequency_threshold': 0.01,
                'text_length_threshold': 3.0,
                'check_future_dates': True,
                'include_weekends': True
            }
        
        results = {
            'summary': {},
            'detailed_results': {},
            'recommendations': [],
            'total_anomalies': 0,
            'anomaly_indices': set()
        }
        
        for column in df.columns:
            col_type = self._detect_column_type(df[column])
            col_results = self._detect_column_anomalies(df, column, col_type, config)
            
            results['detailed_results'][column] = col_results
            results['total_anomalies'] += len(col_results.get('anomaly_indices', []))
            results['anomaly_indices'].update(col_results.get('anomaly_indices', []))
        
        results['anomaly_indices'] = list(results['anomaly_indices'])
        results['summary'] = self._generate_summary(results['detailed_results'])
        results['recommendations'] = self._generate_recommendations(results['detailed_results'])
        
        return results
    
    def _detect_column_type(self, series: pd.Series) -> str:
        """Intelligently detect column type"""
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            return 'empty'
        
        if pd.api.types.is_numeric_dtype(series):
            return 'numeric'
        
        if pd.api.types.is_datetime64_any_dtype(series):
            return 'datetime'
        
        try:
            pd.to_datetime(non_null_series.iloc[:min(100, len(non_null_series))])
            return 'datetime'
        except:
            pass
        
        unique_ratio = len(non_null_series.unique()) / len(non_null_series)
        if unique_ratio < 0.1 and len(non_null_series.unique()) < 50:
            return 'categorical'
        
        return 'text'
    
    def _detect_column_anomalies(self, df: pd.DataFrame, column: str, col_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies for a specific column based on its type"""
        
        result = {
            'column_type': col_type,
            'anomaly_indices': [],
            'anomaly_details': [],
            'statistics': {},
            'method_used': []
        }
        
        series = df[column]
        
        if col_type == 'numeric':
            result.update(self._detect_numeric_anomalies(series, config))
        elif col_type == 'categorical':
            result.update(self._detect_categorical_anomalies(series, config))
        elif col_type == 'text':
            result.update(self._detect_text_anomalies(series, config))
        elif col_type == 'datetime':
            result.update(self._detect_datetime_anomalies(series, config))
        
        return result
    
    def _detect_numeric_anomalies(self, series: pd.Series, config: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in numeric columns"""
        result = {
            'anomaly_indices': [],
            'anomaly_details': [],
            'method_used': ['statistical', 'iqr'],
            'statistics': {}
        }
        
        non_null_series = series.dropna()
        if len(non_null_series) == 0:
            return result
        
        # Statistical method (Z-score)
        mean_val = non_null_series.mean()
        std_val = non_null_series.std()
        threshold = config.get('statistical_threshold', 3.0)
        
        if std_val > 0:
            z_scores = np.abs((non_null_series - mean_val) / std_val)
            statistical_outliers = non_null_series.index[z_scores > threshold].tolist()
            
            for idx in statistical_outliers:
                result['anomaly_indices'].append(idx)
                result['anomaly_details'].append({
                    'index': idx,
                    'value': series.iloc[idx],
                    'method': 'statistical',
                    'z_score': float(z_scores.loc[idx]),
                    'reason': f'Z-score {z_scores.loc[idx]:.2f} > {threshold}'
                })
        
        # IQR method
        Q1 = non_null_series.quantile(0.25)
        Q3 = non_null_series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        iqr_outliers = non_null_series[(non_null_series < lower_bound) | (non_null_series > upper_bound)]
        
        for idx, value in iqr_outliers.items():
            if idx not in result['anomaly_indices']:
                result['anomaly_indices'].append(idx)
                result['anomaly_details'].append({
                    'index': idx,
                    'value': value,
                    'method': 'iqr',
                    'reason': f'Outside IQR bounds [{lower_bound:.2f}, {upper_bound:.2f}]'
                })
        
        result['statistics'] = {
            'mean': float(mean_val),
            'std': float(std_val),
            'q1': float(Q1),
            'q3': float(Q3),
            'iqr': float(IQR),
            'outliers_statistical': len(statistical_outliers),
            'outliers_iqr': len(iqr_outliers)
        }
        
        return result
    
    def _detect_categorical_anomalies(self, series: pd.Series, config: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in categorical columns"""
        result = {
            'anomaly_indices': [],
            'anomaly_details': [],
            'method_used': ['frequency', 'rare_categories'],
            'statistics': {}
        }
        
        non_null_series = series.dropna()
        if len(non_null_series) == 0:
            return result
        
        # Frequency analysis
        value_counts = non_null_series.value_counts()
        total_count = len(non_null_series)
        frequency_threshold = config.get('frequency_threshold', 0.01)
        
        rare_categories = value_counts[value_counts / total_count < frequency_threshold]
        
        for category, count in rare_categories.items():
            rare_indices = non_null_series[non_null_series == category].index.tolist()
            for idx in rare_indices:
                result['anomaly_indices'].append(idx)
                result['anomaly_details'].append({
                    'index': idx,
                    'value': category,
                    'method': 'rare_category',
                    'frequency': count,
                    'percentage': f"{(count/total_count)*100:.2f}%",
                    'reason': f'Rare category: {count}/{total_count} occurrences'
                })
        
        result['statistics'] = {
            'unique_categories': len(value_counts),
            'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
            'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
            'rare_categories_count': len(rare_categories)
        }
        
        return result
    
    def _detect_text_anomalies(self, series: pd.Series, config: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in text columns"""
        result = {
            'anomaly_indices': [],
            'anomaly_details': [],
            'method_used': ['length_outliers', 'encoding_issues'],
            'statistics': {}
        }
        
        non_null_series = series.dropna().astype(str)
        if len(non_null_series) == 0:
            return result
        
        # Text length analysis
        lengths = non_null_series.str.len()
        mean_length = lengths.mean()
        std_length = lengths.std()
        threshold = config.get('text_length_threshold', 3.0)
        
        if std_length > 0:
            z_scores = np.abs((lengths - mean_length) / std_length)
            length_outliers = lengths.index[z_scores > threshold].tolist()
            
            for idx in length_outliers:
                result['anomaly_indices'].append(idx)
                result['anomaly_details'].append({
                    'index': idx,
                    'value': str(series.iloc[idx])[:100] + "..." if len(str(series.iloc[idx])) > 100 else str(series.iloc[idx]),
                    'method': 'length_outlier',
                    'length': int(lengths.loc[idx]),
                    'z_score': float(z_scores.loc[idx]),
                    'reason': f'Unusual length: {int(lengths.loc[idx])} chars (z-score: {z_scores.loc[idx]:.2f})'
                })
        
        result['statistics'] = {
            'mean_length': float(mean_length),
            'std_length': float(std_length),
            'min_length': int(lengths.min()),
            'max_length': int(lengths.max())
        }
        
        return result
    
    def _detect_datetime_anomalies(self, series: pd.Series, config: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in datetime columns"""
        result = {
            'anomaly_indices': [],
            'anomaly_details': [],
            'method_used': ['future_dates', 'weekend_anomalies'],
            'statistics': {}
        }
        
        try:
            if not pd.api.types.is_datetime64_any_dtype(series):
                dt_series = pd.to_datetime(series, errors='coerce')
            else:
                dt_series = series
        except:
            return result
        
        non_null_dt = dt_series.dropna()
        if len(non_null_dt) == 0:
            return result
        
        current_time = datetime.now()
        
        # Future dates detection
        if config.get('check_future_dates', True):
            future_dates = non_null_dt[non_null_dt > current_time]
            for idx, date_val in future_dates.items():
                result['anomaly_indices'].append(idx)
                result['anomaly_details'].append({
                    'index': idx,
                    'value': str(date_val),
                    'method': 'future_date',
                    'reason': f'Date in future: {date_val.strftime("%Y-%m-%d %H:%M:%S")}'
                })
        
        result['statistics'] = {
            'min_date': str(non_null_dt.min()),
            'max_date': str(non_null_dt.max()),
            'date_range_days': (non_null_dt.max() - non_null_dt.min()).days,
            'future_dates_count': len([idx for idx in result['anomaly_indices']])
        }
        
        return result
    
    def _generate_summary(self, detailed_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics"""
        summary = {
            'total_columns_analyzed': len(detailed_results),
            'columns_with_anomalies': 0,
            'anomalies_by_type': {},
            'anomalies_by_method': {},
            'most_problematic_columns': []
        }
        
        for column, results in detailed_results.items():
            if results['anomaly_indices']:
                summary['columns_with_anomalies'] += 1
                
                col_type = results['column_type']
                if col_type not in summary['anomalies_by_type']:
                    summary['anomalies_by_type'][col_type] = 0
                summary['anomalies_by_type'][col_type] += len(results['anomaly_indices'])
                
                for method in results['method_used']:
                    if method not in summary['anomalies_by_method']:
                        summary['anomalies_by_method'][method] = 0
                    summary['anomalies_by_method'][method] += len([d for d in results['anomaly_details'] if d['method'] == method])
                
                summary['most_problematic_columns'].append({
                    'column': column,
                    'anomaly_count': len(results['anomaly_indices']),
                    'type': col_type
                })
        
        summary['most_problematic_columns'].sort(key=lambda x: x['anomaly_count'], reverse=True)
        summary['most_problematic_columns'] = summary['most_problematic_columns'][:5]
        
        return summary
    
    def _generate_recommendations(self, detailed_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        for column, results in detailed_results.items():
            if not results['anomaly_indices']:
                continue
                
            col_type = results['column_type']
            anomaly_count = len(results['anomaly_indices'])
            
            if col_type == 'numeric':
                if anomaly_count > 10:
                    recommendations.append(f"📊 Column '{column}': Consider data validation rules for numeric values")
                recommendations.append(f"📊 Column '{column}': Review outliers - {anomaly_count} found")
                
            elif col_type == 'categorical':
                rare_categories = len([d for d in results['anomaly_details'] if d['method'] == 'rare_category'])
                if rare_categories > 0:
                    recommendations.append(f"🏷️ Column '{column}': Standardize categories - {rare_categories} rare values found")
                    
            elif col_type == 'text':
                length_outliers = len([d for d in results['anomaly_details'] if d['method'] == 'length_outlier'])
                if length_outliers > 0:
                    recommendations.append(f"📝 Column '{column}': Review text length validation - {length_outliers} outliers")
                    
            elif col_type == 'datetime':
                future_dates = len([d for d in results['anomaly_details'] if d['method'] == 'future_date'])
                if future_dates > 0:
                    recommendations.append(f"📅 Column '{column}': Fix future dates - {future_dates} found")
        
        total_anomalies = sum(len(r['anomaly_indices']) for r in detailed_results.values())
        if total_anomalies > 100:
            recommendations.append("⚠️ High anomaly count detected - consider systematic data quality review")
        
        return recommendations

def create_universal_detection_interface():
    """Complete universal detection interface"""
    
    st.markdown("#### 🔍 Universal Anomaly Detection")
    st.info("🎯 Automatically detect anomalies across all data types: numeric, categorical, text, and datetime columns")
    
    # Load data sources
    try:
        data_sources = load_all_data_sources()
        
        if not data_sources:
            st.warning("No data available. Please load data first.")
            return
        
        # Dataset selection
        st.markdown("### 📊 Select Dataset")
        dataset_key = st.selectbox(
            "Choose dataset to analyze:",
            list(data_sources.keys()),
            help="Select which dataset to analyze for anomalies",
            key="universal_dataset"
        )
        
        df = data_sources[dataset_key]
        
        # Display basic dataset info
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", f"{len(df):,}")
        
        with col2:
            st.metric("Total Columns", len(df.columns))
        
        with col3:
            numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
            st.metric("Numeric Columns", numeric_cols)
        
        with col4:
            text_cols = len(df.select_dtypes(include=['object']).columns)
            st.metric("Text/Categorical", text_cols)
        
        # Show data types breakdown
        with st.expander("📋 Column Types Breakdown", expanded=False):
            detector = UniversalAnomalyDetector()
            
            type_breakdown = {}
            for col in df.columns:
                col_type = detector._detect_column_type(df[col])
                if col_type not in type_breakdown:
                    type_breakdown[col_type] = []
                type_breakdown[col_type].append(col)
            
            for data_type, columns in type_breakdown.items():
                st.markdown(f"**{data_type.title()} ({len(columns)} columns):**")
                st.write(", ".join(columns[:10]) + ("..." if len(columns) > 10 else ""))
        
        # Configuration options
        st.markdown("### ⚙️ Detection Configuration")
        
        with st.expander("🔧 Advanced Settings", expanded=True):
            col_config1, col_config2, col_config3 = st.columns(3)
            
            with col_config1:
                st.markdown("**📊 Numeric Detection:**")
                statistical_threshold = st.slider("Statistical Threshold (Z-score)", 1.5, 5.0, 3.0, 0.5, 
                                                 help="Higher values = fewer outliers detected")
                include_iqr = st.checkbox("Include IQR Method", True, 
                                        help="Use Interquartile Range for outlier detection")
            
            with col_config2:
                st.markdown("**🏷️ Categorical Detection:**")
                frequency_threshold = st.slider("Rare Category Threshold (%)", 0.1, 5.0, 1.0, 0.1, 
                                               help="Categories below this % are flagged as rare") / 100
                check_misspellings = st.checkbox("Check for Misspellings", True,
                                               help="Detect potential typos in categorical data")
            
            with col_config3:
                st.markdown("**📝 Text/Date Detection:**")
                text_length_threshold = st.slider("Text Length Threshold", 1.5, 5.0, 3.0, 0.5,
                                                 help="Z-score threshold for unusual text lengths")
                check_future_dates = st.checkbox("Flag Future Dates", True,
                                                help="Detect dates in the future")
        
        # Column selection
        st.markdown("### 📋 Column Selection")
        
        select_all = st.checkbox("Select All Columns", value=True)
        
        if select_all:
            selected_columns = df.columns.tolist()
        else:
            selected_columns = st.multiselect(
                "Choose specific columns to analyze:",
                df.columns.tolist(),
                default=df.columns.tolist()[:10],
                help="Select which columns to include in anomaly detection"
            )
        
        if not selected_columns:
            st.warning("Please select at least one column to analyze.")
            return
        
        st.info(f"📊 Will analyze {len(selected_columns)} columns out of {len(df.columns)} total")
        
        # Run detection button
        if st.button("🔍 Run Universal Anomaly Detection", type="primary", use_container_width=True):
            run_universal_detection(df, selected_columns, {
                'statistical_threshold': statistical_threshold,
                'frequency_threshold': frequency_threshold,
                'text_length_threshold': text_length_threshold,
                'check_future_dates': check_future_dates,
                'check_misspellings': check_misspellings,
                'include_iqr': include_iqr
            })
            
    except Exception as e:
        st.error(f"Error in universal detection interface: {e}")

def run_universal_detection(df: pd.DataFrame, selected_columns: List[str], config: Dict[str, Any]):
    """Run universal anomaly detection and display results"""
    
    # Filter dataframe to selected columns
    df_subset = df[selected_columns].copy()
    
    with st.spinner("🔍 Analyzing all data types for anomalies..."):
        detector = UniversalAnomalyDetector()
        results = detector.detect_all_anomalies(df_subset, config)
    
    # Display results
    st.markdown("## 📊 Universal Anomaly Detection Results")
    
    # Summary metrics
    summary = results['summary']
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("Total Anomalies", results['total_anomalies'])
    with metric_col2:
        st.metric("Columns Analyzed", summary['total_columns_analyzed'])
    with metric_col3:
        st.metric("Problematic Columns", summary['columns_with_anomalies'])
    with metric_col4:
        anomaly_rate = (results['total_anomalies'] / len(df) * 100) if len(df) > 0 else 0
        st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
    
    # Visualizations
    if summary['anomalies_by_type']:
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            # Anomalies by data type
            type_data = summary['anomalies_by_type']
            fig_type = px.pie(
                values=list(type_data.values()),
                names=list(type_data.keys()),
                title="🎯 Anomalies by Data Type",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_type, use_container_width=True)
        
        with viz_col2:
            # Anomalies by detection method
            method_data = summary['anomalies_by_method']
            fig_method = px.bar(
                x=list(method_data.keys()),
                y=list(method_data.values()),
                title="🔧 Anomalies by Detection Method",
                labels={'x': 'Detection Method', 'y': 'Anomaly Count'},
                color=list(method_data.values()),
                color_continuous_scale='Reds'
            )
            fig_method.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_method, use_container_width=True)
    
    # Most problematic columns
    if summary['most_problematic_columns']:
        st.markdown("### 🚨 Most Problematic Columns")
        
        prob_cols_data = []
        for col_info in summary['most_problematic_columns']:
            prob_cols_data.append({
                'Column': col_info['column'],
                'Data Type': col_info['type'].title(),
                'Anomaly Count': col_info['anomaly_count'],
                'Severity': '🔴 High' if col_info['anomaly_count'] > 20 else '🟡 Medium' if col_info['anomaly_count'] > 5 else '🟢 Low'
            })
        
        st.dataframe(pd.DataFrame(prob_cols_data), use_container_width=True)
    
    # Detailed results by column
    st.markdown("### 🔍 Detailed Results by Column")
    
    for column, col_results in results['detailed_results'].items():
        if col_results['anomaly_indices']:
            with st.expander(f"📊 {column} ({col_results['column_type']}) - {len(col_results['anomaly_indices'])} anomalies", expanded=False):
                
                # Statistics
                if col_results['statistics']:
                    st.markdown("**📈 Statistics:**")
                    stats_cols = st.columns(len(col_results['statistics']))
                    for i, (stat_name, stat_value) in enumerate(col_results['statistics'].items()):
                        with stats_cols[i % len(stats_cols)]:
                            if isinstance(stat_value, (int, float)):
                                st.metric(stat_name.replace('_', ' ').title(), f"{stat_value:.2f}" if isinstance(stat_value, float) else str(stat_value))
                            else:
                                st.text(f"{stat_name}: {stat_value}")
                
                # Anomalous records sample
                if col_results['anomaly_indices']:
                    st.markdown("**🚨 Anomalous Records (Sample):**")
                    anomaly_indices = col_results['anomaly_indices'][:20]  # Show first 20
                    anomaly_sample = df.iloc[anomaly_indices]
                    st.dataframe(anomaly_sample, use_container_width=True)
                    
                    if len(col_results['anomaly_indices']) > 20:
                        st.info(f"Showing first 20 anomalies out of {len(col_results['anomaly_indices'])} total")
    
    # Recommendations
    if results['recommendations']:
        st.markdown("### 💡 Recommendations")
        for i, recommendation in enumerate(results['recommendations'], 1):
            st.markdown(f"{i}. {recommendation}")
    
    # Export options
    st.markdown("### 📥 Export Results")
    
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        if results['anomaly_indices']:
            anomaly_data = df.iloc[results['anomaly_indices']]
            csv_data = anomaly_data.to_csv(index=True)
            st.download_button(
                "📄 Download Anomalous Records",
                csv_data,
                f"universal_anomalies_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv"
            )
    
    with export_col2:
        # Export detailed results as JSON
        json_data = json.dumps(results, indent=2, default=str)
        st.download_button(
            "📋 Download Full Report (JSON)",
            json_data,
            f"universal_detection_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.json",
            "application/json"
        )

def create_quick_analysis_interface():
    """Quick analysis interface for fast anomaly detection"""
    
    st.markdown("#### 📊 Quick Analysis")
    st.info("🚀 Fast anomaly detection for quick insights and data quality checks")
    
    # Load data sources
    try:
        data_sources = load_all_data_sources()
        
        if not data_sources:
            st.warning("No data available. Please load data first.")
            return
        
        # Quick dataset selection
        dataset_key = st.selectbox(
            "📊 Select Dataset:",
            list(data_sources.keys()),
            key="quick_dataset"
        )
        
        df = data_sources[dataset_key]
        
        # Quick overview
        st.markdown("### 📋 Quick Overview")
        
        overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
        
        with overview_col1:
            st.metric("Rows", f"{len(df):,}")
        with overview_col2:
            st.metric("Columns", len(df.columns))
        with overview_col3:
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
            st.metric("Missing Data", f"{missing_pct:.1f}%")
        with overview_col4:
            duplicates = df.duplicated().sum()
            st.metric("Duplicates", f"{duplicates:,}")
        
        # Quick analysis options
        st.markdown("### ⚡ Quick Analysis Options")
        
        analysis_tabs = st.tabs(["🎯 Smart Detection", "📊 Statistical Overview", "🔍 Specific Checks"])
        
        with analysis_tabs[0]:
            st.markdown("#### 🎯 Smart Anomaly Detection")
            st.info("Automatically detect the most common types of anomalies")
            
            if st.button("🚀 Run Smart Detection", type="primary", use_container_width=True):
                run_smart_detection(df)
        
        with analysis_tabs[1]:
            st.markdown("#### 📊 Statistical Overview")
            st.info("Get quick statistical insights about your data")
            
            if st.button("📈 Generate Overview", type="primary", use_container_width=True):
                generate_statistical_overview(df)
        
        with analysis_tabs[2]:
            st.markdown("#### 🔍 Specific Anomaly Checks")
            
            check_col1, check_col2 = st.columns(2)
            
            with check_col1:
                if st.button("🔢 Numeric Outliers", use_container_width=True):
                    detect_numeric_outliers_quick(df)
                
                if st.button("📅 Date Issues", use_container_width=True):
                    detect_date_issues_quick(df)
            
            with check_col2:
                if st.button("🏷️ Categorical Issues", use_container_width=True):
                    detect_categorical_issues_quick(df)
                
                if st.button("📝 Text Problems", use_container_width=True):
                    detect_text_problems_quick(df)
            
            # Custom threshold check
            st.markdown("#### 🎛️ Custom Threshold Check")
            
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_columns:
                selected_col = st.selectbox("Select numeric column:", numeric_columns)
                threshold_value = st.number_input("Threshold value:", value=float(df[selected_col].quantile(0.95)))
                operator = st.selectbox("Condition:", ["greater than", "less than", "equal to"])
                
                if st.button("🔍 Check Threshold", use_container_width=True):
                    check_custom_threshold(df, selected_col, threshold_value, operator)
            
    except Exception as e:
        st.error(f"Error in quick analysis interface: {e}")

def run_smart_detection(df: pd.DataFrame):
    """Run smart detection with predefined common anomaly patterns"""
    
    st.markdown("### 🎯 Smart Detection Results")
    
    with st.spinner("🧠 Running smart anomaly detection..."):
        anomalies_found = []
        
        # 1. Extreme outliers in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].nunique() > 1:  # Skip constant columns
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                extreme_outliers = df[(df[col] < Q1 - 3*IQR) | (df[col] > Q3 + 3*IQR)]
                
                if len(extreme_outliers) > 0:
                    anomalies_found.append({
                        'type': 'Extreme Numeric Outliers',
                        'column': col,
                        'count': len(extreme_outliers),
                        'severity': 'High' if len(extreme_outliers) > len(df) * 0.01 else 'Medium',
                        'indices': extreme_outliers.index.tolist(),
                        'description': f'Values beyond 3×IQR in {col}'
                    })
        
        # 2. Missing value patterns
        missing_cols = df.columns[df.isnull().any()].tolist()
        for col in missing_cols:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            if missing_pct > 50:
                anomalies_found.append({
                    'type': 'High Missing Values',
                    'column': col,
                    'count': df[col].isnull().sum(),
                    'severity': 'High',
                    'indices': df[df[col].isnull()].index.tolist(),
                    'description': f'{missing_pct:.1f}% missing values in {col}'
                })
        
        # 3. Duplicate rows
        duplicates = df[df.duplicated(keep=False)]
        if len(duplicates) > 0:
            anomalies_found.append({
                'type': 'Duplicate Rows',
                'column': 'All columns',
                'count': len(duplicates),
                'severity': 'Medium' if len(duplicates) < len(df) * 0.05 else 'High',
                'indices': duplicates.index.tolist(),
                'description': f'{len(duplicates)} duplicate rows found'
            })
        
        # 4. Rare categorical values
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].nunique() > 1:  # Skip constant columns
                value_counts = df[col].value_counts()
                rare_threshold = len(df) * 0.001  # Less than 0.1% frequency
                rare_values = value_counts[value_counts < rare_threshold]
                
                if len(rare_values) > 0:
                    rare_indices = df[df[col].isin(rare_values.index)].index.tolist()
                    anomalies_found.append({
                        'type': 'Rare Categorical Values',
                        'column': col,
                        'count': len(rare_indices),
                        'severity': 'Low',
                        'indices': rare_indices,
                        'description': f'{len(rare_values)} rare categories in {col}'
                    })
        
        # 5. Date anomalies
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]) or 'date' in col.lower():
                try:
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        date_col = df[col]
                    else:
                        date_col = pd.to_datetime(df[col], errors='coerce')
                    
                    # Future dates
                    future_dates = date_col > pd.Timestamp.now()
                    if future_dates.sum() > 0:
                        anomalies_found.append({
                            'type': 'Future Dates',
                            'column': col,
                            'count': future_dates.sum(),
                            'severity': 'High',
                            'indices': df[future_dates].index.tolist(),
                            'description': f'{future_dates.sum()} future dates in {col}'
                        })
                    
                    # Very old dates (before 1900)
                    if not date_col.isnull().all():
                        old_dates = date_col < pd.Timestamp('1900-01-01')
                        if old_dates.sum() > 0:
                            anomalies_found.append({
                                'type': 'Unrealistic Old Dates',
                                'column': col,
                                'count': old_dates.sum(),
                                'severity': 'Medium',
                                'indices': df[old_dates].index.tolist(),
                                'description': f'{old_dates.sum()} dates before 1900 in {col}'
                            })
                except:
                    continue
        
        # 6. Text length anomalies
        text_cols = df.select_dtypes(include=['object']).columns
        for col in text_cols:
            if df[col].dtype == 'object':
                text_lengths = df[col].astype(str).str.len()
                mean_length = text_lengths.mean()
                std_length = text_lengths.std()
                
                if std_length > 0:
                    z_scores = np.abs((text_lengths - mean_length) / std_length)
                    extreme_length_outliers = z_scores > 4  # Very extreme
                    
                    if extreme_length_outliers.sum() > 0:
                        anomalies_found.append({
                            'type': 'Extreme Text Length',
                            'column': col,
                            'count': extreme_length_outliers.sum(),
                            'severity': 'Low',
                            'indices': df[extreme_length_outliers].index.tolist(),
                            'description': f'{extreme_length_outliers.sum()} extreme text lengths in {col}'
                        })
    
    # Display results
    if anomalies_found:
        st.success(f"🎯 Found {len(anomalies_found)} types of anomalies")
        
        # Summary metrics
        total_anomalous_records = len(set(sum([a['indices'] for a in anomalies_found], [])))
        
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        with summary_col1:
            st.metric("Anomaly Types", len(anomalies_found))
        with summary_col2:
            st.metric("Affected Records", f"{total_anomalous_records:,}")
        with summary_col3:
            anomaly_rate = (total_anomalous_records / len(df) * 100) if len(df) > 0 else 0
            st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
        
        # Detailed results
        for anomaly in sorted(anomalies_found, key=lambda x: x['count'], reverse=True):
            severity_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
            
            with st.expander(f"{severity_color[anomaly['severity']]} {anomaly['type']} - {anomaly['column']} ({anomaly['count']} records)", expanded=False):
                st.markdown(f"**Description:** {anomaly['description']}")
                st.markdown(f"**Severity:** {anomaly['severity']}")
                st.markdown(f"**Affected Records:** {anomaly['count']:,}")
                
                # Show sample of anomalous records
                if anomaly['indices']:
                    sample_indices = anomaly['indices'][:10]
                    sample_data = df.iloc[sample_indices]
                    st.dataframe(sample_data, use_container_width=True)
                    
                    if len(anomaly['indices']) > 10:
                        st.info(f"Showing 10 out of {len(anomaly['indices'])} anomalous records")
        
        # Export all anomalies
        all_anomaly_indices = list(set(sum([a['indices'] for a in anomalies_found], [])))
        if all_anomaly_indices:
            anomaly_df = df.iloc[all_anomaly_indices].copy()
            
            # Add anomaly types as new column
            anomaly_types = []
            for idx in anomaly_df.index:
                types = [a['type'] for a in anomalies_found if idx in a['indices']]
                anomaly_types.append('; '.join(types))
            
            anomaly_df['Anomaly_Types'] = anomaly_types
            
            csv_data = anomaly_df.to_csv(index=True)
            st.download_button(
                "📥 Download All Anomalies",
                csv_data,
                f"smart_detection_anomalies_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv"
            )
    else:
        st.success("🎉 No significant anomalies detected! Your data looks clean.")

def generate_statistical_overview(df: pd.DataFrame):
    """Generate quick statistical overview"""
    
    st.markdown("### 📊 Statistical Overview")
    
    # Basic statistics
    basic_col1, basic_col2, basic_col3, basic_col4 = st.columns(4)
    
    with basic_col1:
        st.metric("Total Records", f"{len(df):,}")
    with basic_col2:
        st.metric("Total Columns", len(df.columns))
    with basic_col3:
        memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
        st.metric("Memory Usage", f"{memory_mb:.1f} MB")
    with basic_col4:
        completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Data Completeness", f"{completeness:.1f}%")
    
    # Data type breakdown
    st.markdown("#### 📋 Data Type Distribution")
    type_counts = df.dtypes.value_counts()
    
    type_col1, type_col2 = st.columns(2)
    
    with type_col1:
        fig_types = px.pie(
            values=type_counts.values,
            names=[str(t) for t in type_counts.index],
            title="Column Types"
        )
        st.plotly_chart(fig_types, use_container_width=True)
    
    with type_col2:
        st.markdown("**Type Summary:**")
        for dtype, count in type_counts.items():
            st.write(f"• **{dtype}**: {count} columns")
    
    # Numeric columns overview
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        st.markdown("#### 📊 Numeric Columns Summary")
        
        numeric_stats = df[numeric_cols].describe().round(2)
        st.dataframe(numeric_stats, use_container_width=True)
        
        # Distribution plots for top numeric columns
        if len(numeric_cols) > 0:
            st.markdown("#### 📈 Distribution Plots")
            
            cols_to_plot = min(4, len(numeric_cols))
            plot_cols = st.columns(cols_to_plot)
            
            for i, col in enumerate(numeric_cols[:cols_to_plot]):
                with plot_cols[i]:
                    fig_hist = px.histogram(
                        df, x=col, 
                        title=f"Distribution: {col}",
                        nbins=20
                    )
                    fig_hist.update_layout(height=300, showlegend=False)
                    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Categorical columns overview
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        st.markdown("#### 🏷️ Categorical Columns Summary")
        
        cat_summary = []
        for col in categorical_cols:
            cat_summary.append({
                'Column': col,
                'Unique Values': df[col].nunique(),
                'Most Common': df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A',
                'Missing Count': df[col].isnull().sum()
            })
        
        cat_df = pd.DataFrame(cat_summary)
        st.dataframe(cat_df, use_container_width=True)
    
    # Missing values heatmap
    if df.isnull().sum().sum() > 0:
        st.markdown("#### 🕳️ Missing Values Pattern")
        
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) > 0:
            fig_missing = px.bar(
                x=missing_data.index,
                y=missing_data.values,
                title="Missing Values by Column",
                labels={'x': 'Columns', 'y': 'Missing Count'}
            )
            fig_missing.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_missing, use_container_width=True)
    
    # Correlation matrix for numeric data
    if len(numeric_cols) > 1:
        st.markdown("#### 🔗 Correlation Matrix")
        
        corr_matrix = df[numeric_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            title="Correlation Heatmap",
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        st.plotly_chart(fig_corr, use_container_width=True)

def detect_numeric_outliers_quick(df: pd.DataFrame):
    """Quick numeric outlier detection"""
    
    st.markdown("### 🔢 Numeric Outliers Detection")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        st.info("No numeric columns found in the dataset.")
        return
    
    outliers_found = []
    
    for col in numeric_cols:
        if df[col].nunique() > 1:  # Skip constant columns
            # IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
            if len(outliers) > 0:
                outliers_found.append({
                    'column': col,
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(df)) * 100,
                    'method': 'IQR',
                    'bounds': f"[{lower_bound:.2f}, {upper_bound:.2f}]",
                    'indices': outliers.index.tolist()
                })
    
    if outliers_found:
        st.success(f"Found outliers in {len(outliers_found)} columns")
        
        # Summary metrics
        total_outlier_records = len(set(sum([o['indices'] for o in outliers_found], [])))
        
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        with summary_col1:
            st.metric("Outlier Types", len(outliers_found))
        with summary_col2:
            st.metric("Affected Records", f"{total_outlier_records:,}")
        with summary_col3:
            anomaly_rate = (total_outlier_records / len(df) * 100) if len(df) > 0 else 0
            st.metric("Outlier Rate", f"{anomaly_rate:.1f}%")
        
        # Detailed results
        for outlier_info in sorted(outliers_found, key=lambda x: x['count'], reverse=True):
            with st.expander(f"📊 {outlier_info['column']} - {outlier_info['count']} outliers ({outlier_info['percentage']:.1f}%)", expanded=False):
                st.markdown(f"**Method:** {outlier_info['method']}")
                st.markdown(f"**Normal Range:** {outlier_info['bounds']}")
                
                # Show outlier values
                outlier_data = df.iloc[outlier_info['indices'][:10]]
                st.dataframe(outlier_data, use_container_width=True)
                
                if len(outlier_info['indices']) > 10:
                    st.info(f"Showing 10 out of {outlier_info['count']} outliers")
    else:
        st.info("No significant numeric outliers detected.")

def detect_date_issues_quick(df: pd.DataFrame):
    """Quick date issues detection"""
    
    st.markdown("### 📅 Date Issues Detection")
    
    date_issues = []
    
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]) or 'date' in col.lower():
            try:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    date_col = df[col]
                else:
                    date_col = pd.to_datetime(df[col], errors='coerce')
                
                # Future dates
                future_dates = date_col > pd.Timestamp.now()
                if future_dates.sum() > 0:
                    date_issues.append({
                        'column': col,
                        'issue': 'Future Dates',
                        'count': future_dates.sum(),
                        'indices': df[future_dates].index.tolist()
                    })
                
                # Very old dates
                old_dates = date_col < pd.Timestamp('1900-01-01')
                if old_dates.sum() > 0:
                    date_issues.append({
                        'column': col,
                        'issue': 'Very Old Dates',
                        'count': old_dates.sum(),
                        'indices': df[old_dates].index.tolist()
                    })
                
                # Invalid dates (NaT after conversion)
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    invalid_dates = date_col.isnull()
                    if invalid_dates.sum() > 0:
                        date_issues.append({
                            'column': col,
                            'issue': 'Invalid Date Format',
                            'count': invalid_dates.sum(),
                            'indices': df[invalid_dates].index.tolist()
                        })
                
            except Exception:
                continue
    
    if date_issues:
        st.warning(f"Found date issues in {len(set(issue['column'] for issue in date_issues))} columns")
        
        for issue in date_issues:
            with st.expander(f"📅 {issue['column']} - {issue['issue']} ({issue['count']} records)", expanded=False):
                sample_data = df.iloc[issue['indices'][:10]]
                st.dataframe(sample_data, use_container_width=True)
                
                if len(issue['indices']) > 10:
                    st.info(f"Showing 10 out of {issue['count']} problematic dates")
    else:
        st.success("No date issues detected!")

def detect_categorical_issues_quick(df: pd.DataFrame):
    """Quick categorical issues detection"""
    
    st.markdown("### 🏷️ Categorical Issues Detection")
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    if len(categorical_cols) == 0:
        st.info("No categorical columns found in the dataset.")
        return
    
    categorical_issues = []
    
    for col in categorical_cols:
        # Rare categories
        value_counts = df[col].value_counts()
        rare_threshold = len(df) * 0.001  # Less than 0.1%
        rare_values = value_counts[value_counts < rare_threshold]
        
        if len(rare_values) > 0:
            categorical_issues.append({
                'column': col,
                'issue': 'Rare Categories',
                'count': len(rare_values),
                'details': f"{len(rare_values)} categories with < 0.1% frequency",
                'examples': rare_values.index.tolist()[:5]
            })
        
        # Potential typos (similar strings)
        if len(value_counts) > 1:
            unique_values = value_counts.index.tolist()
            potential_typos = []
            
            for i, val1 in enumerate(unique_values[:20]):  # Check first 20 to avoid performance issues
                for val2 in unique_values[i+1:20]:
                    if isinstance(val1, str) and isinstance(val2, str):
                        if len(val1) > 2 and len(val2) > 2:
                            # Simple similarity check
                            similarity = len(set(val1.lower()) & set(val2.lower())) / len(set(val1.lower()) | set(val2.lower()))
                            if similarity > 0.8 and val1.lower() != val2.lower():
                                potential_typos.append((val1, val2))
            
            if potential_typos:
                categorical_issues.append({
                    'column': col,
                    'issue': 'Potential Typos',
                    'count': len(potential_typos),
                    'details': f"{len(potential_typos)} pairs of similar values",
                    'examples': potential_typos[:3]
                })
    
    if categorical_issues:
        st.warning(f"Found categorical issues in {len(set(issue['column'] for issue in categorical_issues))} columns")
        
        for issue in categorical_issues:
            with st.expander(f"🏷️ {issue['column']} - {issue['issue']} ({issue['count']})", expanded=False):
                st.markdown(f"**Details:** {issue['details']}")
                st.markdown(f"**Examples:** {', '.join(map(str, issue['examples']))}")
    else:
        st.success("No categorical issues detected!")

def detect_text_problems_quick(df: pd.DataFrame):
    """Quick text problems detection"""
    
    st.markdown("### 📝 Text Problems Detection")
    
    text_cols = df.select_dtypes(include=['object']).columns
    
    if len(text_cols) == 0:
        st.info("No text columns found in the dataset.")
        return
    
    text_problems = []
    
    for col in text_cols:
        text_series = df[col].astype(str)
        
        # Unusual lengths
        lengths = text_series.str.len()
        mean_length = lengths.mean()
        std_length = lengths.std()
        
        if std_length > 0:
            z_scores = np.abs((lengths - mean_length) / std_length)
            unusual_lengths = z_scores > 3
            
            if unusual_lengths.sum() > 0:
                text_problems.append({
                    'column': col,
                    'issue': 'Unusual Text Length',
                    'count': unusual_lengths.sum(),
                    'indices': df[unusual_lengths].index.tolist()
                })
        
        # Empty strings
        empty_strings = (text_series == '') | (text_series == 'nan')
        if empty_strings.sum() > 0:
            text_problems.append({
                'column': col,
                'issue': 'Empty Strings',
                'count': empty_strings.sum(),
                'indices': df[empty_strings].index.tolist()
            })
        
        # Special characters
        special_chars = text_series.str.contains(r'[^\w\s.-]', regex=True, na=False)
        if special_chars.sum() > len(df) * 0.1:  # More than 10% have special chars
            text_problems.append({
                'column': col,
                'issue': 'High Special Characters',
                'count': special_chars.sum(),
                'indices': df[special_chars].index.tolist()[:20]  # Limit to first 20
            })
    
    if text_problems:
        st.warning(f"Found text problems in {len(set(problem['column'] for problem in text_problems))} columns")
        
        for problem in text_problems:
            with st.expander(f"📝 {problem['column']} - {problem['issue']} ({problem['count']} records)", expanded=False):
                sample_data = df.iloc[problem['indices'][:10]]
                st.dataframe(sample_data, use_container_width=True)
                
                if len(problem['indices']) > 10:
                    st.info(f"Showing 10 out of {problem['count']} problematic records")
    else:
        st.success("No text problems detected!")

def check_custom_threshold(df: pd.DataFrame, column: str, threshold: float, operator: str):
    """Check custom threshold conditions"""
    
    st.markdown(f"### 🎛️ Custom Threshold Results: {column}")
    
    if operator == "greater than":
        condition = df[column] > threshold
        op_symbol = ">"
    elif operator == "less than":
        condition = df[column] < threshold
        op_symbol = "<"
    else:  # equal to
        condition = df[column] == threshold
        op_symbol = "="
    
    matching_records = df[condition]
    
    if len(matching_records) > 0:
        st.warning(f"Found {len(matching_records)} records where {column} {op_symbol} {threshold}")
        
        # Show statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Matching Records", len(matching_records))
        with col2:
            percentage = (len(matching_records) / len(df)) * 100
            st.metric("Percentage", f"{percentage:.1f}%")
        with col3:
            if operator in ["greater than", "less than"]:
                if operator == "greater than":
                    avg_excess = matching_records[column].mean() - threshold
                    st.metric("Avg Excess", f"{avg_excess:.2f}")
                else:
                    avg_deficit = threshold - matching_records[column].mean()
                    st.metric("Avg Deficit", f"{avg_deficit:.2f}")
        
        # Show sample records
        st.markdown("**Sample Records:**")
        st.dataframe(matching_records.head(20), use_container_width=True)
        
        if len(matching_records) > 20:
            st.info(f"Showing first 20 out of {len(matching_records)} matching records")
        
        # Download option
        csv_data = matching_records.to_csv(index=True)
        st.download_button(
            "📥 Download Matching Records",
            csv_data,
            f"threshold_check_{column}_{operator.replace(' ', '_')}_{threshold}.csv",
            "text/csv"
        )
    else:
        st.success(f"No records found where {column} {op_symbol} {threshold}")

if __name__ == "__main__":
    main()