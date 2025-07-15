# File: app/pages/5_Anomaly_Detection.py
# COMPLETE UPDATED VERSION WITH ENHANCED NATURAL LANGUAGE RULES

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
warnings.filterwarnings('ignore')

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
    page_title="Universal Anomaly Detection",
    page_icon="🔍",
    layout="wide"
)

if STYLING_AVAILABLE:
    apply_clean_styling()

# Enhanced Natural Language Rule Parser (embedded in this file for completeness)
class NaturalLanguageRuleParser:
    """
    Advanced natural language rule parser for anomaly detection
    Supports complex rules across all data types
    """
    
    def __init__(self):
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


# Universal Anomaly Detector Class
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


# Enhanced Natural Language Interface Functions
def create_enhanced_natural_language_rules():
    """Enhanced natural language rule interface for Streamlit"""
    
    st.subheader("🧠 Enhanced Natural Language Anomaly Rules")
    st.info("Create sophisticated anomaly detection rules using plain English!")
    
    # Initialize the parser
    if 'nl_parser' not in st.session_state:
        st.session_state.nl_parser = NaturalLanguageRuleParser()
    
    # Enhanced examples
    with st.expander("📚 Advanced Rule Examples", expanded=True):
        
        example_tabs = st.tabs(["📊 Numeric Rules", "🏷️ Categorical Rules", "📝 Text Rules", "📅 Date Rules", "🔗 Complex Rules"])
        
        with example_tabs[0]:
            st.markdown("""
            **📊 Advanced Numeric Rules:**
            - `sales_amount is outlier`
            - `price is greater than 1000 and discount is less than 0.1`
            - `age is between 18 and 65`
            - `balance is outside -1000 and 10000`
            - `score is unusual`
            """)
        
        with example_tabs[1]:
            st.markdown("""
            **🏷️ Smart Categorical Rules:**
            - `status is not in ['active', 'pending', 'completed']`
            - `category is rare`
            - `department contains 'test'`
            - `country_code is unusual`
            - `product_type is duplicate`
            """)
        
        with example_tabs[2]:
            st.markdown("""
            **📝 Intelligent Text Rules:**
            - `description is empty`
            - `email does not contain '@'`
            - `comment is outlier` (by length)
            - `name contains 'test' or name contains 'demo'`
            - `address is missing`
            """)
        
        with example_tabs[3]:
            st.markdown("""
            **📅 Smart Date Rules:**
            - `order_date is future`
            - `created_at is weekend`
            - `last_login is old`
            - `birthday is recent`
            - `timestamp is missing`
            """)
        
        with example_tabs[4]:
            st.markdown("""
            **🔗 Complex Multi-Condition Rules:**
            - `amount is greater than 5000 and category is not 'premium'`
            - `age is less than 18 or age is greater than 80`
            - `status is 'active' and last_activity is old`
            - `price is outlier or quantity is unusual`
            """)
    
    # Main rule input interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 🎯 Create Your Rule")
        
        rule_name = st.text_input(
            "Rule Name:",
            placeholder="High Value Weekend Transactions",
            help="Give your rule a descriptive name"
        )
        
        rule_description = st.text_area(
            "Rule Description (optional):",
            placeholder="Detect unusually high transactions that occur on weekends",
            height=80
        )
        
        natural_rule = st.text_area(
            "Natural Language Rule:",
            placeholder="amount is greater than 5000 and transaction_date is weekend",
            height=120,
            help="Write your rule in plain English using the examples above as guidance"
        )
    
    with col2:
        st.markdown("### 📋 Available Columns")
        
        try:
            data_sources = load_all_data_sources()
            
            if data_sources:
                dataset_key = st.selectbox("Select Dataset:", list(data_sources.keys()))
                df = data_sources[dataset_key]
                
                st.markdown("**Columns in your data:**")
                for col in df.columns:
                    col_type = str(df[col].dtype)
                    col_icon = "📊" if pd.api.types.is_numeric_dtype(df[col]) else "📅" if pd.api.types.is_datetime64_any_dtype(df[col]) else "📝"
                    st.text(f"{col_icon} {col}")
                
                st.session_state.selected_dataset = dataset_key
            else:
                st.info("No data available. Please load data first.")
                
        except Exception as e:
            st.error(f"Error loading column info: {e}")
    
    # Parse and test rule
    if natural_rule and st.button("🔍 Parse and Test Rule", type="primary"):
        
        try:
            data_sources = load_all_data_sources()
            
            if not data_sources:
                st.error("No data available. Please load data first.")
                return
            
            dataset_key = getattr(st.session_state, 'selected_dataset', list(data_sources.keys())[0])
            df = data_sources[dataset_key]
            
            # Parse the rule
            with st.spinner("🧠 Parsing natural language rule..."):
                parser = st.session_state.nl_parser
                rule_structure = parser.parse_rule(natural_rule, df)
            
            # Display parsing results
            st.markdown("### 📝 Rule Parsing Results")
            
            if rule_structure['is_valid']:
                st.success("✅ Rule parsed successfully!")
                
                # Show parsed conditions
                with st.expander("🔍 Parsed Conditions", expanded=True):
                    for i, condition in enumerate(rule_structure['conditions'], 1):
                        st.markdown(f"**Condition {i}:**")
                        
                        condition_info = {
                            "Original Text": condition['original_text'],
                            "Column": condition['column'],
                            "Operator": condition['operator'],
                            "Value": str(condition.get('value', 'N/A')),
                            "Logical Operator": condition.get('logical_operator', 'N/A')
                        }
                        
                        st.json(condition_info)
                
                # Execute the rule
                with st.spinner("⚡ Executing rule on your data..."):
                    execution_result = parser.execute_rule(rule_structure, df)
                
                # Display execution results
                st.markdown("### 📊 Execution Results")
                
                if execution_result['success']:
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Records", len(df))
                    with col2:
                        st.metric("Anomalies Found", execution_result['anomaly_count'])
                    with col3:
                        st.metric("Anomaly Rate", f"{execution_result['anomaly_percentage']:.2f}%")
                    with col4:
                        severity = "🔴 High" if execution_result['anomaly_percentage'] > 10 else "🟡 Medium" if execution_result['anomaly_percentage'] > 2 else "🟢 Low"
                        st.metric("Severity", severity)
                    
                    # Show anomalous records
                    if execution_result['anomaly_count'] > 0:
                        st.markdown("**🚨 Anomalous Records:**")
                        
                        anomaly_df = df.iloc[execution_result['anomaly_indices']]
                        display_limit = min(50, len(anomaly_df))
                        st.dataframe(anomaly_df.head(display_limit), use_container_width=True)
                        
                        if len(anomaly_df) > display_limit:
                            st.info(f"Showing first {display_limit} anomalies out of {len(anomaly_df)} total")
                        
                        # Export options
                        st.markdown("### 📥 Export Anomalies")
                        
                        export_col1, export_col2 = st.columns(2)
                        
                        with export_col1:
                            csv_data = anomaly_df.to_csv(index=True)
                            st.download_button(
                                "📄 Download Anomalies (CSV)",
                                csv_data,
                                f"{rule_name.replace(' ', '_')}_anomalies.csv" if rule_name else "anomalies.csv",
                                "text/csv"
                            )
                        
                        with export_col2:
                            # Create detailed report
                            full_report = {
                                "rule_metadata": {
                                    "name": rule_name,
                                    "description": rule_description,
                                    "natural_language_rule": natural_rule,
                                    "created_at": datetime.now().isoformat()
                                },
                                "execution_results": execution_result,
                                "dataset_info": {
                                    "total_records": len(df),
                                    "columns": list(df.columns),
                                    "dataset_name": dataset_key
                                }
                            }
                            
                            json_data = json.dumps(full_report, indent=2, default=str)
                            st.download_button(
                                "📋 Download Report (JSON)",
                                json_data,
                                f"{rule_name.replace(' ', '_')}_report.json" if rule_name else "rule_report.json",
                                "application/json"
                            )
                    
                    else:
                        st.success("🎉 No anomalies found! Your data looks clean according to this rule.")
                
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
                5. **Check examples**: Refer to the examples above for proper syntax
                """)
        
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            st.info("Please check your rule syntax and try again.")


def create_universal_detection_interface(df: pd.DataFrame):
    """Create interface for universal anomaly detection"""
    
    st.subheader("🔍 Universal Anomaly Detection")
    st.info("Detect anomalies across all data types: numeric, categorical, text, and datetime columns")
    
    # Configuration options
    with st.expander("⚙️ Detection Configuration", expanded=True):
        col_config1, col_config2, col_config3 = st.columns(3)
        
        with col_config1:
            st.markdown("**Numeric Detection:**")
            statistical_threshold = st.slider("Statistical Threshold (Z-score)", 1.5, 5.0, 3.0, 0.5)
            include_iqr = st.checkbox("Include IQR Method", True)
        
        with col_config2:
            st.markdown("**Categorical Detection:**")
            frequency_threshold = st.slider("Rare Category Threshold (%)", 0.1, 5.0, 1.0, 0.1) / 100
            check_misspellings = st.checkbox("Check for Misspellings", True)
        
        with col_config3:
            st.markdown("**Text/Date Detection:**")
            text_length_threshold = st.slider("Text Length Threshold", 1.5, 5.0, 3.0, 0.5)
            check_future_dates = st.checkbox("Flag Future Dates", True)
    
    # Column selection
    all_columns = df.columns.tolist()
    selected_columns = st.multiselect(
        "Select columns to analyze:",
        all_columns,
        default=all_columns[:10],  # Select first 10 by default
        help="Select which columns to include in anomaly detection"
    )
    
    if not selected_columns:
        st.warning("Please select at least one column to analyze.")
        return
    
    # Run detection
    if st.button("🔍 Run Universal Anomaly Detection", type="primary"):
        
        config = {
            'statistical_threshold': statistical_threshold,
            'frequency_threshold': frequency_threshold,
            'text_length_threshold': text_length_threshold,
            'check_future_dates': check_future_dates,
            'check_misspellings': check_misspellings,
            'include_iqr': include_iqr
        }
        
        # Filter dataframe to selected columns
        df_subset = df[selected_columns].copy()
        
        with st.spinner("🔍 Analyzing all data types for anomalies..."):
            detector = UniversalAnomalyDetector()
            results = detector.detect_all_anomalies(df_subset, config)
        
        # Display results
        display_universal_results(results, df_subset)


def display_universal_results(results: Dict[str, Any], df: pd.DataFrame):
    """Display comprehensive anomaly detection results"""
    
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
                f"anomalies_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv"
            )
    
    with export_col2:
        # Export detailed results as JSON
        json_data = json.dumps(results, indent=2, default=str)
        st.download_button(
            "📋 Download Full Report (JSON)",
            json_data,
            f"anomaly_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.json",
            "application/json"
        )


def main():
    """Main function for universal anomaly detection"""
    
    # Header
    if STYLING_AVAILABLE:
        create_clean_header(
            "🔍 Universal Anomaly Detection", 
            "Advanced anomaly detection for all data types: numeric, categorical, text, and datetime"
        )
    else:
        st.title("🔍 Universal Anomaly Detection")
        st.markdown("Advanced anomaly detection for all data types: numeric, categorical, text, and datetime")
    
    # Navigation
    nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])
    with nav_col1:
        st.page_link("pages/4_Chat_with_Data.py", label="⬅ Chat with Data", icon="💬")
    with nav_col3:
        st.page_link("Home.py", label="Home ➡", icon="🏠")
    
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
    
    # Data selection
    st.markdown("### 📊 Select Dataset")
    
    selected_dataset = st.selectbox(
        "Choose dataset to analyze:",
        list(data_sources.keys()),
        help="Select which dataset to analyze for anomalies"
    )
    
    df = data_sources[selected_dataset]
    
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
    
    # Main interface tabs
    tab1, tab2 = st.tabs(["🔍 Universal Detection", "🧠 Natural Language Rules"])
    
    with tab1:
        create_universal_detection_interface(df)
    
    with tab2:
        # Enhanced Natural Language Rules with full functionality
        create_enhanced_natural_language_rules()


if __name__ == "__main__":
    main()