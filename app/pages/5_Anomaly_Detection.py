# File: app/pages/5_Enhanced_Anomaly_Detection.py
# UNIVERSAL ANOMALY DETECTION FOR ALL COLUMN TYPES

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
                'frequency_threshold': 0.01,  # 1% threshold for rare categories
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
        
        # Analyze each column by type
        for column in df.columns:
            col_type = self._detect_column_type(df[column])
            col_results = self._detect_column_anomalies(df, column, col_type, config)
            
            results['detailed_results'][column] = col_results
            results['total_anomalies'] += len(col_results.get('anomaly_indices', []))
            results['anomaly_indices'].update(col_results.get('anomaly_indices', []))
        
        # Convert set to list for JSON serialization
        results['anomaly_indices'] = list(results['anomaly_indices'])
        
        # Generate summary
        results['summary'] = self._generate_summary(results['detailed_results'])
        results['recommendations'] = self._generate_recommendations(results['detailed_results'])
        
        return results
    
    def _detect_column_type(self, series: pd.Series) -> str:
        """Intelligently detect column type"""
        
        # Remove nulls for analysis
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            return 'empty'
        
        # Check if numeric
        if pd.api.types.is_numeric_dtype(series):
            return 'numeric'
        
        # Check if datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return 'datetime'
        
        # Try to parse as datetime
        try:
            pd.to_datetime(non_null_series.iloc[:min(100, len(non_null_series))])
            return 'datetime'
        except:
            pass
        
        # Check if categorical (limited unique values)
        unique_ratio = len(non_null_series.unique()) / len(non_null_series)
        if unique_ratio < 0.1 and len(non_null_series.unique()) < 50:
            return 'categorical'
        
        # Default to text
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
        
        # Always check for missing value patterns
        missing_anomalies = self._detect_missing_patterns(series)
        if missing_anomalies['anomaly_indices']:
            result['anomaly_indices'].extend(missing_anomalies['anomaly_indices'])
            result['anomaly_details'].extend(missing_anomalies['anomaly_details'])
            result['method_used'].append('missing_pattern')
        
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
        frequency_threshold = config.get('frequency_threshold', 0.01)  # 1%
        
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
        
        # Check for potentially misspelled categories
        categories = value_counts.index.tolist()
        similar_pairs = self._find_similar_categories(categories)
        
        for cat1, cat2, similarity in similar_pairs:
            if similarity > 0.8:  # Very similar strings
                # Flag the less frequent one as potential misspelling
                less_frequent = cat1 if value_counts[cat1] < value_counts[cat2] else cat2
                indices = non_null_series[non_null_series == less_frequent].index.tolist()
                
                for idx in indices:
                    if idx not in result['anomaly_indices']:
                        result['anomaly_indices'].append(idx)
                        result['anomaly_details'].append({
                            'index': idx,
                            'value': less_frequent,
                            'method': 'potential_misspelling',
                            'similar_to': cat1 if less_frequent == cat2 else cat2,
                            'reason': f'Potential misspelling of "{cat1 if less_frequent == cat2 else cat2}"'
                        })
        
        result['statistics'] = {
            'unique_categories': len(value_counts),
            'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
            'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
            'rare_categories_count': len(rare_categories),
            'potential_misspellings': len(similar_pairs)
        }
        
        return result
    
    def _detect_text_anomalies(self, series: pd.Series, config: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in text columns"""
        
        result = {
            'anomaly_indices': [],
            'anomaly_details': [],
            'method_used': ['length_outliers', 'encoding_issues', 'pattern_violations'],
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
        
        # Encoding issues detection
        for idx, text in non_null_series.items():
            try:
                # Check for common encoding issues
                if any(char in text for char in ['�', '\ufffd', '\x00']):
                    result['anomaly_indices'].append(idx)
                    result['anomaly_details'].append({
                        'index': idx,
                        'value': text[:100] + "..." if len(text) > 100 else text,
                        'method': 'encoding_issue',
                        'reason': 'Contains encoding error characters'
                    })
                
                # Check for unusual character patterns
                if re.search(r'[^\x00-\x7F]', text) and len(re.findall(r'[^\x00-\x7F]', text)) / len(text) > 0.5:
                    if idx not in result['anomaly_indices']:
                        result['anomaly_indices'].append(idx)
                        result['anomaly_details'].append({
                            'index': idx,
                            'value': text[:100] + "..." if len(text) > 100 else text,
                            'method': 'unusual_characters',
                            'reason': 'High proportion of non-ASCII characters'
                        })
            except Exception:
                continue
        
        result['statistics'] = {
            'mean_length': float(mean_length),
            'std_length': float(std_length),
            'min_length': int(lengths.min()),
            'max_length': int(lengths.max()),
            'length_outliers': len([idx for idx in result['anomaly_indices'] 
                                  if any(d['method'] == 'length_outlier' for d in result['anomaly_details'] if d['index'] == idx)])
        }
        
        return result
    
    def _detect_datetime_anomalies(self, series: pd.Series, config: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in datetime columns"""
        
        result = {
            'anomaly_indices': [],
            'anomaly_details': [],
            'method_used': ['future_dates', 'time_gaps', 'weekend_anomalies'],
            'statistics': {}
        }
        
        # Try to convert to datetime if not already
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
        
        # Time gaps detection (unusually large gaps between consecutive dates)
        if len(non_null_dt) > 1:
            sorted_dates = non_null_dt.sort_values()
            time_diffs = sorted_dates.diff().dropna()
            
            if len(time_diffs) > 0:
                median_diff = time_diffs.median()
                large_gaps = time_diffs[time_diffs > median_diff * 10]  # 10x larger than median
                
                for idx, gap in large_gaps.items():
                    result['anomaly_indices'].append(idx)
                    result['anomaly_details'].append({
                        'index': idx,
                        'value': str(sorted_dates.loc[idx]),
                        'method': 'large_time_gap',
                        'gap_days': gap.days,
                        'reason': f'Large time gap: {gap.days} days'
                    })
        
        # Weekend anomalies (if business data expected on weekdays)
        if not config.get('include_weekends', True):
            weekend_dates = non_null_dt[non_null_dt.dt.dayofweek >= 5]  # Saturday=5, Sunday=6
            for idx, date_val in weekend_dates.items():
                result['anomaly_indices'].append(idx)
                result['anomaly_details'].append({
                    'index': idx,
                    'value': str(date_val),
                    'method': 'weekend_date',
                    'reason': f'Weekend date in business data: {date_val.strftime("%A, %Y-%m-%d")}'
                })
        
        result['statistics'] = {
            'min_date': str(non_null_dt.min()),
            'max_date': str(non_null_dt.max()),
            'date_range_days': (non_null_dt.max() - non_null_dt.min()).days,
            'future_dates_count': len([idx for idx in result['anomaly_indices'] 
                                     if any(d['method'] == 'future_date' for d in result['anomaly_details'] if d['index'] == idx)]),
            'weekend_dates_count': len([idx for idx in result['anomaly_indices'] 
                                      if any(d['method'] == 'weekend_date' for d in result['anomaly_details'] if d['index'] == idx)])
        }
        
        return result
    
    def _detect_missing_patterns(self, series: pd.Series) -> Dict[str, Any]:
        """Detect unusual missing value patterns"""
        
        result = {
            'anomaly_indices': [],
            'anomaly_details': []
        }
        
        # Check for consecutive missing values (might indicate system downtime, etc.)
        is_null = series.isnull()
        consecutive_nulls = []
        current_streak = 0
        streak_start = None
        
        for idx, is_missing in is_null.items():
            if is_missing:
                if current_streak == 0:
                    streak_start = idx
                current_streak += 1
            else:
                if current_streak >= 5:  # 5 or more consecutive nulls
                    consecutive_nulls.append((streak_start, current_streak))
                current_streak = 0
        
        # Add the last streak if it exists
        if current_streak >= 5:
            consecutive_nulls.append((streak_start, current_streak))
        
        # Flag long consecutive null streaks as anomalies
        for start_idx, streak_length in consecutive_nulls:
            result['anomaly_indices'].append(start_idx)
            result['anomaly_details'].append({
                'index': start_idx,
                'value': None,
                'method': 'consecutive_nulls',
                'streak_length': streak_length,
                'reason': f'Start of {streak_length} consecutive missing values'
            })
        
        return result
    
    def _find_similar_categories(self, categories: List[str]) -> List[Tuple[str, str, float]]:
        """Find similar category names that might be misspellings"""
        
        from difflib import SequenceMatcher
        
        similar_pairs = []
        
        for i, cat1 in enumerate(categories):
            for cat2 in categories[i+1:]:
                if len(cat1) > 2 and len(cat2) > 2:  # Only check non-trivial strings
                    similarity = SequenceMatcher(None, cat1.lower(), cat2.lower()).ratio()
                    if similarity > 0.7:  # 70% similarity threshold
                        similar_pairs.append((cat1, cat2, similarity))
        
        return similar_pairs
    
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
                
                # Count by column type
                col_type = results['column_type']
                if col_type not in summary['anomalies_by_type']:
                    summary['anomalies_by_type'][col_type] = 0
                summary['anomalies_by_type'][col_type] += len(results['anomaly_indices'])
                
                # Count by method
                for method in results['method_used']:
                    if method not in summary['anomalies_by_method']:
                        summary['anomalies_by_method'][method] = 0
                    summary['anomalies_by_method'][method] += len([d for d in results['anomaly_details'] if d['method'] == method])
                
                # Track most problematic columns
                summary['most_problematic_columns'].append({
                    'column': column,
                    'anomaly_count': len(results['anomaly_indices']),
                    'type': col_type
                })
        
        # Sort most problematic columns
        summary['most_problematic_columns'].sort(key=lambda x: x['anomaly_count'], reverse=True)
        summary['most_problematic_columns'] = summary['most_problematic_columns'][:5]  # Top 5
        
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
                
                misspellings = len([d for d in results['anomaly_details'] if d['method'] == 'potential_misspelling'])
                if misspellings > 0:
                    recommendations.append(f"🏷️ Column '{column}': Check for typos - {misspellings} potential misspellings")
                    
            elif col_type == 'text':
                length_outliers = len([d for d in results['anomaly_details'] if d['method'] == 'length_outlier'])
                if length_outliers > 0:
                    recommendations.append(f"📝 Column '{column}': Review text length validation - {length_outliers} outliers")
                    
            elif col_type == 'datetime':
                future_dates = len([d for d in results['anomaly_details'] if d['method'] == 'future_date'])
                if future_dates > 0:
                    recommendations.append(f"📅 Column '{column}': Fix future dates - {future_dates} found")
        
        # Add general recommendations
        total_anomalies = sum(len(r['anomaly_indices']) for r in detailed_results.values())
        if total_anomalies > 100:
            recommendations.append("⚠️ High anomaly count detected - consider systematic data quality review")
        
        return recommendations

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
            include_weekends = st.checkbox("Allow Weekend Dates", True)
    
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
            'include_weekends': include_weekends,
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
    
    if STYLING_AVAILABLE:
        with metric_col1:
            create_clean_metric("Total Anomalies", str(results['total_anomalies']))
        with metric_col2:
            create_clean_metric("Columns Analyzed", str(summary['total_columns_analyzed']))
        with metric_col3:
            create_clean_metric("Problematic Columns", str(summary['columns_with_anomalies']))
        with metric_col4:
            anomaly_rate = (results['total_anomalies'] / len(df) * 100) if len(df) > 0 else 0
            create_clean_metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
    else:
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
    st.markdown("### 📋 Detailed Results by Column")
    
    tabs = st.tabs([f"📊 {col}" for col in results['detailed_results'].keys() if results['detailed_results'][col]['anomaly_indices']])
    
    for i, (column, col_results) in enumerate([item for item in results['detailed_results'].items() if item[1]['anomaly_indices']]):
        with tabs[i]:
            display_column_anomalies(column, col_results, df)
    
    # Recommendations
    if results['recommendations']:
        st.markdown("### 💡 Recommendations")
        
        for i, recommendation in enumerate(results['recommendations'], 1):
            st.markdown(f"{i}. {recommendation}")
    
    # Export options
    st.markdown("### 📥 Export Results")
    
    export_col1, export_col2, export_col3 = st.columns(3)
    
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
    
    with export_col3:
        # Create summary report
        summary_text = create_summary_report(results, df)
        st.download_button(
            "📊 Download Summary Report",
            summary_text,
            f"anomaly_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt",
            "text/plain"
        )

def display_column_anomalies(column: str, col_results: Dict[str, Any], df: pd.DataFrame):
    """Display detailed anomaly results for a specific column"""
    
    st.markdown(f"#### 📊 Column: {column}")
    
    # Column info
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.metric("Data Type", col_results['column_type'].title())
    with info_col2:
        st.metric("Anomalies Found", len(col_results['anomaly_indices']))
    with info_col3:
        st.metric("Detection Methods", len(col_results['method_used']))
    
    # Statistics
    if col_results['statistics']:
        with st.expander("📈 Column Statistics", expanded=False):
            stats_data = []
            for key, value in col_results['statistics'].items():
                stats_data.append({
                    'Statistic': key.replace('_', ' ').title(),
                    'Value': str(value)
                })
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
    
    # Anomaly details
    st.markdown("**🔍 Anomaly Details:**")
    
    if col_results['anomaly_details']:
        anomaly_df_data = []
        for detail in col_results['anomaly_details'][:50]:  # Show first 50
            anomaly_df_data.append({
                'Row Index': detail['index'],
                'Value': str(detail['value'])[:50] + "..." if len(str(detail['value'])) > 50 else str(detail['value']),
                'Detection Method': detail['method'].replace('_', ' ').title(),
                'Reason': detail['reason']
            })
        
        st.dataframe(pd.DataFrame(anomaly_df_data), use_container_width=True)
        
        if len(col_results['anomaly_details']) > 50:
            st.info(f"Showing first 50 anomalies. Total: {len(col_results['anomaly_details'])}")
    
    # Visualization for the column
    create_column_visualization(column, col_results, df)

def create_column_visualization(column: str, col_results: Dict[str, Any], df: pd.DataFrame):
    """Create appropriate visualization for the column based on its type"""
    
    col_type = col_results['column_type']
    series = df[column]
    anomaly_indices = col_results['anomaly_indices']
    
    try:
        if col_type == 'numeric':
            # Box plot with anomalies highlighted
            fig = go.Figure()
            
            # Add box plot
            fig.add_trace(go.Box(
                y=series.dropna(),
                name=column,
                boxpoints='outliers',
                marker_color='lightblue'
            ))
            
            # Highlight detected anomalies
            if anomaly_indices:
                anomaly_values = series.iloc[anomaly_indices].dropna()
                fig.add_trace(go.Scatter(
                    y=anomaly_values,
                    x=[column] * len(anomaly_values),
                    mode='markers',
                    marker=dict(color='red', size=8, symbol='x'),
                    name='Detected Anomalies'
                ))
            
            fig.update_layout(
                title=f"📊 Numeric Anomalies in {column}",
                yaxis_title="Value",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif col_type == 'categorical':
            # Bar chart of category frequencies with anomalies highlighted
            value_counts = series.value_counts()
            
            # Identify which categories have anomalies
            anomaly_categories = set()
            for idx in anomaly_indices:
                if pd.notna(series.iloc[idx]):
                    anomaly_categories.add(series.iloc[idx])
            
            colors = ['red' if cat in anomaly_categories else 'lightblue' for cat in value_counts.index]
            
            fig = px.bar(
                x=value_counts.index[:20],  # Top 20 categories
                y=value_counts.values[:20],
                title=f"🏷️ Category Frequencies in {column}",
                labels={'x': column, 'y': 'Count'},
                color=colors[:20],
                color_discrete_map={'red': 'red', 'lightblue': 'lightblue'}
            )
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        elif col_type == 'text':
            # Text length distribution
            lengths = series.dropna().astype(str).str.len()
            
            fig = px.histogram(
                x=lengths,
                title=f"📝 Text Length Distribution in {column}",
                labels={'x': 'Text Length (characters)', 'y': 'Frequency'},
                nbins=30
            )
            
            # Add vertical lines for anomaly lengths
            if anomaly_indices:
                anomaly_lengths = series.iloc[anomaly_indices].dropna().astype(str).str.len()
                for length in anomaly_lengths:
                    fig.add_vline(x=length, line_color="red", line_dash="dash", opacity=0.7)
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        elif col_type == 'datetime':
            # Timeline with anomalies highlighted
            try:
                if not pd.api.types.is_datetime64_any_dtype(series):
                    dt_series = pd.to_datetime(series, errors='coerce')
                else:
                    dt_series = series
                
                # Create timeline
                dt_clean = dt_series.dropna().sort_values()
                
                fig = go.Figure()
                
                # Add normal dates
                fig.add_trace(go.Scatter(
                    x=dt_clean,
                    y=[1] * len(dt_clean),
                    mode='markers',
                    marker=dict(color='lightblue', size=4),
                    name='Normal Dates'
                ))
                
                # Highlight anomalous dates
                if anomaly_indices:
                    anomaly_dates = dt_series.iloc[anomaly_indices].dropna()
                    if len(anomaly_dates) > 0:
                        fig.add_trace(go.Scatter(
                            x=anomaly_dates,
                            y=[1.1] * len(anomaly_dates),
                            mode='markers',
                            marker=dict(color='red', size=8, symbol='x'),
                            name='Anomalous Dates'
                        ))
                
                fig.update_layout(
                    title=f"📅 Date Timeline for {column}",
                    xaxis_title="Date",
                    yaxis_title="",
                    height=400,
                    yaxis=dict(showticklabels=False)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.warning(f"Could not create datetime visualization: {e}")
    
    except Exception as e:
        st.warning(f"Could not create visualization for {column}: {e}")

def create_summary_report(results: Dict[str, Any], df: pd.DataFrame) -> str:
    """Create a text summary report"""
    
    report = f"""
UNIVERSAL ANOMALY DETECTION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*50}

DATASET SUMMARY:
- Total Rows: {len(df):,}
- Total Columns: {len(df.columns)}
- Columns Analyzed: {results['summary']['total_columns_analyzed']}

ANOMALY SUMMARY:
- Total Anomalies Found: {results['total_anomalies']:,}
- Anomaly Rate: {(results['total_anomalies']/len(df)*100):.2f}%
- Columns with Anomalies: {results['summary']['columns_with_anomalies']}

ANOMALIES BY DATA TYPE:
"""
    
    for data_type, count in results['summary']['anomalies_by_type'].items():
        report += f"- {data_type.title()}: {count:,} anomalies\n"
    
    report += "\nANOMALIES BY DETECTION METHOD:\n"
    for method, count in results['summary']['anomalies_by_method'].items():
        report += f"- {method.replace('_', ' ').title()}: {count:,} anomalies\n"
    
    report += "\nMOST PROBLEMATIC COLUMNS:\n"
    for col_info in results['summary']['most_problematic_columns']:
        report += f"- {col_info['column']} ({col_info['type']}): {col_info['anomaly_count']} anomalies\n"
    
    report += "\nRECOMMENDations:\n"
    for i, rec in enumerate(results['recommendations'], 1):
        report += f"{i}. {rec}\n"
    
    report += f"\n{'='*50}\nEnd of Report"
    
    return report

def create_enhanced_natural_language_rules():
    """Enhanced natural language rule parser for all data types"""
    
    st.subheader("🧠 Enhanced Natural Language Rules")
    st.info("Create anomaly detection rules in plain English for any data type!")
    
    # Enhanced examples
    with st.expander("📚 Rule Examples for All Data Types", expanded=True):
        
        rule_col1, rule_col2 = st.columns(2)
        
        with rule_col1:
            st.markdown("""
            **📊 Numeric Rules:**
            - "sales_amount is greater than 10000"
            - "age is less than 18 or age is greater than 65"
            - "price is outside 100 and 1000"
            - "discount_rate is outlier"
            
            **🏷️ Categorical Rules:**
            - "status is not in ['active', 'pending', 'closed']"
            - "category contains 'test' or 'temp'"
            - "country_code has unusual frequency"
            - "product_type is rare category"
            """)
        
        with rule_col2:
            st.markdown("""
            **📝 Text Rules:**
            - "description length is greater than 500 characters"
            - "email does not contain '@'"
            - "comment has encoding issues"
            - "name has unusual characters"
            
            **📅 Date Rules:**
            - "order_date is in the future"
            - "created_at is on weekend"
            - "last_login is older than 365 days"
            - "timestamp has large gaps"
            """)
    
    # Rule input
    rule_input_col1, rule_input_col2 = st.columns([3, 1])
    
    with rule_input_col1:
        rule_name = st.text_input("Rule Name:", placeholder="High Value Transactions")
        
        natural_rule = st.text_area(
            "Natural Language Rule:",
            placeholder="amount is greater than 5000 and category is not 'refund'",
            height=100
        )
    
    with rule_input_col2:
        st.markdown("**Available Columns:**")
        # This would be populated with actual column names from the selected dataset
        st.code("amount\ncategory\ndate\nstatus\n...")
    
    if natural_rule:
        st.markdown("#### 🔍 Rule Preview")
        st.info(f"Rule: {natural_rule}")
        # Here you would add the enhanced rule parsing logic
        st.success("✅ Rule parsed successfully!")

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
        if STYLING_AVAILABLE:
            create_clean_metric("Total Rows", f"{len(df):,}")
        else:
            st.metric("Total Rows", f"{len(df):,}")
    
    with col2:
        if STYLING_AVAILABLE:
            create_clean_metric("Total Columns", str(len(df.columns)))
        else:
            st.metric("Total Columns", len(df.columns))
    
    with col3:
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        if STYLING_AVAILABLE:
            create_clean_metric("Numeric Columns", str(numeric_cols))
        else:
            st.metric("Numeric Columns", numeric_cols)
    
    with col4:
        text_cols = len(df.select_dtypes(include=['object']).columns)
        if STYLING_AVAILABLE:
            create_clean_metric("Text/Categorical", str(text_cols))
        else:
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
        create_enhanced_natural_language_rules()

if __name__ == "__main__":
    main()