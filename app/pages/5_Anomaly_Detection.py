# File: app/pages/5_Anomaly_Detection.py
# ENHANCED VERSION WITH NATURAL LANGUAGE RULES

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import re
from datetime import datetime
from typing import Dict, List, Any

# Simple imports that should always work
try:
    from core_logic.data_loader import load_all_data_sources
    from core_logic.llm_config import configure_llm_backend
except ImportError:
    def load_all_data_sources():
        if "csv_dataframes" in st.session_state:
            return {f"CSV: {k.replace('csv_', '')}": v for k, v in st.session_state["csv_dataframes"].items()}
        return {}
    
    def configure_llm_backend():
        return "pandasai", "huggingface", "gpt2", ""

# Try to get clean styling
try:
    from app.assets.clean_styles import apply_clean_styling, create_clean_header, create_clean_metric
    STYLING_AVAILABLE = True
except ImportError:
    STYLING_AVAILABLE = False

st.set_page_config(
    page_title="Anomaly Detection & Rules",
    page_icon="🔍",
    layout="wide"
)

if STYLING_AVAILABLE:
    apply_clean_styling()

class NaturalLanguageRuleParser:
    """Parse natural language rules into executable anomaly detection rules"""
    
    def __init__(self):
        self.rule_patterns = {
            # Threshold patterns
            'threshold_gt': r'(\w+)\s+(?:is\s+)?(?:greater than|more than|above|>)\s+(\d+(?:\.\d+)?)',
            'threshold_lt': r'(\w+)\s+(?:is\s+)?(?:less than|below|under|<)\s+(\d+(?:\.\d+)?)',
            'threshold_eq': r'(\w+)\s+(?:is\s+)?(?:equals?|=)\s+(\d+(?:\.\d+)?)',
            
            # Range patterns
            'outside_range': r'(\w+)\s+(?:is\s+)?(?:outside|not between)\s+(\d+(?:\.\d+)?)\s+(?:and|to)\s+(\d+(?:\.\d+)?)',
            'inside_range': r'(\w+)\s+(?:is\s+)?(?:between|within)\s+(\d+(?:\.\d+)?)\s+(?:and|to)\s+(\d+(?:\.\d+)?)',
            
            # Statistical patterns
            'outlier': r'(\w+)\s+(?:is\s+)?(?:outlier|anomal|unusual|abnormal)',
            'extreme': r'(\w+)\s+(?:is\s+)?(?:extreme|very high|very low)',
            
            # Percentage patterns
            'percent_above': r'(\w+)\s+(?:is\s+)?above\s+(\d+)%\s+of\s+(?:normal|average|typical)',
            'percent_below': r'(\w+)\s+(?:is\s+)?below\s+(\d+)%\s+of\s+(?:normal|average|typical)',
            
            # Combination patterns
            'and_condition': r'(.+)\s+and\s+(.+)',
            'or_condition': r'(.+)\s+or\s+(.+)',
        }
    
    def parse_rule(self, rule_text: str, df_columns: List[str]) -> Dict[str, Any]:
        """Parse natural language rule into structured format"""
        
        rule_text = rule_text.lower().strip()
        
        parsed_rule = {
            "original_text": rule_text,
            "conditions": [],
            "logic": "and",  # default logic
            "confidence": 0.0,
            "executable_code": "",
            "description": ""
        }
        
        # Check for AND/OR logic
        if " and " in rule_text:
            parsed_rule["logic"] = "and"
            parts = rule_text.split(" and ")
        elif " or " in rule_text:
            parsed_rule["logic"] = "or"
            parts = rule_text.split(" or ")
        else:
            parts = [rule_text]
        
        # Parse each part
        for part in parts:
            condition = self._parse_single_condition(part.strip(), df_columns)
            if condition:
                parsed_rule["conditions"].append(condition)
                parsed_rule["confidence"] += 0.3
        
        # Generate executable code
        if parsed_rule["conditions"]:
            parsed_rule["executable_code"] = self._generate_code(parsed_rule, df_columns)
            parsed_rule["description"] = self._generate_description(parsed_rule)
        
        return parsed_rule
    
    def _parse_single_condition(self, condition_text: str, df_columns: List[str]) -> Dict[str, Any]:
        """Parse a single condition"""
        
        # Try to match against patterns
        for pattern_name, pattern in self.rule_patterns.items():
            if pattern_name in ['and_condition', 'or_condition']:
                continue
                
            match = re.search(pattern, condition_text)
            if match:
                return self._create_condition_from_match(pattern_name, match, df_columns)
        
        return None
    
    def _create_condition_from_match(self, pattern_name: str, match, df_columns: List[str]) -> Dict[str, Any]:
        """Create condition object from regex match"""
        
        groups = match.groups()
        
        if pattern_name == 'threshold_gt':
            column, value = groups
            if column in df_columns:
                return {
                    "type": "threshold",
                    "column": column,
                    "operator": ">",
                    "value": float(value),
                    "pattern": pattern_name
                }
        
        elif pattern_name == 'threshold_lt':
            column, value = groups
            if column in df_columns:
                return {
                    "type": "threshold",
                    "column": column,
                    "operator": "<",
                    "value": float(value),
                    "pattern": pattern_name
                }
        
        elif pattern_name == 'outside_range':
            column, min_val, max_val = groups
            if column in df_columns:
                return {
                    "type": "range",
                    "column": column,
                    "operator": "outside",
                    "min_value": float(min_val),
                    "max_value": float(max_val),
                    "pattern": pattern_name
                }
        
        elif pattern_name == 'outlier':
            column = groups[0]
            if column in df_columns:
                return {
                    "type": "statistical",
                    "column": column,
                    "operator": "outlier",
                    "threshold": 3.0,
                    "pattern": pattern_name
                }
        
        return None
    
    def _generate_code(self, parsed_rule: Dict[str, Any], df_columns: List[str]) -> str:
        """Generate executable Python code from parsed rule"""
        
        conditions_code = []
        
        for condition in parsed_rule["conditions"]:
            if condition["type"] == "threshold":
                code = f"(df['{condition['column']}'] {condition['operator']} {condition['value']})"
                conditions_code.append(code)
            
            elif condition["type"] == "range":
                if condition["operator"] == "outside":
                    code = f"((df['{condition['column']}'] < {condition['min_value']}) | (df['{condition['column']}'] > {condition['max_value']}))"
                else:
                    code = f"((df['{condition['column']}'] >= {condition['min_value']}) & (df['{condition['column']}'] <= {condition['max_value']}))"
                conditions_code.append(code)
            
            elif condition["type"] == "statistical":
                code = f"""(
                    np.abs((df['{condition['column']}'] - df['{condition['column']}'].mean()) / df['{condition['column']}'].std()) > {condition['threshold']}
                )"""
                conditions_code.append(code)
        
        if conditions_code:
            logic_operator = " & " if parsed_rule["logic"] == "and" else " | "
            final_code = f"anomaly_mask = {logic_operator.join(conditions_code)}"
            return final_code
        
        return ""
    
    def _generate_description(self, parsed_rule: Dict[str, Any]) -> str:
        """Generate human-readable description"""
        
        descriptions = []
        for condition in parsed_rule["conditions"]:
            if condition["type"] == "threshold":
                desc = f"{condition['column']} {condition['operator']} {condition['value']}"
                descriptions.append(desc)
            elif condition["type"] == "range":
                if condition["operator"] == "outside":
                    desc = f"{condition['column']} outside range {condition['min_value']}-{condition['max_value']}"
                else:
                    desc = f"{condition['column']} within range {condition['min_value']}-{condition['max_value']}"
                descriptions.append(desc)
            elif condition["type"] == "statistical":
                desc = f"{condition['column']} is statistical outlier"
                descriptions.append(desc)
        
        logic_word = " AND " if parsed_rule["logic"] == "and" else " OR "
        return logic_word.join(descriptions)

class RuleManager:
    """Manage saved anomaly detection rules"""
    
    def __init__(self):
        if "anomaly_rules" not in st.session_state:
            st.session_state.anomaly_rules = []
    
    def save_rule(self, rule_data: Dict[str, Any]) -> bool:
        """Save a rule to session state"""
        try:
            rule_data["id"] = len(st.session_state.anomaly_rules) + 1
            rule_data["created_at"] = datetime.now().isoformat()
            st.session_state.anomaly_rules.append(rule_data)
            return True
        except Exception as e:
            st.error(f"Failed to save rule: {e}")
            return False
    
    def get_rules(self) -> List[Dict[str, Any]]:
        """Get all saved rules"""
        return st.session_state.anomaly_rules
    
    def delete_rule(self, rule_id: int) -> bool:
        """Delete a rule"""
        try:
            st.session_state.anomaly_rules = [
                rule for rule in st.session_state.anomaly_rules 
                if rule["id"] != rule_id
            ]
            return True
        except Exception as e:
            st.error(f"Failed to delete rule: {e}")
            return False
    
    def execute_rule(self, rule: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """Execute a saved rule on data"""
        try:
            # Execute the generated code
            local_vars = {"df": df, "np": np, "pd": pd}
            exec(rule["executable_code"], globals(), local_vars)
            
            if "anomaly_mask" in local_vars:
                anomaly_indices = df.index[local_vars["anomaly_mask"]].tolist()
                return {
                    "success": True,
                    "anomalies": anomaly_indices,
                    "total_checked": len(df),
                    "rule_name": rule["name"]
                }
            else:
                return {"success": False, "error": "No anomaly_mask generated"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}

class SimpleAnomalyDetector:
    """Simple, reliable anomaly detection with rule support"""
    
    def detect_statistical_anomalies(self, df: pd.DataFrame, columns: list, threshold: float = 3.0):
        """Simple statistical anomaly detection using Z-scores"""
        results = {
            "anomalies": [],
            "total_checked": 0,
            "method": "Statistical Z-Score",
            "threshold": threshold
        }
        
        try:
            numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
            
            if not numeric_cols:
                return {"error": "No numeric columns found"}
            
            anomaly_indices = set()
            
            for col in numeric_cols:
                mean_val = df[col].mean()
                std_val = df[col].std()
                
                if std_val > 0:
                    z_scores = np.abs((df[col] - mean_val) / std_val)
                    col_anomalies = df.index[z_scores > threshold].tolist()
                    anomaly_indices.update(col_anomalies)
            
            results["anomalies"] = list(anomaly_indices)
            results["total_checked"] = len(df)
            results["columns_analyzed"] = numeric_cols
            
            return results
            
        except Exception as e:
            return {"error": f"Detection failed: {str(e)}"}

def create_natural_language_interface(df: pd.DataFrame):
    """Create interface for natural language rule definition"""
    
    st.subheader("🧠 Natural Language Anomaly Rules")
    
    # Initialize components
    parser = NaturalLanguageRuleParser()
    rule_manager = RuleManager()
    
    # Get available columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Rule creation interface
    with st.expander("📝 Create New Rule", expanded=True):
        st.markdown("**Define anomaly rules in natural language:**")
        
        # Examples
        st.info("""
        **Example rules:**
        - "sales_amount is greater than 5000"
        - "customer_age is less than 18 or customer_age is greater than 80"
        - "order_quantity is outside 1 and 10"
        - "discount_percent is outlier"
        - "sales_amount is above 200% of normal"
        """)
        
        col_rule1, col_rule2 = st.columns([3, 1])
        
        with col_rule1:
            rule_name = st.text_input(
                "Rule Name:",
                placeholder="e.g., 'High Value Orders'"
            )
            
            natural_rule = st.text_area(
                "Natural Language Rule:",
                placeholder="e.g., 'sales_amount is greater than 5000 and customer_age is less than 25'",
                height=80
            )
        
        with col_rule2:
            st.markdown("**Available Columns:**")
            for col in numeric_columns[:8]:  # Show first 8 columns
                st.code(col)
            if len(numeric_columns) > 8:
                st.text(f"... and {len(numeric_columns) - 8} more")
        
        # Parse and preview rule
        if natural_rule:
            st.markdown("#### 🔍 Rule Analysis")
            
            parsed_rule = parser.parse_rule(natural_rule, numeric_columns)
            
            analysis_col1, analysis_col2 = st.columns(2)
            
            with analysis_col1:
                confidence = parsed_rule["confidence"]
                confidence_color = "🟢" if confidence >= 0.6 else "🟡" if confidence >= 0.3 else "🔴"
                st.markdown(f"**Parsing Confidence:** {confidence_color} {confidence:.1%}")
                
                if parsed_rule["conditions"]:
                    st.markdown("**Parsed Conditions:**")
                    for i, condition in enumerate(parsed_rule["conditions"]):
                        st.write(f"{i+1}. {condition}")
                else:
                    st.warning("⚠️ Could not parse the rule. Please check syntax.")
            
            with analysis_col2:
                if parsed_rule["description"]:
                    st.markdown("**Rule Description:**")
                    st.info(parsed_rule["description"])
                
                if parsed_rule["executable_code"]:
                    st.markdown("**Generated Code:**")
                    st.code(parsed_rule["executable_code"], language="python")
        
        # Save rule
        if st.button("💾 Save Rule", type="primary"):
            if not rule_name or not natural_rule:
                st.error("Please provide both rule name and natural language rule")
            elif not parsed_rule["conditions"]:
                st.error("Could not parse the rule. Please check your syntax.")
            else:
                rule_data = {
                    "name": rule_name,
                    "natural_language": natural_rule,
                    "parsed_rule": parsed_rule,
                    "executable_code": parsed_rule["executable_code"],
                    "description": parsed_rule["description"]
                }
                
                if rule_manager.save_rule(rule_data):
                    st.success(f"✅ Rule '{rule_name}' saved successfully!")
                    st.rerun()
    
    # Saved rules management
    saved_rules = rule_manager.get_rules()
    
    if saved_rules:
        st.subheader("📚 Saved Rules")
        
        for rule in saved_rules:
            with st.expander(f"📋 {rule['name']}", expanded=False):
                rule_col1, rule_col2, rule_col3 = st.columns([2, 1, 1])
                
                with rule_col1:
                    st.markdown(f"**Natural Language:** `{rule['natural_language']}`")
                    st.markdown(f"**Description:** {rule['description']}")
                    st.markdown(f"**Created:** {rule['created_at'][:19]}")
                
                with rule_col2:
                    if st.button(f"🔍 Execute", key=f"exec_{rule['id']}"):
                        execute_saved_rule(rule, df, rule_manager)
                
                with rule_col3:
                    if st.button(f"🗑️ Delete", key=f"del_{rule['id']}"):
                        if rule_manager.delete_rule(rule["id"]):
                            st.success("Rule deleted!")
                            st.rerun()

def execute_saved_rule(rule: Dict[str, Any], df: pd.DataFrame, rule_manager: RuleManager):
    """Execute a saved rule and show results"""
    
    with st.spinner(f"Executing rule '{rule['name']}'..."):
        result = rule_manager.execute_rule(rule, df)
    
    if result["success"]:
        anomaly_count = len(result["anomalies"])
        total_count = result["total_checked"]
        anomaly_rate = (anomaly_count / total_count * 100) if total_count > 0 else 0
        
        st.markdown(f"### 📊 Results for '{rule['name']}'")
        
        result_col1, result_col2, result_col3 = st.columns(3)
        
        if STYLING_AVAILABLE:
            with result_col1:
                create_clean_metric("Anomalies Found", str(anomaly_count))
            with result_col2:
                create_clean_metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
            with result_col3:
                create_clean_metric("Rule Type", "Natural Language")
        else:
            with result_col1:
                st.metric("Anomalies Found", anomaly_count)
            with result_col2:
                st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
            with result_col3:
                st.metric("Rule Type", "Natural Language")
        
        if anomaly_count > 0:
            st.markdown("#### 🔍 Anomalous Records")
            anomaly_data = df.loc[result["anomalies"]]
            st.dataframe(anomaly_data, use_container_width=True)
            
            # Download option
            csv = anomaly_data.to_csv(index=False)
            st.download_button(
                "📥 Download Anomalies",
                csv,
                f"anomalies_{rule['name'].replace(' ', '_')}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv"
            )
        else:
            st.success("✅ No anomalies found using this rule!")
    
    else:
        st.error(f"❌ Rule execution failed: {result['error']}")

def main():
    """Main function with natural language rule support"""
    
    # Header
    if STYLING_AVAILABLE:
        create_clean_header("🔍 Anomaly Detection & Rules", "Find patterns with ML algorithms and natural language rules")
    else:
        st.title("🔍 Anomaly Detection & Rules")
        st.markdown("Find patterns with ML algorithms and natural language rules")
    
    # Load data
    try:
        data_sources = load_all_data_sources()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please load data first using the 'Load Data' page.")
        return
    
    if not data_sources:
        st.warning("No data found. Please load CSV files or connect to a database first.")
        return
    
    # Data selection
    st.subheader("📊 Select Data")
    
    selected_dataset = st.selectbox(
        "Choose dataset:",
        list(data_sources.keys()),
        help="Select which dataset to analyze"
    )
    
    df = data_sources[selected_dataset]
    
    # Basic info
    col1, col2, col3 = st.columns(3)
    
    if STYLING_AVAILABLE:
        with col1:
            create_clean_metric("Rows", f"{len(df):,}")
        with col2:
            create_clean_metric("Columns", str(len(df.columns)))
        with col3:
            create_clean_metric("Numeric Cols", str(len(df.select_dtypes(include=[np.number]).columns)))
    else:
        with col1:
            st.metric("Rows", f"{len(df):,}")
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
    
    # Tabbed interface
    tab1, tab2 = st.tabs(["🤖 ML Algorithms", "🧠 Natural Language Rules"])
    
    with tab1:
        # Traditional ML-based anomaly detection (existing code)
        st.subheader("🤖 Machine Learning Detection")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_columns:
            st.error("❌ No numeric columns found.")
            return
        
        col_config1, col_config2 = st.columns(2)
        
        with col_config1:
            selected_columns = st.multiselect(
                "Select columns to analyze:",
                numeric_columns,
                default=numeric_columns[:min(3, len(numeric_columns))],
                help="Choose which numeric columns to check for anomalies"
            )
        
        with col_config2:
            method = st.selectbox("Detection method:", ["Statistical", "IQR"])
            
            if method == "Statistical":
                threshold = st.slider(
                    "Sensitivity (higher = fewer anomalies):",
                    min_value=1.5,
                    max_value=4.0,
                    value=3.0,
                    step=0.5
                )
            else:
                threshold = None
        
        if selected_columns and st.button("🔍 Find Anomalies", type="primary"):
            with st.spinner("Analyzing data..."):
                detector = SimpleAnomalyDetector()
                results = detector.detect_statistical_anomalies(df, selected_columns, threshold or 3.0)
            
            if "error" not in results:
                anomaly_count = len(results["anomalies"])
                st.metric("Anomalies Found", anomaly_count)
                
                if anomaly_count > 0:
                    anomaly_data = df.loc[results["anomalies"]]
                    st.dataframe(anomaly_data[selected_columns])
    
    with tab2:
        # Natural language rule interface
        create_natural_language_interface(df)

if __name__ == "__main__":
    main()