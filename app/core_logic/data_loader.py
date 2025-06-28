# core_logic/data_loader.py
import streamlit as st
import pandas as pd
from sqlalchemy import inspect
import os # <--- ADD THIS LINE

def load_all_data_sources():
    """
    Loads dataframes from CSVs and database tables stored in session state.
    Returns a dictionary of {display_name: dataframe}.
    """
    data_sources = {}

    if "csv_dataframes" in st.session_state:
        for name_key, df in st.session_state["csv_dataframes"].items():
            display_name = name_key.replace("csv_", "")
            data_sources[f"CSV: {display_name}"] = df

    if "engine" in st.session_state:
        engine = st.session_state["engine"]
        try:
            inspector = inspect(engine)
            for table in inspector.get_table_names():
                # Limit rows to avoid excessively large dataframes in memory for chat
                df = pd.read_sql(f"SELECT * FROM {table} LIMIT 10000", engine)
                data_sources[f"DB Table: {table}"] = df
        except Exception as e:
            st.error(f"❌ Failed to load database tables for chat: {e}")
    
    return data_sources

def get_selected_dfs(data_sources, selected_keys):
    """
    Prepares selected dataframes as a list of (name, dataframe) tuples
    for use with AI models.
    """
    return [(key.replace("DB Table: ", "").replace("CSV: ", ""), data_sources[key]) for key in selected_keys]

def get_base_name_from_selection(selected_keys):
    """
    Extracts a base name from the first selected key for file naming.
    """
    if selected_keys:
        base_name = selected_keys[0].replace("DB Table: ", "").replace("CSV: ", "")
        return os.path.splitext(base_name)[0] # This line needs 'os'
    return ""