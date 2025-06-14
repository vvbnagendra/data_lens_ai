# Directory: app/pages/3_Profile_Tables.py
# Profile CSV or DB Tables visually and via quality checks

import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import inspect
from data_quality.quality_checks import run_quality_checks
# from data_quality.profiler import generate_profile # We will modify this or assume it uses ydata-profiling
from data_quality.convert_dates import convert_dates

# Import ProfileReport directly to ensure we are using ydata-profiling
try:
    from ydata_profiling import ProfileReport
except ImportError:
    st.error("Error: ydata-profiling not found. Please install it using pip install ydata-profiling.")
    st.stop() # Stop the app if the crucial library is missing

# Streamlit component for displaying HTML
import streamlit.components.v1 as components
import os # For path manipulation

# --- Page Configuration ---
st.set_page_config(
    page_title="Profile Tables",
    page_icon="📊", # Icon for this page in the sidebar
    layout="wide"
)

# --- Header Section with Navigation ---
col_nav1, col_nav2, col_nav3 = st.columns([1, 4, 1])
with col_nav1:
    st.page_link("pages/2_Load_Data_CSV_or_Database.py", label="⬅ Load Data", icon="📂")
with col_nav2:
    st.markdown("## 📊 Profile Tables")
with col_nav3:
    st.page_link("pages/4_Chat_with_data.py", label="Chat with Data ➡", icon="💬")

st.markdown("---")

df_sources = []   # To hold dataframes from CSV or DB

# --- 1. Check for uploaded CSV(s) ---
st.subheader("Select Data to Profile")
csv_dataframes_available = "csv_dataframes" in st.session_state and st.session_state["csv_dataframes"]
db_engine_available = "engine" in st.session_state

selected_sources_options = []
if csv_dataframes_available:
    # Ensure keys are consistently handled, e.g., 'csv_my_file.csv' vs 'my_file.csv'
    selected_sources_options.extend([f"CSV: {name.replace('csv_', '')}" for name in st.session_state["csv_dataframes"].keys()])
if db_engine_available:
    try:
        inspector = inspect(st.session_state["engine"])
        db_tables = inspector.get_table_names()
        selected_sources_options.extend([f"DB: {table}" for table in db_tables])
    except Exception as e:
        st.warning(f"⚠ Could not retrieve DB tables: {e}")

selected_sources = st.multiselect(
    "Choose datasets to profile:",
    selected_sources_options,
    help="Select one or more CSV files or database tables that you have loaded in the previous step."
)

# Populate df_sources based on selections
for source_id in selected_sources:
    if source_id.startswith("CSV:"):
        # Reconstruct the key used in session_state from the display name
        original_key = f"csv_{source_id.replace('CSV: ', '')}"
        if original_key in st.session_state["csv_dataframes"]:
            df_sources.append((source_id, st.session_state["csv_dataframes"][original_key]))
        else:
            st.error(f"❌ Error: CSV data for '{source_id}' not found in session state. Please re-upload.")
    elif source_id.startswith("DB:"):
        table_name = source_id.replace("DB: ", "")
        try:
            with st.spinner(f"Loading table '{table_name}'..."):
                # Fetch only a sample for profiling to prevent memory issues with very large tables
                df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 10000", st.session_state["engine"])
                df_sources.append((source_id, df))
        except Exception as e:
            st.error(f"❌ Could not load DB table *{table_name}*: {e}")

if not df_sources:
    st.warning("Please go back to 'Load Data' and ensure you've uploaded CSVs or successfully connected to a database and selected them here.")
    st.stop()

st.markdown("---")

# --- Function to generate and display profile ---
# @st.cache_resource is typically used for heavy objects like LLMs, not for generating reports
# Since report generation can be slow and depends on user interaction, we won't cache the function directly,
# but rather focus on caching the result (the report HTML) if needed, or re-generating on demand.
# For simplicity, we will regenerate the report every time the page reruns (if the user selects a dataframe).
def generate_and_display_profile(df_name, df_data):
    st.header(f"📈 Profiling: {df_name}")
    
    # Set base name for report and ensure output directory exists
    base_name = df_name.replace(" ", "").replace(":", "").replace(".", "").strip()
    output_dir = "app/outputs"
    os.makedirs(output_dir, exist_ok=True) # Ensure directory exists
    report_path = os.path.join(output_dir, f"{base_name}_profile_report.html")

    # Display source information and preview
    st.subheader(f"Data Preview for {df_name}:")
    with st.expander("Click to see raw data head", expanded=False):
        st.dataframe(df_data.head())

    # Convert date columns (if applicable) - use a copy to avoid modifying original df_data
    df_processed = convert_dates(df_data.copy())
    if not df_processed.equals(df_data): # Check if any dates were converted
        st.info("ℹ Date columns have been converted for better profiling.")
    
    # Clean DataFrame before profiling
    df_cleaned = df_processed.copy()
    initial_columns = df_cleaned.shape[1]
    
    # Drop columns with all NaNs - ensure it works on a copy
    df_cleaned.dropna(axis=1, how="all", inplace=True)
    if df_cleaned.shape[1] < initial_columns:
        st.info(f"ℹ Dropped {initial_columns - df_cleaned.shape[1]} column(s) with all missing values for profiling.")
    
    # Replace infinities with NA for better profiling compatibility
    df_cleaned.replace([float("inf"), float("-inf")], pd.NA, inplace=True)

    # Check if DataFrame is empty after cleaning
    if df_cleaned.empty:
        st.warning(f"⚠ DataFrame '{df_name}' is empty after cleaning. Cannot generate profile.")
        return # Exit the function if DataFrame is empty

    # Generate profile and display results
    st.subheader("Profiling Report")
    with st.spinner(f"Generating detailed profile for {df_name}... This might take a moment."):
        try:
            # Generate the profile using ydata_profiling.ProfileReport
            # You can add more arguments to ProfileReport if needed, e.g., minimal=True for faster reports
            profile = ProfileReport(df_cleaned, title=f"{df_name} Profile Report", html={"style":{"full_width":True}})

            # Display overview data from the profile object
            with st.expander("📋 Overview Summary", expanded=True):
                # Accessing properties directly from the ProfileReport object or its description
                # ydata-profiling 4.x has a more direct way to get summary statistics
                
                # These are attributes of the Report object after computation,
                # or accessed via profile.get_description().table
                try:
                    summary_data = profile.get_description().table
                    st.markdown(f"- *Rows*: {summary_data['n']} \n"
                                f"- *Columns*: {summary_data['n_var']} \n"
                                f"- *Missing cells*: {summary_data['n_cells_missing']} \n"
                                f"- *Duplicate rows*: {summary_data['n_duplicates']}")

                    st.markdown("*Data Types:*")
                    # Convert to dictionary if it's a pandas Series
                    types_counts = summary_data['types'].to_dict() if hasattr(summary_data['types'], 'to_dict') else summary_data['types']
                    for dtype, count in types_counts.items():
                        st.markdown(f"- *{dtype}*: {count}")
                except Exception as e:
                    st.warning(f"⚠ Failed to extract profiling summary: {e}. Raw description might be different. Error: {e}")
                    st.json(profile.get_description()) # Show full description for debugging

            # Run quality checks (using the df_processed before all-NaN column dropping)
# Run quality checks
            # Run quality checks
            st.subheader("✅ Data Quality Checks:")
            with st.expander("View Quality Check Results", expanded=False):
                quality_results = run_quality_checks(df_processed) # Use df_processed for quality checks
                
                # --- Debugging Quality Results --- (Keep this for now!)
                st.write("--- Debugging Quality Results ---")
                st.write(f"Type of quality_results: {type(quality_results)}")
                st.write("Content of quality_results:")
                st.json(quality_results) # Show the full structure
                st.write("--- End Debugging ---")

                # Check if quality_results is a dictionary
                if isinstance(quality_results, dict):
                    all_checks_passed = True # Assume all pass initially
                    for check_name, check_data in quality_results.items():
                        if isinstance(check_data, dict) and "status" in check_data:
                            # Check for "FAIL" or "WARN" statuses specifically
                            if check_data["status"] == "FAIL" or check_data["status"] == "WARN":
                                all_checks_passed = False
                                # Do NOT break here, continue to find other warnings/failures
                        else:
                            st.warning(f"⚠ Quality check '{check_name}' has unexpected format (missing 'status' key or not a dict).")
                            all_checks_passed = False # If format is wrong, consider overall status as "not passed"

                    if all_checks_passed:
                        st.success("All basic quality checks passed!")
                    else:
                        # This is the line that's printing now
                        st.error("Some quality checks failed or had an unexpected format. Please review the results.")
                else:
                    st.error("❌ Quality check results are not in the expected dictionary format.")
                    st.json(quality_results) # Display raw result if not a dict

            # Save profiling report to HTML file
            profile.to_file(report_path)
            st.success(f"Profiling report generated and saved to {report_path}")

            # Read the HTML file and display it using Streamlit components
            st.subheader("Interactive Profiling Report")
            with st.expander("View Interactive Report", expanded=True):
                with open(report_path, "r", encoding="utf-8") as f:
                    html_report = f.read()
                    # Use Streamlit components to embed the HTML
                    components.html(html_report, height=800, scrolling=True)

            # Download profiling report
            with open(report_path, "rb") as f:
                st.download_button(
                    "📥 Download Full HTML Report",
                    f,
                    f"{base_name}.html",
                    "text/html",
                    help="Download the complete interactive profiling report as an HTML file."
                )

        except Exception as e:
            st.error(f"❌ An error occurred during profiling for {df_name}: {e}")
            st.info("Please ensure your data is correctly formatted and ydata-profiling is installed.")
    
    st.markdown("### 📈 Visual Distributions:")
    # Show distribution charts (only for numeric columns for simplicity)
    numeric_cols = df_cleaned.select_dtypes("number").columns
    if not numeric_cols.empty:
        for col in numeric_cols:
            try:
                fig = px.histogram(df_cleaned, x=col, title=f"Distribution of {col}")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not plot histogram for {col}: {e}")
    else:
        st.info("No numeric columns found to plot distributions.")

    st.markdown("---") # Separator between multiple profiled dataframes

# --- 4. Profile each DataFrame ---
for source_name, df in df_sources:
    generate_and_display_profile(source_name, df)

# --- Navigation Buttons (Bottom of Page) ---
st.markdown("<br>", unsafe_allow_html=True) # Add some space
col_bottom_nav1, col_bottom_nav2 = st.columns([1, 1])
with col_bottom_nav1:
    st.page_link("pages/2_Load_Data_CSV_or_Database.py", label="⬅ Load More Data", icon="📂")
with col_bottom_nav2:
    st.page_link("pages/4_Chat_with_data.py", label="Proceed to Chat with Data➡",icon="💬")