# 1. Imports (only those absolutely necessary *before* page config)
import streamlit as st
import pandas as pd # pandas import doesn't trigger st commands
# Import authentication decorator *after* st.set_page_config
from auth.page_protection import require_auth

# --- Page Configuration ---
st.set_page_config(
    page_title="Profile Tables",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Original page content below (indented)
    # app/pages/3_Profile_Tables.py

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import inspect
from data_quality.quality_checks import run_quality_checks
from data_quality.convert_dates import convert_dates
import streamlit.components.v1 as components
import os

# Import ProfileReport directly
try:
    from ydata_profiling import ProfileReport
except ImportError:
    st.error("Error: ydata-profiling not found. Please install it using pip install ydata-profiling.")
    st.stop()


from assets.streamlit_styles import apply_professional_styling, create_nav_header

apply_professional_styling()


@require_auth()
def main():
    """Protected main function for Profile Tables"""
   # --- Navigation Header ---
    create_nav_header("📊 Data Profiling & Quality Analysis", "Generate comprehensive insights and quality reports for your datasets")
    
    # --- Data Source Collection ---
    df_sources = []
    
    # Check for CSV data
    csv_dataframes_available = "csv_dataframes" in st.session_state and st.session_state["csv_dataframes"]
    db_engine_available = "engine" in st.session_state
    
    selected_sources_options = []
    if csv_dataframes_available:
        selected_sources_options.extend([f"📁 {name.replace('csv_', '')}" for name in st.session_state["csv_dataframes"].keys()])
    if db_engine_available:
        try:
            inspector = inspect(st.session_state["engine"])
            db_tables = inspector.get_table_names()
            selected_sources_options.extend([f"🗄️ {table}" for table in db_tables])
        except Exception as e:
            st.warning(f"⚠ Could not retrieve DB tables: {e}")
    
    # --- Dataset Selection ---
    
    if not selected_sources_options:
        st.error("❌ No data sources found. Please go back to 'Load Data' and upload CSV files or connect to a database.")
        st.stop()
    
    selected_sources = st.multiselect(
        "Choose datasets to profile:",
        selected_sources_options,
        default=selected_sources_options[:3],  # Select first 3 by default
        help="Select the datasets you want to profile. You can choose multiple datasets."
    )
    
    if not selected_sources:
        st.warning("⚠️ Please select at least one dataset to profile.")
        st.stop()
    
    # Populate df_sources based on selections
    for source_id in selected_sources:
        if source_id.startswith("📁"):
            original_key = f"csv_{source_id.replace('📁 ', '')}"
            if original_key in st.session_state["csv_dataframes"]:
                df_sources.append((source_id, st.session_state["csv_dataframes"][original_key]))
            else:
                st.error(f"❌ Error: CSV data for '{source_id}' not found.")
        elif source_id.startswith("🗄️"):
            table_name = source_id.replace("🗄️ ", "")
            try:
                with st.spinner(f"📥 Loading table '{table_name}'..."):
                    df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 10000", st.session_state["engine"])
                    df_sources.append((source_id, df))
            except Exception as e:
                st.error(f"❌ Could not load DB table **{table_name}**: {e}")
    
    # --- Dataset Overview ---
    if df_sources:
        st.markdown("### 📋 Dataset Overview")
        
        overview_data = []
        total_rows = 0
        total_cols = 0
        
        for name, df in df_sources:
            overview_data.append({
                "Dataset": name,
                "Rows": f"{len(df):,}",
                "Columns": len(df.columns),
                "Memory (MB)": f"{df.memory_usage(deep=True).sum() / (1024*1024):.2f}",
                "Missing %": f"{(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100):.1f}%"
            })
            total_rows += len(df)
            total_cols += len(df.columns)
        
        # Display overview metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Total Datasets", len(df_sources))
        with metric_col2:
            st.metric("Total Rows", f"{total_rows:,}")
        with metric_col3:
            st.metric("Total Columns", total_cols)
        with metric_col4:
            total_memory = sum(df.memory_usage(deep=True).sum() for _, df in df_sources) / (1024*1024)
            st.metric("Total Memory", f"{total_memory:.1f} MB")
        
        # Overview table
        st.dataframe(pd.DataFrame(overview_data), use_container_width=True)
    
    st.markdown("---")
    
    # --- Enhanced Profiling Function ---
    def generate_enhanced_profile(df_name, df_data):
        """Generate and display enhanced profile with better visualizations"""
        
        st.markdown(f"""
        <div class="profile-container">
            <h2>📈 Profiling: {df_name}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Basic info
        info_col1, info_col2, info_col3, info_col4 = st.columns(4)
        
        with info_col1:
            st.metric("Rows", f"{len(df_data):,}")
        with info_col2:
            st.metric("Columns", len(df_data.columns))
        with info_col3:
            st.metric("Memory", f"{df_data.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
        with info_col4:
            missing_pct = (df_data.isnull().sum().sum() / (len(df_data) * len(df_data.columns)) * 100)
            st.metric("Missing %", f"{missing_pct:.1f}%")
        
        # Data preview with enhanced styling
        with st.expander("🔍 Data Preview & Column Information", expanded=False):
            preview_col1, preview_col2 = st.columns([2, 1])
            
            with preview_col1:
                st.markdown("**Sample Data:**")
                st.dataframe(df_data.head(10), use_container_width=True)
            
            with preview_col2:
                st.markdown("**Column Details:**")
                col_info = []
                for col in df_data.columns:
                    dtype = str(df_data[col].dtype)  # Convert to string
                    null_count = int(df_data[col].isnull().sum())  # Convert to int
                    unique_count = int(df_data[col].nunique())  # Convert to int
                    col_info.append({
                        "Column": str(col),  # Ensure string
                        "Type": dtype,
                        "Nulls": null_count,
                        "Unique": unique_count
                    })
                st.dataframe(pd.DataFrame(col_info), use_container_width=True)
        
        # Data processing
        df_processed = convert_dates(df_data.copy())
        df_cleaned = df_processed.copy()
        
        initial_columns = df_cleaned.shape[1]
        df_cleaned.dropna(axis=1, how="all", inplace=True)
        df_cleaned.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
        
        if df_cleaned.shape[1] < initial_columns:
            st.info(f"ℹ️ Dropped {initial_columns - df_cleaned.shape[1]} column(s) with all missing values")
        
        if df_cleaned.empty:
            st.warning(f"⚠️ DataFrame '{df_name}' is empty after cleaning. Cannot generate profile.")
            return
        
        # Enhanced Quality Checks
        st.markdown("### ✅ Data Quality Assessment")
        
        with st.spinner("🔍 Running quality checks..."):
            quality_results = run_quality_checks(df_processed)
            
            quality_col1, quality_col2, quality_col3 = st.columns(3)
            
            with quality_col1:
                missing_vals = quality_results.get("Missing Values (%)", {})
                avg_missing = sum(missing_vals.values()) / len(missing_vals) if missing_vals else 0
                quality_class = "quality-pass" if avg_missing < 5 else "quality-warn" if avg_missing < 20 else "quality-fail"
                
                st.markdown(f"""
                <div class="metric-card {quality_class}">
                    <h4>📊 Missing Values</h4>
                    <p><strong>{avg_missing:.1f}%</strong> average missing</p>
                    <small>{len([v for v in missing_vals.values() if v > 0])} columns affected</small>
                </div>
                """, unsafe_allow_html=True)
            
            with quality_col2:
                duplicate_count = quality_results.get("Duplicate Rows", 0)
                duplicate_pct = (duplicate_count / len(df_data) * 100) if len(df_data) > 0 else 0
                quality_class = "quality-pass" if duplicate_pct < 1 else "quality-warn" if duplicate_pct < 5 else "quality-fail"
                
                st.markdown(f"""
                <div class="metric-card {quality_class}">
                    <h4>🔄 Duplicate Rows</h4>
                    <p><strong>{duplicate_count:,}</strong> duplicates</p>
                    <small>{duplicate_pct:.1f}% of total rows</small>
                </div>
                """, unsafe_allow_html=True)
            
            with quality_col3:
                constant_cols = quality_results.get("Constant Columns", [])
                quality_class = "quality-pass" if len(constant_cols) == 0 else "quality-warn"
                
                st.markdown(f"""
                <div class="metric-card {quality_class}">
                    <h4>📐 Constant Columns</h4>
                    <p><strong>{len(constant_cols)}</strong> constant</p>
                    <small>Columns with single value</small>
                </div>
                """, unsafe_allow_html=True)
        
        # Enhanced Visualizations
        st.markdown("### 📈 Visual Analysis")
        
        # Data type distribution
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            try:
                # Convert dtypes to string to avoid JSON serialization issues
                dtype_counts = df_cleaned.dtypes.astype(str).value_counts()
                
                # Create labels and values for the pie chart
                labels = [str(dtype) for dtype in dtype_counts.index]
                values = dtype_counts.values.tolist()
                
                fig_dtype = px.pie(
                    values=values,
                    names=labels,
                    title="📊 Data Types Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_dtype.update_layout(height=400)
                st.plotly_chart(fig_dtype, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not create data types chart: {e}")
                # Fallback display
                dtype_df = pd.DataFrame({
                    'Data Type': [str(dtype) for dtype in df_cleaned.dtypes.astype(str).value_counts().index],
                    'Count': df_cleaned.dtypes.astype(str).value_counts().values
                })
                st.dataframe(dtype_df, use_container_width=True)
        
        with viz_col2:
            # Missing values heatmap for top columns with missing data
            missing_data = df_cleaned.isnull().sum().sort_values(ascending=False)
            top_missing = missing_data.head(10)
            
            if top_missing.sum() > 0:
                try:
                    # Convert to basic Python types to avoid serialization issues
                    column_names = [str(col) for col in top_missing.index]
                    missing_counts = [int(count) for count in top_missing.values]
                    
                    fig_missing = px.bar(
                        x=column_names,
                        y=missing_counts,
                        title="🕳️ Top Columns with Missing Values",
                        labels={'x': 'Columns', 'y': 'Missing Count'},
                        color=missing_counts,
                        color_continuous_scale='Reds'
                    )
                    fig_missing.update_layout(height=400, xaxis_tickangle=-45)
                    st.plotly_chart(fig_missing, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not create missing values chart: {e}")
                    # Fallback display
                    missing_df = pd.DataFrame({
                        'Column': [str(col) for col in top_missing.index],
                        'Missing Count': [int(count) for count in top_missing.values]
                    })
                    st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("✅ No missing values found!")
        
        # Numeric columns analysis
        numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            st.markdown("#### 📊 Numeric Columns Analysis")
            
            # Create distribution plots for numeric columns
            num_cols_to_show = min(4, len(numeric_cols))
            if num_cols_to_show > 0:
                cols = st.columns(num_cols_to_show)
                
                for i, col in enumerate(numeric_cols[:num_cols_to_show]):
                    with cols[i]:
                        try:
                            fig_hist = px.histogram(
                                df_cleaned,
                                x=col,
                                title=f"📈 {col}",
                                nbins=30,
                                color_discrete_sequence=['#667eea']
                            )
                            fig_hist.update_layout(height=300, showlegend=False)
                            st.plotly_chart(fig_hist, use_container_width=True)
                            
                            # Show basic stats
                            stats = df_cleaned[col].describe()
                            st.metric(f"Mean", f"{float(stats['mean']):.2f}")
                            st.metric(f"Std", f"{float(stats['std']):.2f}")
                            
                        except Exception as e:
                            st.warning(f"Could not plot {col}: {e}")
        
        # Categorical columns analysis
        categorical_cols = df_cleaned.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            st.markdown("#### 🏷️ Categorical Columns Analysis")
            
            cat_analysis_col1, cat_analysis_col2 = st.columns(2)
            
            with cat_analysis_col1:
                # Show unique value counts for categorical columns
                cat_stats = []
                for col in categorical_cols[:10]:  # Show top 10
                    try:
                        unique_count = int(df_cleaned[col].nunique())  # Convert to int
                        most_common_series = df_cleaned[col].mode()
                        most_common = str(most_common_series.iloc[0]) if not most_common_series.empty else "N/A"
                        
                        # Truncate long strings
                        if len(most_common) > 20:
                            most_common = most_common[:20] + "..."
                        
                        cat_stats.append({
                            "Column": str(col),  # Ensure string
                            "Unique Values": unique_count,
                            "Most Common": most_common
                        })
                    except Exception as e:
                        # Skip problematic columns
                        st.warning(f"Could not analyze column {col}: {e}")
                        continue
                
                if cat_stats:
                    st.dataframe(pd.DataFrame(cat_stats), use_container_width=True)
                else:
                    st.info("No categorical data to display")
            
            with cat_analysis_col2:
                # Visualize top categorical column
                if len(categorical_cols) > 0:
                    try:
                        top_cat_col = categorical_cols[0]
                        value_counts = df_cleaned[top_cat_col].value_counts().head(10)
                        
                        if len(value_counts) > 0:
                            # Convert to basic Python types
                            categories = [str(cat) for cat in value_counts.index]
                            counts = [int(count) for count in value_counts.values]
                            
                            fig_cat = px.bar(
                                x=categories,
                                y=counts,
                                title=f"🏷️ Top Values in {top_cat_col}",
                                labels={'x': str(top_cat_col), 'y': 'Count'},
                                color=counts,
                                color_continuous_scale='Viridis'
                            )
                            fig_cat.update_layout(height=400, xaxis_tickangle=-45)
                            st.plotly_chart(fig_cat, use_container_width=True)
                        else:
                            st.info("No categorical data to visualize")
                    except Exception as e:
                        st.warning(f"Could not create categorical chart: {e}")
                        # Show fallback table
                        if len(categorical_cols) > 0:
                            top_cat_col = categorical_cols[0]
                            value_counts = df_cleaned[top_cat_col].value_counts().head(10)
                            fallback_df = pd.DataFrame({
                                'Category': [str(cat) for cat in value_counts.index],
                                'Count': [int(count) for count in value_counts.values]
                            })
                            st.dataframe(fallback_df, use_container_width=True)
        
        # Correlation analysis for numeric data
        if len(numeric_cols) > 1:
            st.markdown("#### 🔗 Correlation Analysis")
            
            try:
                correlation_matrix = df_cleaned[numeric_cols].corr()
                
                # Convert to basic types and handle any NaN values
                correlation_matrix = correlation_matrix.fillna(0)
                
                fig_corr = px.imshow(
                    correlation_matrix,
                    title="🔗 Correlation Heatmap",
                    color_continuous_scale='RdBu',
                    aspect='auto',
                    text_auto=True
                )
                fig_corr.update_layout(height=500)
                st.plotly_chart(fig_corr, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not create correlation heatmap: {e}")
                # Show correlation as a table instead
                try:
                    correlation_matrix = df_cleaned[numeric_cols].corr().fillna(0)
                    st.dataframe(correlation_matrix, use_container_width=True)
                except Exception as e2:
                    st.error(f"Could not display correlation data: {e2}")
        
        # Generate full profile report
        st.markdown("### 📋 Comprehensive Profile Report")
        
        base_name = df_name.replace(" ", "").replace(":", "").replace(".", "").strip()
        output_dir = "app/outputs"
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, f"{base_name}_profile_report.html")
        
        with st.spinner(f"📊 Generating comprehensive profile for {df_name}..."):
            try:
                # Generate the profile with optimized settings
                profile = ProfileReport(
                    df_cleaned,
                    title=f"{df_name} Profile Report",
                    explorative=True,
                    html={'style': {'full_width': True}},
                    progress_bar=False
                )
                
                # Save the report
                profile.to_file(report_path)
                
                # Display summary statistics
                with st.expander("📊 Profile Summary", expanded=True):
                    try:
                        summary = profile.get_description()
                        table_stats = summary.table;
                        
                        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                        
                        with summary_col1:
                            st.metric("Total Variables", table_stats['n_var'])
                        with summary_col2:
                            st.metric("Observations", f"{table_stats['n']:,}")
                        with summary_col3:
                            st.metric("Missing Cells", f"{table_stats['n_cells_missing']:,}")
                        with summary_col4:
                            st.metric("Duplicate Rows", f"{table_stats['n_duplicates']:,}")
                        
                        # Variable types breakdown
                        st.markdown("**Variable Types:**")
                        types_data = table_stats['types']
                        for var_type, count in types_data.items():
                            if count > 0:
                                st.write(f"• **{var_type.title()}**: {count}")
                                
                    except Exception as e:
                        st.warning(f"Could not extract detailed summary: {e}")
                
                # Embed the interactive report
                st.markdown("#### 🔍 Interactive Profile Report")
                
                with st.expander("View Full Interactive Report", expanded=False):
                    try:
                        with open(report_path, "r", encoding="utf-8") as f:
                            html_report = f.read()
                            components.html(html_report, height=800, scrolling=True)
                    except Exception as e:
                        st.error(f"Could not display interactive report: {e}")
                
                # Download option
                with open(report_path, "rb") as f:
                    st.download_button(
                        "📥 Download Full HTML Report",
                        f,
                        f"{base_name}_profile_report.html",
                        "text/html",
                        help="Download the complete interactive profiling report"
                    )
                
                st.success(f"✅ Profile report generated successfully!")
                
            except Exception as e:
                st.error(f"❌ Error generating profile for {df_name}: {e}")
                st.info("💡 This might be due to data complexity. Try with a smaller sample or check data quality.")
    
        st.markdown("---")
    
    # --- Process Each Dataset with Error Handling ---
    for source_name, df in df_sources:
        try:
            generate_enhanced_profile(source_name, df)
        except Exception as e:
            st.error(f"❌ Error profiling {source_name}: {str(e)}")
            st.info("💡 Try with a smaller dataset or check data quality")
            
            # Show basic fallback information
            with st.expander(f"Basic Info for {source_name}", expanded=True):
                basic_col1, basic_col2, basic_col3 = st.columns(3)
                with basic_col1:
                    st.metric("Rows", f"{len(df):,}")
                with basic_col2:
                    st.metric("Columns", len(df.columns))
                with basic_col3:
                    st.metric("Memory", f"{df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
                
                st.dataframe(df.head(), use_container_width=True)
    
    # --- Summary Dashboard ---
    if len(df_sources) > 1:
        st.markdown("## 📊 Multi-Dataset Summary Dashboard")
        
        # Aggregate statistics
        summary_stats = []
        for name, df in df_sources:
            try:
                stats = {
                    "Dataset": str(name),
                    "Rows": int(len(df)),
                    "Columns": int(len(df.columns)),
                    "Numeric Cols": int(len(df.select_dtypes(include=['number']).columns)),
                    "Text Cols": int(len(df.select_dtypes(include=['object']).columns)),
                    "Missing %": f"{float(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100):.1f}%",
                    "Memory (MB)": f"{float(df.memory_usage(deep=True).sum() / (1024*1024)):.2f}"
                }
                summary_stats.append(stats)
            except Exception as e:
                st.warning(f"Could not generate summary for {name}: {e}")
                continue
        
        if summary_stats:  # Only proceed if we have valid summary stats
            summary_df = pd.DataFrame(summary_stats)
            
            # Display summary table
            st.dataframe(summary_df, use_container_width=True)
            
            # Comparative visualizations
            comp_col1, comp_col2 = st.columns(2)
            
            with comp_col1:
                try:
                    # Dataset size comparison
                    fig_size = px.bar(
                        summary_df,
                        x="Dataset",
                        y="Rows",
                        title="📊 Dataset Size Comparison",
                        color="Rows",
                        color_continuous_scale="Blues"
                    )
                    fig_size.update_layout(height=400, xaxis_tickangle=-45)
                    st.plotly_chart(fig_size, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not create size comparison chart: {e}")
            
            with comp_col2:
                try:
                    # Column type distribution
                    fig_cols = px.bar(
                        summary_df,
                        x="Dataset",
                        y=["Numeric Cols", "Text Cols"],
                        title="🏷️ Column Types by Dataset",
                        barmode="stack"
                    )
                    fig_cols.update_layout(height=400, xaxis_tickangle=-45)
                    st.plotly_chart(fig_cols, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not create column types chart: {e}")
        else:
            st.warning("No valid summary statistics could be generated.")
    
    # --- Next Steps Section ---
    st.markdown("---")
    st.markdown("## 🚀 Next Steps")
    
    next_col1, next_col2, next_col3 = st.columns(3)
    
    with next_col1:
        st.markdown("""
        <div class="metric-card">
            <h4>💬 Chat with Your Data</h4>
            <p>Ask questions in natural language and get intelligent answers from your analyzed datasets.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with next_col2:
        st.markdown("""
        <div class="metric-card">
            <h4>📊 Advanced Analysis</h4>
            <p>Use the profiling insights to guide deeper statistical analysis and machine learning tasks.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with next_col3:
        st.markdown("""
        <div class="metric-card">
            <h4>📁 Export Results</h4>
            <p>Download your profiling reports and share insights with your team or stakeholders.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # --- Navigation Buttons ---
    
    nav_button_col1, nav_button_col2, nav_button_col3 = st.columns([1, 2, 1])
    
    with nav_button_col1:
        st.page_link("pages/2_Load_Data_CSV_or_Database.py", label="⬅ Load More Data", icon="📂")
    
    
    with nav_button_col3:
        st.page_link("pages/4_Chat_with_Data.py", label="Chat with Data ➡", icon="💬")
    
    # --- Performance Tips ---
    with st.expander("⚡ Performance Tips & Insights", expanded=False):
        st.markdown("""
        ### 💡 Optimization Recommendations
        
        **For Large Datasets:**
        - Consider sampling your data for faster profiling
        - Focus on key columns for initial analysis
        - Use database views to pre-filter data
        
        **Data Quality Improvements:**
        - Address high missing value percentages
        - Remove or investigate constant columns
        - Consider data type optimizations
        
        **Next Analysis Steps:**
        - Use chat functionality for specific questions
        - Export reports for documentation
        - Share insights with stakeholders
        """)
    
    # --- Footer ---
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px; margin-top: 2rem;">
        <h4>🎉 Profiling Complete!</h4>
        <p style="margin: 0; color: #666;">
            Your datasets have been thoroughly analyzed. Proceed to chat with your data for deeper insights!
        </p>
    </div>
    """, unsafe_allow_html=True)

# Call the protected main function
if __name__ == "__main__":
    main()
