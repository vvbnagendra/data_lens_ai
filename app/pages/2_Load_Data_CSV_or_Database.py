# app/pages/2_Load_Data_CSV_or_Database.py
import streamlit as st
import pandas as pd
from utils.db_connector import get_sqlalchemy_engine
from utils.file_handler import save_connection, load_connections
from assets.streamlit_styles import apply_professional_styling, create_nav_header

# --- Page Configuration ---
st.set_page_config(
    page_title="Load Data",
    page_icon="📂",
    layout="wide"
)

apply_professional_styling()

# --- Initialize data_loaded_successfully flag ---
# This is the ONLY change needed to resolve the NameError.
# It ensures the variable is always defined from the start.
data_loaded_successfully = False

# --- Navigation Header ---
create_nav_header("📂 Load Your Data", "Connect to databases or upload CSV files to begin your data analysis journey")

# --- Enhanced CSS ---
st.markdown("""
<style>
    .page-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .data-source-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 2px solid transparent;
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .data-source-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        border-color: #667eea;
    }
    
    .data-source-card.selected {
        border-color: #667eea;
        background: linear-gradient(135deg, #f8f9ff 0%, #f0f2ff 100%);
    }
    
    .upload-zone {
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 3rem;
        text-align: center;
        background: linear-gradient(135deg, #f8f9ff 0%, #f0f2ff 100%);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-zone:hover {
        border-color: #764ba2;
        background: linear-gradient(135deg, #f0f2ff 0%, #e8ebff 100%);
    }
    
    .success-card {
        background: linear-gradient(135deg, #d4f5d4 0%, #b8e6b8 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    
    .connection-card {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    
    .progress-step {
        display: flex;
        align-items: center;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .progress-step.completed {
        background: linear-gradient(135deg, #d4f5d4 0%, #b8e6b8 100%);
    }
    
    .progress-step.active {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
    }
</style>
""", unsafe_allow_html=True)

# --- Data Source Selection ---
st.markdown("### 🎯 Choose Your Data Source")

# Create two columns for data source options
source_col1, source_col2 = st.columns(2)

with source_col1:
    csv_card = st.container()
    with csv_card:
        st.markdown("""
        <div class="data-source-card">
            <div style="text-align: center;">
                <span style="font-size: 3rem;">📁</span>
                <h3>Upload CSV Files</h3>
                <p>Quick and easy - drag & drop your CSV files</p>
                <ul style="text-align: left; margin-top: 1rem;">
                    <li>✅ Support for multiple files</li>
                    <li>✅ Automatic data type detection</li>
                    <li>✅ Instant preview</li>
                    <li>✅ No setup required</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

with source_col2:
    db_card = st.container()
    with db_card:
        st.markdown("""
        <div class="data-source-card">
            <div style="text-align: center;">
                <span style="font-size: 3rem;">🗄️</span>
                <h3>Connect to Database</h3>
                <p>Connect to your existing PostgreSQL or MySQL databases</p>
                <ul style="text-align: left; margin-top: 1rem;">
                    <li>✅ Live data connection</li>
                    <li>✅ Multiple table support</li>
                    <li>✅ Saved connections</li>
                    <li>✅ Secure authentication</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Data source selection
data_source = st.radio(
    "Select your preferred data source:",
    ["📁 Upload CSV Files", "🗄️ Connect to Database"],
    horizontal=True,
    key="data_source_radio"
)

st.markdown("---")

# --- CSV Upload Section ---
if data_source == "📁 Upload CSV Files":
    st.markdown("### 📁 CSV File Upload")
    
    # Enhanced upload zone
    st.markdown("""
    <div class="upload-zone">
        <span style="font-size: 3rem;">📤</span>
        <h3 style="margin: 1rem 0;">Drag & Drop Your CSV Files</h3>
        <p>Or click the button below to browse and select files</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Choose CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help="You can upload multiple CSV files at once. Each file will be processed separately.",
        label_visibility="collapsed"
    )

    if uploaded_files:
        if "csv_dataframes" not in st.session_state:
            st.session_state["csv_dataframes"] = {}

        st.markdown("### 📋 Processing Your Files")
        
        success_count = 0
        total_files = len(uploaded_files)
        
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file in enumerate(uploaded_files):
            file_key = f"csv_{file.name}"
            
            # Update progress
            progress = (i + 1) / total_files
            progress_bar.progress(progress)
            status_text.text(f"Processing {file.name}... ({i+1}/{total_files})")
            
            try:
                if file_key not in st.session_state["csv_dataframes"]:
                    df = pd.read_csv(file, low_memory=False)
                    st.session_state["csv_dataframes"][file_key] = df
                    
                    # Success card for each file
                    st.markdown(f"""
                    <div class="success-card">
                        <h4>✅ {file.name}</h4>
                        <p><strong>Rows:</strong> {len(df):,} | <strong>Columns:</strong> {len(df.columns)} | 
                        <strong>Size:</strong> {file.size / 1024:.1f} KB</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Preview in expander
                    with st.expander(f"🔍 Preview: {file.name}", expanded=False):
                        col_preview1, col_preview2 = st.columns([2, 1])
                        with col_preview1:
                            st.dataframe(df.head(), use_container_width=True)
                        with col_preview2:
                            st.markdown("*Column Info:*")
                            for col in df.columns[:10]:  # Show first 10 columns
                                dtype = str(df[col].dtype)
                                null_count = df[col].isnull().sum()
                                st.text(f"{col}: {dtype} ({null_count} nulls)")
                            if len(df.columns) > 10:
                                st.text(f"... and {len(df.columns) - 10} more columns")
                    
                    success_count += 1
                else:
                    st.info(f"ℹ️ {file.name} is already loaded.")
                    success_count += 1
                    
            except Exception as e:
                st.error(f"❌ Error reading {file.name}: {e}")
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        if success_count > 0:
            data_loaded_successfully = True # Set the flag to True if CSVs are successfully processed
            st.balloons()  # Celebration animation

    elif "csv_dataframes" in st.session_state and st.session_state["csv_dataframes"]:
        st.markdown("""
        <div class="connection-card">
            <h4>📚 Previously Loaded Files</h4>
            <p>You have CSV files from a previous session. Upload new ones or proceed to the next step.</p>
        </div>
        """, unsafe_allow_html=True)
        data_loaded_successfully = True # Set the flag to True if existing CSVs are in session state
    else:
        st.info("👆 Please upload CSV files using the area above.")

# --- Database Connection Section ---
else:  # Database connection
    st.markdown("### 🗄️ Database Connection")
    
    # Load saved connections
    saved_connections = load_connections()
    connection_options = ["🆕 Create New Connection"] + [f"💾 {name}" for name in saved_connections.keys()]
    
    connection_choice = st.selectbox(
        "Choose connection option:",
        connection_options,
        help="Select a saved connection or create a new one"
    )

    if connection_choice.startswith("🆕"):
        # New connection form
        st.markdown("#### 🔧 Database Configuration")
        
        with st.container():
            st.markdown("""
            <div class="data-source-card">
                <h4>🔗 New Connection Setup</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Database type selection with icons
            db_col1, db_col2 = st.columns(2)
            
            with db_col1:
                if st.button("🐘 PostgreSQL", use_container_width=True):
                    st.session_state.db_type = "PostgreSQL"
            
            with db_col2:
                if st.button("🐬 MySQL", use_container_width=True):
                    st.session_state.db_type = "MySQL"
            
            # Get database type
            db_type = getattr(st.session_state, 'db_type', 'PostgreSQL')
            st.info(f"Selected: *{db_type}*")
            
            # Connection form
            with st.form("db_connection_form", clear_on_submit=False):
                form_col1, form_col2 = st.columns(2)
                
                with form_col1:
                    host = st.text_input("🌐 Host", "localhost", help="Database server hostname or IP")
                    user = st.text_input("👤 Username", help="Database username")
                    db_name = st.text_input("🗃️ Database Name", help="Name of the database to connect to")
                
                with form_col2:
                    port = st.text_input("🔌 Port", "5432" if db_type == "PostgreSQL" else "3306", help="Database port number")
                    password = st.text_input("🔐 Password", type="password", help="Database password")
                    label = st.text_input("🏷️ Save As (optional)", help="Friendly name for this connection")
                
                submitted = st.form_submit_button("🔗 Connect to Database", type="primary", use_container_width=True)
                
                if submitted:
                    with st.spinner("🔄 Establishing connection..."):
                        try:
                            engine = get_sqlalchemy_engine(db_type, host, port, user, password, db_name)
                            
                            # Test connection
                            with engine.connect() as conn:
                                pass  # Connection successful
                            
                            st.session_state["engine"] = engine
                            st.session_state["db_connection_info"] = {"db_type": db_type, "db_name": db_name}
                            
                            st.markdown(f"""
                            <div class="success-card">
                                <h4>✅ Connection Successful!</h4>
                                <p>Connected to <strong>{db_name}</strong> ({db_type}) on {host}:{port}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            data_loaded_successfully = True # Set the flag to True if DB connected
                            
                            # Save connection if label provided
                            if label:
                                save_connection(label, {
                                    "type": db_type, "host": host, "port": port,
                                    "user": user, "password": password, "db_name": db_name
                                })
                                st.success(f"💾 Connection saved as '{label}'")
                            
                            st.balloons()
                            
                        except Exception as e:
                            st.error(f"❌ Connection failed: {e}")
                            st.info("💡 Please check your credentials and ensure the database server is running")

    else:
        # Saved connection
        connection_name = connection_choice.replace("💾 ", "")
        conn_info = saved_connections[connection_name]
        
        st.markdown(f"#### 💾 Using Saved Connection: {connection_name}")
        
        # Display connection info
        st.markdown(f"""
        <div class="connection-card">
            <h4>🔗 Connection Details</h4>
            <p><strong>Type:</strong> {conn_info.get('type', 'N/A')}</p>
            <p><strong>Host:</strong> {conn_info.get('host', 'N/A')}:{conn_info.get('port', 'N/A')}</p>
            <p><strong>Database:</strong> {conn_info.get('db_name', 'N/A')}</p>
            <p><strong>User:</strong> {conn_info.get('user', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button(f"🔗 Connect to {connection_name}", type="primary", use_container_width=True):
            with st.spinner(f"🔄 Connecting to {connection_name}..."):
                try:
                    engine = get_sqlalchemy_engine(
                        conn_info["type"], conn_info["host"], conn_info["port"],
                        conn_info["user"], conn_info["password"], conn_info["db_name"]
                    )
                    
                    with engine.connect() as conn:
                        pass  # Test connection
                    
                    st.session_state["engine"] = engine
                    st.session_state["db_connection_info"] = {"db_type": conn_info["type"], "db_name": conn_info["db_name"]}
                    
                    st.markdown(f"""
                    <div class="success-card">
                        <h4>✅ Connected Successfully!</h4>
                        <p>Using saved connection to <strong>{conn_info['db_name']}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    data_loaded_successfully = True # Set the flag to True if saved DB connected
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Connection failed: {e}")

# --- Status Summary ---
st.markdown("---")
st.markdown("### 📊 Current Status")

status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:
    csv_count = len(st.session_state.get("csv_dataframes", {}))
    st.metric("CSV Files Loaded", csv_count, help="Number of CSV files currently loaded")

with status_col2:
    db_status = "Connected" if "engine" in st.session_state else "Not Connected"
    st.metric("Database Status", db_status, help="Current database connection status")

with status_col3:
    next_step = "Ready ✅" if data_loaded_successfully else "Waiting 🔄"
    st.metric("Next Step", next_step, help="Whether you can proceed to data profiling")

# --- Next Steps ---
if data_loaded_successfully:
    st.markdown("---")
    st.markdown("""
    <div class="success-card">
        <h3>🎉 Great! Your data is ready for analysis</h3>
        <p>You can now proceed to profile your data and generate comprehensive insights.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation buttons
    # st.markdown("### 🚀 Continue Your Journey")
    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
    
    with nav_col3:
        st.page_link(
            "pages/3_Profile_Tables.py",
            label="📊 Continue to Data Profiling",
            icon="➡️",
            help="Proceed to generate comprehensive data profiles and quality reports"
        )

else:
    st.markdown("---")
    st.info("💡 Please load your data (CSV files or database connection) to proceed to the next step.")

# --- Help Section ---
with st.expander("❓ Need Help?", expanded=False):
    help_col1, help_col2 = st.columns(2)
    
    with help_col1:
        st.markdown("""
        *📁 CSV File Tips:*
        - Supported format: .csv files only
        - Maximum file size: 200MB per file
        - Ensure your CSV has headers in the first row
        - Multiple files will be treated as separate datasets
        """)
    
    with help_col2:
        st.markdown("""
        *🗄️ Database Tips:*
        - Supported: PostgreSQL and MySQL
        - Ensure your database server is accessible
        - Check firewall settings if connection fails
        - Saved connections are stored locally and encrypted
        """)

# --- Footer ---
st.markdown("""
<div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px; margin-top: 2rem;">
    <p style="margin: 0; color: #666;">
        🔒 Your data is processed locally and never shared with third parties
    </p>
</div>
""", unsafe_allow_html=True)