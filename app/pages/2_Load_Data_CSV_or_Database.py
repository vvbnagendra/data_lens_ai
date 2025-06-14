# Directory: app/pages/2__Load_Data_CSV_or_Database.py
# this page is to load csvs or connect databases and connection

import streamlit as st
import pandas as pd
from utils.db_connector import get_sqlalchemy_engine
from utils.file_handler import save_connection, load_connections

# --- Page Configuration ---
st.set_page_config(
    page_title="Load Data",
    page_icon="📂", # Icon for this page in the sidebar
    layout="wide"
)

# --- Header Section with Navigation ---
col_nav1, col_nav2, col_nav3 = st.columns([1, 4, 1])
with col_nav1:
    st.page_link("Home.py", label="🏠 Home", icon="🏠")
with col_nav2:
    st.markdown("## 📂 Load Data: CSV or Database")
with col_nav3:
    # This button will be shown conditionally at the bottom
    pass

st.markdown("---")

data_loaded_successfully = False  # Flag to track if data is ready for profiling

# --- Data Source Selection ---
st.subheader("Select Your Data Source")
data_source = st.radio(
    "Choose how you want to load your data:",
    ["Upload CSV", "Connect to Database"],
    index=0, # Default to CSV upload
    help="Select 'Upload CSV' to load files from your computer, or 'Connect to Database' for SQL connections."
)

st.markdown("---")

# --- CSV Upload Section ---
if data_source == "Upload CSV":
    st.subheader("Upload CSV Files")
    uploaded_files = st.file_uploader(
        "Drag & Drop or Browse for one or more CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help="You can upload multiple CSV files at once."
    )

    if uploaded_files:
        if "csv_dataframes" not in st.session_state:
            st.session_state["csv_dataframes"] = {}

        success_count = 0
        for file in uploaded_files:
            file_key = f"csv_{file.name}" # Unique key for session state
            try:
                # Ensure the file is not already loaded
                if file_key not in st.session_state["csv_dataframes"]:
                    with st.spinner(f"Loading {file.name}..."):
                        df = pd.read_csv(file, low_memory=False)
                        st.session_state["csv_dataframes"][file_key] = df
                        st.success(f"✅ Successfully loaded *{file.name}*")
                        success_count += 1
                        with st.expander(f"Preview of *{file.name}*"):
                            st.dataframe(df.head())
                else:
                    st.info(f"ℹ *{file.name}* is already loaded.")
                    success_count += 1 # Count as already loaded for the flag
            except Exception as e:
                st.error(f"❌ Error reading *{file.name}*: {e}")
        
        if success_count > 0:
            data_loaded_successfully = True

    if not uploaded_files and "csv_dataframes" in st.session_state and st.session_state["csv_dataframes"]:
        st.info("You have previously loaded CSV files. Upload new ones or proceed.")
        data_loaded_successfully = True # Assume data is present from previous loads
    elif not uploaded_files:
        st.info("No CSV files uploaded yet.")


# --- Database Connection Section ---
else: # data_source == "Connect to Database"
    st.subheader("Connect to a Database")

    # Load previously saved connections
    saved_connections = load_connections()
    saved_names = ["New Connection"] + list(saved_connections.keys())

    connection_name = st.selectbox(
        "Choose a saved connection or create a new one:",
        saved_names,
        help="Select 'New Connection' to enter fresh credentials, or pick from your saved connections."
    )

    db_details_changed = False
    if connection_name == "New Connection":
        with st.expander("Enter New Database Credentials", expanded=True):
            db_type = st.selectbox("Database Type", ["PostgreSQL", "MySQL"], help="Select the type of database you want to connect to.")
            
            col_host, col_port = st.columns(2)
            with col_host:
                host = st.text_input("Host", "localhost", help="The hostname or IP address of your database server.")
            with col_port:
                port = st.text_input("Port", "5432" if db_type == "PostgreSQL" else "3306", help="The port number your database is listening on.")
            
            user = st.text_input("Username", help="The username for connecting to the database.")
            password = st.text_input("Password", type="password", help="The password for the specified username.")
            db_name = st.text_input("Database Name", help="The name of the database you want to connect to.")
            label = st.text_input("Save connection as (label)", "", help="Optional: A friendly name to save this connection for future use.")

            if st.button("Connect to Database", key="connect_new_db_button"):
                with st.spinner("Attempting connection..."):
                    try:
                        engine = get_sqlalchemy_engine(db_type, host, port, user, password, db_name)
                        engine.connect() # Attempt to connect to validate credentials
                        st.session_state["engine"] = engine
                        st.session_state["db_connection_info"] = {"db_type": db_type, "db_name": db_name} # Store relevant info
                        st.success(f"✅ Successfully connected to *{db_name}* ({db_type})!")
                        data_loaded_successfully = True

                        if label:
                            save_connection(label, {
                                "type": db_type, "host": host, "port": port,
                                "user": user, "password": password, "db_name": db_name
                            })
                            st.info(f"Connection saved as *'{label}'*.")
                    except Exception as e:
                        st.error(f"❌ Failed to connect: {e}")
                        data_loaded_successfully = False # Ensure flag is false on failure

    else: # Using a saved connection
        conn_info = saved_connections[connection_name]
        with st.expander(f"Details for saved connection: *{connection_name}*", expanded=True):
            st.write(f"*Type:* {conn_info.get('type', 'N/A')}")
            st.write(f"*Host:* {conn_info.get('host', 'N/A')}")
            st.write(f"*Port:* {conn_info.get('port', 'N/A')}")
            st.write(f"*User:* {conn_info.get('user', 'N/A')}")
            st.write(f"*Database:* {conn_info.get('db_name', 'N/A')}")
            # Do not display password for security

            if st.button(f"Connect to {connection_name}", key="connect_saved_db_button"):
                with st.spinner(f"Connecting to '{connection_name}'..."):
                    try:
                        engine = get_sqlalchemy_engine(
                            conn_info["type"], conn_info["host"], conn_info["port"],
                            conn_info["user"], conn_info["password"], conn_info["db_name"]
                        )
                        engine.connect()
                        st.session_state["engine"] = engine
                        st.session_state["db_connection_info"] = {"db_type": conn_info["type"], "db_name": conn_info["db_name"]}
                        st.success(f"✅ Connected to *{conn_info['db_name']}* using saved connection *'{connection_name}'*!")
                        data_loaded_successfully = True
                    except Exception as e:
                        st.error(f"❌ Failed to connect to '{connection_name}': {e}")
                        data_loaded_successfully = False

# --- Conditional Navigation Button ---
st.markdown("---")
if data_loaded_successfully:
    st.success("Data source is ready! You can now proceed to profile your data.")
    st.page_link(
        "pages/3_Profile_Tables.py",
        label="Proceed to Profile Tables ➡",
        icon="📊",
        help="Click to go to the data profiling page."
    )
else:
    st.info("💡 Please upload CSV files or successfully connect to a database to enable profiling.")