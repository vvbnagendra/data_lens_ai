# app/Home.py
import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="Smart Data Profiler",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for enhanced visuals ---
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .main-title {
        color: white;
        font-size: 3rem;
        font-weight: bold;
        margin: 0;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-subtitle {
        color: #f0f0f0;
        font-size: 1.2rem;
        text-align: center;
        margin-top: 0.5rem;
        font-style: italic;
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .feature-title {
        color: #333;
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        color: #666;
        line-height: 1.6;
    }
    
    .workflow-step {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    
    .cta-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 25px;
        text-decoration: none;
        font-weight: bold;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
        display: inline-block;
        margin: 1rem 0;
    }
    
    .cta-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        text-decoration: none;
        color: white;
    }
    
    .stats-container {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
    }
    
    .footer-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin-top: 3rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- Hero Section ---
st.markdown("""
<div class="main-header">
    <h1 class="main-title">🧠 Smart Data Profiler</h1>
    <p class="main-subtitle">Unlock insights from your data with artificial intelligence</p>
</div>
""", unsafe_allow_html=True)

# --- Introduction & Overview ---
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### Welcome to Your Intelligent Data Hub! 🚀
    
    Transform your data exploration experience with our cutting-edge AI-powered platform. 
    Whether you're analyzing CSV files or connecting to databases, our Smart Data Profiler 
    makes data discovery intuitive, fast, and insightful.
    """)

with col2:
    st.image("https://via.placeholder.com/300x200/667eea/white?text=Data+Analytics", 
             caption="AI-Powered Data Analysis", use_column_width=True)

# --- Key Features Grid ---
st.markdown("## 🌟 Powerful Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📊</div>
        <div class="feature-title">Smart Data Profiling</div>
        <div class="feature-desc">
            Generate comprehensive statistical reports, detect data quality issues, 
            and visualize distributions automatically. Get insights in seconds, not hours.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">💬</div>
        <div class="feature-title">AI-Powered Chat</div>
        <div class="feature-desc">
            Ask questions in natural language and get instant answers. Choose between 
            PandasAI for analysis or Lotus for semantic search and filtering.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🔗</div>
        <div class="feature-title">Multi-Source Support</div>
        <div class="feature-desc">
            Connect to PostgreSQL, MySQL databases or upload CSV files. 
            Work with multiple data sources simultaneously.
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- AI Models Section ---
st.markdown("## 🤖 AI Models & Backends")

model_col1, model_col2 = st.columns(2)

with model_col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🐼</div>
        <div class="feature-title">PandasAI Backend</div>
        <div class="feature-desc">
            <strong>Perfect for:</strong><br>
            • Statistical analysis & calculations<br>
            • Data visualizations & charts<br>
            • Code generation<br>
            • Complex data transformations
        </div>
    </div>
    """, unsafe_allow_html=True)

with model_col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🪷</div>
        <div class="feature-title">Lotus Backend</div>
        <div class="feature-desc">
            <strong>Perfect for:</strong><br>
            • Semantic search & filtering<br>
            • Natural language queries<br>
            • Text understanding<br>
            • Intelligent data discovery
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Workflow Steps ---
st.markdown("## 🛣️ Get Started in 3 Simple Steps")

step_col1, step_col2, step_col3 = st.columns(3)

with step_col1:
    st.markdown("""
    <div class="workflow-step">
        <h3>1. 📂 Load Data</h3>
        Upload CSV files or connect to your database with our intuitive interface
    </div>
    """, unsafe_allow_html=True)

with step_col2:
    st.markdown("""
    <div class="workflow-step">
        <h3>2. 📊 Profile & Analyze</h3>
        Generate comprehensive reports and discover data quality insights automatically
    </div>
    """, unsafe_allow_html=True)

with step_col3:
    st.markdown("""
    <div class="workflow-step">
        <h3>3. 💬 Chat & Explore</h3>
        Ask questions in natural language and get intelligent answers from your data
    </div>
    """, unsafe_allow_html=True)

# --- Stats Section ---
st.markdown("""
<div class="stats-container">
    <h3 style="text-align: center; margin-bottom: 1rem;">📈 Platform Capabilities</h3>
""", unsafe_allow_html=True)

stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

with stat_col1:
    st.metric("AI Models", "15+", help="Multiple LLM backends including Ollama, HuggingFace, Google")

with stat_col2:
    st.metric("Data Sources", "Multiple", help="CSV files, PostgreSQL, MySQL databases")

with stat_col3:
    st.metric("Query Types", "Unlimited", help="Statistical, visual, semantic, and filtering queries")

with stat_col4:
    st.metric("Export Formats", "3+", help="HTML reports, images, CSV exports")

st.markdown("</div>", unsafe_allow_html=True)

# --- Call to Action ---
st.markdown("---")
st.markdown("### 🚀 Ready to Transform Your Data Analysis?")

# Center the button
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    st.page_link(
        "pages/2_Load_Data_CSV_or_Database.py",
        label="🚀 Start Your Data Journey",
        icon="👉",
        help="Click to begin by loading your data from CSV or database"
    )

# --- Feature Highlights ---
st.markdown("## ✨ What Makes Us Different")

highlight_col1, highlight_col2 = st.columns(2)

with highlight_col1:
    st.markdown("""
    ### 🎯 Intelligent Analysis
    - **Auto-detect data types** and suggest appropriate analyses
    - **Smart quality checks** identify issues before they become problems  
    - **Context-aware responses** understand your domain and data
    - **Multi-modal AI** combines text, numbers, and visual insights
    """)

with highlight_col2:
    st.markdown("""
    ### 🔧 Enterprise Ready
    - **Secure local processing** keeps your data private
    - **Scalable architecture** handles datasets of any size
    - **Multiple AI backends** for different use cases
    - **Export capabilities** for reports and visualizations
    """)

# --- Footer ---
st.markdown("""
<div class="footer-section">
    <h4>🎉 Start Exploring Your Data Today!</h4>
    <p>Join thousands of data professionals who trust Smart Data Profiler for their analysis needs.</p>
    <p style="margin-top: 2rem; color: #666; font-size: 0.9rem;">
        © 2023-2025 Smart Data Profiler. Built with ❤️ using Streamlit, PandasAI & Lotus
    </p>
</div>
""", unsafe_allow_html=True)