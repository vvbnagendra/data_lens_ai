# app/core_logic/llm_config.py
import streamlit as st
from core_logic.utils import get_ollama_models, get_huggingface_models, get_google_models

def configure_llm_backend():
    """
    Enhanced UI for LLM backend and model configuration with standardized models.
    Both PandasAI and Lotus use the EXACT same models.
    Returns (llm_backend, model_backend, model_name, user_token).
    """
    
    # Initialize session state for model selection persistence
    if "selected_model_backend" not in st.session_state:
        st.session_state.selected_model_backend = "huggingface"
    if "selected_model_name" not in st.session_state:
        st.session_state.selected_model_name = get_huggingface_models()[0]
    
    # Create a more visually appealing header
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="color: white; margin: 0; text-align: center;">
            🤖 Configure your AI Assistant
        </h3>
        <p style="color: #f0f0f0; margin: 5px 0 0 0; text-align: center;">
            PandasAI and Lotus use identical models - only the prompting approach differs
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Backend selection with enhanced styling
    col1, col2 = st.columns([2, 1])
    
    with col1:
        llm_backend = st.selectbox(
            "🚀 Choose Processing Approach:",
            ["pandasai", "lotus"],
            key="llm_backend_select",
            help="**PandasAI**: Analysis, visualizations, statistical code generation\n**Lotus**: Semantic search, natural language filtering, intelligent data discovery",
            format_func=lambda x: f"🐼 PandasAI (Analysis)" if x == "pandasai" else f"🪷 Lotus (Semantic)"
        )
    
    with col2:
        # Show backend info card
        if llm_backend == "pandasai":
            st.info("🐼 **PandasAI**\n\n✨ Same models\n📊 Analysis focus\n📈 Visualizations")
        else:
            st.info("🪷 **Lotus**\n\n✨ Same models\n🔍 Semantic focus\n🎯 Smart filtering")
    
    # Emphasize model consistency
    st.success("💡 **Key Point**: Both backends use the EXACT same models. The only difference is how they process your questions!")
    
    # API Token input with better styling
    st.markdown("### 🔐 Authentication")
    user_token = st.text_input(
        "API Token (optional for some models):", 
        type="password", 
        key="api_token_input",
        help="Enter your API key for cloud models. Local models don't require tokens.",
        placeholder="sk-... or your API key"
    )

    # UNIFIED model selection for both backends
    st.markdown("### 🎯 Model Selection (Shared by Both Backends)")
    st.info("ℹ️ These models work identically for both PandasAI and Lotus")
    
    # Create tabs for different model types
    tab1, tab2, tab3 = st.tabs(["🏠 Local (Ollama)", "☁️ Cloud (HuggingFace)", "🔮 Google AI"])
    
    with tab1:
        st.markdown("**Local Models** - Run on your machine, no API key needed")
        ollama_models = get_ollama_models()
        
        if ollama_models:
            selected_ollama = st.selectbox(
                "Select Ollama Model:",
                ollama_models,
                key="ollama_model_select",
                help="Local models provide privacy and don't require internet connection"
            )
            if st.button("✅ Use This Ollama Model", key="use_ollama", type="primary"):
                st.session_state.selected_model_backend = "ollama"
                st.session_state.selected_model_name = selected_ollama
                st.success(f"🎉 Selected: **{selected_ollama}** (Ollama)")
                st.info(f"This model will work for both 🐼 PandasAI and 🪷 Lotus!")
                st.rerun()  # Refresh to update the display
        else:
            st.warning("⚠️ No Ollama models found. Please install Ollama and pull some models.")
            st.code("ollama pull mistral")
    
    with tab2:
        st.markdown("**Cloud Models** - Powerful models hosted online")
        hf_models = get_huggingface_models()
        
        selected_hf = st.selectbox(
            "Select HuggingFace Model:",
            hf_models,
            key="hf_model_select",
            help="Cloud models offer latest capabilities but require API tokens"
        )
        if st.button("✅ Use This HuggingFace Model", key="use_hf", type="primary"):
            st.session_state.selected_model_backend = "huggingface"
            st.session_state.selected_model_name = selected_hf
            st.success(f"🎉 Selected: **{selected_hf}** (HuggingFace)")
            st.info(f"This model will work for both 🐼 PandasAI and 🪷 Lotus!")
            if not user_token:
                st.warning("⚠️ Consider adding your HuggingFace token for better performance")
            st.rerun()  # Refresh to update the display
    
    with tab3:
        st.markdown("**Google AI Models** - Gemini and other Google models")
        google_models = get_google_models()
        
        selected_google = st.selectbox(
            "Select Google Model:",
            google_models,
            key="google_model_select",
            help="Google's advanced AI models with strong reasoning capabilities"
        )
        if st.button("✅ Use This Google Model", key="use_google", type="primary"):
            st.session_state.selected_model_backend = "google"
            st.session_state.selected_model_name = selected_google
            st.success(f"🎉 Selected: **{selected_google}** (Google)")
            st.info(f"This model will work for both 🐼 PandasAI and 🪷 Lotus!")
            if not user_token:
                st.warning("⚠️ Google models require an API key")
            st.rerun()  # Refresh to update the display
    
    # Get the current selections from session state
    model_backend = st.session_state.selected_model_backend
    model_name = st.session_state.selected_model_name
    
    # Show current configuration summary
    st.markdown("---")
    st.markdown("### 📋 Current Configuration")
    
    config_col1, config_col2, config_col3 = st.columns(3)
    
    with config_col1:
        st.metric(
            label="🤖 Processing Style", 
            value=llm_backend.title(),
            help="How your questions will be processed"
        )
    
    with config_col2:
        st.metric(
            label="🏭 Model Provider", 
            value=model_backend.title(),
            help="Where the model runs (Local vs Cloud)"
        )
    
    with config_col3:
        st.metric(
            label="🧠 AI Model", 
            value=model_name.split('/')[-1] if model_name else "Default",
            help="The specific AI model being used"
        )
    
    # Show token status
    token_status = "🔑 Configured" if user_token else "🔓 Not Required" if model_backend == "ollama" else "🔓 Not Set"
    st.markdown(f"**API Token Status:** {token_status}")
    
    # Final clarification with current selections
    st.markdown("---")
    st.markdown("### 🎯 What This Means")
    
    clarify_col1, clarify_col2 = st.columns(2)
    with clarify_col1:
        st.markdown(f"""
        **🐼 If you choose PandasAI:**
        - Uses: `{model_name}` ({model_backend})
        - Creates: Analysis & visualization code
        - Good for: Statistics, charts, calculations
        """)
    
    with clarify_col2:
        st.markdown(f"""
        **🪷 If you choose Lotus:**
        - Uses: `{model_name}` ({model_backend}) (SAME MODEL!)
        - Creates: Filtering & search code  
        - Good for: Finding data, semantic queries
        """)
    
    return llm_backend, model_backend, model_name, user_token