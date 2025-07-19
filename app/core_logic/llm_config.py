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
    
    # Show token status
    token_status = "🔑 Configured" if user_token else "🔓 Not Required" if model_backend == "ollama" else "🔓 Not Set"
    st.markdown(f"**API Token Status:** {token_status}")
    

    
    return llm_backend, model_backend, model_name, user_token