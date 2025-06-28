# core_logic/llm_config.py
import streamlit as st
from core_logic.utils import get_ollama_models, get_huggingface_models, get_google_models

def configure_llm_backend():
    """
    Displays UI for LLM backend and model configuration.
    Returns (llm_backend, model_backend, model_name, user_token).
    """
    st.subheader("Configure your AI Assistant")
    llm_backend = st.selectbox(
        "Choose LLM Backend:",
        ["pandasai", "lotus"],
        key="llm_backend_select",
        help="*PandasAI* for more general data analysis questions. *Lotus* (if configured) for specific functionality."
    )
    user_token = st.text_input("🔑 Enter your API Token (optional, for some LLMs)", type="password", key="api_token_input")

    model_name = None
    model_backend = None

    if llm_backend == "pandasai":
        ollama_models = get_ollama_models() or []
        hf_models = get_huggingface_models() or []
        google_models = get_google_models() or []

        combined_models = (
            [f"ollama: {m}" for m in ollama_models] +
            [f"hf: {m}" for m in hf_models] +
            [f"google: {m}" for m in google_models]
        )

        if not combined_models:
            combined_models = ["hf: gpt2"]

        model_name_label = st.selectbox(
            "Select Model:",
            combined_models,
            key="model_name_select",
            help="Choose the specific language model to power your chat."
        )

        if model_name_label.startswith("ollama: "):
            model_backend = "ollama"
            model_name = model_name_label.replace("ollama: ", "")
        elif model_name_label.startswith("hf: "):
            model_backend = "huggingface"
            model_name = model_name_label.replace("hf: ", "")
        elif model_name_label.startswith("google: "):
            model_backend = "google"
            model_name = model_name_label.replace("google: ", "")
        else:
            model_backend = "huggingface"
            model_name = model_name_label

        if model_backend == "google" and not user_token:
            st.warning("Please enter your Google API Key above if using Google models.")

    else: # Lotus backend
        model_name = st.text_input("Lotus Model Name", "lotus-mixtral", key="lotus_model_name_input", help="Specify the model name for the Lotus backend.")
        if not user_token:
            st.warning("Please enter your Lotus API Token above.")
    
    return llm_backend, model_backend, model_name, user_token