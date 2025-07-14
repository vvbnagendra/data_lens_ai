# app/data_quality/pandasai_chat.py
from pandasai import SmartDataframe, SmartDatalake
from data_quality.ollama_llm_adapter import OllamaLLM
from data_quality.google_llm_adapter import GoogleLLM
from data_quality.huggingface_llm_adapter import HuggingFaceLLM
import pandas as pd

SUPPORTED_BACKENDS = ["ollama", "huggingface", "google"]

def get_llm_model(backend: str, model_name: str = None, api_key: str = None):
    """Get the appropriate LLM model based on backend"""
    backend = backend.lower()

    if backend == "ollama":
        return OllamaLLM(model=model_name or "mistral")
    
    elif backend == "huggingface":
        if not model_name:
            raise ValueError("Model name is required for Hugging Face backend.")
        return HuggingFaceLLM(model=model_name, token=api_key)  # Use 'token' parameter
    
    elif backend == "google":
        if not model_name:
            raise ValueError("Model name is required for Google backend.")
        return GoogleLLM(model=model_name, api_key=api_key)
    
    else:
        raise ValueError(f"Unknown backend: {backend}. Supported backends: {SUPPORTED_BACKENDS}")

def get_smart_chat(dataframes, backend, model_name=None, api_key=None, **kwargs):
    """Get SmartDataframe or SmartDatalake with the specified LLM backend"""
    
    # Validate backend
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Backend '{backend}' not supported. Use one of: {SUPPORTED_BACKENDS}")
    
    # Get LLM model
    try:
        llm_model = get_llm_model(backend, model_name=model_name, api_key=api_key)
    except Exception as e:
        raise ValueError(f"Failed to initialize {backend} LLM: {str(e)}")

    # Create configuration
    config = {
        "llm": llm_model,
        "custom_whitelisted_libraries": ["pandasai"],
        "verbose": False,  # Reduce verbosity for cleaner output
        "enforce_privacy": True,  # Enable privacy mode
    }
    
    print(f"✅ Using LLM: {backend} - {model_name}")
    print(f"📊 Processing {len(dataframes) if isinstance(dataframes, list) else 1} dataframe(s)")
    
    # Determine whether to use SmartDataframe or SmartDatalake
    try:
        if isinstance(dataframes, list) and len(dataframes) > 1:
            # Multiple dataframes - use SmartDatalake
            return SmartDatalake(dataframes, config=config)
        elif isinstance(dataframes, list) and len(dataframes) == 1:
            # Single dataframe in list - use SmartDataframe
            return SmartDataframe(dataframes[0], config=config)
        elif isinstance(dataframes, pd.DataFrame):
            # Single dataframe - use SmartDataframe
            return SmartDataframe(dataframes, config=config)
        else:
            raise ValueError("Invalid dataframes input type")
            
    except Exception as e:
        raise ValueError(f"Failed to create Smart chat instance: {str(e)}")

def get_smart_df(df: pd.DataFrame, provider: str = "huggingface", model_name: str = "gpt2", api_key: str = None):
    """Create SmartDataframe with specified provider and model"""
    llm = get_llm_model(provider, model_name, api_key)
    return SmartDataframe(df, config={
        "llm": llm,
        "custom_whitelisted_libraries": ["pandasai"],
        "verbose": False,
        "enforce_privacy": True,
    })

def get_smart_datalake(dfs: list, provider: str = "huggingface", model_name: str = "gpt2", api_key: str = None):
    """Create SmartDatalake with specified provider and model"""
    llm = get_llm_model(provider, model_name, api_key)
    return SmartDatalake(dfs, config={
        "llm": llm,
        "custom_whitelisted_libraries": ["pandasai"],
        "verbose": False,
        "enforce_privacy": True,
    })