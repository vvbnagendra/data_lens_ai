from pandasai import SmartDataframe, SmartDatalake
from data_quality.ollama_llm_adapter import OllamaLLM
from data_quality.google_llm_adapter import GoogleLLM
from data_quality.huggingface_llm_adapter import HuggingFaceLLM
from data_quality.lotus_llm_adapter import LotusLLM
import pandas as pd

SUPPORTED_BACKENDS = ["ollama", "huggingface", "lotus"]

def get_llm_model(backend: str, model_name: str = None, api_key: str=None):
    backend = backend.lower()

    if backend == "ollama":
        return OllamaLLM(model=model_name or "mistral")
    elif backend == "huggingface":
        if not model_name:
            raise ValueError("Model name is required for Hugging Face backend.")
        return HuggingFaceLLM(model=model_name, api_key=api_key)
    elif backend == "google":
        if not model_name:
            raise ValueError("Model name is required for Google backend.")
        return GoogleLLM(model=model_name, api_key=api_key)

    elif backend == "lotus":
        return LotusLLM(model=model_name or "lotus-mixtral")
    else:
        raise ValueError(f"Unknown backend: {backend}")

def get_smart_chat(dataframes, backend, model_name=None, api_key=None, **kwargs):
    llm_model = get_llm_model(backend, model_name=model_name,api_key=api_key, **kwargs)

    config = {
        "llm": llm_model,
        "custom_whitelisted_libraries": ["pandasai"],  # ✅ Key fix
    }
    print(f"Using LLM model: {llm_model}")
    print(f"Config: {config}")
    print(f"Dataframes: {dataframes}")
    print(f"Backend: {backend}")
    print(f"Model Name: {model_name}")
    print(f"kwargs: {kwargs}")
    print(f"Supported backends: {SUPPORTED_BACKENDS}")
    print(f"Type of dataframes: {type(dataframes)}")
    print(f"Length of dataframes: {len(dataframes) if isinstance(dataframes, list) else 'N/A'}")
    print(f"Is dataframes a list? {isinstance(dataframes, list)}")
    print(f"Is dataframes a DataFrame? {isinstance(dataframes, pd.DataFrame)}")
    print(f"Is dataframes a Datalake? {isinstance(dataframes, SmartDatalake)}")
    print(f"Is dataframes a SmartDataframe? {isinstance(dataframes, SmartDataframe)}")

    if isinstance(dataframes, list) and len(dataframes) > 1:
        return SmartDatalake(dataframes, config=config)
    elif isinstance(dataframes, list):
        return SmartDataframe(dataframes[0], config=config)
    else:
        return SmartDataframe(dataframes, config=config)

def get_smart_df(df: pd.DataFrame, provider: str = "huggingface", model_name: str = "gpt2"):
    llm = get_llm_model(provider, model_name)
    return SmartDataframe(df, config={
        "llm": llm,
        "custom_whitelisted_libraries": ["pandasai"],  # ✅ Consistent
    })

def get_smart_datalake(dfs: list, provider: str = "huggingface", model_name: str = "gpt2"):
    llm = get_llm_model(provider, model_name)
    return SmartDatalake(dfs, config={
        "llm": llm,
        "custom_whitelisted_libraries": ["pandasai"],  # ✅ Consistent
    })
