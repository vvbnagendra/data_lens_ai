# app/core_logic/utils.py
import subprocess

def get_ollama_models():
    """Get available Ollama models with fallback to standard list"""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")[1:]  # skip header
            models = [line.split()[0] for line in lines if line.strip()]
            return models if models else get_standard_ollama_models()
        else:
            return get_standard_ollama_models()
    except Exception:
        return get_standard_ollama_models()

def get_standard_ollama_models():
    """Standard Ollama models list"""
    return [
        "mistral",
        "llama3.2", 
        "codellama",
        "deepseek-r1:1.5b",
        "phi3",
        "gemma2",
        "qwen2.5",
        "llama3.1"
    ]

def get_huggingface_models():
    """Get HuggingFace models list"""
    return [
        "deepseek/deepseek-r1-0528",
        "microsoft/DialoGPT-medium",
        "bigcode/starcoder2-15b", 
        "codellama/CodeLlama-7b-hf",
        "mistralai/Mistral-7B-v0.1",
        "microsoft/CodeBERT-base",
        "huggingface/CodeBERTa-small-v1",
        "gpt2",
        "facebook/opt-1.3b"
    ]

def get_google_models():
    """Get Google AI models list"""
    return [
        "gemini-2.0-flash",
        "gemini-1.5-pro", 
        "gemini-1.5-flash",
        "gemini-1.0-pro",
        "google/flan-t5-base",
        "google/flan-t5-large",
        "google/flan-t5-xl"
    ]

def get_lotus_models():
    """Get Lotus-specific models"""
    return [
        "lotus-mistral",
        "lotus-llama",
        "lotus-gpt", 
        "lotus-claude",
        "lotus-gemini",
        "lotus-deepseek"
    ]

def validate_model_availability(backend, model_name):
    """Validate if a model is available for the given backend"""
    if backend == "ollama":
        available_models = get_ollama_models()
        return model_name in available_models
    elif backend == "huggingface":
        # For HuggingFace, we assume models are available online
        return True
    elif backend == "google":
        # For Google, we assume models are available with proper API key
        return True
    elif backend == "lotus":
        available_models = get_lotus_models()
        return model_name in available_models
    
    return False

def get_model_info(backend, model_name):
    """Get information about a specific model"""
    model_info = {
        "ollama": {
            "mistral": {"size": "7B", "type": "General", "description": "Fast and efficient general-purpose model"},
            "llama3.2": {"size": "3B-70B", "type": "General", "description": "Latest Llama model with improved capabilities"},
            "codellama": {"size": "7B-34B", "type": "Code", "description": "Specialized for code generation and analysis"},
            "deepseek-r1:1.5b": {"size": "1.5B", "type": "Code", "description": "Compact model optimized for reasoning"},
            "phi3": {"size": "3.8B", "type": "General", "description": "Microsoft's efficient small language model"},
            "gemma2": {"size": "2B-27B", "type": "General", "description": "Google's open model family"}
        },
        "huggingface": {
            "deepseek/deepseek-r1-0528": {"type": "Code", "description": "Advanced reasoning model for code and analysis"},
            "microsoft/DialoGPT-medium": {"type": "Conversational", "description": "Conversational AI model"},
            "bigcode/starcoder2-15b": {"type": "Code", "description": "Large code generation model"},
            "mistralai/Mistral-7B-v0.1": {"type": "General", "description": "Efficient 7B parameter model"},
            "gpt2": {"type": "General", "description": "Classic general-purpose model"}
        },
        "google": {
            "gemini-2.0-flash": {"type": "Multimodal", "description": "Latest Gemini with fast response times"},
            "gemini-1.5-pro": {"type": "Multimodal", "description": "Professional-grade model with large context"},
            "gemini-1.5-flash": {"type": "Multimodal", "description": "Balanced speed and capability"},
            "google/flan-t5-base": {"type": "Text", "description": "Instruction-tuned T5 model"}
        },
        "lotus": {
            "lotus-mistral": {"type": "Semantic", "description": "Balanced performance for semantic tasks"},
            "lotus-llama": {"type": "Semantic", "description": "Strong reasoning capabilities"},
            "lotus-gpt": {"type": "Semantic", "description": "Fast semantic search and filtering"},
            "lotus-claude": {"type": "Semantic", "description": "Advanced natural language understanding"},
            "lotus-gemini": {"type": "Semantic", "description": "Cutting-edge semantic capabilities"}
        }
    }
    
    return model_info.get(backend, {}).get(model_name, {
        "type": "Unknown", 
        "description": "Model information not available"
    })

def format_model_display_name(backend, model_name):
    """Format model name for display"""
    info = get_model_info(backend, model_name)
    model_type = info.get("type", "")
    size = info.get("size", "")
    
    display_name = model_name.split("/")[-1]  # Get the model name part
    
    if size:
        display_name += f" ({size})"
    if model_type:
        display_name += f" - {model_type}"
    
    return display_name