# app/data_quality/utils.py
import subprocess

def get_ollama_models():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        lines = result.stdout.strip().split("\n")[1:]  # skip header
        return [line.split()[0] for line in lines if line]
    except Exception:
        return []

def get_huggingface_models():
    # You can customize this list or fetch dynamically if you want
    return [
        "bigcode/starcoder2-15b",
        "gpt2",
        # add more models you want to support
    ]

def get_google_models():
    # You can customize this list or fetch dynamically if you want
    return [
        "gemini-2.0-flash",
        "google/flan-t5-base",
        # add more models you want to support
    ]