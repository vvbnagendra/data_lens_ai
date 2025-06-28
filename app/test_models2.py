import os
import pandas as pd
import certifi
import ssl

from data_quality.quality_checks import run_quality_checks
from data_quality.profiler import generate_profile
from data_quality.convert_dates import convert_dates
from data_quality.pandasai_chat import get_smart_chat
from app.core_logic.utils import get_ollama_models, get_huggingface_models

# Ensure SSL certificate verification using certifi
ssl._create_default_https_context = ssl.create_default_context(cafile=certifi.where())

# Load the dataset
csv_path = r'C:\Users\LENOVO\Downloads\glaciers.csv'
df = pd.read_csv(csv_path, low_memory=False)

# Convert date columns
df = convert_dates(df)

# Clean the dataset
df_cleaned = df.copy()
df_cleaned.dropna(axis=1, how="all", inplace=True)
df_cleaned.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
df_cleaned.dropna(axis=0, how="any", inplace=True)

# Print basic info
print("Cleaned DataFrame shape:", df_cleaned.shape)
print(df_cleaned.head())

# Initialize smart chat
df_or_smart = get_smart_chat(df_cleaned, backend='google', model_name='gemini-2.0-flash')
print("Smart chat initialized.")

# Get Python code response from smart chat
response = df_or_smart.chat(
    "You are a Python data analyst. Only return valid Python code using a DataFrame called df. "
    "Don't explain or output anything else. Summarise the data and assign the result to a variable named result, "
    "in the format: {'type': 'string', 'value': '...'}"
) 

# Print the raw response
print("Raw response from chat model:\n", response)

# Execute response safely
try:
    # Basic sanity check
    if "result" in response and ("=" in response or "def" in response):
        exec(response, globals())
        print("Summary result:\n", result)
    else:
        print("Response does not contain valid Python code. Skipping exec.")
except Exception as e:
    print("Failed to execute chat response:", e)

# Optional: Generate profile
# profile = generate_profile(df_cleaned)
# profile.to_file(r'C:\Users\LENOVO\Downloads\Crash_Reporting_-_Drivers_Data_profile.html')