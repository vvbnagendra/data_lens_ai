import pandas as pd
import plotly.express as px
from sqlalchemy import inspect
from data_quality.quality_checks import run_quality_checks
from data_quality.profiler import generate_profile
from data_quality.convert_dates import convert_dates
from data_quality.pandasai_chat import get_smart_chat
from data_quality.utils import get_ollama_models, get_huggingface_models

df = pd.read_csv(r'C:\Users\LENOVO\Downloads\glaciers.csv', low_memory=False)
df = convert_dates(df)
print(df.shape)
print(df.head())

df_cleaned = df.copy()
print(df_cleaned.shape)
print(df_cleaned.head())
df_cleaned.dropna(axis=1, how="all", inplace=True)
print(df_cleaned.shape)
print(df_cleaned.head())
df_cleaned.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
print(df_cleaned.shape)
print(df_cleaned.head())
df_cleaned.dropna(axis=0, how="any", inplace=True)
print(df_cleaned.shape)
print(df_cleaned.head())

df_cleaned1 = [df_cleaned]

df_or_smart = get_smart_chat(df_cleaned, backend='huggingface', model_name='deepseek/deepseek-r1-0528')
print(df_or_smart.head(5))

response = get_smart_chat(df_cleaned, backend='huggingface', model_name='deepseek/deepseek-r1-0528').chat('summarise the data')


print(response) 

# profile = generate_profile(df_cleaned)
# profile.to_file(r'C:\Users\LENOVO\Downloads\Crash_Reporting_-_Drivers_Data_profile.html')

