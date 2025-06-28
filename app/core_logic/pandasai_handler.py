# core_logic/pandasai_handler.py
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import os
import io

from data_quality.pandasai_chat import get_smart_chat # Assuming this is still where your SmartDatalake/SmartDataframe init is
from core_logic.file_operations import save_chat_image

@st.cache_resource(hash_funcs={pd.DataFrame: lambda _: None})
def get_pandasai_chat_instance(dataframes_list, backend_type, model_name_arg, api_key_arg):
    return get_smart_chat(
        dataframes_list,
        backend=backend_type,
        model_name=model_name_arg,
        api_key=api_key_arg
    )

def handle_pandasai_query(user_question, dfs, model_backend, model_name, user_token, selected_base_name):
    """
    Handles a query using the PandasAI backend.
    Returns a dictionary with 'type' and 'content' for the response.
    """
    response_content = None
    response_type = "text"

    try:
        smart_chat_instance = get_pandasai_chat_instance(
            [df_item for _, df_item in dfs],
            model_backend,
            model_name,
            user_token
        )
        
        raw_pandasai_response = smart_chat_instance.chat(user_question)

        if isinstance(raw_pandasai_response, str) and (
            raw_pandasai_response.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')) or "fig =" in raw_pandasai_response
        ):
            if os.path.exists(raw_pandasai_response):
                response_content = raw_pandasai_response
                response_type = "image"
            elif "fig =" in raw_pandasai_response:
                try:
                    local_env = {"pd": pd, "px": px, "plt": plt}
                    if dfs:
                        if len(dfs) == 1:
                            local_env["df"] = dfs[0][1]
                        else:
                            local_env["df"] = pd.concat([df_item for _, df_item in dfs], ignore_index=True)

                    exec(raw_pandasai_response, {"_builtins_": None}, local_env)
                    fig = local_env.get("fig")

                    if fig:
                        saved_chart_path = save_chat_image(fig, selected_base_name)
                        if saved_chart_path:
                            response_content = saved_chart_path
                            response_type = "image"
                        else:
                            response_content = f"Failed to save chart from generated code. Raw code:\n```python\n{raw_pandasai_response}\n```"
                            response_type = "error"
                    else:
                        response_content = f"Plot code executed, but no 'fig' object found. Raw code:\n```python\n{raw_pandasai_response}\n```"
                        response_type = "code"
                except Exception as plot_e:
                    response_content = f"Error executing plot code: {plot_e}\nRaw code:\n```python\n{raw_pandasai_response}\n```"
                    response_type = "error"
            else:
                response_content = raw_pandasai_response
                response_type = "text"
        else:
            response_content = raw_pandasai_response
            response_type = "text"

    except Exception as e:
        response_content = f"Error from PandasAI: {e}"
        response_type = "error"
    
    return {"type": response_type, "content": response_content}