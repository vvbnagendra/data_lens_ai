# core_logic/file_operations.py
import streamlit as st
import os
import datetime
import uuid
import matplotlib.pyplot as plt # Import only if needed for saving mpl plots

def save_chat_image(fig_obj, base_name, export_base_folder="exports/outputs"):
    """
    Saves a Plotly figure or Matplotlib figure object to a structured export folder
    with a unique filename.

    Args:
        fig_obj: The Plotly figure object (plotly.graph_objects.Figure) or Matplotlib figure object (matplotlib.figure.Figure).
        base_name (str): The base name for the subfolder (e.g., "data", "customers").
        export_base_folder (str): The main folder where exported content will reside (e.g., "exports/outputs").
    Returns:
        str: The full path to the saved image file, or None if saving failed.
    """
    if not base_name:
        st.warning("Cannot save image: No file/table selected to determine base name.")
        return None

    specific_export_folder = os.path.join(export_base_folder, base_name)
    os.makedirs(specific_export_folder, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    
    file_name = f"chart_{timestamp}_{unique_id}.png"
    file_path = os.path.join(specific_export_folder, file_name)

    try:
        if hasattr(fig_obj, 'write_image'):
            fig_obj.write_image(file_path)
            st.toast(f"📈 Chart saved to: {file_path}", icon="✅")
            return file_path
        elif isinstance(fig_obj, plt.Figure):
            fig_obj.savefig(file_path, bbox_inches='tight', dpi=300)
            st.toast(f"📈 Chart saved to: {file_path}", icon="✅")
            plt.close(fig_obj)
            return file_path
        else:
            st.error("Unsupported plot type for saving. Please provide a Plotly figure or Matplotlib figure object.")
            return None

    except ImportError:
        st.error("Plotly image export requires `kaleido`. Please install it: `pip install kaleido`")
        return None
    except Exception as e:
        st.error(f"❌ Error saving chart to `{file_path}`: {e}")
        return None