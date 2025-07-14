# app/core_logic/chat_history_manager.py
import streamlit as st
import os
import pandas as pd

def add_to_chat_history(user_question, response_dict):
    """
    Adds a question and its structured response to the chat history session state.
    response_dict should be {"type": ..., "content": ...}
    """
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    st.session_state.chat_history.append({
        "question": user_question,
        "response": response_dict
    })

def display_chat_history():
    """
    Displays the chat history from session state.
    """
    if not hasattr(st.session_state, 'chat_history') or not st.session_state.chat_history:
        st.info("Your chat history will appear here. Ask a question to begin!")
        return

    st.markdown("---")
    st.subheader("🧠 Chat History")

    with st.expander("View Full Chat History", expanded=True):
        for i, entry in enumerate(reversed(st.session_state.chat_history)):
            st.markdown(f"**🧑 You:** {entry['question']}")
            
            response_data = entry["response"]

            # Fallback for older entries or unexpected formats
            if not isinstance(response_data, dict) or "type" not in response_data or "content" not in response_data:
                response_data = {"type": "text", "content": str(response_data)}

            response_type = response_data["type"]
            response_content = response_data["content"]

            try:
                if response_type == "image":
                    if os.path.exists(response_content):
                        st.image(response_content, caption="📊 Generated Chart", use_column_width=True)
                        # If content is a Base64 string (new capability)
                    elif isinstance(response_content, str) and response_content.startswith("data:image/"):
                        st.image(response_content, caption="📊 Generated Chart", use_column_width=True)
                    elif isinstance(response_content, str): # Assume it's just the base64 string
                        st.image(f"data:image/png;base64,{response_content}", caption="📊 Generated Chart", use_column_width=True)
                    else:
                        st.error("Invalid image format in response")

                elif response_type == "dataframe":
                    st.markdown("**🤖 Response (Data Table):**")
                    if isinstance(response_content, pd.DataFrame):
                        st.dataframe(response_content, use_container_width=True)
                    else:
                        st.error("Invalid dataframe format in response")

                elif response_type == "dataframe_json":
                    st.markdown("**🤖 Response (Data Table):**")
                    try:
                        if isinstance(response_content, dict) and "data" in response_content:
                            df_display = pd.DataFrame(response_content["data"], columns=response_content.get("columns", []))
                            st.dataframe(df_display, use_container_width=True)
                        else:
                            st.json(response_content)
                    except Exception as e:
                        st.error(f"Error displaying dataframe: {e}")
                        st.json(response_content)

                elif response_type == "summary_text":
                    st.markdown("**🤖 Response (Data Summary):**")
                    st.info(response_content)

                elif response_type == "code":
                    st.markdown("**🤖 Response (Generated Code):**")
                    st.code(response_content.strip(), language="python")

                elif response_type == "error":
                    st.error(f"🤖 Error: {response_content}")

                elif isinstance(response_content, (int, float)):
                    st.markdown("**🤖 Response (Numeric Value):**")
                    st.write(f"**Value:** {response_content}")

                elif response_type == "text":
                    st.markdown("**🤖 Response:**")
                    if isinstance(response_content, pd.DataFrame):
                        # If somehow a DataFrame ended up as text type
                        st.dataframe(response_content, use_container_width=True)
                    elif hasattr(response_content, 'to_string'):
                        # Handle pandas Series or other objects with to_string method
                        formatted_output = f"```\n{response_content.to_string()}\n```"
                        st.markdown(formatted_output)
                    else:
                        st.write(response_content)

                else:
                    # Unknown type, try to display as text
                    st.markdown("**🤖 Response:**")
                    st.write(str(response_content))

            except Exception as e:
                st.error(f"Error displaying response: {e}")
                st.write(f"Raw response: {response_content}")

            # Add separator between entries
            if i < len(st.session_state.chat_history) - 1:
                st.markdown("---")

def clear_chat_history():
    """Clears the chat history in session state and reruns the app."""
    if st.button("🗑 Clear Chat History", help="Removes all past questions and answers from this session.", key="clear_chat_button"):
        st.session_state.chat_history = []
        st.rerun()