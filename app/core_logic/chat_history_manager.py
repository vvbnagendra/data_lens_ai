# core_logic/chat_history_manager.py
import streamlit as st
import os # For os.path.exists

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
    st.markdown("---")
    st.subheader("🧠 Chat History")

    if not st.session_state.chat_history:
        st.info("Your chat history will appear here. Ask a question to begin!")

    with st.expander("View Full Chat History", expanded=True):
        for entry in reversed(st.session_state.chat_history):
            st.markdown(f"🧑 You:** {entry['question']}")
            
            response_data = entry["response"]

            # Fallback for older entries or unexpected formats
            if not isinstance(response_data, dict) or "type" not in response_data or "content" not in response_data:
                response_data = {"type": "text", "content": str(response_data)}

            response_type = response_data["type"]
            response_content = response_data["content"]

            if response_type == "image":
                try:
                    # Optional: Add debugging prints to Streamlit console
                    # st.write(f"Attempting to load image from path: `{response_content}`")
                    # print(f"DEBUG: Image path received for display: {response_content}")

                    if os.path.exists(response_content):
                        # print(f"DEBUG: Image file EXISTS at {response_content}")
                        st.image(response_content, caption="📊 Generated Chart", use_column_width=True)
                    else:
                        # print(f"DEBUG: Image file DOES NOT EXIST at {response_content}")
                        st.error(f"❌ Image file not found at: `{response_content}`. Please check the path and generation process.")

                except Exception as e:
                    st.error(f"⚠️ Could not load image from path: {e}")
                    st.markdown(f"**🤖 Response Path (Error):** {response_content}")

            elif response_type == "dataframe":
                st.markdown("🤖 Response (Data Table):")
                st.dataframe(response_content, use_container_width=True)

            elif response_type == "summary_text":
                st.markdown("🤖 Response (Data Summary):")
                st.info(response_content)

            elif response_type == "code":
                st.markdown("🤖 Response (Code Generated):")
                st.code(response_content.strip(), language="python")

            elif response_type == "error":
                st.error(f"🤖 Error: {response_content}")

            elif response_type == "text":
                st.markdown(f"🤖 Response:\n\n{response_content}")

            st.markdown("---")

def clear_chat_history():
    """Clears the chat history in session state and reruns the app."""
    if st.button("🗑 Clear Chat History", help="Removes all past questions and answers from this session.", key="clear_chat_button"):
        st.session_state.chat_history = []
        st.rerun()