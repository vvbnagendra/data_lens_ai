import streamlit as st
from .models import AuthenticationDB

class AuthenticationMiddleware:
    def __init__(self):
        self.db = AuthenticationDB()

        if 'auth_user' not in st.session_state:
            st.session_state.auth_user = None
        if 'auth_session_id' not in st.session_state:
            st.session_state.auth_session_id = None

    def login_user(self, email, password):
        user = self.db.verify_password(email, password)
        if user:
            session_id = self.db.create_session(user['user_id'])
            st.session_state.auth_user = user
            st.session_state.auth_session_id = session_id
            return {'success': True, 'user': user}
        return {'success': False, 'error': 'Invalid credentials'}

    def logout_user(self):
        st.session_state.auth_user = None
        st.session_state.auth_session_id = None
        # IMPORTANT: Add a rerun after logout to update the UI
        st.rerun() 

    def validate_session(self):
        if not st.session_state.auth_session_id:
            return False

        session_data = self.db.validate_session(st.session_state.auth_session_id)
        if session_data:
            st.session_state.auth_user = dict(session_data)
            return True
        else:
            st.session_state.auth_user = None
            st.session_state.auth_session_id = None
            return False

    def get_current_user(self):
        return st.session_state.auth_user

def show_login_page():
    # REMOVE THIS LINE: st.set_page_config(page_title="Login - Data Lens AI", page_icon="🔐")

    st.title("Data Lens AI Login")

    auth = AuthenticationMiddleware()

    tab1, tab2 = st.tabs(["Login", "Info"])

    with tab1:
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="admin@datalensai.com")
            password = st.text_input("Password", type="password", placeholder="DataLens2024!")

            if st.form_submit_button("Login", type="primary"):
                if email and password:
                    result = auth.login_user(email, password)
                    if result['success']:
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("ERROR: " + result['error'])
                else:
                    st.error("Please enter both email and password")

    with tab2:
        st.info("Default credentials: admin@datalensai.com / DataLens2024!")
        st.warning("Please change the default password after first login!")