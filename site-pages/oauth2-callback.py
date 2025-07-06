import streamlit as st
from src.utils.discord import exchange_code, get_user_data
from src.utils.frontend import reset_app
from src.utils.pages import HOME_PAGE

reset_app()
st.session_state.access_token = exchange_code(st.query_params['code'])
st.session_state.logged_user_data = get_user_data(
    st.session_state.access_token
)
st.session_state.login_requested = True
st.session_state.has_new_chat = True
st.switch_page(HOME_PAGE)
