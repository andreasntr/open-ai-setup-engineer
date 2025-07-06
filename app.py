from os import getenv
from src.utils.frontend import (
    get_cars_data,
    get_tracks_data,
    init_page_layout
)
from src.utils.pages import HOME_PAGE, CHAT_PAGE, OAUTH2_CALLBACK_PAGE
import streamlit as st

if not getenv('ENVIRONMENT'):
    from dotenv import load_dotenv
    load_dotenv(override=True)

init_page_layout()

default_page = st.navigation(
    [
        HOME_PAGE,
        CHAT_PAGE,
        OAUTH2_CALLBACK_PAGE
    ],
    position='hidden'
)

st.session_state.car_names_mapping, st.session_state.cars_data = get_cars_data()
st.session_state.tracks_data = get_tracks_data()

if 'logout_requested' not in st.session_state:
    st.session_state.logout_requested = False

default_page.run()
