from operator import itemgetter
from src.utils.frontend import (
    start_chat,
    start_context_selection,
    SETUP_MODE,
    TIPS_MODE
)
from src.utils.pages import CHAT_PAGE
import streamlit as st


@st.fragment
def handle_settings_selection():
    """
    Handles the selection of settings for the application.
    """
    # if chat mode selected
    if st.session_state.get("chat_mode"):
        # select car/setup and track form
        with st.form('car_track_selector'):
            # for tips mode, select car
            if st.session_state.chat_mode == TIPS_MODE:
                car_names = list(
                    map(itemgetter('name'), st.session_state.cars_data)
                )
                st.selectbox(
                    'Select your car',
                    options=car_names,
                    index=None,
                    key='car_selection'
                )
            track_names = list(
                map(itemgetter('name'), st.session_state.tracks_data)
            )
            st.selectbox(
                '**Select track**',
                options=track_names,
                index=None,
                key='track_selection'
            )
            # for setup mode, car is derived from setup
            if st.session_state.chat_mode == SETUP_MODE:
                uploader_key = "new_setup_json_uploader"
                st.file_uploader(
                    "**Upload your setup file**",
                    type="json",
                    key=uploader_key
                )
            # when starting a chat in setup mode, ensure a setup is uploaded
            if st.form_submit_button('Start chat'):
                if st.session_state.chat_mode == SETUP_MODE and \
                        not st.session_state.get(uploader_key):
                    # emit setup not found alert
                    st.toast(":red[Please upload your setup file]")
                else:
                    try:
                        # setup chat metadata
                        start_chat()
                        st.session_state.selection_process_ended = True
                        # go to chat page
                        st.switch_page(CHAT_PAGE)
                    except IndexError:
                        # ensure all fields are filled
                        st.toast(":red[Please fill all the required fields]")
                        pass
        # return to chat mode selector
        if st.button('⏮️ Back'):
            st.session_state.pop('chat_mode')
            st.rerun(scope='fragment')
    else:
        # chat mode selector
        with st.form('chat_mode_selector'):
            st.selectbox(
                '**Select chat mode**',
                options=[
                    TIPS_MODE,
                    SETUP_MODE
                ],
                index=None,
                key='chat_mode_selection'
            )

            st.form_submit_button(
                'Apply',
                on_click=start_context_selection
            )


# default starting metadata, allow selection process to be started correctly
st.session_state.is_chat_mode_selected = False
st.session_state.is_car_selected = False
st.session_state.is_track_selected = False
st.session_state.selection_process_ended = False


st.write((
    "## 🔧 AI Setup Engineer  \n"
    "#### Optimize your car setups in ACC using artificial intelligence.  \n"
    "To get the most out of the AI Setup Engineer, **be specific about what you're looking to achieve**. Instead of simply asking for a \"better setup\", describe the issues you're encountering or the performance improvements you desire. For example:  \n"
    "- \"My car is understeering at corner entry.\"  ➡️ The tool might suggest adjusting suspension settings or tire pressures.  \n"
    "- \"I want to improve my top speed on the straights.\" ➡️ The tool could recommend optimizing aerodynamics or gearing.  \n  \n"
    "The AI Setup Engineer offers two modes of operation:  \n"
    "- 🏎️ **Tips Mode**:  Get general advice on setup improvements which fulfill your needs.  \n"
    "- ⚙️ **Setup Mode**: Upload your existing JSON setup for tailored recommendations.  \n  \n"
    "No matter which mode you choose, the AI Setup Engineer provides valuable insights to help you fine-tune your car and achieve better results on the track! 🏆"
))
st.divider()

handle_settings_selection()
if st.session_state.selection_process_ended:
    st.session_state.pop('selection_process_ended')
    # go to chat page
    st.switch_page(CHAT_PAGE)
