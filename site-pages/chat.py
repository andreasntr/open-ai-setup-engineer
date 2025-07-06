from typing import Literal
import streamlit as st
from src.utils.frontend import (
    get_track_layout_image,
    init_llm,
    load_json_setup,
    reset_app,
    reset_chat,
    show_preliminary_info,
    ALLOW_LLM_INFERENCE,
    CHAT_MODES_MAPPING,
    SETUP_MODE,
    TIPS_MODE
)
from src.utils.llm import human_message, assistant_message
from src.utils.pages import HOME_PAGE
from src.AISetupEngineerApp import ERROR_MESSAGE
from time import sleep

FeedbackScore = Literal['positive', 'negative']


def disable_chat_input():
    st.session_state.is_generating = True


def enable_chat_input():
    st.session_state.is_generating = False


def sumbit_feedback(message_id: int, feedback: FeedbackScore) -> None:
    content = st.session_state.get(
        f'msg_{message_id}_{feedback}',
        default=''
    )
    feedback = {
        # -1 is for pointing at the corresponding user message
        'msg_id': (message_id-2-1)//2,
        'feedback': feedback,
        'content': content if len(content) else None
    }
    st.session_state.app.submit_feedback(feedback)
    st.session_state.feedback_submitted = True


@st.dialog('Please leave a comment')
def handle_feedback(message_id: int) -> None:
    is_positive = st.session_state.get(f'msg_{message_id}_feedback_form')
    feedback = 'positive' if is_positive else 'negative'
    # write empty feedback to avoid losing it if the user closes the dialog
    sumbit_feedback(
        message_id=message_id,
        feedback=feedback
    )
    with st.container():
        st.text_area(
            value="",
            key=f'msg_{message_id}_{feedback}',
            label='Feedback message (optional)',
            placeholder='Your feedback here'
        )
        if st.button('Submit feedback'):
            sumbit_feedback(
                message_id=message_id,
                feedback=feedback
            )
            st.rerun()


@st.fragment
def add_feedback_buttons(
    message_id: int,
    # keep for future use
    previous_feedback: FeedbackScore | None = None
) -> None:
    st.feedback(
        key=f'msg_{message_id}_feedback_form',
        on_change=handle_feedback,
        kwargs={
            'message_id': message_id
        },
        disabled=previous_feedback is not None
    )
    # if previous_feedback is not None:
    #     feedback_bool = previous_feedback.get('feedback') == 'positive'
    #     st.session_state[f'msg_{message_id}_feedback_form'] = int(
    #         feedback_bool)


@st.fragment
def handle_qa(question: str) -> str:
    progress = st.progress(0, 'Detecting corners...')
    message_placeholder = st.empty()
    if ALLOW_LLM_INFERENCE:
        answer, is_successfully_handled = st.session_state.app.ask(
            question,
            st.session_state.selected_car,
            st.session_state.selected_track,
            CHAT_MODES_MAPPING[st.session_state.chat_mode],
            setup_json=st.session_state.get("setup_json"),
            container=message_placeholder,
            progress=progress
        )
    else:
        sleep(2)
        answer = ERROR_MESSAGE
    st.session_state.last_message_good = True
    if answer.startswith('Sorry') or not is_successfully_handled:
        st.session_state.last_message_good = False
    return answer


if st.session_state.get('start_chat'):

    chat_container = st.container()
    menu_container = st.container()

    with chat_container:
        caption = ''
        if not st.session_state.selected_track["track_map_path"].endswith('nurburgring_24h.png'):
            caption = 'All credits for the base track layout (without corner numbers) ' + \
                'belong to: racingcircuits.info'
        st.image(
            get_track_layout_image(
                st.session_state.selected_track["track_map_path"]
            ),
            caption=caption
        )

        with st.spinner("Loading chat..."):
            for msg_id, message in enumerate(st.session_state.messages):
                with st.chat_message(message["role"]):
                    is_error = message["content"].startswith('Sorry')
                    if is_error:
                        message['content'] = message['content']
                    st.markdown(message["content"])
                    if msg_id >= 2 and \
                            message["role"] == "assistant" and \
                            not is_error:
                        prev_feedback = message.get("feedback")
                        add_feedback_buttons(msg_id, prev_feedback)

        if "app" not in st.session_state:
            with st.spinner('Loading AI model...'):
                # Initialize app
                init_llm()
                show_preliminary_info()

    if st.session_state.get("start_chat"):
        with menu_container:
            with st.popover(
                label="⏮️ Reset chat",
                disabled=st.session_state.is_generating
            ):
                if st.button(
                    "Change car/track",
                    key='reset_btn',
                    use_container_width=True
                ):
                    reset_app()
                    st.switch_page(HOME_PAGE)
                if st.button(
                    "Reset chat history",
                    key='reset_chat_btn',
                    use_container_width=True,
                    disabled=len(st.session_state.messages) == 2
                ):
                    reset_chat()
                    st.rerun()
                if st.session_state.chat_mode == SETUP_MODE and \
                        len(st.session_state.messages) > 2 and \
                        st.session_state.last_message_good:
                    uploader_key = "updated_setup_json_uploader"
                    if st.file_uploader(
                        "**Update setup**",
                        key=uploader_key,
                        type="json"
                    ):
                        try:
                            load_json_setup(uploader_key)
                            if st.button(
                                "**Update setup**",
                                key='update_setup_btn',
                                use_container_width=True
                            ):
                                reset_chat()
                                st.rerun()
                        except Exception:
                            error_msg = "There was an error parsing your setup.  \n" \
                                "Please retry or reload this page."
                            st.error(error_msg)

    if st.session_state.chat_mode == TIPS_MODE:
        chat_placeholder = "How can I help you?"
    else:
        chat_placeholder = "How do you want me to help you with this setup?"

    if st.session_state.get('feedback_submitted', False):
        st.toast("Thank you for your feedback!")
        st.session_state.feedback_submitted = False

    # if st.session_state.chat_mode == SETUP_MODE:
    #     with st.spinner("Reading setup data..."):
    #         while not st.session_state.setup_info:
    #             sleep(0.2)
    # Accept user input
    if question := st.chat_input(
        chat_placeholder,
        max_chars=500,
        disabled=(
            st.session_state.is_generating or
            (
                not st.session_state.is_generating and
                st.session_state.last_message_good and
                st.session_state.chat_mode == SETUP_MODE
            )
        ),
        on_submit=disable_chat_input
    ):
        # Add user message to chat history
        st.session_state.messages.append(human_message(question))
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(question)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            answer = handle_qa(question)

            enable_chat_input()

        st.session_state.messages.append(
            assistant_message(answer)
        )
        sleep(0.02)
        st.rerun()
else:
    st.error("Cut detected!", icon="✂️")
    st.write("You must start your chat from the Home page.")
    if st.button(label="🏠 Go to the Home page"):
        reset_app()
        st.switch_page(HOME_PAGE)
