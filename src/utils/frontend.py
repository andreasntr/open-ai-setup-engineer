from operator import itemgetter
from os import listdir, getenv
from src.AISetupEngineerApp import AISetupEngineerApp, ChatModeEnum
from src.utils.discord import (
    get_login_url,
    revoke_token,
    revoke_token
)
from src.utils.mongo import get_sessions, get_session
from src.utils.pages import PagesEnum
from bson.objectid import ObjectId
from os.path import join
from typing import Any
import streamlit as st
from json import load, loads

# set to False to test the UI without generating any answer from the LLM
ALLOW_LLM_INFERENCE = True
DISCORD_IS_SET = getenv('DISCORD_CLIENT_ID') is not None

TIPS_MODE = "Tips Mode: Ask about general driving/setup tips"
SETUP_MODE = "Setup Mode: Get tailored tips based on your current setup"
CHAT_MODES_MAPPING = {
    TIPS_MODE: ChatModeEnum.TIPS_MODE,
    SETUP_MODE: ChatModeEnum.SETUP_MODE
}
# useful to retrieve the description given the chat mode
CODED_CHAT_MODES_MAPPING = dict(map(reversed, CHAT_MODES_MAPPING.items()))

CHAT_ICON = ':speech_balloon:'
CAR_ICON = ':racing_car:'
TRACK_ICON = ':pushpin:'

N_WELCOME_MESSAGES = 2
INSTRUCTIONS_MSG = 'Hi, I\'m your AI setup engineer and I\'m here to help you build your car setup step by step!  \n' \
    'For example, ask me _reduce braking distance_ or _reduce oversteer at corner entry_ and I\'ll give you some tips to reach your goal.  \n' \
    'To get help about specific corners, please use the numbering provided in the above image (better than corner names, since they may change).'
DATA_PATH = 'data'
ASSETS_PATH = 'assets'
CARS_DATA_PATH = join(DATA_PATH, 'cars')
TRACKS_DATA_PATH = join(DATA_PATH, 'tracks')


@st.cache_resource
def get_cars_data() -> tuple[dict, list[dict]]:
    """
    Loads car data from JSON files in the CARS_DATA_PATH directory.

    Returns:
        tuple[dict, list[dict]]: A tuple containing a dictionary mapping car IDs to car names,
        and a list of cars metadata. Each car dictionary contains car data loaded from its JSON file.
    """
    car_filenames = listdir(CARS_DATA_PATH)
    cars_data = []
    car_names_mapping = {}
    for car_filename in car_filenames:
        file_path = join(CARS_DATA_PATH, car_filename)
        with open(file_path, 'r') as f:
            car = load(f)
        car_id = car_filename.split('.')[0]
        car_names_mapping[car_id] = car['name']
        cars_data.append(car)
    cars_data = sorted(cars_data, key=itemgetter('name'))
    return car_names_mapping, cars_data


@st.cache_resource
def get_tracks_data() -> list[dict]:
    """
    Loads track data from JSON files in the TRACKS_DATA_PATH directory.

    Returns:
       list[dict]: A list of track dictionaries. Each track dictionary contains track data loaded from its JSON file.
    """
    track_filenames = listdir(TRACKS_DATA_PATH)
    tracks_data = []
    for track_filename in track_filenames:
        file_path = join(TRACKS_DATA_PATH, track_filename)
        track_id = track_filename.split('.')[0]
        with open(file_path, 'r') as f:
            track = load(f)
        track["layout"] = track.get(
            "layout", 'No information available')
        track["track_map_path"] = join(ASSETS_PATH, f'{track_id}.png')
        tracks_data.append(track)
    tracks_data = sorted(tracks_data, key=itemgetter('name'))
    return tracks_data


@st.cache_resource
def get_track_layout_image(track_map_path):
    """
    Opens and returns a track layout image.

    Args:
        track_map_path (str): The path to the track layout image file.

    Returns:
        PIL.Image.Image: The opened track layout image.
    """
    from PIL import Image
    return Image.open(track_map_path)


def init_sidebar() -> None:
    if DISCORD_IS_SET:
        show_user_data()
        st.divider()

    mailto_link = "mailto:andreasantoro.pvt@gmail.com?subject=AI Setup Engineer"

    st.markdown(
        "Made by Andrea Santoro"
    )
    st.link_button(
        label="🍕 Donate",
        url="https://www.buymeacoffee.com/andreasntr",
        use_container_width=True
    )
    st.link_button(
        label="💬 Discord",
        url="https://discord.com/users/453277976490934282",
        use_container_width=True
    )
    st.link_button(
        label="📧 Email",
        url=mailto_link,
        use_container_width=True
    )


def init_page_layout() -> None:
    st.set_page_config(
        layout='wide',
        page_icon="🔧"
    )
    with st.sidebar:
        init_sidebar()


def init_chat(additional_interactions: list[dict[str, str]] = []) -> None:
    """
    Initializes the chat session state.
    Args:
        additional_interactions (list[dict[str, str]]): A list of dictionaries,
            where each dictionary contains the previous QnAs.

    Returns:
        None
    """
    st.session_state.feedback_submitted = False
    st.session_state.messages = [
        {"role": "assistant", "content": INSTRUCTIONS_MSG},
        {
            "role": "assistant",
            "content": f"You selected:  \n"
            f" - **{CHAT_ICON}**: {st.session_state.chat_mode}  \n" +
            f" - **{CAR_ICON}**: {st.session_state.selected_car['name']}  \n" +
            f" - **{TRACK_ICON}**: {st.session_state.selected_track['name']}  \n\n" +
            "To change car/track, please reset this chat."
        }
    ]
    if len(additional_interactions):
        for interaction in additional_interactions:
            st.session_state.messages.extend([
                {"role": "user", "content": interaction['original_q']},
                {"role": "assistant", "content": interaction['a']}
            ])
            if interaction.get("feedback"):
                st.session_state.messages[-1]["feedback"] = interaction["feedback"]


@st.dialog('⚠️ Please read carefully', width='large')
def show_preliminary_info():
    st.markdown(
        'Before starting this chat, keep in mind the following info:  \n' +
        "1) **This chatbot is _NOT_ made to give you exact numbers**, only tips to build/improve your setup step by step. Don't waste your time asking questions like '_give me a complete setup_' or similar. If that's what you need, visit other websites; \n" +
        '2) Some of the provided suggestions may sometimes be inaccurate. **Make sure you test suggestions one at a time**;  \n' +
        '3) If you are looking for **engine maps**, refer to [this post](https://www.assettocorsa.net/forum/index.php?threads/ecu-maps-implementation.54472/);  \n' +
        '4) If you see the chatbot is **repeating the same answer** over and over, try to **reset the chat**;  \n' +
        '5) **The way you ask your questions influences the quality of the response**. If you want precise answers, ask precise questions by including as much details as you can about what you are currently experiencing;  \n' +
        '6) **Please consider leaving a feedback** after each response to help me improve the service.'
    )
    if st.button("Got it, let's start!"):
        st.rerun()


def init_llm():
    """
    Initializes the LLM and sets up the application state.
    """
    user = st.session_state.get('logged_user_data')
    user_data = None
    if user:
        user_data = user.copy()
        user_data.pop('past_sessions')
    st.session_state.app = AISetupEngineerApp(
        user_data=user_data
    )


def reset_llm():
    """
    Reset the LLM state and clears previous interactions.
    """
    user = st.session_state.get('logged_user_data')
    user_data = None
    if user:
        user_data = user.copy()
        user_data.pop('past_sessions', None)
    st.session_state.app.reset(user_data)


def restore_chat(past_chat_data: dict[str, Any]) -> None:
    st.session_state.start_chat = True
    st.session_state.is_generating = False
    st.session_state.last_message_good = True
    st.session_state.chat_mode = CODED_CHAT_MODES_MAPPING[past_chat_data['chat_mode']]
    st.session_state.setup_json = past_chat_data.get('setup_json')
    st.session_state.selected_car = list(filter(
        lambda car: car['name'] == past_chat_data['car'],
        st.session_state.cars_data))[0]
    st.session_state.selected_track = list(filter(
        lambda track: track['name'] == past_chat_data['track'],
        st.session_state.tracks_data))[0]
    if not 'app' in st.session_state:
        init_llm()
    else:
        reset_llm()
    st.session_state.app.restore(past_chat_data)
    init_chat(past_chat_data['interactions'])
    st.session_state.has_new_chat = len(st.session_state.messages) > 2


def reset_chat():
    """
    Resets the chat by removing every message.
    """
    if st.session_state.get('app'):
        reset_llm()
    if len(st.session_state.messages) > 2:
        st.session_state.has_new_chat = True
    st.session_state.messages = st.session_state.messages[:2]
    st.session_state.is_generating = False
    st.session_state.last_message_good = False
    st.session_state.feedback_submitted = False


def reset_app():
    """
    Resets the whole app by going back to the chat mode selection page.
    """
    if st.session_state.get('start_chat'):
        st.session_state.start_chat = False
        if st.session_state.chat_mode == SETUP_MODE:
            st.session_state.pop("setup_json")
        st.session_state.pop("chat_mode")
        reset_chat()


def load_json_setup(uploader_name: str = 'new_setup_json_uploader') -> None:
    """
    Loads the setup json file from the given uploader.

    Args:
        uploader_name (str): The name of the uploader to load the setup json file from.

    Returns:
        None
    """
    try:
        st.session_state.setup_json = loads(
            st.session_state[uploader_name].getvalue().decode('utf-8'))
        selected_car_name = st.session_state.car_names_mapping[
            st.session_state.setup_json['carName']]
        st.session_state.selected_car = list(filter(
            lambda car: car['name'] == selected_car_name,
            st.session_state.cars_data
        ))[0]
    except:
        error_msg = "There was an error parsing your setup.  \n" \
            "Please retry or reload this page."
        st.error(error_msg)
        raise Exception('Setup error')


def start_chat() -> None:
    """
    Start the chat by initializing the chat history and setting the necessary metadata.
    """
    st.session_state.start_chat = True
    st.session_state.is_generating = False
    st.session_state.last_message_good = False
    st.session_state.has_new_chat = False
    if st.session_state.chat_mode == TIPS_MODE:
        st.session_state.selected_car = list(filter(
            lambda car: car['name'] == st.session_state.car_selection,
            st.session_state.cars_data
        ))[0]
    else:
        load_json_setup()

    st.session_state.selected_track = list(filter(
        lambda track: track['name'] == st.session_state.track_selection,
        st.session_state.tracks_data))[0]
    # Initialize chat history
    init_chat()


def start_context_selection():
    st.session_state.chat_mode = st.session_state.chat_mode_selection


def create_link_button(key: str, url: str, label: str | None = None, icon: str | None = None) -> None:
    st.html(
        f'''
        <style>
            .{key} {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                font-weight: 400;
                padding: 0.25rem 0.75rem;
                border-radius: 0.5rem;
                min-height: 38.4px;
                margin: 0px;
                line-height: 1.6;
                text-decoration: none;
                width: 100%;
                user-select: none;
                background-color: rgb(43, 44, 54);
                color: rgb(250, 250, 250) !important;
                border: 1px solid rgba(250, 250, 250, 0.2);
            }}

            .{key}:hover {{
                border-color: rgb(255, 75, 75);
                color: rgb(255, 75, 75);
            }}

            .{key} p, .{key} img {{
                margin: 0;
            }}
        </style>

        <div class="row-widget stLinkButton" data-testid="stLinkButton">
          <a class="{key}" data-testid="baseLinkButton-secondary" 
            kind="secondary" href="{url}">
            <div data-testid="stMarkdownContainer">
                {f'<img src="{icon}" width="50px" height="50px">' if icon else ''}
                {f'<p>{label}</p>' if label else ''}
            </div>
        </a>
        </div>
      '''
    )


def logout() -> None:
    revoke_token(st.session_state.access_token)
    st.session_state.pop('logged_user_data')
    st.session_state.pop('access_token')
    st.session_state.logout_requested = True


def show_user_icon() -> None:
    st.html(f'''
        <div>
        <img class="profileImage" src="{st.session_state.logged_user_data.get('avatar_url')}"/>
        </div>
    ''')


def get_user_past_sessions(sessions_limit: int = 20) -> list[dict[str, Any]]:
    if st.session_state.get('has_new_chat'):
        st.session_state.has_new_chat = False
        retrieval_limit = 1
        curr_sessions = st.session_state.logged_user_data.get('past_sessions')

        if len(curr_sessions) == 0:
            retrieval_limit = sessions_limit
        user = st.session_state.logged_user_data
        sessions = get_sessions(
            filter={
                "user.provider": user.get('provider'),
                "user.id": user.get('id')
            },
            projection={
                "q": {"$first": "$interactions.original_q"},
                "chat_mode": 1,
                "car": 1,
                "track": 1
            },
            sort={'updated_at': -1},
            limit=retrieval_limit
        )
        sessions = list(
            filter(
                lambda s: s not in curr_sessions,
                sessions
            )
        )
        if len(sessions):
            if len(st.session_state.logged_user_data.get('past_sessions')):
                st.session_state.logged_user_data['past_sessions'] = [sessions[0]] + \
                    curr_sessions
            else:
                st.session_state.logged_user_data['past_sessions'] = sessions
            st.session_state.logged_user_data['past_sessions'] = st.session_state.logged_user_data['past_sessions'][:sessions_limit]
    return st.session_state.logged_user_data.get('past_sessions')


@st.fragment
def update_past_sessions(sessions_limit: int = 20) -> None:
    with st.spinner('Loading past sessions...'):
        sessions = get_user_past_sessions(sessions_limit)
    with st.popover("Previous chats", use_container_width=True):
        with st.container(height=300, border=False):
            for i, session in enumerate(sessions):
                session_first_question = session.get('q')
                if len(session_first_question) >= 50:
                    session_first_question = session_first_question[:50] + '...'
                chat_mode = ChatModeEnum(session.get('chat_mode'))
                with st.expander(session_first_question):
                    st.markdown(
                        f"**{CHAT_ICON}**: {CODED_CHAT_MODES_MAPPING[chat_mode]}  \n" +
                        f"**{CAR_ICON}**: {session.get('car')}  \n" +
                        f"**{TRACK_ICON}**: {session.get('track')}")
                    if st.button(
                        key=f'session_{i}',
                        label='Resume chat',
                        use_container_width=True,
                        on_click=load_chat,
                        kwargs={'id': session.get('_id')}
                    ):
                        st.switch_page(PagesEnum.CHAT_PAGE)


def load_chat(id: ObjectId) -> None:
    past_chat_data = get_session(
        {'_id': id},
        {
            "_id": 0,
            "interactions.metadata": 0,
            "user": 0,
            "session_start_at": 0,
            "chat_start_at": 0
        }
    )
    restore_chat(past_chat_data)


@st.fragment
def show_user_data() -> None:
    if 'logged_user_data' not in st.session_state:
        st.write('Login to restart your past chats at any time')
        create_link_button(
            "login_btn",
            get_login_url(),
            icon="https://cdn3.emoji.gg/emojis/7561-discord-clyde.png"
        )
        if st.session_state.get('logout_requested'):
            st.toast('Logged out')
            st.session_state.logout_requested = False
            if st.session_state.get('start_chat'):
                reset_app()
            st.switch_page(PagesEnum.HOME_PAGE)
    else:
        st.write(
            f"Logged in as {st.session_state.logged_user_data.get('username')}")
        show_user_icon()
        if st.session_state.get('login_requested'):
            st.toast(
                f"Welcome {st.session_state.logged_user_data.get('username')}!")
            st.session_state.login_requested = False
            st.session_state.logged_user_data['past_sessions'] = []
        st.button(
            key="logout_btn",
            label="Logout",
            on_click=logout,
            disabled=st.session_state.logout_requested
        )

        sessions_limit = 20
        update_past_sessions(sessions_limit)
