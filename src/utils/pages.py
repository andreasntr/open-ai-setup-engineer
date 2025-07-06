from streamlit import Page
from enum import StrEnum

PAGES_DIR = 'site-pages'
PAGES_NAME_SUFFIX = "- AI Setup Engineer"


class PagesEnum(StrEnum):
    HOME_PAGE = f'{PAGES_DIR}/home.py'
    CHAT_PAGE = f'{PAGES_DIR}/chat.py'
    OAUTH2_CALLBACK_PAGE = f'{PAGES_DIR}/oauth2-callback.py'


HOME_PAGE = Page(
    PagesEnum.HOME_PAGE,
    title=f"Home {PAGES_NAME_SUFFIX}",
    icon="🔧",
    url_path="/home",
    default=True
)
CHAT_PAGE = Page(
    PagesEnum.CHAT_PAGE,
    title=f"Chat {PAGES_NAME_SUFFIX}",
    icon="🔧",
    url_path="/chat"
)
OAUTH2_CALLBACK_PAGE = Page(
    PagesEnum.OAUTH2_CALLBACK_PAGE,
    title=f"OAuth2 Callback {PAGES_NAME_SUFFIX}",
    icon="🔧",
    url_path="/oauth2-callback"
)
