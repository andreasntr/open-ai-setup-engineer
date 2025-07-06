from os import getenv
from requests import post, get

if not getenv('ENVIRONMENT'):
    from dotenv import load_dotenv
    load_dotenv(override=True)

DISCORD_API_ENDPOINT = 'https://discord.com/api'
DISCORD_CDN_URL = 'https://cdn.discordapp.com'
REDIRECT_URI = f"{getenv('BASE_URI')}/oauth2-callback"
DISCORD_CLIENT_ID = getenv('DISCORD_CLIENT_ID')
DISCORD_CLIENT_SECRET = getenv('DISCORD_CLIENT_SECRET')

LOGIN_URL = f'https://discord.com/oauth2/authorize?client_id={DISCORD_CLIENT_ID}&response_type=code&redirect_uri={REDIRECT_URI}&scope=identify&prompt=none'
LOGOUT_URL = f'{DISCORD_API_ENDPOINT}/oauth2/token/revoke'


def get_login_url() -> str:
    """
    Returns the URL for the Discord OAuth2 login flow.

    Returns:
        str: The login URL.
    """
    return LOGIN_URL


def exchange_code(code) -> str:
    """
    Exchange an authorization code for an access token.

    This function takes an authorization code obtained from the Discord OAuth2 flow
    and exchanges it for an access token that can be used to make authenticated
    requests to the Discord API.

    Args:
        code (str): The authorization code to exchange.

    Returns:
        str: The access token obtained from the exchange.
    """
    data = {
        'grant_type': 'authorization_code',
        'code': code,
        'redirect_uri': REDIRECT_URI
    }
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    r = post(f'{DISCORD_API_ENDPOINT}/oauth2/token', data=data,
             params={'state': '1234'},
             headers=headers, auth=(DISCORD_CLIENT_ID, DISCORD_CLIENT_SECRET))
    access_token = r.json()['access_token']
    return access_token


def get_user_data(token) -> dict[str, str]:
    """
    Retrieve user data from the Discord API using the provided access token.

    This function makes a GET request to the Discord API endpoint `/users/@me` with the
    provided access token in the Authorization header. It then extracts and processes
    the user data from the response, including the user's avatar URL and provider.

    Args:
        token (str): The access token to use for authenticating the request.

    Returns:
        dict[str, str]: A dictionary containing the user's data, including the user's
        ID, username, avatar URL, and provider (set to 'discord').
    """
    headers = {
        'Authorization': f'Bearer {token}'
    }
    r = get(f'{DISCORD_API_ENDPOINT}/users/@me', headers=headers)
    r.raise_for_status()
    user_data = r.json()

    if user_data.get('avatar'):
        avatar_url = f"{DISCORD_CDN_URL}/avatars/{user_data.get('id')}/{user_data.get('avatar')}.png"
    else:
        avatar_url = f"{DISCORD_CDN_URL}/embed/avatars/{(int(user_data.get('id')) >> 22) % 6}.png"

    user_data['avatar_url'] = avatar_url
    user_data['provider'] = 'discord'
    user_data.pop('avatar')
    return user_data


def revoke_token(access_token) -> None:
    """
    Revoke the provided access token.

    This function sends a POST request to the Discord OAuth2 token revocation
    endpoint to revoke the provided access token. This effectively logs the
    user out of the application.

    Args:
        access_token (str): The access token to be revoked.

    Raises:
        requests.exceptions.HTTPError: If the revocation request fails.
    """
    data = {
        'token': access_token,
        'token_type_hint': 'access_token'
    }
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    r = post(
        LOGOUT_URL,
        headers=headers,
        data=data,
        auth=(DISCORD_CLIENT_ID, DISCORD_CLIENT_SECRET)
    )
    r.raise_for_status()
