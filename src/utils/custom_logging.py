from cachetools.func import lru_cache
import logging
from os.path import join
from os import getenv

# always relative to the entrypoint of the app
# with the native implementation it is:
# - /app/ in Docker
# - <your-repo-path>/ when run locally
LOGS_FOLDER = 'logs'
APP_LOGGER_NAME = 'app'


@lru_cache(maxsize=1024)
def __get_logger__(name: str = APP_LOGGER_NAME) -> logging.Logger:
    logger = logging.getLogger(name)

    log_level = logging._nameToLevel[getenv('LOG_LEVEL', 'INFO')]

    file_handler = logging.FileHandler(
        join(LOGS_FOLDER, 'ai-setup-engineer.log'),
        encoding='utf-8',
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(
        logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s  ',
            '%Y-%m-%d %H:%M:%S'
        )
    )
    if name == APP_LOGGER_NAME:
        file_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(chat_id)s - %(function_name)s - %(message)s  ',
                '%Y-%m-%d %H:%M:%S'
            )
        )

    logger.addHandler(file_handler)
    logger.setLevel(log_level)

    return logger


class AppLogger:
    def __init__(
        self,
        name: str = APP_LOGGER_NAME,
        chat_id: str | None = None
    ):
        self.name = name
        self.logger = __get_logger__(self.name)
        self.chat_id = chat_id
        if chat_id:
            self.info(
                msg='Initiating chat...',
                function_name='__init__'
            )

    def __log__(
        self,
        fn: callable,
        msg: str | dict,
        function_name: str | None = None
    ):
        fn(
            msg,
            extra={
                'function_name': function_name,
                'chat_id': self.chat_id
            }
        )

    def error(
        self,
        msg: str,
        function_name: str | None = None
    ) -> None:
        self.__log__(
            self.logger.error,
            msg,
            function_name
        )

    def info(
        self,
        msg: str | dict,
        function_name: str | None = None
    ) -> None:
        self.__log__(
            self.logger.info,
            msg,
            function_name
        )

    def debug(
        self,
        msg: str | dict,
        function_name: str | None = None
    ) -> None:
        self.__log__(
            self.logger.debug,
            msg,
            function_name
        )
