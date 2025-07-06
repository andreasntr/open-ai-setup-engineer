from collections.abc import Callable, Generator
from typing import Any, Optional
from enum import StrEnum
from datetime import datetime, UTC
from pydantic import BaseModel
from traceback import format_exc
from re import fullmatch, findall, IGNORECASE
from json import loads
from itertools import chain
from uuid import uuid4 as uuid
from re import escape, sub
from streamlit.delta_generator import DeltaGenerator
from src.utils.prompts import *
from src.utils.llm import (
    human_message,
    get_embeddings,
    get_streaming_response,
    Usage,
    StreamingResponse,
    _ModelNames
)
from src.utils.mongo import (
    get_docs,
    create_session,
    update_session,
    get_cached_queries
)
from src.utils.setup import (
    extract_setup_info,
    get_available_components,
    get_unavailable_components,
    get_cars_with_same_unavailable_components
)
from src.utils.responses import *
from src.utils.custom_logging import AppLogger
from time import sleep
from random import choice, uniform
from os import getenv


SUPPORT_LLM_ALIAS = "support"
SUPPORT_LLM_TEMPERATURE = 0.1
RECOMMENDATION_LLM_ALIAS = "recommender"
RECOMMENDATION_LLM_TEMPERATURE = 0.2
CHUNK_LAG = 0.025
CHUNK_SIZE = 5
# The reverse % is made to sample from the uniform distribution.
# Defaults to 20% of the times
CACHE_HIT_DESIRED_RATIO = 1 - float(getenv('CACHE_HIT_DESIRED_RATIO', 0.8))

MAX_RETRIES = 3
MAX_CHAT_HISTORY_MESSAGES = 10
STEPS_MAPPING = {
    25: 'Analyzing your question...',
    50: 'Crawling my knowledge base...',
    75: 'Thinking...',
}

SERVICE_DOWN_MESSAGE = "Sorry, it looks like there was an error and I'm not able to process your request.  \n" + \
    "If the issue persists, please try again later or report the error using the contacts in the sidebar."


class ChatModeEnum(StrEnum):
    """
    An enumeration of the available chat modes for the AI-powered setup engineer application.

    The `ChatModeEnum` class defines two possible chat modes:

    - `TIPS_MODE`: This mode is used when the user is asking for general setup tips, without
      providing a specific car setup.
    - `SETUP_MODE`: This mode is used when the user is asking for feedback on a specific car
      setup, and the application should evaluate the setup and provide recommendations.

    The `StrEnum` base class ensures that the chat mode values are represented as strings.
    """
    TIPS_MODE = "tips"
    SETUP_MODE = "setup"


def replace_exact_sequence(text: str, sequence: str, replacement: str) -> str:
    """
    Replace an exact sequence of characters in a given text with a replacement string.

    This function takes a text string, a sequence of characters to be replaced, and a replacement
    string. It creates a regular expression pattern that matches the sequence only if it is not
    preceded or followed by alphanumeric characters. The function then uses the `re.sub` function
    to replace the matched pattern with the replacement string.

    Args:
        text (str): The input text string in which the replacement will be made.
        sequence (str): The sequence of characters to be replaced.
        replacement (str): The string to replace the matched sequence.

    Returns:
        str: The modified text string with the specified sequence replaced.
    """
    parsed_replacement = replacement.split(':')[0]

    # if a range of corners is found, concatenate them
    involved_corners = findall(
        r'(Turn(?:s){0,1} [0-9]{1,2}(?:\-[0-9]{1,2})*)',
        replacement
    )
    if len(involved_corners) > 1:
        parsed_replacement = ', '.join([
            corner.split(':')[0]
            for corner in involved_corners
        ])

    # Create a pattern that matches the sequence
    # only if it's not preceded or followed by alphanumeric characters
    pattern = r'(?<!\w)' + escape(sequence) + r'(?!\w)'

    # Use re.sub to replace the matched pattern with the replacement
    return sub(pattern, parsed_replacement, text)


class AISetupEngineerApp:
    """
    A class that represents an AI-powered setup engineer application.

    This class is responsible for handling user queries, processing them, and generating
    responses based on the user's chat mode (tips mode or setup mode). It interacts with
    various language models and data sources to provide relevant setup tips and evaluate
    user-provided car setups.

    The class maintains a session-level state, including the session ID, chat ID, chat
    history, and interaction history. It also tracks the token usage for the support and
    recommender language models, which is used for logging and billing purposes.

    Attributes:
        __session_id__ (str): The unique identifier for the current session.
        __chat_id__ (str): The unique identifier for the current chat.
        __session_start_at__ (datetime): The timestamp when the session started.
        __chat_start_at__ (datetime): The timestamp when the chat started.
        __chat_history__ (list[list[str]]): The history of questions and answers in the chat.
        __interactions__ (list[tuple[str, str, str, dict[str, int | str]]]): The history of interactions,
            including the original question, rephrased question, answer, and metadata.
        __curr_progress__ (int): The current progress of the response generation.
        __input_tokens_support__ (int): The number of input tokens used for the support language model.
        __cached_input_tokens_support__ (int): The number of cached input tokens used for the support language model.
        __output_tokens_support__ (int): The number of output tokens used for the support language model.
        __input_tokens_recommender__ (int): The number of input tokens used for the recommender language model.
        __cached_input_tokens_recommender__ (int): The number of cached input tokens used for the recommender language model.
        __output_tokens_recommender__ (int): The number of output tokens used for the recommender language model.
        __has_corners__ (bool): A flag indicating whether the user's query mentions specific corners.
        __user_data__ (dict[str, str]): Additional user data associated with the session.
        __logger__ (AppLogger): The logger for the application.
    """
    __session_id__: str
    __chat_id__: str
    __session_start_at__: datetime
    __chat_start_at__: datetime
    __chat_history__: list[list[str]]
    __interactions__: list[tuple[str, str, str, dict[str, int | str]]]
    __curr_progress__: int
    __input_tokens_support__: int
    __cached_input_tokens_support__: int
    __output_tokens_support__: int
    __input_tokens_recommender__: int
    __cached_input_tokens_recommender__: int
    __output_tokens_recommender__: int
    __has_corners__: bool
    __user_data__: dict[str, str]
    __logger__: AppLogger

    def __init__(
        self,
        user_data: Optional[dict[str, str]] = None,
    ):

        self.__session_id__ = str(uuid())
        self.__chat_id__ = str(uuid())
        self.__session_start_at__ = datetime.now(UTC)
        self.__chat_start_at__ = self.__session_start_at__
        self.__chat_history__ = []
        self.__interactions__ = []
        self.__curr_progress__ = 0
        self.__input_tokens_support__ = 0
        self.__output_tokens_support__ = 0
        self.__input_tokens_recommender__ = 0
        self.__output_tokens_recommender__ = 0
        self.__has_corners__ = False
        self.__user_data__ = user_data
        self.__logger__ = AppLogger(chat_id=self.__chat_id__)

    def __del__(self):
        pass

    def log_output(f: Callable) -> Callable:
        """
        A decorator that logs the inputs and outputs of a function call.

        This decorator wraps a function and logs the output of the function call
        using the `__logger__` attribute of the class. It also logs the input
        arguments of the function, excluding any arguments that are instances of
        `DeltaGenerator`.

        For the `__rephrase_question__` method, the `chat_history` argument is
        added to the inputs using the class `__parse_chat_history__` method.

        Args:
            f (Callable): The function to be wrapped.

        Returns:
            Callable: The wrapped function that execute the input callable and 
                      logs its inputs and outputs.
        """

        def call_and_log_output(self, *args):

            arg_names = list(
                # skip self
                f.__code__.co_varnames[1:]
            )
            filtered_args = {
                arg_name: (
                    arg_value if not isinstance(arg_value, BaseModel)
                    else loads(arg_value.model_dump_json())
                )
                for arg_name, arg_value in zip(arg_names, args)
                if not isinstance(arg_value, DeltaGenerator)
            }
            if f.__name__ == '__rephrase_question__':
                filtered_args['chat_history'] = self.__parse_chat_history__()

            outputs = f(self, *args)

            self.__logger__.debug(
                msg=f'{f.__name__}: {outputs}',
                function_name=f.__name__
            )

            return outputs

        return call_and_log_output

    def __register_token_usage__(self, usage: Usage) -> None:
        """
        Register the token usage for the specified language model.

        This method updates the token usage counters for the support and recommender
        language models based on the provided model name and usage information.

        If the model name matches the support LLM name, the input and output tokens
        are added to the support LLM counters. Otherwise, they are added to the
        recommender LLM counters.
        Args:
            model (str): The name of the language model for which to register the token usage.
            usage (Usage): A `Usage` object containing the input and output token counts.

        Returns:
            None
        """
        if usage.model == SUPPORT_LLM_ALIAS:
            self.__input_tokens_support__ += usage.input_tokens
            self.__cached_input_tokens_support__ += usage.cached_input_tokens
            self.__output_tokens_support__ += usage.output_tokens
        else:
            self.__input_tokens_recommender__ += usage.input_tokens
            self.__cached_input_tokens_recommender__ += usage.cached_input_tokens
            self.__output_tokens_recommender__ += usage.output_tokens

    def __stream_response__(
        self,
        streaming: Generator[StreamingResponse],
        container: Optional[DeltaGenerator] = None,
        progress: Optional[DeltaGenerator] = None
    ) -> str:
        """
        Stream the response from a language model and display it in the user interface.

        This method takes a list of streaming events from a language model and processes them
        to generate the final response. It iterates through the streaming events, appending
        the generated text to the response string. If a container is provided, it displays
        the response in the container as it is being generated.

        The method also updates the progress bar, if provided, to simulate the progress of
        the response generation. Additionally, it registers the token usage for the language
        model, which is used for tracking and billing purposes.

        Args:
            streaming (Generator[StreamingResponse]): A list of streaming events from the language model.
            container (Optional[DeltaGenerator]): A Streamlit container for displaying the
                response as it is being generated. Defaults to None.
            progress (Optional[DeltaGenerator]): A Streamlit progress bar for tracking the
                progress of the response generation. Defaults to None.

        Returns:
            str: The final response generated by the language model.
        """
        response = ''
        for event in streaming:
            text = event.text
            usage = event.usage

            if text:
                if container:
                    for i in range(0, len(text), CHUNK_SIZE):
                        response += text[i:i+CHUNK_SIZE]
                        container.markdown(response)
                        sleep(CHUNK_LAG)
                else:
                    response += text
            if usage:
                if progress:
                    self.__advance_to_next_step__(
                        progress
                    )
                self.__register_token_usage__(usage)

        return response

    def __mock_streaming__(
        self,
        input: str,
        progress: DeltaGenerator,
        container: DeltaGenerator
    ) -> str:
        """
        Simulates a streaming response by gradually displaying the output in the container.

        This method takes an input string, a Streamlit progress bar, and a Streamlit container,
        and gradually displays the input string in the container, giving the impression of a
        streaming response. It updates the progress bar to simulate the progress of the response
        generation.

        Args:
            input (str): The input string to be displayed in the container.
            progress (DeltaGenerator): The Streamlit progress bar to be updated.
            container (DeltaGenerator): The Streamlit container where the input string will be displayed.

        Returns:
            str: The input string that was displayed in the container.
        """
        while self.__curr_progress__ != 75:
            self.__advance_to_next_step__(progress)
        for i in range(0, len(input), CHUNK_SIZE):
            container.markdown(input[:i])
            sleep(CHUNK_LAG)
        container.markdown(input)
        self.__advance_to_next_step__(progress)

    def __call_support_llm__(
        self,
        system: str,
        messages: list[dict[str, str]],
        examples: Optional[list[tuple[str, str]]] = None,
        max_tokens: int = 1000
    ) -> str:
        """
        Call the support language model and stream the response.

        This method takes a system prompt, a list of message dictionaries (containing 'role' and 'content' keys),
        an optional list of example question-answer pairs, and a maximum number of tokens to generate. It then
        creates a streamable request to the support language model using the provided inputs, and streams the
        response back to the caller.

        Args:
            system (str): The system prompt to provide context to the language model.
            messages (list[dict[str, str]]): A list of message dictionaries, where each dictionary contains a
                'role' (either 'system', 'human', or 'assistant') and 'content' (the message text).
            examples (Optional[list[tuple[str, str]]]): An optional list of example question-answer pairs to
                provide additional context to the language model.
            max_tokens (int, optional): The maximum number of tokens to generate in the response. Defaults to 1000.

        Returns:
            str: The generated response from the support language model.
        """
        streaming = get_streaming_response(
            system=system,
            messages=messages,
            examples=examples,
            model=SUPPORT_LLM_ALIAS,
            temperature=SUPPORT_LLM_TEMPERATURE,
            max_tokens=max_tokens
        )
        return self.__stream_response__(streaming)

    @log_output
    def __call_specific_tips_llm__(
        self,
        query: str
    ) -> Recommendations:
        """
        Generates a response from a language model with specific tips based on the user's query.

        This method is responsible for calling the language model to provide suggestions or tips
        to the user based on their query. It first determines the available components based on
        the car metadata, and then constructs a system prompt that instructs the model to only
        provide suggestions for the available components.

        The method then prepares the input messages, creates a streamable request to the language
        model, and streams the response back to the user interface using the provided container
        and progress bar.

        Args:
            query (str): The user's query or question.

        Returns:
            Recommendations: a Recommendations object containing the logical steps to formalize
                the final answer and the answer itself.
        """
        car_name = self.car.get('name')
        unavailable_components = get_unavailable_components(
            car_name
        )
        system_prompt = TIPS_PROMPT
        if len(unavailable_components):
            available_components = get_available_components(
                car_name
            )
            available_components_list = ';\n- '.join(available_components)
            system_prompt += f"""
                You must only output suggestions for the following components:
                - {available_components_list}.
            """
        response = Recommendations.model_validate(
            self.__call_json_llm__(
                system=system_prompt,
                messages=[human_message(query)],
                model=RECOMMENDATION_LLM_ALIAS,
                temperature=RECOMMENDATION_LLM_TEMPERATURE,
                max_tokens=2000,
                response_class=Recommendations
            )
        )
        return response

    @log_output
    def __call_setup_llm__(
        self,
        query: str
    ) -> Recommendations:
        """
        Call the setup LLM to generate a response to the given query.

        This method takes a query, a Streamlit container, and a Streamlit progress bar as input.
        It prepares the input messages, including a system message with setup tips instructions
        and the user's query, and then creates a streamable request to the setup LLM model.

        The method then streams the response from the LLM and returns the generated text.

        Args:
            query (str): The user's query to be processed by the setup LLM.
            container (DeltaGenerator): A Streamlit container for displaying the response.
            progress (DeltaGenerator): A Streamlit progress bar for tracking the response generation.

        Returns:
            Recommendations: a Recommendations object containing the logical steps to formalize
                the final answer and the answer itself.
        """
        car_name = self.car.get('name')
        unavailable_components = get_unavailable_components(
            car_name
        )
        system_prompt = SETUP_TIPS_INSTRUCTIONS
        if len(unavailable_components):
            available_components = get_available_components(
                car_name
            )
            available_components_list = ';\n- '.join(available_components)
            system_prompt += f"""
                You must only output suggestions for the following components:
                - {available_components_list}.
            """

        response = Recommendations.model_validate(
            self.__call_json_llm__(
                system=system_prompt,
                messages=[human_message(query)],
                model=RECOMMENDATION_LLM_ALIAS,
                temperature=RECOMMENDATION_LLM_TEMPERATURE,
                max_tokens=2000,
                response_class=Recommendations
            )
        )
        return response

    def __call_json_llm__(
        self,
        system: str,
        messages: list[dict[str, str]],
        examples: Optional[list[tuple[str, str]]] = None,
        response_class: Optional[BaseModel] = None,
        model: _ModelNames = SUPPORT_LLM_ALIAS,
        temperature: float = SUPPORT_LLM_TEMPERATURE,
        max_tokens: int = 1000
    ) -> dict[str, Any]:
        """
        Call a language model and return the response as a JSON object.

        This method takes a list of input messages, which can include a system message and
        a list of human and assistant messages. It then creates a streamable request to the
        language model, using the SUPPORT_LLM model, with a temperature of 0 and a maximum
        of 1000 tokens.

        The output type is set to 'json', and an optional response_class can be provided to
        validate the JSON response against a Pydantic model.

        The method then streams the response from the language model, converts the JSON
        response to a Python dictionary, and returns the result.

        Args:
            system (str): A system message to be included in the input messages.
            messages (list[dict[str, str]]): A list of input messages, where each message is a
                dictionary with 'role' and 'content' keys.
            examples (Optional[list[tuple[str, str]]]): A list of example input-output pairs.
            response_class (Optional[BaseModel]): A Pydantic model class to validate the
                JSON response against. Defaults to None.

        Returns:
            dict[str, Any]: The JSON response from the language model, as a Python dictionary.
        """
        streaming = get_streaming_response(
            system=system,
            messages=messages,
            examples=examples,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_class=response_class
        )
        response = self.__stream_response__(streaming)
        return loads(response)

    @log_output
    def __get_unavailable_components__(self) -> dict[str, dict[str, str]]:
        """
        Generates a filter for unavailable components based on the car's coded boundaries.

        This method checks the car's coded boundaries to determine which components are unavailable
        for the current car. It then creates a list of filters to exclude the unavailable components
        from the search results.

        The method checks for the following unavailable components:
        - Front splitter
        - Rear wing
        - Differential preload
        - Front and rear wheel rate
        - Front and rear anti-roll bar
        - Traction control (TC1 and TC2)
        - ABS
        - Front caster
        - Bumpstops
        - Dampers

        The resulting filter is a list of dictionaries, where each dictionary represents a filter
        for a specific unavailable component.

        Returns:
            dict[str, dict[str, str]]: A list of filters for the unavailable components.
        """
        unavailable_filter = []
        unavailable_components = get_unavailable_components(
            self.car.get('name')
        )
        if 'front splitter' in unavailable_components:
            unavailable_filter.append({'item': {'$ne': 'front splitter'}})
        if 'rear wing' in unavailable_components:
            unavailable_filter.append({'item': {'$ne': 'rear wing'}})
        if 'preload' in unavailable_components:
            unavailable_filter.append({'item': {'$ne': 'differential'}})
        if 'front wheel rate' in unavailable_components and \
                'rear wheel rate' in unavailable_components:
            unavailable_filter.append({'item': {'$ne': 'wheel rate'}})
        if 'front anti-roll bar' in unavailable_components:
            unavailable_filter.append({'item': {'$ne': 'front anti-roll bar'}})
        if 'rear anti-roll bar' in unavailable_components:
            unavailable_filter.append({'item': {'$ne': 'rear anti-roll bar'}})
        if 'TC1' in unavailable_components:
            unavailable_filter.append({'item': {'$ne': 'TC1'}})
        if 'TC2' in unavailable_components:
            unavailable_filter.append({'item': {'$ne': 'TC2'}})
        if 'ABS' in unavailable_components:
            unavailable_filter.append({'item': {'$ne': 'ABS'}})
        if 'front caster' in unavailable_components:
            unavailable_filter.append({'item': {'$ne': 'caster'}})
        if 'front bumpstop rate' in unavailable_components and \
                'rear bumpstop rate' in unavailable_components and \
                'front bumpstop range' in unavailable_components and \
                'rear bumpstop range' in unavailable_components:
            unavailable_filter.append({'item': {'$ne': 'bumpstops'}})
        if 'front slow bump' in unavailable_components and \
                'front fast bump' in unavailable_components and \
                'rear slow bump' in unavailable_components and \
                'rear fast bump' in unavailable_components:
            unavailable_filter.append({'title': {'$ne': 'dampers'}})
        return unavailable_filter

    @log_output
    def __get_relevant_docs__(self, query: str, retry_penalty=0) -> list[dict[str, Any]]:
        """
        Retrieves a list of relevant documents based on the given query.

        This method takes a query string and an optional retry penalty as input, and returns a list of
        relevant documents. The method uses a vector search to find the most relevant documents based
        on the embeddings of the query. It also applies a filter to exclude certain documents based on
        the unavailable components for the current car.

        Args:
            query (str): The query string to search for relevant documents.
            retry_penalty (int, optional): An optional penalty to apply to the score threshold for
                the vector search. Defaults to 0.

        Returns:
            list[dict[str, Any]]: A list of relevant documents, where each document is represented as
                a dictionary with the 'content' key containing the document text.
        """
        k = 5
        embedded_query = get_embeddings(query)
        unavailable_components_filter = self.__get_unavailable_components__()
        car_name = self.car.get('name')
        docs = get_docs([
            {
                "$vectorSearch": {
                    "index": "docs_embeddings",
                    "path": "embeddings",
                    "queryVector": embedded_query,
                    "numCandidates": k * 2,
                    "limit": k,
                    "filter":  {
                        '$or': [
                            {
                                '$and': [
                                    {'title': {'$eq': 'engine_maps'}},
                                    {'car': {'$eq': car_name.lower()}}
                                ]
                            },
                            {
                                '$and': [
                                    {'title': {'$ne': 'engine_maps'}},
                                    *unavailable_components_filter
                                ]
                            }
                        ]
                    }
                }
            },
            {
                '$project': {
                    'content': 1,
                    'title': 1,
                    'item': 1,
                    '_id': 0,
                    'score': {
                        '$meta': 'vectorSearchScore'
                    }
                }
            },
            {
                "$match": {"score": {"$gte": 0.7 - 0.1 * retry_penalty}}
            },
            {
                "$project": {"score": 0}
            }
        ])
        return docs

    @log_output
    def __extract_corners__(self, query: str, track_layout: str) -> tuple[str, CornersList]:
        """
        Extract the corners mentioned in the given query and the track layout.

        This method takes a user query and the track layout as input, and returns a
        tuple containing a boolean flag indicating whether the query mentions specific
        corners, and a CornersList object containing the details of the involved corners.

        The method first defines a set of example corner references and their corresponding
        corner descriptions. It then iterates through the track layout to find the format
        of the corner descriptions, and creates additional examples based on the found format.

        The method then uses the examples to call a language model (via the `__call_json_llm__`
        method) to extract the corners mentioned in the user's query. If any corners are
        found, the method returns `True` and the CornersList object containing the details.
        If no corners are found, the method returns `False` and an empty CornersList.

        Args:
            query (str): The user's input query.
            track_layout (str): The layout of the track, containing the descriptions of the corners.

        Returns:
            tuple[str, CornersList]: A tuple containing the parsed query and a CornersList object
                with the details of the involved corners.
        """
        parsed_query = query
        corners = track_layout.split('\n\n')
        corner_number_example = corners[0]
        corner_number_example_lower = CornersList(
            corners=[
                CornerDetails(
                    reference='t1',
                    corner=corner_number_example
                )
            ]
        ).model_dump_json()
        corner_number_example_upper = CornersList(
            corners=[
                CornerDetails(
                    reference='T 1',
                    corner=corner_number_example
                )
            ]
        ).model_dump_json()

        corner_examples = [
            ("oversteer in t1?", corner_number_example_lower),
            ("understeer T 1?", corner_number_example_upper),
            ("how to fix oversteer?", CornersList(corners=[]).model_dump_json()),
            ("what is toe?", CornersList(corners=[]).model_dump_json()),
            ("what is 5 doing?", CornersList(corners=[]).model_dump_json()),
            ("talk about 3", CornersList(corners=[]).model_dump_json()),
            (
                "improve stability t1 to 3",
                CornersList(corners=[
                    CornerDetails(
                        reference='t1 to 3',
                        corner=', '.join(corners[:3])
                    )
                ]).model_dump_json()
            ),
            (
                "more grip from t2 all the way to turn 4",
                CornersList(corners=[
                    CornerDetails(
                        reference='from t2 all the way to turn 4',
                        corner=', '.join(corners[1:4])
                    )
                ]).model_dump_json()
            )
        ]

        try:
            corner_name_example = None

            for corner in corners:
                corner_name_example = fullmatch(
                    r'Turn [0-9]{1,2} \(.+\): .*', corner)
                if corner_name_example is not None:
                    corner_name_example = corner_name_example.group(
                        0)
                    corner_name_example_name = corner_name_example.split('(')[1].split(')')[
                        0]
                    corner_name_example_lower = CornersList(
                        corners=[
                            CornerDetails(
                                reference=corner_name_example_name.lower(),
                                corner=corner_name_example
                            )
                        ]
                    ).model_dump_json()
                    corner_name_example_upper = CornersList(
                        corners=[
                            CornerDetails(
                                reference=corner_name_example_name,
                                corner=corner_name_example
                            )
                        ]
                    ).model_dump_json()
                    corner_examples.extend([
                        (f"how to fix oversteer at {corner_name_example_name.lower()}?",
                            corner_name_example_lower),
                        (f"how to increase grip at {corner_name_example_name}?",
                            corner_name_example_upper),
                    ])
                    break

            # replace explicit corner references for better accuracy
            prompt = EXTRACT_CORNERS_PROMPT.format(
                track_layout=track_layout)
            corner_indicators = findall(
                r'(t(?:urns?)?\s*[0-9]{1,2})',
                query,
                flags=IGNORECASE
            )
            for corner_indicator in corner_indicators:
                corner_number = findall(
                    r'([0-9]+)',
                    corner_indicator
                )[0]
                parsed_query = replace_exact_sequence(
                    parsed_query,
                    corner_indicator,
                    f'turn {corner_number}'
                )

            involved_corners = CornersList.model_validate(
                self.__call_json_llm__(
                    system=prompt,
                    messages=[human_message(parsed_query)],
                    examples=corner_examples,
                    response_class=CornersList
                )
            )

        except:
            involved_corners = CornersList(corners=[])

        self.__has_corners__ = len(involved_corners.corners) > 0

        return parsed_query, involved_corners

    @log_output
    def __rephrase_question__(self, followup: str) -> Rephrase:
        """
        Rephrase a follow-up question based on the chat history.

        This method takes a follow-up question and uses the chat history to generate
        a rephrased version of the question. It calls a language model to process
        the input and return a structured response.

        Args:
            followup (str): The follow-up question to be rephrased.

        Returns:
            Rephrase: A Rephrase object containing the rephrased question and additional
                      information about whether the question is ambiguous, needs
                      clarification, or is not addressable.
        """
        rephrase_output = Rephrase.model_validate(
            self.__call_json_llm__(
                system=REPHRASE_STANDALONE_QUESTION,
                messages=[human_message(
                    (
                        f'{self.__parse_chat_history__()}\n'
                        f'<followup>\n{followup}\n</followup>'
                    )
                )],
                response_class=Rephrase
            )
        )
        return rephrase_output

    def __update_chat_history__(self, question: str, answer: str) -> None:
        """
        Update the chat history with the latest question and answer.

        This method appends the given question and answer to the chat history list.
        If the chat history exceeds the maximum allowed number of messages (defined
        by the `MAX_CHAT_HISTORY_MESSAGES` constant), the oldest messages are
        removed to keep the history within the limit.

        Args:
            question (str): The user's question.
            answer (str): The assistant's response to the question.

        Returns:
            None
        """
        self.__chat_history__.append([question, answer])
        # keep only the last MAX_CHAT_HISTORY_MESSAGES interactions
        self.__chat_history__ = self.__chat_history__[
            -MAX_CHAT_HISTORY_MESSAGES:]

    @log_output
    def __augment_with_corners__(self, question: str, corners: CornersList) -> str:
        """
        Augment the given question with context about the involved corners.

        This method takes a question and a list of corner details, and generates a new
        question that incorporates the information about the involved corners. If there
        are no specific corners mentioned in the original question, the method simply
        returns the original question.

        Args:
            question (str): The original question to be augmented.
            corners (CornersList): A list of corner details, containing the reference
                and the full description of the corners.

        Returns:
            str: The augmented question, with the context about the involved corners.
        """
        if len(corners.corners) == 0:
            return question
        involved_corners_str = '\n'.join(
            list(map(lambda c: c.corner, corners.corners))
        )
        return self.__call_support_llm__(
            system=ADD_CORNERS_CONTEXT_PROMPT.format(
                involved_corners=involved_corners_str
            ),
            messages=[human_message(question)]
        )

    def __get_questions_from_chat_history__(self) -> list[str]:
        """
        Retrieve the questions from the chat history.

        This method extracts the list of questions from the chat history, which is stored as a list of
        question-answer pairs. It uses the `chain.from_iterable` function from the `itertools` module
        to flatten the list of pairs, and then selects every other element starting from the first
        one (the questions).

        Returns:
            list[str]: A list of questions from the chat history.
        """
        return list(chain.from_iterable(self.__chat_history__))[::2]

    def __get_answers_from_chat_history__(self) -> list[str]:
        """
        Retrieve the answers from the chat history.

        This method extracts the list of answers from the chat history, which is stored as a list of
        question-answer pairs. It uses the `chain.from_iterable` function from the `itertools` module
        to flatten the list of pairs, and then selects every other element starting from the second
        one (the answers).

        Returns:
            list[str]: A list of answers from the chat history.
        """
        return list(chain.from_iterable(self.__chat_history__))[1::2]

    def __parse_qa__(self, questions: list[str], answers: list[str]):
        """
        Parse the question-answer pairs into a formatted string.

        This method takes two lists of questions and answers, and formats them into a string
        with XML-like tags to represent the question and answer sections.

        Args:
            questions (list[str]): A list of questions.
            answers (list[str]): A list of answers, corresponding to the questions.

        Returns:
            str: A formatted string containing the question-answer pairs.
        """
        parsed_qa = ''
        for q, a in zip(
            questions,
            answers
        ):
            parsed_qa += f'<question>\n{q}\n</question>\n<answer>\n{a}\n</answer>\n'
        return parsed_qa

    def __parse_chat_history__(self) -> str:
        """
        Parse the chat history into a formatted string.

        This method retrieves the questions and answers from the chat history, and formats them
        into a string with XML-like tags to represent the chat history section.

        Returns:
            str: A formatted string containing the chat history.
        """
        questions = self.__get_questions_from_chat_history__()
        answers = self.__get_answers_from_chat_history__()
        parsed_chat_history = f'<chat_history>\n{self.__parse_qa__(questions, answers)}\n</chat_history>'
        return parsed_chat_history

    def __save_progress__(self) -> None:
        """
        Save the current progress of the chat session to the database.

        This method updates the chat session in the database with the latest information,
        including the session ID, chat ID, user data, session start time, chat start time,
        chat mode, car details, and track details. It also adds the latest interaction
        (question, rephrased question, answer, and metadata) to the session's interaction
        history.

        If this is the first interaction in the session, the method creates a new session
        document in the database. Otherwise, it updates the existing session document by
        appending the new interaction to the interactions array.

        Returns:
            None
        """
        base_update = {
            'session_id': self.__session_id__,
            'chat_id': self.__chat_id__,
            'user': self.__user_data__,
            'session_start_at': self.__session_start_at__,
            'chat_start_at': self.__chat_start_at__,
            'chat_mode': self.chat_mode,
            'car': self.car,
            'track': self.track
        }
        q, rephrased_q, a, metadata = self.__interactions__[-1]
        if len(self.__interactions__) == 1:
            if self.chat_mode == ChatModeEnum.SETUP_MODE:
                base_update.update({'setup_json': self.setup_json})
            create_session({
                **base_update,
                'interactions': [
                    {
                        'created_at': datetime.now(UTC),
                        'original_q': q,
                        'q': rephrased_q,
                        'a': a,
                        'has_corners': self.__has_corners__,
                        'metadata': metadata
                    }
                ]
            })
        else:
            update_session(
                {
                    'session_id': self.__session_id__,
                    'chat_id': self.__chat_id__
                },
                {
                    "$push": {
                        'interactions':
                            {
                                'created_at': datetime.now(UTC),
                                'original_q': q,
                                'q': rephrased_q,
                                'a': a,
                                'has_corners': self.__has_corners__,
                                'metadata': metadata
                            }
                    }
                }
            )

    def __advance_to_next_step__(self, progress: DeltaGenerator) -> None:
        """
        Advances the progress bar to the next step.

        This method updates the progress bar to the next step in the sequence of steps
        defined in the `STEPS_MAPPING` dictionary. It calculates the next step percentage
        and updates the progress bar accordingly. If the next step is 100%, the progress
        bar is emptied, and the current progress is reset to 0.

        Args:
            progress (DeltaGenerator): The Streamlit progress bar to be updated.

        Returns:
            None
        """
        next_step_pct = self.__curr_progress__ + 25
        if next_step_pct == 100:
            progress.empty()
            self.__curr_progress__ = 0
            return
        next_step = STEPS_MAPPING.get(next_step_pct)
        progress.progress(next_step_pct, next_step)
        self.__curr_progress__ = next_step_pct

    def __handle_static_response__(
        self,
        question: str,
        rephrased_question: str,
        response: str,
        comes_from_model: bool,
        progress: DeltaGenerator,
        container: DeltaGenerator,
        is_error: bool = False,
        is_cached: bool = False
    ) -> str:
        """
        Streams a static response to the user interface.

        This method is used to display a pre-generated response to the user in a streaming
        fashion, giving the impression of a real-time response. It updates the chat history,
        appends the interaction to the session, and displays the response using a mock
        streaming approach.

        Args:
            question (str): The original user question.
            response (str): The response to be displayed.
            comes_from_model (bool): Indicates whether the response is generated by a model or not.
            progress (DeltaGenerator): A Streamlit progress bar for tracking the response generation.
            container (DeltaGenerator): A Streamlit container for displaying the response.
            is_error (bool, optional): Indicates whether the response is an error message. Defaults to `False`.

        Returns:
            str: The response string that was displayed.
        """
        metadata = {
            "model": SUPPORT_LLM_ALIAS if comes_from_model else None,
            "is_cached": is_cached,
            "is_error": is_error,
            "input_tokens_support": self.__input_tokens_support__,
            "cached_input_tokens_support": self.__cached_input_tokens_support__,
            "output_tokens_support": self.__output_tokens_support__,
            "input_tokens_recommender": self.__input_tokens_recommender__,
            "cached_input_tokens_recommender": self.__cached_input_tokens_recommender__,
            "output_tokens_recommender": self.__output_tokens_recommender__
        }
        self.__interactions__.append(
            (question, rephrased_question, response, metadata))
        if not is_error:
            self.__update_chat_history__(question, response)
        self.__mock_streaming__(
            response,
            progress,
            container
        )
        return response

    @log_output
    def __get_cached_response__(
        self,
        query: str
    ) -> str | None:
        """
        Retrieve a cached response for a given query based on the car's name and other cars that share the same unavailable components.

        Args:
            query (str): The query string for which a cached response is being sought.

        Returns:
            str | None: A cached response string if found; otherwise, None.
        """
        car_name = self.car.get('name')
        similar_cars_filter = [
            *get_cars_with_same_unavailable_components(
                car_name
            ),
            car_name
        ]
        cached_queries = get_cached_queries(
            query=query,
            similar_cars=similar_cars_filter,
            track_downforce=self.track.get('downforce')
        )

        if not len(cached_queries):
            return None
        # generate a new response only (1-CACHE_HIT_DESIRED_RATIO) % of the times
        # a cached response has been found,
        # so to increase diversity
        return_cached = uniform(0, 1) < CACHE_HIT_DESIRED_RATIO
        chosen_response = choice(cached_queries)

        return chosen_response if return_cached else None

    def __give_tips__(
        self,
        query: str,
        car_details: dict[str, str],
        track_details: dict[str, str],
        container: DeltaGenerator,
        progress: DeltaGenerator
    ) -> tuple[str, bool]:
        """
        Provides setup tips based on a given query.

        This method takes a query, car details, and track details, and processes the query
        to provide relevant setup tips. It performs the following steps:

        1. Extracts any specific corners mentioned in the query.
        2. Rephrases the query to make it more specific.
        3. Checks if the rephrased query is ambiguous or needs clarification.
        4. Augments the rephrased query with the extracted corners.
        5. Retrieves relevant documents based on the rephrased query.
        6. Constructs a contextualized question using the relevant documents, car details,
        and track details.
        7. Calls the recommendation LLM to generate a response with setup tips.
        8. Handles any exceptions that may occur during the process.
        9. Appends the interaction to the chat history and returns the response and a boolean
        indicating whether the query was successfully handled.

        Args:
            query (str): The original query.
            car_details (dict[str, str]): A dictionary containing car-related details.
            track_details (dict[str, str]): A dictionary containing track-related details.
            container (DeltaGenerator): A Streamlit container for displaying the response.
            progress (DeltaGenerator): A Streamlit progress bar for tracking the evaluation process.

        Returns:
            tuple[str, bool]: The response and a boolean indicating whether the query was
            successfully handled.
        """
        try:
            rephrase_output = self.__rephrase_question__(
                query
            )

            rephrased_question = rephrase_output.final_response

            rephrased_question, involved_corners_with_references = self.__extract_corners__(
                rephrased_question, track_details['layout']
            )

            if self.__curr_progress__ == 0:
                self.__advance_to_next_step__(progress)

            if self.__has_corners__:
                for corner in involved_corners_with_references.corners:
                    rephrased_question = replace_exact_sequence(
                        rephrased_question,
                        corner.reference,
                        corner.corner
                    )

            if rephrase_output.is_ambiguous_or_needs_clarification or \
                    rephrase_output.is_not_addressable:
                return self.__handle_static_response__(
                    question=query,
                    rephrased_question=rephrased_question,
                    response=rephrased_question,
                    comes_from_model=False,
                    progress=progress,
                    container=container
                ), False

            if not self.__has_corners__:
                cached_response = self.__get_cached_response__(
                    rephrased_question
                )
                if cached_response:
                    response = cached_response
                    return self.__handle_static_response__(
                        question=query,
                        rephrased_question=rephrased_question,
                        response=response,
                        comes_from_model=False,
                        progress=progress,
                        container=container,
                        is_cached=True
                    ), False

            rephrased_question = self.__augment_with_corners__(
                rephrased_question,
                involved_corners_with_references
            )

            if self.__curr_progress__ == 25:
                self.__advance_to_next_step__(progress)

            is_question_relevant = False
            relevant_docs = []

            for retry_count in range(MAX_RETRIES):
                relevant_docs = self.__get_relevant_docs__(
                    rephrased_question,
                    retry_count
                )

                is_question_relevant = len(relevant_docs) > 0
                if is_question_relevant:
                    break

            context = ''
            for doc in relevant_docs:
                context += (
                    f'\n\n\n{doc.get("title").title()}: {doc.get("item").title()}\n\n'
                    f'{doc.get("content")}\n\n\n'
                )

            if is_question_relevant:

                contextualized_question = CONTEXTUALIZED_PROMPT.format(
                    question=rephrased_question,
                    context=context,
                    engine_position=car_details['engine_position'],
                    track_downforce=track_details['downforce']
                )

                if self.__curr_progress__ == 50:
                    self.__advance_to_next_step__(progress)

                response = self.__call_specific_tips_llm__(
                    contextualized_question
                ).final_answer
                self.__mock_streaming__(response, progress, container)

                metadata = {
                    'model': RECOMMENDATION_LLM_ALIAS,
                    'input_tokens_support': self.__input_tokens_support__,
                    'cached_input_tokens_support': self.__cached_input_tokens_support__,
                    'output_tokens_support': self.__output_tokens_support__,
                    'input_tokens_recommender': self.__input_tokens_recommender__,
                    'cached_input_tokens_recommender': self.__cached_input_tokens_recommender__,
                    'output_tokens_recommender': self.__output_tokens_recommender__
                }

            else:
                return self.__handle_static_response__(
                    question=query,
                    rephrased_question=rephrased_question,
                    response=ERROR_MESSAGE,
                    comes_from_model=True,
                    progress=progress,
                    container=container,
                    is_error=True
                ), False

        except:
            tb_log = format_exc()
            self.__logger__.error(tb_log)
            return self.__handle_static_response__(
                question=query,
                rephrased_question=rephrased_question,
                response=SERVICE_DOWN_MESSAGE,
                comes_from_model=True,
                progress=progress,
                container=container,
                is_error=True
            ), False
        self.__interactions__.append(
            (query, rephrased_question, response, metadata))
        self.__update_chat_history__(
            rephrased_question, response
        )
        return response, True

    def __evaluate_setup__(
        self,
        query: str,
        car_details: dict[str, str],
        track_details: dict[str, str],
        setup_str: str,
        container: DeltaGenerator,
        progress: DeltaGenerator
    ) -> tuple[str, bool]:
        """
        Evaluates a setup query and provides a response.

        This method takes a query, car details, track details, and a setup string, and
        processes the query to provide a relevant response. It performs the following steps:

        1. Extracts any specific corners mentioned in the query.
        2. Rephrases the query to make it more specific.
        3. Checks if the rephrased query is ambiguous or needs clarification.
        4. Augments the rephrased query with the extracted corners.
        5. Retrieves relevant documents based on the rephrased query.
        6. Constructs a contextualized question using the relevant documents, car details,
        track details, and the setup string.
        7. Calls the recommendation LLM to generate a response to the contextualized question.
        8. Handles any exceptions that may occur during the process.
        9. Appends the interaction to the chat history and returns the response and a boolean
        indicating whether the query was successfully handled.

        Args:
            query (str): The original query.
            car_details (dict[str, str]): A dictionary containing car-related details.
            track_details (dict[str, str]): A dictionary containing track-related details.
            setup_str (str): A string representing the setup.
            container (DeltaGenerator): A Streamlit container for displaying the response.
            progress (DeltaGenerator): A Streamlit progress bar for tracking the evaluation process.

        Returns:
            tuple[str, bool]: The response and a boolean indicating whether the query was
            successfully handled.
        """
        try:
            rephrase_output = self.__rephrase_question__(
                query
            )

            rephrased_question = rephrase_output.final_response

            rephrased_question, involved_corners_with_references = self.__extract_corners__(
                rephrased_question, track_details['layout']
            )

            if self.__curr_progress__ == 0:
                self.__advance_to_next_step__(progress)

            if self.__has_corners__:
                for corner in involved_corners_with_references.corners:
                    rephrased_question = replace_exact_sequence(
                        rephrased_question,
                        corner.reference,
                        corner.corner
                    )

            if rephrase_output.is_ambiguous_or_needs_clarification or \
                    rephrase_output.is_not_addressable:
                return self.__handle_static_response__(
                    question=query,
                    rephrased_question=rephrased_question,
                    response=rephrased_question,
                    comes_from_model=False,
                    progress=progress,
                    container=container
                ), False

            rephrased_question = self.__augment_with_corners__(
                rephrased_question,
                involved_corners_with_references
            )

            if self.__curr_progress__ == 25:
                self.__advance_to_next_step__(progress)

            is_question_relevant = False
            relevant_docs = []

            for retry_count in range(MAX_RETRIES):
                relevant_docs = self.__get_relevant_docs__(
                    rephrased_question,
                    retry_count
                )

                is_question_relevant = len(relevant_docs) > 0
                if is_question_relevant:
                    break

            context = ''
            for doc in relevant_docs:
                context += (
                    f'\n\n\n{doc.get("title").title()}: {doc.get("item").title()}\n\n'
                    f'{doc.get("content")}\n\n\n'
                )

            if is_question_relevant:
                contextualized_question = CONTEXTUALIZED_PROMPT_WITH_SETUP.format(
                    question=rephrased_question,
                    context=context,
                    engine_position=car_details['engine_position'],
                    track_downforce=track_details['downforce'],
                    setup=setup_str
                )

                if self.__curr_progress__ == 50:
                    self.__advance_to_next_step__(progress)

                model = RECOMMENDATION_LLM_ALIAS
                response = self.__call_setup_llm__(
                    contextualized_question
                ).final_answer
                self.__mock_streaming__(response, progress, container)

                metadata = {
                    'model': model,
                    'input_tokens_support': self.__input_tokens_support__,
                    'cached_input_tokens_support': self.__cached_input_tokens_support__,
                    'output_tokens_support': self.__output_tokens_support__,
                    'input_tokens_recommender': self.__input_tokens_recommender__,
                    'cached_input_tokens_recommender': self.__cached_input_tokens_recommender__,
                    'output_tokens_recommender': self.__output_tokens_recommender__
                }

            else:
                return self.__handle_static_response__(
                    question=query,
                    rephrased_question=rephrased_question,
                    response=ERROR_MESSAGE,
                    comes_from_model=True,
                    progress=progress,
                    container=container,
                    is_error=True
                ), False

        except:
            tb_log = format_exc()
            self.__logger__.error(tb_log)
            return self.__handle_static_response__(
                question=query,
                rephrased_question=rephrased_question,
                response=SERVICE_DOWN_MESSAGE,
                comes_from_model=True,
                progress=progress,
                container=container,
                is_error=True
            ), False
        self.__interactions__.append(
            (query, rephrased_question, response, metadata))
        self.__update_chat_history__(
            rephrased_question, response
        )
        return response, True

    def reset(self, user_data: Optional[dict[str, str]] = None) -> None:
        """
        Reset the chat session to its initial state.

        This method resets various attributes of the chat session, including:
        - Generating a new chat ID
        - Setting the chat start time to the current time
        - Clearing the chat history and interactions
        - Resetting token counters for support and recommender LLMs
        - Updating user data

        Args:
            user_data (Optional[dict[str, str]]): A dictionary containing user data.
                If provided, it updates the user data for the session. Defaults to None.

        Returns:
            None
        """
        self.__logger__.info(
            msg='Resetting chat...',
            function_name='reset'
        )
        self.__chat_id__ = str(uuid())
        self.__chat_start_at__ = datetime.now(UTC)
        self.__chat_history__ = []
        self.__interactions__ = []
        self.__input_tokens_support__ = 0
        self.__cached_input_tokens_support__ = 0
        self.__output_tokens_support__ = 0
        self.__input_tokens_recommender__ = 0
        self.__cached_input_tokens_recommender__ = 0
        self.__output_tokens_recommender__ = 0
        self.__user_data__ = user_data

    def restore(self, past_chat_data: dict[str, Any]) -> None:
        """
        Restore the chat session from past chat data.

        This method restores the chat session state using the provided past chat data.
        It updates the session ID, chat ID, interactions, and chat history.

        Args:
            past_chat_data (dict[str, Any]): A dictionary containing the past chat data.
                It should include the following keys:
                - "session_id": The ID of the session to restore.
                - "chat_id": The ID of the chat to restore.
                - "interactions": A list of past interactions.

        Returns:
            None
        """
        self.__logger__.info(
            msg=f'Restoring chat {past_chat_data["chat_id"]}...',
            function_name='restore'
        )
        self.__session_id__ = past_chat_data["session_id"]
        self.__chat_id__ = past_chat_data["chat_id"]
        self.__interactions__ = past_chat_data["interactions"]
        self.__chat_history__ = list(map(
            lambda i: [i['q'], i['a']],
            filter(
                lambda i: i['a'] not in (ERROR_MESSAGE, SERVICE_DOWN_MESSAGE),
                self.__interactions__
            )
        ))

    def submit_feedback(self, feedback: dict[str, str]) -> None:
        """
        Submit user feedback for a specific interaction in the chat session.

        This method takes a feedback dictionary, extracts the message ID,
        and updates the corresponding interaction in the database with the
        provided feedback.
        Args:
            feedback (dict[str, str]): A dictionary containing feedback information.
                                       It must include a 'msg_id' key to identify the
                                       specific interaction, and can contain additional
                                       key-value pairs for various feedback aspects.

        Returns:
            None
        """
        self.__logger__.info(
            msg=f'Submitting {feedback} feedback...',
            function_name='submit_feedback'
        )
        msg_id = feedback['msg_id']
        feedback.pop('msg_id')
        update_session(
            {
                'session_id': self.__session_id__,
                'chat_id': self.__chat_id__
            },
            {
                '$set': {
                    f'interactions.{msg_id}.feedback': feedback
                }
            }
        )

    # if setup_json is given , call __evaluate_setup__. Else call __give_tips__
    def ask(
        self,
        query: str,
        car_details: dict[str, str],
        track_details: dict[str, str],
        chat_mode: ChatModeEnum,
        setup_json: Optional[dict[str, Any]] = None,
        container: Optional[DeltaGenerator] = None,
        progress: Optional[DeltaGenerator] = None
    ) -> tuple[str, bool]:
        """
        Process a user query and generate a response based on the chat mode.

        This method handles user queries in both tips and setup evaluation modes.
        It performs input validation, processes the query through the appropriate
        method based on the chat mode, and saves the interaction progress.

        Args:
            query (str): The user's input question.
            car_details (dict[str, str]): Details about the car, including name and boundaries.
            track_details (dict[str, str]): Details about the track, including name.
            chat_mode (ChatModeEnum): The current mode of the chat (TIPS_MODE or SETUP_MODE).
            setup_json (Optional[dict[str, Any]]): JSON representation of the car setup. Defaults to None.
            container (Optional[DeltaGenerator]): Streamlit container for displaying output. Defaults to None.
            progress (Optional[DeltaGenerator]): Streamlit progress bar. Defaults to None.

        Returns:
            tuple[str, bool]: A tuple containing the response string and a boolean indicating
                              if the response is an error, a clarification request or the request is not allowed.
        """
        self.__logger__.info(
            msg=query,
            function_name='ask'
        )
        self.chat_mode = chat_mode
        self.car = {
            'name': car_details.get('name'),
            'engine_position': car_details.get('engine_position')
        }
        self.car_metadata = {
            'coded_boundaries': car_details.get('coded_boundaries'),
            'actual_toe_camber_boundaries': car_details.get('actual_toe_camber_boundaries')
        }
        self.track = {
            'name': track_details.get('name'),
            'downforce': track_details.get('downforce')
        }
        self.setup_json = setup_json
        self.__input_tokens_support__ = 0
        self.__cached_input_tokens_support__ = 0
        self.__output_tokens_support__ = 0
        self.__input_tokens_recommender__ = 0
        self.__cached_input_tokens_recommender__ = 0
        self.__output_tokens_recommender__ = 0
        if fullmatch('[0-9]+', query) or len(query) < 3:
            response = self.__handle_static_response__(
                question=query,
                rephrased_question=query,
                response=ERROR_MESSAGE,
                comes_from_model=False,
                progress=progress,
                container=container,
                is_error=True
            )
            progress.empty()
            self.__save_progress__()
            return response, False

        if chat_mode == ChatModeEnum.TIPS_MODE:
            response, is_clarification_or_not_allowed = self.__give_tips__(
                query, car_details, track_details, container, progress)
        elif chat_mode == ChatModeEnum.SETUP_MODE:
            setup_str = extract_setup_info(self.setup_json, self.car_metadata)
            response, is_clarification_or_not_allowed = self.__evaluate_setup__(
                query, car_details, track_details, setup_str, container, progress)

        progress.empty()
        self.__save_progress__()

        self.__logger__.info(
            msg=response,
            function_name='ask'
        )

        return response, is_clarification_or_not_allowed
