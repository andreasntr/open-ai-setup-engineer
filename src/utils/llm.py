from collections.abc import Generator
from typing import Literal, Optional, Union
from pydantic import BaseModel
from openai import OpenAI
from google.genai import Client as GeminiClient
from google.genai.types import (
    Part,
    Content,
    GenerateContentConfig,
    SafetySetting
)
from tiktoken import encoding_for_model
from os import getenv
from cachetools.func import lru_cache
from json import loads, dumps
import src.utils.responses as responses
from re import sub, IGNORECASE

if not getenv('ENVIRONMENT'):
    from dotenv import load_dotenv
    load_dotenv()


SUPPORT_LLM = getenv('SUPPORT_LLM')
RECOMMENDER_LLM = getenv('RECOMMENDER_LLM')
EMBEDDINGS_MODEL = getenv('EMBEDDINGS_MODEL')

OPENAI_API_KEY = getenv("OPENAI_API_KEY")
OPENAI_ORG_ID = getenv("OPENAI_ORG_ID")
OPENAI_PROJECT_ID = getenv("OPENAI_PROJECT_ID")

_ModelTypes = Union[OpenAI, GeminiClient]
_ModelNames = Literal['support', 'recommender']


class Usage(BaseModel):
    """
    A Pydantic model representing the usage information for a language model response.

    Attributes:
        model (_ModelNames): The name of the language model used for the response.
        input_tokens (int): The number of input tokens used.
        cached_input_tokens (int): The number of cached input tokens used.
        output_tokens (int): The number of output tokens generated.
    """
    model: _ModelNames
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int


class StreamingResponse(BaseModel):
    """A class representing a streaming response from an LLM."""
    text: str
    usage: Optional[Usage]


# this is needed otherwise Google models will not get fields descriptions
@lru_cache(maxsize=1024)
def __unnest_pydantic_model__(model: BaseModel) -> dict:
    """
    Unnests a Pydantic model into a dictionary format.

    This function takes a Pydantic model and returns its unnested representation as a dictionary. 

    Args:
        model (BaseModel): The Pydantic model to be unnested.

    Returns:
        dict: A dictionary representing the unnested model.
    """
    unnested_model = model.model_json_schema()
    definitions = unnested_model.get('$defs')
    if definitions:
        unnested_model.pop('$defs')
        unnested_model = dumps(unnested_model)
        for definition_id, definition_schema in definitions.items():
            model_cls: BaseModel = getattr(
                responses,
                definition_id
            )
            definition_schema = __unnest_pydantic_model__(
                model_cls
            )
            unnested_definition_schema_fields = [
                f'"{k}": {dumps(v)}'
                for k, v in definition_schema.items()
            ]
            unnested_model = unnested_model.replace(
                f'"$ref": "#/$defs/{definition_id}"',
                ', '.join(unnested_definition_schema_fields)
            )
        unnested_model = loads(unnested_model)
    unnested_model['propertyOrdering'] = list(
        unnested_model.get('properties').keys()
    )
    unnested_model = loads(
        sub(
            r"['\"]title['\"]: ['\"][A-Z\s]+['\"], ",
            '',
            dumps(unnested_model),
            flags=IGNORECASE
        )
    )
    return unnested_model


@lru_cache(maxsize=1024)
def __init_model_client__(model_type: _ModelTypes) -> OpenAI | GeminiClient:
    """
    Initialize the appropriate language model client based on the provided model type.

    Args:
        model_type (_ModelTypes): The type of the language model to initialize.

    Returns:
        OpenAI | GeminiClient: The initialized language model client.
    """
    if model_type == OpenAI:
        return OpenAI(
            api_key=OPENAI_API_KEY,
            project=OPENAI_PROJECT_ID,
            organization=OPENAI_ORG_ID,
            timeout=10
        )
    if model_type == GeminiClient:
        if getenv('USE_VERTEXAI'):
            return GeminiClient(
                vertexai=True,
                project=getenv('GOOGLE_PROJECT_ID'),
                location=getenv('GOOGLE_PROJECT_REGION')
            )
        return GeminiClient(
            api_key=getenv('GEMINI_API_KEY')
        )


def system_message(message: str) -> dict:
    return {'role': 'system', 'content': message}


def human_message(message: str) -> dict:
    return {'role': 'user', 'content': message}


def assistant_message(message: str) -> dict:
    return {'role': 'assistant', 'content': message}


def __prepare_input_messages__(
    model_type: _ModelTypes,
    input_messages: list[dict[str, str]],
    system: Optional[str] = None,
) -> tuple[Optional[str], list[dict[str, str]]]:
    """
    Prepare the input messages for the language model.

    Args:
        model_type (_ModelTypes): The type of the language model.
        input_messages (list[dict[str, str]]): The list of input messages to be included in the input.
        system (Optional[str]): The system message to be included in the input.

    Returns:
        tuple[Optional[str], list[dict[str, str]]]: A tuple containing the system message (if any) and the list of prepared input messages.
    """
    if model_type == OpenAI:
        messages = []
        if system:
            messages = [system_message(system)]
        messages.extend(input_messages)
    elif model_type == GeminiClient:
        messages = [
            Content(
                role=message['role'],
                parts=[Part.from_text(text=message['content'])]
            )
            for message in input_messages
        ]
    return messages


def __parse_examples__(
    model_type: _ModelTypes,
    examples: list[tuple[str, str]]
) -> list[dict[str, str]] | str:
    """
    Parse the example messages for the language model.

    Args:
        model_type (_ModelTypes): The type of the language model.
        examples (list[tuple[str, str]]): The list of example messages, where each tuple contains the user message and the assistant message.

    Returns:
        list[dict[str, str]] | str: The parsed examples, depending on the model type.
            - For OpenAI, a list of dictionaries representing the user and assistant messages.
            - For GeminiClient, a string containing the examples in the required format.
    """
    if model_type == OpenAI:
        examples_parsed = []
        for user, assistant in examples:
            examples_parsed.extend([
                human_message(user),
                assistant_message(assistant),
            ])
    elif model_type == GeminiClient:
        examples_parsed = ''
        for user, assistant in examples:
            examples_parsed += f'input: {user}\noutput: {assistant}\n\n'

    return examples_parsed


@lru_cache(maxsize=1024)
def __get_model_type__(model: str) -> GeminiClient | OpenAI:
    """
    Determine the type of language model based on the model name.

    Args:
        model (str): The name of the language model.

    Returns:
        GeminiClient | OpenAI: The type of the language model.
    """
    if 'text-embedding' not in model:
        if model.startswith('gpt') or model.startswith('o'):
            return OpenAI
        if model.startswith('gemini'):
            return GeminiClient
    elif '00' in model:
        return GeminiClient
    return OpenAI


def get_tokens(model: _ModelNames, query: str) -> int:
    """
    Get the number of tokens in the given query for the specified language model.

    Args:
        model (_ModelNames): The type of language model to use, either 'support' or 'recommender'.
        query (str): The input text for which to count the tokens.

    Returns:
        int: The number of tokens in the input query.
    """
    model_name = SUPPORT_LLM if model == 'support' else RECOMMENDER_LLM
    model_type = __get_model_type__(model)
    __init_model_client__(model_type)

    if model_type == OpenAI:
        return len(encoding_for_model(model_name).encode(query))

    if model_type == GeminiClient:
        return GeminiClient(model_name).count_tokens(query).total_tokens


def get_streaming_response(
    messages: list[dict[str, str]],
    system: Optional[str] = None,
    examples: Optional[list[tuple[str, str]]] = None,
    model: _ModelNames = 'support',
    temperature: float = 0,
    max_tokens: int = 1000,
    output_type: Literal['text', 'json'] = 'text',
    response_class: Optional[BaseModel] = None
) -> Generator[StreamingResponse]:
    """
    Retrieves a streaming response from an LLM.

    This function takes a list of messages, system instructions (optional), examples (optional), 
    a model name, temperature, max tokens, output type, and a response class (optional). It then
    generates a stream of responses until the end of the conversation is reached.

    Args:
        messages (list[dict[str, str]]): A list of messages to be sent to the LLM. 
        system (Optional[str]): System instructions for the LLM. 
        examples (Optional[list[tuple[str, str]]]): Examples to provide context.
        model (_ModelNames): The name of the model to use. Defaults to 'support'.
        temperature (float): Controls the randomness of the generated text. Defaults to 0.
        max_tokens (int): Maximum number of tokens in the response. Defaults to 1000.
        output_type (Literal['text', 'json']): The output type for the response. Defaults to 'text'.
        response_class (Optional[BaseModel]): A class representing a specific LLM model. 
                                                    Defaults to None, meaning it will use the default model.
    Returns:
        Generator[StreamingResponse]: A generator that yields StreamingResponse objects for each response chunk.
    """
    model_name = SUPPORT_LLM if model == 'support' else RECOMMENDER_LLM
    model_type = __get_model_type__(model_name)

    if response_class is not None:
        output_type = 'json'

    parsed_messages = __prepare_input_messages__(
        model_type,
        system=system,
        input_messages=messages
    )
    if examples:
        parsed_examples = __parse_examples__(
            model_type,
            examples
        )

    client = __init_model_client__(model_type)

    if model_type == OpenAI:
        if examples:
            parsed_messages = parsed_examples + parsed_messages
        args = {
            'model': model_name,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'stream': True,
            'stream_options': {"include_usage": True},
            'messages': parsed_messages,
            'seed': 0
        }
        if output_type == 'json':
            args['response_format'] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_class.__name__,
                    "schema": response_class.model_json_schema(),
                    "strict": True
                }
            } if response_class else 'json'

        stream = client.chat.completions.create(**args)
        for event in stream:
            text = ''
            usage = None
            if event.choices and event.choices[0].delta.content:
                text = event.choices[0].delta.content
            if event.usage:
                cached_tokens = event.usage.prompt_tokens_details.cached_tokens
                usage = Usage(
                    model=model,
                    input_tokens=event.usage.prompt_tokens - cached_tokens,
                    cached_input_tokens=cached_tokens,
                    output_tokens=event.usage.completion_tokens
                )

            yield StreamingResponse(text=text, usage=usage)

    if model_type == GeminiClient:

        if examples:
            parsed_messages[0].parts[0] = Part.from_text(
                text=parsed_examples + parsed_messages[0].parts[0].text
            )

        safety_settings = [
            SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="OFF"
            ),
            SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="OFF"
            ),
            SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="OFF"
            ),
            SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="OFF"
            ),
        ]

        generation_config = GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            seed=0,
            safety_settings=safety_settings,
            response_mime_type="application/json" if output_type == 'json' else None,
            response_schema=__unnest_pydantic_model__(
                response_class
            ) if output_type == 'json' else None,
            system_instruction=[
                Part.from_text(text=system)
            ] if system else None
        )

        stream = client.models.generate_content_stream(
            model=model_name,
            contents=parsed_messages,
            config=generation_config
        )

        for event in stream:
            text = event.text
            usage = None

            if event.usage_metadata.total_token_count:
                cached_tokens = event.usage_metadata.cached_content_token_count or 0

                usage = Usage(
                    model=model,
                    input_tokens=event.usage_metadata.prompt_token_count - cached_tokens,
                    cached_input_tokens=cached_tokens,
                    output_tokens=event.usage_metadata.candidates_token_count
                )

            yield StreamingResponse(text=text, usage=usage)


def get_embeddings(
    query: str
) -> list[float]:
    """
    Generate embeddings for the given query using the specified language model.

    Args:
        query (str): The input text for which to generate the embeddings.
        task_type (Optional[_TextEmbeddingTypes]): The type of text embedding task, such as "QUESTION_ANSWERING" or "RETRIEVAL_DOCUMENT".

    Returns:
        list[float]: The generated embeddings for the input query.
    """
    model_type = __get_model_type__(EMBEDDINGS_MODEL)
    client = __init_model_client__(model_type)

    if model_type == OpenAI:
        return client.embeddings.create(
            model=EMBEDDINGS_MODEL,
            input=query,
            encoding_format="float"
        ).data[0].embedding
    if model_type == GeminiClient:
        return client.models.embed_content(
            model=EMBEDDINGS_MODEL,
            contents=[query]
        ).embeddings[0].values
