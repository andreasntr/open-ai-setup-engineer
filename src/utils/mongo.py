from os import getenv
from operator import itemgetter
from typing import Any
from pymongo import MongoClient
from pymongo.typings import _Pipeline
from pymongo.server_api import ServerApi
from datetime import datetime, UTC
from cachetools.func import lru_cache
from datetime import datetime
from src.utils.custom_logging import AppLogger

if not getenv('ENVIRONMENT'):
    from dotenv import load_dotenv
    load_dotenv()

LOGGER = AppLogger('db')
EMBEDDINGS_DIMENSIONS = int(getenv('EMBEDDINGS_DIMENSIONS', 1536))
# always relative to the entrypoint of the app
# with the native implementation it is:
# - /app/ in Docker
# - <your-repo-path>/ when run locally
DB_FOLDER = "db"
DOCS_INIT_FILE = "docs.json"
MONGODB_URI = getenv("MONGODB_URI")
MONGODB_INDEX_NAME = getenv("MONGODB_INDEX_NAME")
MONGODB_DOCS_DB_NAME = getenv("MONGODB_DOCS_DB_NAME")
MONGODB_DOCS_COLLECTION_NAME = getenv("MONGODB_DOCS_COLLECTION_NAME")
MONGODB_SESSIONS_DB_NAME = getenv("MONGODB_SESSIONS_DB_NAME")
MONGODB_SESSIONS_COLLECTION_NAME = getenv("MONGODB_SESSIONS_COLLECTION_NAME")
MIN_CACHE_SIZE = 3

# https://www.mongodb.com/docs/atlas/atlas-vector-search/tutorials/local-rag/


@lru_cache(maxsize=1024)
def get_client():
    client = MongoClient(MONGODB_URI, server_api=ServerApi('1'))
    return client


@lru_cache(maxsize=1024)
def get_docs_collection():
    client = get_client()
    docs_collection = client[MONGODB_DOCS_DB_NAME][MONGODB_DOCS_COLLECTION_NAME]

    return docs_collection


@lru_cache(maxsize=1024)
def get_sessions_collection():
    client = get_client()
    sessions_collection = client[MONGODB_SESSIONS_DB_NAME][MONGODB_SESSIONS_COLLECTION_NAME]
    return sessions_collection


def get_docs(aggregation_pipeline: _Pipeline) -> list[dict[str, Any]]:
    """
    Performs an aggregation pipeline over documents for the retrieval process.

    Args:
        aggregation_pipeline (_Pipeline): MongoDB aggregation pipeline.

    Returns:
        list[dict[str, Any]]: the resulting documents as per the aggregation pipeline.
    """
    docs_collection = get_docs_collection()
    docs = docs_collection.aggregate(aggregation_pipeline)
    return list(docs)


def get_cached_queries(
        query: str,
        similar_cars: list[str],
        track_downforce: str
) -> list[str]:
    """
    Retrieves cached queries from the database based on the provided criteria.

    Args:
        query (str): The query string to match.
        similar_cars (list[str]): A list of similar car names to match.
        track_downforce (str): The track downforce to match.

    Returns:
        list[str]: The list of responses whose questions matched the provided criteria.
    """
    sessions_collection = get_sessions_collection()
    cached_queries = sessions_collection.aggregate([
        {
            "$match": {
                "session_start_at": {
                    "$gte": datetime.strptime(
                        "2024-10-20T17:40:00.000+00:00",
                        "%Y-%m-%dT%H:%M:%S.%f%z"
                    )
                },
                "interactions.has_corners": False,
                "interactions.feedback.feedback": {"$ne": "negative"},
                "interactions.metadata.model": {"$ne": None},
                "chat_mode": "tips",
                "car.name": {"$in": similar_cars},
                "track.downforce": track_downforce
            }
        },
        {
            "$project": {
                "q": "$interactions.q",
                "a": "$interactions.a",
            },
        },
        {
            "$project": {
                "interactions": 0,
                "_id": 0
            },
        },
        {
            "$addFields": {
                "qa": {
                    "$zip": {
                        "inputs": ["$q", "$a"],
                    },
                },
            },
        },
        {
            "$unwind": {
                "path": "$qa",
            },
        },
        {
            "$project": {
                "q": {
                    "$toLower": {
                        "$rtrim": {
                            "input": {
                                "$trim": {
                                    "input": {
                                        "$first": "$qa"
                                    }
                                }
                            },
                            "chars": "?",
                        }
                    }
                },
                "a": {
                    "$last": "$qa"
                }
            },
        },
        {
            "$match": {
                "q": query.lower().strip().rstrip('?'),
                "a": {"$ne": ""}
            }
        },
        {
            "$group": {
                "_id": {
                    "q": "$q"
                },
                "cnt": {
                    "$count": {},
                },
                "a": {
                    "$push": "$a",
                }
            },
        },
        {
            "$unwind": {"path": "$a"}
        },
        {
            "$match":
            {
                "cnt": {"$gte": MIN_CACHE_SIZE}
            },
        },
        {
            "$project": {
                "a": 1,
                "_id": 0
            }
        }
    ])
    return list(map(itemgetter('a'), cached_queries))


def get_sessions(
        filter: dict[str, Any],
        projection: dict[str, Any],
        sort: dict[str, Any] = {},
        skip: int = 0,
        limit: int | None = None
) -> list[dict[str, Any]]:
    """
    Retrieves sessions given some input conditions.

    Args:
        filter (dict[str, Any]): The filter to apply to the query.
        projection (dict[str, Any]): The projection to apply to the query.
        sort (dict[str, Any], optional): The sort order to apply to the query. Defaults to {}.
        skip (int, optional): The number of documents to skip. Defaults to 0.
        limit (int | None, optional): The maximum number of documents to return. 
            Defaults to None which means all sessions are returned.

    Returns:
        list[dict[str, Any]]: The list of sessions matching the input criteria.
    """
    sessions_collection = get_sessions_collection()
    sessions = sessions_collection.find(
        filter,
        projection=projection,
        sort=sort,
        skip=skip,
        limit=limit
    )
    return list(sessions)


def get_session(
    filter: dict[str, Any],
    projection: dict[str, Any]
) -> dict[str, Any]:
    """
    Retrieves a single session given some input conditions.

    Args:
        filter (dict[str, Any]): The filter to apply to the query.
        projection (dict[str, Any]): The projection to apply to the query.

    Returns:
        dict[str, Any]: The session matching the input criteria.
    """
    sessions_collection = get_sessions_collection()
    session = sessions_collection.find_one(
        filter,
        projection
    )
    return session


def update_session(
        filter: dict[str, Any],
        replacement: dict[str, Any],
        upsert: bool = False
) -> None:
    """
    Updates a given session with the provided replacements.

    Args:
        filter (dict[str, Any]): The filter to apply to the query to retrieve the correct session.
        replacement (dict[str, Any]): The replacement to apply to the session.
        upsert (bool, optional): Whether to create the session if it doesn't exist. Defaults to Fals

    Returns:
        None
    """
    sessions_collection = get_sessions_collection()
    replacement['$set'] = replacement.get('$set', {}) | {
        'updated_at': datetime.now(UTC)
    }
    sessions_collection.update_one(
        filter,
        replacement,
        upsert=upsert)


def create_session(
        new_session: dict[str, Any]
) -> None:
    """
    Creates a new session.

    Args:
        new_session (dict[str, Any]): The new session to create.

    Returns:
        None
    """
    sessions_collection = get_sessions_collection()
    sessions_collection.insert_one({
        **new_session,
        'updated_at': datetime.now(UTC)
    })
