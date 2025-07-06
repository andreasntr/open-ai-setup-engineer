from os.path import join
from pymongo.operations import SearchIndexModel
from pymongo.errors import InvalidName
from json import load
from src.utils.custom_logging import AppLogger
from src.utils.mongo import (
    MONGODB_DOCS_DB_NAME,
    MONGODB_DOCS_COLLECTION_NAME,
    MONGODB_INDEX_NAME,
    DB_FOLDER,
    DOCS_INIT_FILE,
    EMBEDDINGS_DIMENSIONS,
    get_client
)
from json import load
from time import sleep

LOGGER = AppLogger('db_init')

if __name__ == '__main__':
    client = get_client()

    db = client[MONGODB_DOCS_DB_NAME]
    collection_to_be_built = False
    try:
        # rebuild if no documents is stored
        collection_to_be_built = db[MONGODB_DOCS_COLLECTION_NAME]\
            .count_documents({}) == 0
    except InvalidName:
        # or if collection does not exist
        collection_to_be_built = True

    if collection_to_be_built:
        LOGGER.info('Uninitialized qa.docs collection detected...')
        LOGGER.info('Creating qa DB...')
        # if collection exists from previous runs, remove it
        db.drop_collection(MONGODB_DOCS_COLLECTION_NAME)
        LOGGER.info('Creating qa.docs collection...')
        docs_collection = db.create_collection(
            MONGODB_DOCS_COLLECTION_NAME
        )
        LOGGER.info("Inserting docs into collection...")
        with open(join(DB_FOLDER, DOCS_INIT_FILE), 'r') as f:
            docs_collection.insert_many(load(f))
        LOGGER.info(
            'Creating vector search index "docs_embeddings" (may take a while)...'
        )
        docs_collection.create_search_index(
            SearchIndexModel(
                definition={
                    "fields": [
                        {
                            "numDimensions": EMBEDDINGS_DIMENSIONS,
                            "path": "embeddings",
                            "similarity": "cosine",
                            "type": "vector"
                        },
                        {
                            "path": "title",
                            "type": "filter"
                        },
                        {
                            "path": "item",
                            "type": "filter"
                        },
                        {
                            "path": "car",
                            "type": "filter"
                        }
                    ]
                },
                type='vectorSearch',
                name=MONGODB_INDEX_NAME
            )
        )

        while not list(docs_collection.list_search_indexes(MONGODB_INDEX_NAME))[0]\
                .get('queryable'):
            sleep(5)

        LOGGER.info("Done!")
