from os import listdir
from os.path import join
from src.utils.llm import (
    get_embeddings,
    get_streaming_response,
    human_message
)
from json import dump
from pymongo.errors import InvalidName
from src.utils.mongo import (
    get_client,
    MONGODB_DOCS_DB_NAME,
    MONGODB_DOCS_COLLECTION_NAME
)
from src.utils.custom_logging import AppLogger

LOGGER = AppLogger('regenerate-docs-embeddings')

# sorry, i lost the original prompt but it was a simple one
SUMMARIZATION_PROMPT = """
  ...
"""
# relative to this script's folder
DOCS_FOLDER = join("..", "docs")
DB_FOLDER = join("..", "db")
DOCS_INIT_FILE = "docs.json"


def get_summary(content: str) -> str:
    # your code to summarize the content

    summary = ''
    for chunk in get_streaming_response(
        [human_message(content)],
        system=SUMMARIZATION_PROMPT
    ):
        summary += chunk.text
    return summary


if __name__ == '__main__':
    # get all txt files in the DOCS_FOLDER
    docs = list(filter(lambda fn: fn.endswith('txt'), listdir(DOCS_FOLDER)))
    # your code to extract embeddings from txt files

    docs_with_metadata_and_embeddings = []

    for fn in docs:
        with open(join(DOCS_FOLDER, fn), 'r') as doc_f:
            doc = doc_f.read()
            for split in doc.split('\n\n\n'):
                # extract chapters
                metadata = {'title': fn.split('.')[0].lower()}
                subsplits = split.split('\n\n')
                item = subsplits[0].lower()
                metadata['item'] = item

                # add reference car for engine maps
                if metadata['title'] == 'engine_maps':
                    metadata['item'] = 'engine_maps'
                    metadata['car'] = item

                for subsplit in subsplits[1:]:
                    # extract sub-chapters
                    content = subsplit.strip().strip('\n')

                    split_content = content
                    # avoid summarization for engine maps
                    if metadata['title'] != 'engine_maps':
                        summary = content.split('\n')[0]
                        summary = summary.split('{{')[-1].split('}}')[0]

                        # build summary for better retrieval
                        split_content = '\n'.join(content.split('\n')[1:])
                        metadata['summary'] = summary
                        split_content = f'{summary}\n\n{split_content}'

                    docs_with_metadata_and_embeddings.append({
                        'content': split_content,
                        **metadata,
                        'embeddings': get_embeddings(split_content)
                    })

    with open(join(DOCS_FOLDER, DOCS_INIT_FILE), 'w') as f:
        dump(docs_with_metadata_and_embeddings, f)

    client = get_client()

    db = client[MONGODB_DOCS_DB_NAME]
    collection_exists = True
    try:
        # rebuild if no documents is stored
        db[MONGODB_DOCS_COLLECTION_NAME]
    except InvalidName:
        # or if collection does not exist
        collection_exists = False

    if collection_exists:
        db.drop_collection(MONGODB_DOCS_COLLECTION_NAME)
