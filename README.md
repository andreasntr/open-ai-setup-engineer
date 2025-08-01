# Disclaimer 

This tool is free for everyone but the costs of API calls to the models are on the user. With gemini flash and flash lite, I estimated a cost of 0.001€ per question, around 1€/1000 questions (which is a lot).

---

# AI Setup Engineer

AI Setup Engineer is a tool which aims to enhance the Assetto Corsa Competizione experience for players who are not akin to setups or players who just want a prototype setup to start with, and spend less time tweaking settings. 

### How it works
AI Setup Engineer uses artifical intelligence to guide you setup your car by exploiting domain knowledge, provided in the documents under the `docs` folder. You can edit them to add, remove, edit information as you see fit, but remember they will be the source of the answers you seek.

AI Setup Engineer will _NOT_ give you exact numbers, for example it will not tell you _set your camber to -2.5_. However, it will give you directions to follow, for example _increase your rear wing_. Given this, the level of detail of your inputs is crucial to obtain tailored suggestions: for example, avoid generic questions such as _reduce oversteer_ and prefer something like _reduce oversteer in turn 2 at the exit of the corner_.

Key Features:

*   Identifies corners from your questions;
*   Can use JSON setups as a starting point for outputting tailored suggestions;
*   Uses an LLM-workflow (not a single prompt) to process your questions and get the most context out of the selected track and car, potentially avoiding too generic suggestions;
*   Works with Google and OpenAI models;
*   Allows restoring past chats by logging in with Discord (this is a bit overkill for personal use but can be useful if AI Setup Engineer is hosted in a server for using it with friends/fellow racers);
*   All your data stays local (API calls are usually not used by providers to improve their models, and DB is local).

## Project Structure

The project is organized as follows:

* `data/`: Contains the metadata files used by the application, such as cars and tracks data, and utilities;
* `db/`: Contains embeddings from the original files in `docs` and an init script to initialize the database;
* `customization/`: Contains user-configurable scripts to customize the app.
* `docs/`: Contains the original documents from which knowledge is extracted;
* `logs/`: Once the application is started, logs will be stored and persisted here;
* `envs/`: Contains the `.env` file which stores credentials and settings;
* `src/`: Contains the source code of the application.

## Customization
**Please start your scripts from the main folder so that logs and imports are managed correctly.**

### Customization Folder

The `customization/` folder allows users to modify the behavior of the project without directly altering the core code. Place custom scripts or configuration files here.

Example: If you want to regenerate embeddings for your documents (suppose you made changes to them), use the `customize_embeddings.py` script and your db will be updated on the next restart.

### Environment Variables

The following environment variables can be used to configure the project:

* `OPENAI_API_KEY`: **\[OPTIONAL\]** Your OpenAI API key (if you want to use OpenAI's models);
* `OPENAI_ORG_ID`: **\[OPTIONAL\]** Your OpenAI organization ID (if you want to use OpenAI's models);
* `OPENAI_PROJECT_ID`: **\[OPTIONAL\]** Your OpenAI project ID (if you want to use OpenAI's models);
* `ENVIRONMENT`: Any value you want, e.g. development, staging, production. Remove it to test locally via `python-dotenv`;
* `MONGODB_URI`: Self-explanatory;
* `MONGODB_INDEX_NAME`: The index name fot the embeddings;
* `MONGODB_DOCS_DB_NAME`: Self-explanatory;
* `MONGODB_DOCS_COLLECTION_NAME`: Self-explanatory;
* `MONGODB_SESSIONS_DB_NAME`: Self-explanatory;
* `MONGODB_SESSIONS_COLLECTION_NAME`: Self-explanatory;
* `BASE_URI`: URL of the app, e.g. http://localhost:8080;
* `DISCORD_CLIENT_ID`: **\[OPTIONAL\]** Set-up at https://discord.com/developers/applications;
* `DISCORD_CLIENT_SECRET`: **\[OPTIONAL\]** Set-up at https://discord.com/developers/applications;
* `USE_VERTEXAI`: **\[OPTIONAL\]** Any value to use Gemini models via VertexAI;
* `GOOGLE_PROJECT_ID`: **\[OPTIONAL\]** Your Google Cloud Project ID, if using VertexAI;
* `GOOGLE_APPLICATION_CREDENTIALS`: **\[OPTIONAL\]** Path to your Google Cloud service account credentials file, if using VertexAI. Can be mounted as a volume in Docker by setting it to `/credentials/svc.json` and naming your JSON accordingly;
* `GOOGLE_PROJECT_REGION`: **\[OPTIONAL\]** Your Google Cloud Project region, if using VertexAI;
* `SUPPORT_LLM`: The LLM to use for support tasks;
* `RECOMMENDER_LLM`: The LLM to use to generate recommendations;
* `EMBEDDINGS_MODEL`: The model to use for generating embeddings;
* `EMBEDDINGS_DIMENSIONS`: The number of dimensions of the embeddings model;
* `CACHE_HIT_DESIRED_RATIO`: The desired ratio of cache hits when the same question has been asked in the past. Lower values can increase diversity, higher values can lead to lower costs;
* `LOG_LEVEL`: The desired log level for the whole application.

## Getting Started

### Prerequisites

* A terminal (if you are using Windows, Docker should work but for maximum compatibility use WSL);
* Git;
* Docker;
* **\[OPTIONAL\]** A way to query the underlying MongoDB database, if needed (e.g. VS Code MongoDB extension)

### Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/andreasntr/open-ai-setup-engineer
    cd open-ai-setup-engineer # cd is dir in Windows
    ```

2.  Fill the `envs/.env` file with the envs you need and add Google credentials, if using VertexAI;

3.  Start the app:

    * Automatically (using the provided `start.sh` script):
      ```bash
      sh start.sh
      ```

    * Manually:
      - docker compose build
      - docker compose up -d

### Accessing the Application

Open your web browser and go to `http://localhost:8080`.

## License

This project is licensed under the MIT License.

## A note from the developer

This project has been developed with passion during my spare time over the course of something less than a year. I reckon something is not perfect but I also reckon it would be impossible for me to keep up with the development alone.

I truly believe in open-source software and thus I believe this project can become extremely better with the contribution of any of you fellow developers-racers.

If you want to contribute, feel free to open an issue or a PR (better).

Thank you and enjoy your AI Setup Engineer.