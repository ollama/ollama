import os
from chromadb.config import Settings

# Define the folder for storing database
PERSIST_DIRECTORY = os.environ.get('PERSIST_DIRECTORY', 'db')

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
        persist_directory=PERSIST_DIRECTORY,
        anonymized_telemetry=False
)
