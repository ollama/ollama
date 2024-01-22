import os


import chromadb
from chromadb import Settings


from secrets import token_bytes
from base64 import b64encode

from constants import ERROR_MESSAGES


from pathlib import Path

try:
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv("../.env"))
except ImportError:
    print("dotenv not installed, skipping...")


####################################
# File Upload
####################################

DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()

UPLOAD_DIR = f"{DATA_DIR}/uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

FRONTEND_BUILD_DIR = Path(os.getenv("FRONTEND_BUILD_DIR", "../build"))

####################################
# ENV (dev,test,prod)
####################################

ENV = os.environ.get("ENV", "dev")

####################################
# OLLAMA_API_BASE_URL
####################################

OLLAMA_API_BASE_URL = os.environ.get(
    "OLLAMA_API_BASE_URL", "http://localhost:11434/api"
)

if ENV == "prod":
    if OLLAMA_API_BASE_URL == "/ollama/api":
        OLLAMA_API_BASE_URL = "http://host.docker.internal:11434/api"

####################################
# OPENAI_API
####################################

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_API_BASE_URL = os.environ.get("OPENAI_API_BASE_URL", "")

if OPENAI_API_BASE_URL == "":
    OPENAI_API_BASE_URL = "https://api.openai.com/v1"

####################################
# WEBUI_VERSION
####################################

WEBUI_VERSION = os.environ.get("WEBUI_VERSION", "v1.0.0-alpha.61")

####################################
# WEBUI_AUTH (Required for security)
####################################

WEBUI_AUTH = True

####################################
# WEBUI_JWT_SECRET_KEY
####################################

WEBUI_JWT_SECRET_KEY = os.environ.get("WEBUI_JWT_SECRET_KEY", "t0p-s3cr3t")

if WEBUI_AUTH and WEBUI_JWT_SECRET_KEY == "":
    raise ValueError(ERROR_MESSAGES.ENV_VAR_NOT_FOUND)

####################################
# RAG
####################################

CHROMA_DATA_PATH = f"{DATA_DIR}/vector_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHROMA_CLIENT = chromadb.PersistentClient(
    path=str(CHROMA_DATA_PATH), settings=Settings(allow_reset=True)
)
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 100
