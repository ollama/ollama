from dotenv import load_dotenv, find_dotenv
from pymongo import MongoClient

from secrets import token_bytes
from base64 import b64encode
import os

load_dotenv(find_dotenv())

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
# OLLAMA_WEBUI_VERSION
####################################

OLLAMA_WEBUI_VERSION = os.environ.get("OLLAMA_WEBUI_VERSION", "v1.0.0-alpha.9")

####################################
# OLLAMA_WEBUI_AUTH
####################################

OLLAMA_WEBUI_AUTH = (
    True if os.environ.get("OLLAMA_WEBUI_AUTH", "TRUE") == "TRUE" else False
)


if OLLAMA_WEBUI_AUTH:
    ####################################
    # OLLAMA_WEBUI_DB
    ####################################

    OLLAMA_WEBUI_DB_URL = os.environ.get(
        "OLLAMA_WEBUI_DB_URL", "mongodb://root:root@localhost:27017/"
    )

    DB_CLIENT = MongoClient(f"{OLLAMA_WEBUI_DB_URL}?authSource=admin")
    DB = DB_CLIENT["ollama-webui"]

    ####################################
    # OLLAMA_WEBUI_JWT_SECRET_KEY
    ####################################

    OLLAMA_WEBUI_JWT_SECRET_KEY = os.environ.get(
        "OLLAMA_WEBUI_JWT_SECRET_KEY", "t0p-s3cr3t"
    )

    if ENV == "prod":
        if OLLAMA_WEBUI_JWT_SECRET_KEY == "":
            OLLAMA_WEBUI_JWT_SECRET_KEY = str(b64encode(token_bytes(32)).decode())
