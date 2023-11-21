from dotenv import load_dotenv, find_dotenv
from pymongo import MongoClient
from constants import ERROR_MESSAGES

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
# WEBUI_VERSION
####################################

WEBUI_VERSION = os.environ.get("WEBUI_VERSION", "v1.0.0-alpha.11")

####################################
# WEBUI_AUTH
####################################


WEBUI_AUTH = True if os.environ.get("WEBUI_AUTH", "TRUE") == "TRUE" else False


####################################
# WEBUI_DB
####################################


WEBUI_DB_URL = os.environ.get("WEBUI_DB_URL", "mongodb://root:root@localhost:27017/")

if WEBUI_AUTH and WEBUI_DB_URL == "":
    raise ValueError(ERROR_MESSAGES.ENV_VAR_NOT_FOUND)


DB_CLIENT = MongoClient(f"{WEBUI_DB_URL}?authSource=admin")
DB = DB_CLIENT["ollama-webui"]


####################################
# WEBUI_JWT_SECRET_KEY
####################################

WEBUI_JWT_SECRET_KEY = os.environ.get("WEBUI_JWT_SECRET_KEY", "t0p-s3cr3t")

if WEBUI_AUTH and WEBUI_JWT_SECRET_KEY == "":
    raise ValueError(ERROR_MESSAGES.ENV_VAR_NOT_FOUND)
