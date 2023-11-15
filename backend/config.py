import sys
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

ENV = os.environ.get("ENV", "dev")

OLLAMA_API_BASE_URL = os.environ.get(
    "OLLAMA_API_BASE_URL", "http://localhost:11434/api"
)

if ENV == "prod":
    if OLLAMA_API_BASE_URL == "/ollama/api":
        OLLAMA_API_BASE_URL = "http://host.docker.internal:11434/api"
