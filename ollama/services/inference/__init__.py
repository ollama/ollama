"""Inference service module.

Handles AI model inference operations including text generation, embeddings,
and completions. Manages Ollama client communication and request/response handling.

This module coordinates with the Ollama inference engine for model execution.
"""

from ollama.services.inference.generate_request import GenerateRequest
from ollama.services.inference.generate_response import GenerateResponse
from ollama.services.inference.ollama_client_main import OllamaClient

OllamaClientMain = OllamaClient

__all__: list[str] = [
    "OllamaClient",
    "OllamaClientMain",
    "GenerateRequest",
    "GenerateResponse",
]
