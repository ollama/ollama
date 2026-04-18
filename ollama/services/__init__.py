"""Services module - Business logic and infrastructure layer.

Organized into functional containers:
  * inference/: AI model inference operations
  * cache/: Redis caching and performance optimization
  * models/: Model lifecycle and management
  * persistence/: Data access and ORM layer

Each container follows single responsibility principle with clear domain boundaries.
Maximum 5 levels deep enforced to maintain architectural clarity.
"""

from ollama.services.cache import (
    CacheManager,
    ResilientCacheManager,
    get_cache_manager,
    init_cache,
)
from ollama.services.inference import (
    GenerateRequest,
    GenerateResponse,
    OllamaClient,
    OllamaClientMain,
)
from ollama.services.models import Model, ModelType, OllamaModelManager, VectorStore
from ollama.services.models.vector import VectorManager, get_vector_db, init_vector_db
from ollama.services.persistence import (
    ChatMessage,
    ChatRequest,
    Database,
    DatabaseManager,
    get_db,
    get_db_manager,
    init_database,
)
from ollama.services.resources.manager import ResourceManager
from ollama.services.resources.types import WorkloadType

__all__: list[str] = [
    "CacheManager",
    "ChatMessage",
    "ChatRequest",
    "Database",
    "DatabaseManager",
    "GenerateRequest",
    "GenerateResponse",
    "Model",
    "ModelType",
    "OllamaClient",
    "OllamaClientMain",
    "OllamaModelManager",
    "ResilientCacheManager",
    "ResourceManager",
    "VectorManager",
    "VectorStore",
    "WorkloadType",
    "get_cache_manager",
    "get_db",
    "get_db_manager",
    "get_vector_db",
    "init_cache",
    "init_database",
    "init_vector_db",
]
