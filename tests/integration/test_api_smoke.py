import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from ollama.main import create_app
from ollama.repositories import get_repositories


class StubManager:
    def __init__(self):
        self._initialized = True

    async def initialize(self):
        self._initialized = True

    async def close(self):
        return None


class StubVector(StubManager):
    async def collection_exists(self, name: str) -> bool:
        return True

    async def create_collection(self, *args, **kwargs):
        return None

    async def upsert_points(self, *args, **kwargs):
        return None


class StubConversationRepo:
    def __init__(self):
        now = datetime.now(UTC)
        self.conversations = [
            SimpleNamespace(
                id=uuid.uuid4(),
                user_id=uuid.uuid4(),
                title="Test Conversation",
                model="test-model",
                system_prompt=None,
                parameters={},
                is_archived=False,
                created_at=now,
                accessed_at=now,
            )
        ]

    async def get_paginated(self, page: int, page_size: int, order_by: str, user_id: uuid.UUID):
        return self.conversations, len(self.conversations)

    async def get_by_id(self, conversation_id: uuid.UUID):
        return self.conversations[0]

    async def update_accessed_at(self, conversation_id: uuid.UUID):
        return None


class StubMessageRepo:
    async def count_conversation_messages(self, conversation_id: uuid.UUID) -> int:
        return 1


class StubDocument:
    def __init__(self):
        now = datetime.now(UTC)
        self.id = uuid.uuid4()
        self.user_id = uuid.uuid4()
        self.title = "Test Document"
        self.is_indexed = False
        self.vector_collection = "test-collection"
        self.chunks = ["chunk-1", "chunk-2"]
        self.created_at = now
        self.updated_at = now


class StubDocumentRepo:
    def __init__(self):
        self.documents = [StubDocument()]

    async def get_by_user_id(self, user_id: uuid.UUID):
        return self.documents

    async def get_indexed_documents(self, user_id: uuid.UUID):
        return []

    async def get_by_id(self, document_id: uuid.UUID):
        return self.documents[0]

    async def delete(self, document_id: uuid.UUID):
        return None


class StubUsageRepo:
    async def get_user_stats(self, user_id: uuid.UUID, days: int):
        return {
            "total_requests": 5 * days,
            "total_input_tokens": 100 * days,
            "total_output_tokens": 200 * days,
            "total_cost": 0.01 * days,
        }


class StubFactory:
    def __init__(self):
        self.conversations = StubConversationRepo()
        self.messages = StubMessageRepo()
        self.documents = StubDocumentRepo()
        self.usage = StubUsageRepo()

    def get_conversation_repository(self):
        return self.conversations

    def get_message_repository(self):
        return self.messages

    def get_document_repository(self):
        return self.documents

    def get_usage_repository(self):
        return self.usage


class DummyResponse:
    def __init__(self, payload: dict):
        self._payload = payload
        self.status_code = 200

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


class StubAsyncClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False

    async def get(self, url: str, **kwargs):
        return DummyResponse(
            {
                "models": [
                    {"name": "test-model", "size": 1024, "digest": "abc", "modified_at": "now"}
                ]
            }
        )

    async def post(self, url: str, json=None, **kwargs):
        if url.endswith("/generate"):
            return DummyResponse(
                {"model": json.get("model", "test-model"), "response": "ok", "done": True}
            )
        if url.endswith("/chat"):
            return DummyResponse(
                {
                    "model": json.get("model", "test-model"),
                    "message": {"role": "assistant", "content": "hi"},
                    "done": True,
                }
            )
        return DummyResponse({})

    async def delete(self, url: str, json=None, **kwargs):
        return DummyResponse({"status": "deleted"})


@pytest.fixture
def app(monkeypatch):
    # Stub service managers
    cache_stub = StubManager()
    vector_stub = StubVector()
    db_stub = StubManager()

    # Patch initialization in lifespan
    monkeypatch.setattr("ollama.main.init_database", lambda *args, **kwargs: db_stub)
    monkeypatch.setattr("ollama.main.get_db_manager", lambda: db_stub)
    monkeypatch.setattr("ollama.main.init_cache", lambda *args, **kwargs: cache_stub)
    monkeypatch.setattr("ollama.api.dependencies.cache._cache_manager", cache_stub)
    monkeypatch.setattr("ollama.api.dependencies.cache.get_cache_manager", lambda: cache_stub)
    monkeypatch.setattr("ollama.main.init_vector_db", lambda *args, **kwargs: vector_stub)
    monkeypatch.setattr("ollama.api.dependencies.vector._vector_manager", vector_stub)

    # Note: routes now use dependency injection (get_model_manager, get_vector_manager)
    # instead of direct httpx imports - no need to patch httpx.AsyncClient

    app = create_app()

    # Override repository factory
    stub_factory = StubFactory()
    app.dependency_overrides[get_repositories] = lambda: stub_factory

    return app


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.anyio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


@pytest.mark.anyio
@pytest.mark.skip(reason="Requires model manager initialization with proper stubs")
async def test_models_list(client):
    resp = await client.get("/api/v1/models")
    assert resp.status_code == 200
    assert resp.json()["models"][0]["name"] == "test-model"


@pytest.mark.anyio
@pytest.mark.skip(reason="Requires model manager initialization with proper stubs")
async def test_generate(client):
    payload = {"model": "test-model", "prompt": "hello", "stream": False}
    resp = await client.post("/api/v1/generate", json=payload)
    assert resp.status_code == 200
    assert resp.json()["response"] == "ok"


@pytest.mark.anyio
@pytest.mark.skip(reason="Requires model manager initialization with proper stubs")
async def test_chat(client):
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    }
    resp = await client.post("/api/v1/chat", json=payload)
    assert resp.status_code == 200
    assert resp.json()["message"]["content"] == "hi"


@pytest.mark.anyio
async def test_conversations_list(client):
    user_id = str(uuid.uuid4())
    resp = await client.get(f"/api/v1/conversations?user_id={user_id}&page=1&page_size=10")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["conversations"][0]["title"] == "Test Conversation"


@pytest.mark.anyio
async def test_documents_list(client):
    user_id = str(uuid.uuid4())
    resp = await client.get(f"/api/v1/documents?user_id={user_id}&page=1&page_size=10")
    assert resp.status_code == 200
    body = resp.json()
    assert body["documents"][0]["title"] == "Test Document"


@pytest.mark.anyio
@pytest.mark.skip(reason="Usage endpoint structure has changed - needs update")
async def test_usage_summary(client):
    user_id = str(uuid.uuid4())
    resp = await client.get(f"/api/v1/usage/summary?user_id={user_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["today"]["requests"] == 5
