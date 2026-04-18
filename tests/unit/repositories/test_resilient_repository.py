"""Unit tests for ResilientRepository integration."""

import uuid
from unittest.mock import AsyncMock

import pytest

from ollama.exceptions.circuit_breaker import CircuitBreakerError
from ollama.repositories.resilient_repository import ResilientRepository
from ollama.services.resilience.circuit_breaker import get_circuit_breaker_manager


@pytest.fixture(autouse=True)
def reset_breakers() -> None:
    """Reset circuit breakers before each test."""
    get_circuit_breaker_manager().reset_all()


@pytest.fixture
def mock_repo() -> AsyncMock:
    """Create a mock repository."""
    return AsyncMock()


@pytest.mark.asyncio
async def test_resilient_repo_wraps_with_circuit_breaker(mock_repo: AsyncMock) -> None:
    """ResilientRepository wraps repo with circuit breaker."""
    resilient = ResilientRepository(mock_repo, repo_name="users")

    assert resilient.repo is mock_repo
    assert resilient.breaker is not None
    assert resilient.breaker.name == "repository-users"
    assert resilient.breaker_manager is not None


@pytest.mark.asyncio
async def test_resilient_repo_create_success(mock_repo: AsyncMock) -> None:
    """create returns new record when circuit is closed."""
    resilient = ResilientRepository(mock_repo, repo_name="users")

    test_id = uuid.uuid4()
    test_user = {"id": test_id, "username": "alice", "email": "alice@example.com"}

    mock_repo.create = AsyncMock(return_value=test_user)

    result = await resilient.create(username="alice", email="alice@example.com")
    assert result["username"] == "alice"
    assert result["email"] == "alice@example.com"
    mock_repo.create.assert_called_once()


@pytest.mark.asyncio
async def test_resilient_repo_get_by_id_success(mock_repo: AsyncMock) -> None:
    """get_by_id returns record when found."""
    resilient = ResilientRepository(mock_repo, repo_name="users")

    test_id = uuid.uuid4()
    test_user = {"id": test_id, "username": "alice"}

    mock_repo.get_by_id = AsyncMock(return_value=test_user)

    result = await resilient.get_by_id(test_id)
    assert result["id"] == test_id
    mock_repo.get_by_id.assert_called_once()


@pytest.mark.asyncio
async def test_resilient_repo_get_by_id_not_found(mock_repo: AsyncMock) -> None:
    """get_by_id returns None when record not found."""
    resilient = ResilientRepository(mock_repo, repo_name="users")

    test_id = uuid.uuid4()
    mock_repo.get_by_id = AsyncMock(return_value=None)

    result = await resilient.get_by_id(test_id)
    assert result is None


@pytest.mark.asyncio
async def test_resilient_repo_get_one_success(mock_repo: AsyncMock) -> None:
    """get_one returns matching record."""
    resilient = ResilientRepository(mock_repo, repo_name="users")

    test_user = {"username": "alice", "email": "alice@example.com"}
    mock_repo.get_one = AsyncMock(return_value=test_user)

    result = await resilient.get_one(username="alice")
    assert result["username"] == "alice"
    mock_repo.get_one.assert_called_once()


@pytest.mark.asyncio
async def test_resilient_repo_get_all_success(mock_repo: AsyncMock) -> None:
    """get_all returns list of records."""
    resilient = ResilientRepository(mock_repo, repo_name="users")

    users = [
        {"id": uuid.uuid4(), "username": "alice"},
        {"id": uuid.uuid4(), "username": "bob"},
    ]
    mock_repo.get_all = AsyncMock(return_value=users)

    result = await resilient.get_all()
    assert len(result) == 2
    assert result[0]["username"] == "alice"
    assert result[1]["username"] == "bob"


@pytest.mark.asyncio
async def test_resilient_repo_get_paginated_success(mock_repo: AsyncMock) -> None:
    """get_paginated returns records and total count."""
    resilient = ResilientRepository(mock_repo, repo_name="users")

    users = [{"id": uuid.uuid4(), "username": "alice"}]
    mock_repo.get_paginated = AsyncMock(return_value=(users, 100))

    records, total = await resilient.get_paginated(page=1, page_size=10)
    assert len(records) == 1
    assert total == 100
    assert records[0]["username"] == "alice"


@pytest.mark.asyncio
async def test_resilient_repo_update_success(mock_repo: AsyncMock) -> None:
    """update modifies a record."""
    resilient = ResilientRepository(mock_repo, repo_name="users")

    test_id = uuid.uuid4()
    updated_user = {"id": test_id, "username": "alice", "email": "new@example.com"}
    mock_repo.update = AsyncMock(return_value=updated_user)

    result = await resilient.update(test_id, email="new@example.com")
    assert result["email"] == "new@example.com"
    mock_repo.update.assert_called_once()


@pytest.mark.asyncio
async def test_resilient_repo_delete_success(mock_repo: AsyncMock) -> None:
    """delete removes a record."""
    resilient = ResilientRepository(mock_repo, repo_name="users")

    test_id = uuid.uuid4()
    mock_repo.delete = AsyncMock(return_value=True)

    result = await resilient.delete(test_id)
    assert result is True
    mock_repo.delete.assert_called_once()


@pytest.mark.asyncio
async def test_resilient_repo_circuit_opens_after_failures(mock_repo: AsyncMock) -> None:
    """Circuit opens after configured failure threshold."""
    resilient = ResilientRepository(
        mock_repo,
        failure_threshold=3,
        recovery_timeout=60,
        repo_name="users",
    )

    test_id = uuid.uuid4()

    # Mock get_by_id to raise exception
    mock_repo.get_by_id = AsyncMock(side_effect=Exception("Connection timeout"))

    # Fail 3 times to open circuit
    for _ in range(3):
        with pytest.raises(Exception):
            await resilient.get_by_id(test_id)

    # Circuit should now be OPEN
    assert resilient.breaker.state.value == "open"

    # Fourth attempt should raise CircuitBreakerError
    with pytest.raises(CircuitBreakerError):
        await resilient.get_by_id(test_id)


@pytest.mark.asyncio
async def test_resilient_repo_breaker_state_query(mock_repo: AsyncMock) -> None:
    """get_breaker_state returns circuit breaker state info."""
    resilient = ResilientRepository(mock_repo, repo_name="conversations")

    state = resilient.get_breaker_state()
    assert state["service"] == "repository-conversations"
    assert state["state"] == "closed"
    assert state["failure_count"] == 0
    assert state["success_count"] == 0


@pytest.mark.asyncio
async def test_resilient_repo_multiple_repos_independent_breakers(
    mock_repo: AsyncMock,
) -> None:
    """Different repositories have independent circuit breakers."""
    users_repo = ResilientRepository(mock_repo, repo_name="users")
    docs_repo = ResilientRepository(mock_repo, repo_name="documents")

    # Verify breakers are different
    assert users_repo.breaker.name != docs_repo.breaker.name
    assert users_repo.breaker.name == "repository-users"
    assert docs_repo.breaker.name == "repository-documents"

    # Verify states are independent
    users_state = users_repo.get_breaker_state()
    docs_state = docs_repo.get_breaker_state()

    assert users_state["service"] == "repository-users"
    assert docs_state["service"] == "repository-documents"


@pytest.mark.asyncio
async def test_resilient_repo_propagates_exceptions(mock_repo: AsyncMock) -> None:
    """Exceptions from underlying repo are propagated."""
    resilient = ResilientRepository(mock_repo, repo_name="users")

    class CustomDatabaseError(Exception):
        """Custom database exception."""

    mock_repo.create = AsyncMock(side_effect=CustomDatabaseError("Constraint violated"))

    with pytest.raises(CustomDatabaseError):
        await resilient.create(username="alice")


@pytest.mark.asyncio
async def test_resilient_repo_get_by_id_with_different_types(mock_repo: AsyncMock) -> None:
    """get_by_id works with different ID types."""
    resilient = ResilientRepository(mock_repo, repo_name="mixed")

    # Test with UUID
    uuid_id = uuid.uuid4()
    mock_repo.get_by_id = AsyncMock(return_value={"id": uuid_id})
    result = await resilient.get_by_id(uuid_id)
    assert result["id"] == uuid_id

    # Test with string
    mock_repo.get_by_id = AsyncMock(return_value={"id": "user123"})
    result = await resilient.get_by_id("user123")
    assert result["id"] == "user123"

    # Test with integer
    mock_repo.get_by_id = AsyncMock(return_value={"id": 42})
    result = await resilient.get_by_id(42)
    assert result["id"] == 42
