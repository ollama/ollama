"""
Repository Factory - Creates repository instances with dependency injection.
Provides a clean interface for accessing repositories in FastAPI endpoints.
"""

from collections.abc import AsyncGenerator
from typing import Any, cast

from sqlalchemy.ext.asyncio import AsyncSession

from ollama.repositories.resilient_repository import ResilientRepository

from .api_key_repository import APIKeyRepository
from .conversation_repository import ConversationRepository
from .document_repository import DocumentRepository
from .message_repository import MessageRepository
from .usage_repository import UsageRepository
from .user_repository import UserRepository


class RepositoryFactory:
    """Factory for creating repository instances.

    Optionally wraps repositories in ResilientRepository for circuit breaker protection.
    """

    def __init__(self, session: AsyncSession, use_resilience: bool = True) -> None:
        """Initialize factory with database session.

        Args:
            session: Async SQLAlchemy session
            use_resilience: Whether to wrap repositories with circuit breakers (default: True)
        """
        self.session = session
        self.use_resilience = use_resilience
        self._repositories: dict[str, Any] = {}

    def _wrap_if_resilient(self, repo: Any, repo_name: str) -> Any:
        """Wrap repository in ResilientRepository if resilience is enabled."""
        if self.use_resilience:
            return ResilientRepository(repo, repo_name=repo_name)
        return repo

    def get_user_repository(self) -> UserRepository:
        """Get or create UserRepository instance."""
        if "user" not in self._repositories:
            repo = UserRepository(self.session)
            self._repositories["user"] = self._wrap_if_resilient(repo, "user")
        return cast(UserRepository, self._repositories["user"])

    def get_api_key_repository(self) -> APIKeyRepository:
        """Get or create APIKeyRepository instance."""
        if "api_key" not in self._repositories:
            repo = APIKeyRepository(self.session)
            self._repositories["api_key"] = self._wrap_if_resilient(repo, "api_key")
        return cast(APIKeyRepository, self._repositories["api_key"])

    def get_conversation_repository(self) -> ConversationRepository:
        """Get or create ConversationRepository instance."""
        if "conversation" not in self._repositories:
            repo = ConversationRepository(self.session)
            self._repositories["conversation"] = self._wrap_if_resilient(repo, "conversation")
        return cast(ConversationRepository, self._repositories["conversation"])

    def get_message_repository(self) -> MessageRepository:
        """Get or create MessageRepository instance."""
        if "message" not in self._repositories:
            repo = MessageRepository(self.session)
            self._repositories["message"] = self._wrap_if_resilient(repo, "message")
        return cast(MessageRepository, self._repositories["message"])

    def get_document_repository(self) -> DocumentRepository:
        """Get or create DocumentRepository instance."""
        if "document" not in self._repositories:
            repo = DocumentRepository(self.session)
            self._repositories["document"] = self._wrap_if_resilient(repo, "document")
        return cast(DocumentRepository, self._repositories["document"])

    def get_usage_repository(self) -> UsageRepository:
        """Get or create UsageRepository instance."""
        if "usage" not in self._repositories:
            repo = UsageRepository(self.session)
            self._repositories["usage"] = self._wrap_if_resilient(repo, "usage")
        return cast(UsageRepository, self._repositories["usage"])

    async def close(self) -> None:
        """Close all repository sessions."""
        # All repositories share the same session, just close once
        await self.session.close()


async def get_repositories() -> AsyncGenerator[RepositoryFactory, None]:
    """FastAPI dependency for repository factory.

    Creates a RepositoryFactory instance with a new session from the database manager.

    Usage in endpoints:
        @app.get("/api/v1/conversations")
        async def list_conversations(
            user_id: uuid.UUID,
            repos: RepositoryFactory = Depends(get_repositories)
        ):
            conv_repo = repos.get_conversation_repository()
            conversations = await conv_repo.get_by_user_id(user_id)
            return conversations

    Yields:
        RepositoryFactory instance
    """
    # Get session from database manager
    from ollama.services import get_db_manager

    manager = get_db_manager()
    async for session in manager.get_session():
        factory = RepositoryFactory(session)
        yield factory
