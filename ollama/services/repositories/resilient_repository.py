"""Resilient Repository with circuit breaker pattern for database operations.

This module wraps repositories with circuit breaker logic to handle transient
database connection failures gracefully and prevent cascading failures when
the database is temporarily unavailable.

Example:
    >>> from ollama.repositories.impl.user_repository import UserRepository
    >>> user_repo = UserRepository(session)
    >>> resilient_repo = ResilientRepository(
    ...     user_repo,
    ...     failure_threshold=5,
    ...     recovery_timeout=60
    ... )
    >>> user = await resilient_repo.get_by_id(user_id)
"""

from typing import Any, TypeVar, cast

import structlog

from ollama.exceptions.circuit_breaker import CircuitBreakerError
from ollama.services.resilience.circuit_breaker import (
    get_circuit_breaker_manager,
)

log: Any = structlog.get_logger(__name__)

T = TypeVar("T")


class ResilientRepository:
    """Resilient wrapper for database repositories with circuit breaker pattern.

    Adds fault tolerance by detecting database connection failures and temporarily
    rejecting requests when the database is unavailable. This prevents cascading
    failures and allows the database time to recover.

    Attributes:
        repo: Underlying repository instance.
        breaker_manager: Circuit breaker manager for database service.
        failure_threshold: Number of failures before opening circuit.
        recovery_timeout: Seconds before attempting recovery.
    """

    def __init__(
        self,
        repo: Any,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        repo_name: str = "database",
    ) -> None:
        """Initialize resilient repository wrapper.

        Args:
            repo: Repository instance to wrap with circuit breaker.
            failure_threshold: Failures before opening circuit (default: 5).
            recovery_timeout: Seconds before recovery attempt (default: 60).
            repo_name: Name for circuit breaker tracking (default: database).
        """
        self.repo = repo
        self.breaker_manager = get_circuit_breaker_manager()
        self.repo_name = repo_name

        # Get or create circuit breaker for database service
        self.breaker = self.breaker_manager.get_or_create(
            service_name=f"repository-{repo_name}",
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        )

        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

    async def create(self, **kwargs: Any) -> Any:
        """Create and persist a new record with circuit breaker protection.

        Args:
            **kwargs: Field values for the model.

        Returns:
            Created model instance.

        Raises:
            CircuitBreakerError: If circuit is open and timeout not elapsed.
            sqlalchemy.exc.SQLAlchemyError: If database operation fails.
        """
        try:
            log.info("resilient_repo_create", repo=self.repo_name, kwargs_keys=list(kwargs.keys()))
            result = await self.breaker.call_async(self.repo.create, **kwargs)
            log.info("resilient_repo_create_success", repo=self.repo_name)
            return result
        except CircuitBreakerError:
            log.warning(
                "resilient_repo_circuit_open",
                operation="create",
                repo=self.repo_name,
                service=self.breaker.name,
            )
            raise
        except Exception as e:
            log.error(
                "resilient_repo_create_failed",
                repo=self.repo_name,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    async def get_by_id(self, id: Any) -> Any:
        """Retrieve record by ID with circuit breaker protection.

        Args:
            id: Primary key value.

        Returns:
            Model instance or None if not found.

        Raises:
            CircuitBreakerError: If circuit is open and timeout not elapsed.
            sqlalchemy.exc.SQLAlchemyError: If database operation fails.
        """
        try:
            result = await self.breaker.call_async(self.repo.get_by_id, id)
            log.info("resilient_repo_get_by_id_success", repo=self.repo_name, id=str(id))
            return result
        except CircuitBreakerError:
            log.warning(
                "resilient_repo_circuit_open",
                operation="get_by_id",
                repo=self.repo_name,
                id=str(id),
                service=self.breaker.name,
            )
            raise
        except Exception as e:
            log.error(
                "resilient_repo_get_by_id_failed",
                repo=self.repo_name,
                id=str(id),
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    async def get_one(self, **filters: Any) -> Any:
        """Retrieve single record matching filters with circuit breaker protection.

        Args:
            **filters: Column=value pairs for WHERE clause.

        Returns:
            Model instance or None if not found.

        Raises:
            CircuitBreakerError: If circuit is open and timeout not elapsed.
            sqlalchemy.exc.SQLAlchemyError: If database operation fails.
        """
        try:
            log.info(
                "resilient_repo_get_one", repo=self.repo_name, filters_keys=list(filters.keys())
            )
            result = await self.breaker.call_async(self.repo.get_one, **filters)
            log.info("resilient_repo_get_one_success", repo=self.repo_name)
            return result
        except CircuitBreakerError:
            log.warning(
                "resilient_repo_circuit_open",
                operation="get_one",
                repo=self.repo_name,
                service=self.breaker.name,
            )
            raise
        except Exception as e:
            log.error(
                "resilient_repo_get_one_failed",
                repo=self.repo_name,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    async def get_all(self, **filters: Any) -> list[Any]:
        """Retrieve all records matching filters with circuit breaker protection.

        Args:
            **filters: Column=value pairs for WHERE clause.

        Returns:
            List of model instances.

        Raises:
            CircuitBreakerError: If circuit is open and timeout not elapsed.
            sqlalchemy.exc.SQLAlchemyError: If database operation fails.
        """
        try:
            log.info(
                "resilient_repo_get_all", repo=self.repo_name, filters_keys=list(filters.keys())
            )
            result: list[Any] = cast(
                list[Any], await self.breaker.call_async(self.repo.get_all, **filters)
            )
            log.info("resilient_repo_get_all_success", repo=self.repo_name, count=len(result))
            return result
        except CircuitBreakerError:
            log.warning(
                "resilient_repo_circuit_open",
                operation="get_all",
                repo=self.repo_name,
                service=self.breaker.name,
            )
            raise
        except Exception as e:
            log.error(
                "resilient_repo_get_all_failed",
                repo=self.repo_name,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    async def get_paginated(
        self,
        page: int = 1,
        page_size: int = 10,
        order_by: str | None = None,
        **filters: Any,
    ) -> tuple[list[Any], int]:
        """Retrieve paginated records with circuit breaker protection.

        Args:
            page: Page number (1-indexed).
            page_size: Records per page.
            order_by: Column name to order by.
            **filters: Column=value pairs for WHERE clause.

        Returns:
            Tuple of (records list, total count).

        Raises:
            CircuitBreakerError: If circuit is open and timeout not elapsed.
            sqlalchemy.exc.SQLAlchemyError: If database operation fails.
        """
        try:
            log.info(
                "resilient_repo_get_paginated",
                repo=self.repo_name,
                page=page,
                page_size=page_size,
            )
            result: tuple[list[Any], int] = cast(
                tuple[list[Any], int],
                await self.breaker.call_async(
                    self.repo.get_paginated,
                    page=page,
                    page_size=page_size,
                    order_by=order_by,
                    **filters,
                ),
            )
            records, total = result
            log.info(
                "resilient_repo_get_paginated_success",
                repo=self.repo_name,
                page=page,
                returned_count=len(records),
                total_count=total,
            )
            return result
        except CircuitBreakerError:
            log.warning(
                "resilient_repo_circuit_open",
                operation="get_paginated",
                repo=self.repo_name,
                service=self.breaker.name,
            )
            raise
        except Exception as e:
            log.error(
                "resilient_repo_get_paginated_failed",
                repo=self.repo_name,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    async def update(self, id: Any, **kwargs: Any) -> Any:
        """Update a record with circuit breaker protection.

        Args:
            id: Primary key value.
            **kwargs: Field values to update.

        Returns:
            Updated model instance.

        Raises:
            CircuitBreakerError: If circuit is open and timeout not elapsed.
            sqlalchemy.exc.SQLAlchemyError: If database operation fails.
        """
        try:
            log.info("resilient_repo_update", repo=self.repo_name, id=str(id))
            result = await self.breaker.call_async(self.repo.update, id, **kwargs)
            log.info("resilient_repo_update_success", repo=self.repo_name, id=str(id))
            return result
        except CircuitBreakerError:
            log.warning(
                "resilient_repo_circuit_open",
                operation="update",
                repo=self.repo_name,
                id=str(id),
                service=self.breaker.name,
            )
            raise
        except Exception as e:
            log.error(
                "resilient_repo_update_failed",
                repo=self.repo_name,
                id=str(id),
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    async def delete(self, id: Any) -> bool:
        """Delete a record with circuit breaker protection.

        Args:
            id: Primary key value.

        Returns:
            True if record was deleted, False if not found.

        Raises:
            CircuitBreakerError: If circuit is open and timeout not elapsed.
            sqlalchemy.exc.SQLAlchemyError: If database operation fails.
        """
        try:
            log.info("resilient_repo_delete", repo=self.repo_name, id=str(id))
            result: bool = cast(bool, await self.breaker.call_async(self.repo.delete, id))
            log.info("resilient_repo_delete_success", repo=self.repo_name, id=str(id))
            return result
        except CircuitBreakerError:
            log.warning(
                "resilient_repo_circuit_open",
                operation="delete",
                repo=self.repo_name,
                id=str(id),
                service=self.breaker.name,
            )
            raise
        except Exception as e:
            log.error(
                "resilient_repo_delete_failed",
                repo=self.repo_name,
                id=str(id),
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    def get_breaker_state(self) -> dict[str, Any]:
        """Get current circuit breaker state.

        Returns:
            Dict with circuit breaker state information.
        """
        return self.breaker.get_state()

    def __getattr__(self, name: str) -> Any:
        """Proxy unknown attributes to the underlying repository.

        This allows calling repo-specific methods (e.g. get_by_email) while
        still benefiting from the circuit breaker. Unknown methods are
        automatically wrapped in the same circuit breaker.
        """
        attr = getattr(self.repo, name)

        if callable(attr):

            async def wrapped(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await self.breaker.call_async(attr, *args, **kwargs)
                except Exception as e:
                    log.error(
                        "resilient_repo_proxied_call_failed",
                        repo=self.repo_name,
                        method=name,
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                    raise

            return wrapped

        return attr


__all__ = ["ResilientRepository"]
