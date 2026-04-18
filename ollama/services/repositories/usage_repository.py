"""
Usage Repository - CRUD operations for Usage analytics model.
"""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ollama.models import Usage
from ollama.repositories.base_repository import BaseRepository


class UsageRepository(BaseRepository[Usage]):  # type: ignore[misc]
    """Repository for Usage analytics operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(Usage, session)

    async def log_usage(
        self,
        user_id: uuid.UUID,
        endpoint: str,
        method: str,
        response_time_ms: int,
        status_code: int,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost: float = 0.0,
        usage_metadata: dict[str, Any] | None = None,
    ) -> Usage:
        """Log an API request.

        Args:
            user_id: User ID
            endpoint: API endpoint
            method: HTTP method
            response_time_ms: Response time in milliseconds
            status_code: HTTP status code
            input_tokens: Tokens used in input
            output_tokens: Tokens generated in output
            cost: Cost of the request
            usage_metadata: Additional metadata for analytics

        Returns:
            Created Usage instance
        """
        usage = await self.create(
            user_id=user_id,
            endpoint=endpoint,
            method=method,
            response_time_ms=response_time_ms,
            status_code=status_code,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            usage_metadata=usage_metadata or {},
        )
        await self.commit()
        return usage

    async def get_user_usage(self, user_id: uuid.UUID, days: int = 30) -> list[Usage]:
        """Get usage logs for a user over period.

        Args:
            user_id: User ID
            days: Number of days to look back

        Returns:
            List of usage records
        """
        since = datetime.now(UTC) - timedelta(days=days)
        query = select(Usage).where(and_(Usage.user_id == user_id, Usage.created_at >= since))
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_endpoint_usage(self, endpoint: str, days: int = 30) -> list[Usage]:
        """Get all usage for a specific endpoint.

        Args:
            endpoint: API endpoint
            days: Number of days to look back

        Returns:
            List of usage records
        """
        since = datetime.now(UTC) - timedelta(days=days)
        query = select(Usage).where(and_(Usage.endpoint == endpoint, Usage.created_at >= since))
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_user_token_usage(self, user_id: uuid.UUID, days: int = 30) -> tuple[int, int]:
        """Get total tokens used by a user.

        Args:
            user_id: User ID
            days: Number of days to look back

        Returns:
            Tuple of (total_input_tokens, total_output_tokens)
        """
        usage_records = await self.get_user_usage(user_id, days)

        total_input = sum(u.input_tokens for u in usage_records)
        total_output = sum(u.output_tokens for u in usage_records)

        return total_input, total_output

    async def get_user_cost(self, user_id: uuid.UUID, days: int = 30) -> float:
        """Get total cost for a user.

        Args:
            user_id: User ID
            days: Number of days to look back

        Returns:
            Total cost in dollars
        """
        usage_records = await self.get_user_usage(user_id, days)
        return sum(u.cost for u in usage_records)  # type: ignore[no-any-return]

    async def get_average_response_time(self, user_id: uuid.UUID, days: int = 30) -> float:
        """Get average response time for a user.

        Args:
            user_id: User ID
            days: Number of days to look back

        Returns:
            Average response time in milliseconds
        """
        usage_records = await self.get_user_usage(user_id, days)
        if not usage_records:
            return 0.0

        total_time = sum(u.response_time_ms for u in usage_records)
        return total_time / len(usage_records)  # type: ignore[no-any-return]

    async def get_endpoint_stats(self, endpoint: str, days: int = 30) -> dict[str, Any]:
        """Get statistics for an endpoint.

        Args:
            endpoint: API endpoint
            days: Number of days to look back

        Returns:
            Dict with usage statistics
        """
        usage_records = await self.get_endpoint_usage(endpoint, days)

        if not usage_records:
            return {
                "endpoint": endpoint,
                "total_requests": 0,
                "successful_requests": 0,
                "error_requests": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "avg_response_time_ms": 0.0,
            }

        successful = [u for u in usage_records if u.status_code < 400]
        errors = [u for u in usage_records if u.status_code >= 400]

        return {
            "endpoint": endpoint,
            "total_requests": len(usage_records),
            "successful_requests": len(successful),
            "error_requests": len(errors),
            "total_tokens": sum(u.input_tokens + u.output_tokens for u in usage_records),
            "total_cost": sum(u.cost for u in usage_records),
            "avg_response_time_ms": sum(u.response_time_ms for u in usage_records)
            / len(usage_records),
        }

    async def get_user_stats(self, user_id: uuid.UUID, days: int = 30) -> dict[str, Any]:
        """Get comprehensive statistics for a user.

        Args:
            user_id: User ID
            days: Number of days to look back

        Returns:
            Dict with user statistics
        """
        usage_records = await self.get_user_usage(user_id, days)

        if not usage_records:
            return {
                "user_id": str(user_id),
                "total_requests": 0,
                "successful_requests": 0,
                "error_requests": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cost": 0.0,
                "avg_response_time_ms": 0.0,
            }

        successful = [u for u in usage_records if u.status_code < 400]
        errors = [u for u in usage_records if u.status_code >= 400]

        return {
            "user_id": str(user_id),
            "total_requests": len(usage_records),
            "successful_requests": len(successful),
            "error_requests": len(errors),
            "total_input_tokens": sum(u.input_tokens for u in usage_records),
            "total_output_tokens": sum(u.output_tokens for u in usage_records),
            "total_cost": sum(u.cost for u in usage_records),
            "avg_response_time_ms": sum(u.response_time_ms for u in usage_records)
            / len(usage_records),
        }

    async def get_daily_usage(self, user_id: uuid.UUID, days: int = 30) -> dict[str, Any]:
        """Get daily usage breakdown for a user.

        Args:
            user_id: User ID
            days: Number of days to look back

        Returns:
            Dict with daily usage by date
        """
        usage_records = await self.get_user_usage(user_id, days)

        daily_stats: dict[str, dict[str, Any]] = {}
        for record in usage_records:
            date_key = record.created_at.date().isoformat()
            if date_key not in daily_stats:
                daily_stats[date_key] = {
                    "requests": 0,
                    "tokens": 0,
                    "cost": 0.0,
                }

            daily_stats[date_key]["requests"] += 1
            daily_stats[date_key]["tokens"] += record.input_tokens + record.output_tokens
            daily_stats[date_key]["cost"] += record.cost

        return daily_stats

    async def delete_old_usage(self, days: int = 90) -> int:
        """Delete usage records older than specified days.

        Args:
            days: Age threshold in days

        Returns:
            Number of records deleted
        """
        cutoff = datetime.now(UTC) - timedelta(days=days)
        query = select(Usage).where(Usage.created_at < cutoff)

        result = await self.session.execute(query)
        old_records = result.scalars().all()

        for record in old_records:
            await self.session.delete(record)

        await self.commit()
        return len(old_records)
