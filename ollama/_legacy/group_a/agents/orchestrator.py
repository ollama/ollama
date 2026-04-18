"""Agent orchestrator for multi-step task execution.

Responsible for:
- Task distribution and scheduling
- Retry logic and backoff strategies
- State management across agent executions
- Performance metrics and monitoring
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import structlog

log = structlog.get_logger(__name__)


class TaskStatus(str, Enum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class TaskExecutionResult:
    """Result of a task execution."""

    task_id: str
    agent_id: str
    status: TaskStatus
    output: Optional[Any] = None
    error: Optional[str] = None
    duration_ms: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "tokens_used": self.tokens_used,
            "cost_usd": self.cost_usd,
            "timestamp": self.timestamp or datetime.utcnow().isoformat(),
        }


class AgentOrchestrator:
    """Orchestrates agent task execution and distribution."""

    def __init__(
        self,
        project_id: str,
        queue_name: str,
        agents_service_url: str,
        max_retries: int = 3,
        timeout_seconds: int = 300,
    ) -> None:
        """Initialize orchestrator.

        Args:
            project_id: GCP project ID
            queue_name: Cloud Tasks queue name
            agents_service_url: URL to agents Cloud Run service
            max_retries: Maximum retry attempts per task
            timeout_seconds: Task execution timeout in seconds
        """
        self.project_id = project_id
        self.queue_name = queue_name
        self.agents_service_url = agents_service_url
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds

    async def execute_task(
        self, task_id: str, agent_id: str, input_data: dict[str, Any]
    ) -> TaskExecutionResult:
        """Execute a single task through an agent.

        Args:
            task_id: Unique task identifier
            agent_id: Agent to execute the task
            input_data: Input data for the task

        Returns:
            TaskExecutionResult with execution status and output
        """
        log.info(
            "task_execution_started",
            task_id=task_id,
            agent_id=agent_id,
            input_keys=list(input_data.keys()),
        )

        start_time = datetime.utcnow()

        try:
            # Execute task with timeout
            result = await asyncio.wait_for(
                self._execute_with_retries(task_id, agent_id, input_data),
                timeout=self.timeout_seconds,
            )

            duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            result.duration_ms = duration_ms
            result.timestamp = datetime.utcnow().isoformat()

            log.info(
                "task_execution_completed",
                task_id=task_id,
                status=result.status.value,
                duration_ms=duration_ms,
            )

            return result

        except asyncio.TimeoutError:
            log.error(
                "task_execution_timeout",
                task_id=task_id,
                timeout_seconds=self.timeout_seconds,
            )
            return TaskExecutionResult(
                task_id=task_id,
                agent_id=agent_id,
                status=TaskStatus.TIMEOUT,
                error=f"Task timed out after {self.timeout_seconds}s",
                duration_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000),
                timestamp=datetime.utcnow().isoformat(),
            )

        except Exception as e:
            log.error(
                "task_execution_failed",
                task_id=task_id,
                error=str(e),
                exc_info=True,
            )
            return TaskExecutionResult(
                task_id=task_id,
                agent_id=agent_id,
                status=TaskStatus.FAILED,
                error=str(e),
                duration_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000),
                timestamp=datetime.utcnow().isoformat(),
            )

    async def _execute_with_retries(
        self,
        task_id: str,
        agent_id: str,
        input_data: dict[str, Any],
        attempt: int = 0,
    ) -> TaskExecutionResult:
        """Execute task with exponential backoff retry logic.

        Args:
            task_id: Task identifier
            agent_id: Agent identifier
            input_data: Task input data
            attempt: Current attempt number (0-indexed)

        Returns:
            TaskExecutionResult
        """
        try:
            # TODO: Implement actual agent invocation
            # This would call the agents service via Cloud Run
            result = TaskExecutionResult(
                task_id=task_id,
                agent_id=agent_id,
                status=TaskStatus.SUCCESS,
                output={"status": "placeholder"},
                tokens_used=0,
            )
            return result

        except Exception:
            if attempt < self.max_retries:
                # Exponential backoff: 1s, 2s, 4s, 8s
                backoff_seconds = 2**attempt
                log.warning(
                    "task_execution_retry",
                    task_id=task_id,
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    backoff_seconds=backoff_seconds,
                )
                await asyncio.sleep(backoff_seconds)
                return await self._execute_with_retries(
                    task_id, agent_id, input_data, attempt=attempt + 1
                )
            else:
                raise
