"""Base agent class for autonomous task execution.

Defines the interface and common functionality for all Ollama agents.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import structlog

log = structlog.get_logger(__name__)


class AgentCapability(str, Enum):
    """Capabilities that agents can declare."""

    # Basic generation/retrieval capabilities used by PMO/agent implementations
    GENERATE = "generate"
    RETRIEVE = "retrieve"

    REASONING = "reasoning"
    PLANNING = "planning"
    TOOL_USE = "tool_use"
    MEMORY = "memory"
    SELF_CORRECTION = "self_correction"
    MULTI_STEP = "multi_step"
    STREAMING = "streaming"
    CONTEXT_WINDOW = "context_window"


@dataclass
class AgentConfig:
    """Configuration for an agent."""

    agent_id: str
    model: str  # e.g., "llama3.2", "neural-chat"
    capabilities: list[AgentCapability]
    max_tokens: int = 2048
    temperature: float = 0.7
    context_window: int = 4096
    timeout_seconds: int = 300
    max_retries: int = 3


class Agent(ABC):
    """Base class for Ollama agents.

    All agents inherit from this class and must implement the execute method.
    """

    def __init__(self, config: AgentConfig) -> None:
        """Initialize agent.

        Args:
            config: Agent configuration
        """
        self.config = config
        # Support dict-based test fixtures by falling back to common keys
        agent_id = None
        if isinstance(config, dict):
            agent_id = config.get("agent_id") or config.get("session_id") or "agent"
        else:
            agent_id = getattr(config, "agent_id", "agent")
        self.logger = structlog.get_logger(f"agent.{agent_id}")
        # Provide a lightweight in-memory audit log for tests and simple usage.
        class _SimpleAuditLog:
            def __init__(self) -> None:
                self.intents: list[dict[str, Any]] = []
                self.results: list[dict[str, Any]] = []

            def log_intent(self, data: dict[str, Any]) -> None:
                self.intents.append(data)

            def log_result(self, data: dict[str, Any]) -> None:
                self.results.append(data)

        self.audit_log: Any = _SimpleAuditLog()

    @abstractmethod
    async def execute(self, input_prompt: str) -> dict[str, Any]:
        """Execute the agent on an input prompt.

        Args:
            input_prompt: User input or task description

        Returns:
            Dictionary containing:
                - output: Generated response
                - tokens_used: Number of tokens consumed
                - cost_usd: Estimated cost
                - metadata: Additional execution metadata
        """

    async def think(self, context: str) -> str:
        """Internal reasoning step (can be overridden).

        Args:
            context: Context for reasoning

        Returns:
            Reasoning output
        """
        self.logger.info("thinking", context_length=len(context))
        return context

    async def plan(self, task: str) -> list[str]:
        """Create a plan for multi-step execution (can be overridden).

        Args:
            task: Task description

        Returns:
            List of plan steps
        """
        self.logger.info("planning", task=task)
        return [task]  # Default: single-step execution

    async def tool_use(self, tool_name: str, **kwargs: Any) -> Any:
        """Call an external tool (can be overridden).

        Args:
            tool_name: Name of the tool to call
            **kwargs: Tool arguments

        Returns:
            Tool output
        """
        self.logger.info("tool_use", tool=tool_name, kwargs_keys=list(kwargs.keys()))
        return None

    def get_capabilities(self) -> list[str]:
        """Get list of agent capabilities.

        Returns:
            List of capability names
        """
        return [cap.value for cap in self.config.capabilities]

    def __str__(self) -> str:
        """String representation of agent."""
        return (
            f"Agent({self.config.agent_id}, model={self.config.model}, "
            f"capabilities={self.get_capabilities()})"
        )
