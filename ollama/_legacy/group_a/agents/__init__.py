"""Agent Framework for Ollama.

This module provides the agentic infrastructure for Ollama, enabling autonomous
AI agents to perform complex tasks using the Ollama model server.

The agents framework includes:
- Agent base class with lifecycle management
- Orchestrator for agent coordination and task distribution
- Tool calling interface for multi-step workflows
- State persistence in Firestore
- Monitoring and observability hooks
- Elite FAANG-tier agent templates for specialized domains
"""

from ollama.agents.compliance_agent import ComplianceAgent
from ollama.agents.cost_agent import CostAgent
from ollama.agents.data_agent import DataAgent
from ollama.agents.deployment_agent import DeploymentAgent
from ollama.agents.performance_agent import PerformanceAgent
from ollama.agents.reliability_agent import ReliabilityAgent
from ollama.agents.security_agent import SecurityAgent
from ollama.agents.templates import (
    AgentExecutionResult,
    AgentFactory,
    AgentMetrics,
    AgentSpecialization,
    AgentStatus,
    BaseEliteAgent,
    SpecializedAgentTemplate,
    create_agent,
    get_global_factory,
    register_agent_template,
)

__all__ = [
    # Base framework
    "BaseEliteAgent",
    "SpecializedAgentTemplate",
    "AgentFactory",
    "AgentMetrics",
    "AgentExecutionResult",
    "AgentSpecialization",
    "AgentStatus",
    # Global factory
    "get_global_factory",
    "create_agent",
    "register_agent_template",
    # Specialized agents
    "SecurityAgent",
    "PerformanceAgent",
    "DeploymentAgent",
    "ComplianceAgent",
    "ReliabilityAgent",
    "CostAgent",
    "DataAgent",
]
