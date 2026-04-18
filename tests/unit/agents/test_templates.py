"""Comprehensive tests for Elite Agent Templates.

Tests cover:
- BaseEliteAgent resilience patterns
- AgentFactory creation and metrics
- All 6 specialized agents
- Integration scenarios
"""

import asyncio

import pytest

from ollama.agents.compliance_agent import ComplianceAgent
from ollama.agents.cost_agent import CostAgent
from ollama.agents.data_agent import DataAgent
from ollama.agents.deployment_agent import DeploymentAgent
from ollama.agents.performance_agent import PerformanceAgent
from ollama.agents.reliability_agent import ReliabilityAgent
from ollama.agents.security_agent import SecurityAgent
from ollama.agents.templates import (
    AgentFactory,
    AgentMetrics,
    AgentSpecialization,
    AgentStatus,
    SpecializedAgentTemplate,
    get_global_factory,
)


class TestAgentMetrics:
    """Test AgentMetrics dataclass."""

    def test_metrics_initialization(self) -> None:
        """Metrics initialized with correct defaults."""
        metrics = AgentMetrics()
        assert metrics.executions_total == 0
        assert metrics.executions_successful == 0

    def test_metrics_update_execution_success(self) -> None:
        """Update execution on success."""
        metrics = AgentMetrics()
        metrics.update_execution(latency_ms=125.5, success=True)

        assert metrics.executions_total == 1
        assert metrics.executions_successful == 1
        assert metrics.executions_failed == 0

    def test_metrics_update_execution_failure(self) -> None:
        """Update execution on failure."""
        metrics = AgentMetrics()
        metrics.update_execution(latency_ms=200.0, success=False)

        assert metrics.executions_total == 1
        assert metrics.executions_successful == 0
        assert metrics.executions_failed == 1

    def test_metrics_percentiles(self) -> None:
        """Calculate latency percentiles correctly."""
        metrics = AgentMetrics()
        latencies = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550]
        for lat in latencies:
            metrics.update_execution(latency_ms=float(lat), success=True)

        assert metrics.p95_latency_ms is not None
        assert metrics.p99_latency_ms is not None
        assert metrics.p95_latency_ms <= metrics.p99_latency_ms

    def test_metrics_to_dict(self) -> None:
        """Serialize metrics to dict."""
        metrics = AgentMetrics()
        metrics.update_execution(latency_ms=100.0, success=True)

        # Metrics dataclass has asdict method
        from dataclasses import asdict
        result = asdict(metrics)
        assert result["executions_total"] == 1
        assert result["executions_successful"] == 1


class TestAgentFactory:
    """Test AgentFactory pattern."""

    @pytest.fixture
    def factory(self) -> AgentFactory:
        """Create fresh factory for each test with all templates registered."""
        factory = AgentFactory()
        # Register all agent templates
        factory.register_template(AgentSpecialization.SECURITY, SecurityAgent)
        factory.register_template(AgentSpecialization.PERFORMANCE, PerformanceAgent)
        factory.register_template(AgentSpecialization.DEPLOYMENT, DeploymentAgent)
        factory.register_template(AgentSpecialization.COMPLIANCE, ComplianceAgent)
        factory.register_template(AgentSpecialization.RELIABILITY, ReliabilityAgent)
        factory.register_template(AgentSpecialization.COST, CostAgent)
        factory.register_template(AgentSpecialization.DATA, DataAgent)
        return factory

    def test_factory_create_security_agent(self, factory: AgentFactory) -> None:
        """Factory creates SecurityAgent correctly."""
        agent = factory.create_agent(
            AgentSpecialization.SECURITY,
            {"config": "test"},
        )

        assert isinstance(agent, SecurityAgent)
        assert agent.specialization == AgentSpecialization.SECURITY

    def test_factory_create_performance_agent(self, factory: AgentFactory) -> None:
        """Factory creates PerformanceAgent correctly."""
        agent = factory.create_agent(
            AgentSpecialization.PERFORMANCE,
            {},
        )

        assert isinstance(agent, PerformanceAgent)
        assert agent.specialization == AgentSpecialization.PERFORMANCE

    def test_factory_register_template(self, factory: AgentFactory) -> None:
        """Factory registers custom templates."""
        new_factory = AgentFactory()
        new_factory.register_template(
            AgentSpecialization.SECURITY,
            SecurityAgent,
        )

        assert AgentSpecialization.SECURITY in new_factory._templates

    def test_factory_list_agents(self, factory: AgentFactory) -> None:
        """List all created agents."""
        agent1 = factory.create_agent(AgentSpecialization.SECURITY, {})
        agent2 = factory.create_agent(AgentSpecialization.PERFORMANCE, {})

        agents = factory.list_agents()
        assert len(agents) >= 2

    def test_factory_get_agent_metrics(self, factory: AgentFactory) -> None:
        """Get metrics for specific agent."""
        agent = factory.create_agent(AgentSpecialization.SECURITY, {})
        metrics = factory.get_agent_metrics(agent.agent_id)

        assert metrics is not None
        assert isinstance(metrics, dict)

    def test_factory_list_all_metrics(self, factory: AgentFactory) -> None:
        """Aggregate metrics across all agents."""
        factory.create_agent(AgentSpecialization.SECURITY, {})
        factory.create_agent(AgentSpecialization.PERFORMANCE, {})

        all_metrics = factory.list_all_metrics()
        assert len(all_metrics) == 2


@pytest.mark.asyncio
class TestSecurityAgent:
    """Test SecurityAgent specialization."""

    @pytest.fixture
    async def agent(self) -> SecurityAgent:
        """Create security agent for tests."""
        factory = AgentFactory()
        # Register templates
        factory.register_template(AgentSpecialization.SECURITY, SecurityAgent)
        agent = factory.create_agent(
            AgentSpecialization.SECURITY,
            {"model": "test"},
        )
        return agent

    async def test_security_agent_vulnerability_scan(self, agent: SecurityAgent) -> None:
        """Security agent performs vulnerability scanning."""
        result = await agent.execute("perform vulnerability scan")

        assert result.status == AgentStatus.COMPLETED
        assert result.specialization == AgentSpecialization.SECURITY
        assert isinstance(result.output, dict)
        assert "vulnerabilities" in result.output or "type" in result.output

    async def test_security_agent_threat_modeling(self, agent: SecurityAgent) -> None:
        """Security agent performs threat modeling."""
        result = await agent.execute("perform threat modeling")

        assert result.status == AgentStatus.COMPLETED
        assert result.output is not None

    async def test_security_agent_compliance_check(self, agent: SecurityAgent) -> None:
        """Security agent checks compliance."""
        result = await agent.execute("check compliance")

        assert result.status == AgentStatus.COMPLETED
        assert result.output is not None

    async def test_security_agent_caching(self, agent: SecurityAgent) -> None:
        """Security agent caches results."""
        prompt = "perform general assessment"

        # First call
        result1 = await agent.execute(prompt)
        latency1 = result1.latency_ms

        # Second call (should be cached)
        result2 = await agent.execute(prompt)
        latency2 = result2.latency_ms

        # Cached call should be faster
        assert latency2 < latency1


@pytest.mark.asyncio
class TestPerformanceAgent:
    """Test PerformanceAgent specialization."""

    @pytest.fixture
    async def agent(self) -> PerformanceAgent:
        """Create performance agent for tests."""
        factory = AgentFactory()
        # Register templates
        factory.register_template(AgentSpecialization.PERFORMANCE, PerformanceAgent)
        agent = factory.create_agent(
            AgentSpecialization.PERFORMANCE,
            {},
        )
        return agent

    async def test_performance_agent_profiling(self, agent: PerformanceAgent) -> None:
        """Performance agent performs profiling."""
        result = await agent.execute("perform profiling")

        assert result.status == AgentStatus.COMPLETED
        assert "type" in result.output
        assert result.latency_ms < 500  # Should be fast

    async def test_performance_agent_optimization(self, agent: PerformanceAgent) -> None:
        """Performance agent generates optimizations."""
        result = await agent.execute("optimize latency")

        assert result.status == AgentStatus.COMPLETED
        assert "optimizations" in result.output or "type" in result.output

    async def test_performance_agent_metrics_tracking(self, agent: PerformanceAgent) -> None:
        """Performance agent tracks execution metrics."""
        await agent.execute("perform profiling")

        metrics = agent.get_metrics()
        assert metrics["executions_total"] >= 1
        assert metrics["executions_successful"] >= 1


@pytest.mark.asyncio
class TestDeploymentAgent:
    """Test DeploymentAgent specialization."""

    @pytest.fixture
    async def agent(self) -> DeploymentAgent:
        """Create deployment agent for tests."""
        factory = AgentFactory()
        # Register templates
        factory.register_template(AgentSpecialization.DEPLOYMENT, DeploymentAgent)
        agent = factory.create_agent(
            AgentSpecialization.DEPLOYMENT,
            {},
        )
        return agent

    async def test_deployment_agent_validation(self, agent: DeploymentAgent) -> None:
        """Deployment agent validates deployment readiness."""
        result = await agent.execute("validate deployment readiness")

        assert result.status == AgentStatus.COMPLETED
        assert "type" in result.output

    async def test_deployment_agent_pipeline_check(self, agent: DeploymentAgent) -> None:
        """Deployment agent checks CI/CD pipeline."""
        result = await agent.execute("validate CI/CD pipeline")

        assert result.status == AgentStatus.COMPLETED
        assert result.output is not None


@pytest.mark.asyncio
class TestComplianceAgent:
    """Test ComplianceAgent specialization."""

    @pytest.fixture
    async def agent(self) -> ComplianceAgent:
        """Create compliance agent for tests."""
        factory = AgentFactory()
        # Register templates
        factory.register_template(AgentSpecialization.COMPLIANCE, ComplianceAgent)
        agent = factory.create_agent(
            AgentSpecialization.COMPLIANCE,
            {},
        )
        return agent

    async def test_compliance_agent_soc2_check(self, agent: ComplianceAgent) -> None:
        """Compliance agent checks SOC 2 compliance."""
        result = await agent.execute("check SOC 2 compliance")

        assert result.status == AgentStatus.COMPLETED
        assert "framework" in result.output or "type" in result.output

    async def test_compliance_agent_gdpr_check(self, agent: ComplianceAgent) -> None:
        """Compliance agent checks GDPR compliance."""
        result = await agent.execute("check GDPR compliance")

        assert result.status == AgentStatus.COMPLETED
        assert result.output is not None


@pytest.mark.asyncio
class TestReliabilityAgent:
    """Test ReliabilityAgent specialization."""

    @pytest.fixture
    async def agent(self) -> ReliabilityAgent:
        """Create reliability agent for tests."""
        factory = AgentFactory()
        # Register templates
        factory.register_template(AgentSpecialization.RELIABILITY, ReliabilityAgent)
        agent = factory.create_agent(
            AgentSpecialization.RELIABILITY,
            {},
        )
        return agent

    async def test_reliability_agent_slo_definition(self, agent: ReliabilityAgent) -> None:
        """Reliability agent defines SLOs."""
        result = await agent.execute("define SLOs")

        assert result.status == AgentStatus.COMPLETED
        assert "type" in result.output

    async def test_reliability_agent_dr_planning(self, agent: ReliabilityAgent) -> None:
        """Reliability agent plans disaster recovery."""
        result = await agent.execute("plan disaster recovery")

        assert result.status == AgentStatus.COMPLETED
        assert result.output is not None


@pytest.mark.asyncio
class TestCostAgent:
    """Test CostAgent specialization."""

    @pytest.fixture
    async def agent(self) -> CostAgent:
        """Create cost agent for tests."""
        factory = AgentFactory()
        # Register templates
        factory.register_template(AgentSpecialization.COST, CostAgent)
        agent = factory.create_agent(
            AgentSpecialization.COST,
            {},
        )
        return agent

    async def test_cost_agent_optimization(self, agent: CostAgent) -> None:
        """Cost agent identifies cost optimizations."""
        result = await agent.execute("optimize cloud costs")

        assert result.status == AgentStatus.COMPLETED
        assert "type" in result.output

    async def test_cost_agent_forecast(self, agent: CostAgent) -> None:
        """Cost agent forecasts future costs."""
        result = await agent.execute("forecast costs")

        assert result.status == AgentStatus.COMPLETED
        assert result.output is not None


@pytest.mark.asyncio
class TestDataAgent:
    """Test DataAgent specialization."""

    @pytest.fixture
    async def agent(self) -> DataAgent:
        """Create data agent for tests."""
        factory = AgentFactory()
        # Register templates
        factory.register_template(AgentSpecialization.DATA, DataAgent)
        agent = factory.create_agent(
            AgentSpecialization.DATA,
            {},
        )
        return agent

    async def test_data_agent_quality_assessment(self, agent: DataAgent) -> None:
        """Data agent assesses data quality."""
        result = await agent.execute("assess data quality")

        assert result.status == AgentStatus.COMPLETED
        assert "type" in result.output

    async def test_data_agent_pipeline_validation(self, agent: DataAgent) -> None:
        """Data agent validates data pipeline."""
        result = await agent.execute("validate data pipeline")

        assert result.status == AgentStatus.COMPLETED
        assert result.output is not None


@pytest.mark.asyncio
class TestAgentIntegration:
    """Integration tests for agent framework."""

    async def test_create_all_agents(self) -> None:
        """Create all 7 specialized agents."""
        factory = AgentFactory()
        # Register all templates
        factory.register_template(AgentSpecialization.SECURITY, SecurityAgent)
        factory.register_template(AgentSpecialization.PERFORMANCE, PerformanceAgent)
        factory.register_template(AgentSpecialization.DEPLOYMENT, DeploymentAgent)
        factory.register_template(AgentSpecialization.COMPLIANCE, ComplianceAgent)
        factory.register_template(AgentSpecialization.RELIABILITY, ReliabilityAgent)
        factory.register_template(AgentSpecialization.COST, CostAgent)
        factory.register_template(AgentSpecialization.DATA, DataAgent)

        agents = [
            factory.create_agent(AgentSpecialization.SECURITY, {}),
            factory.create_agent(AgentSpecialization.PERFORMANCE, {}),
            factory.create_agent(AgentSpecialization.DEPLOYMENT, {}),
            factory.create_agent(AgentSpecialization.COMPLIANCE, {}),
            factory.create_agent(AgentSpecialization.RELIABILITY, {}),
            factory.create_agent(AgentSpecialization.COST, {}),
            factory.create_agent(AgentSpecialization.DATA, {}),
        ]

        assert len(agents) == 7
        assert all(isinstance(a, SpecializedAgentTemplate) for a in agents)

    async def test_concurrent_agent_execution(self) -> None:
        """Execute multiple agents concurrently."""
        factory = AgentFactory()
        # Register templates
        factory.register_template(AgentSpecialization.SECURITY, SecurityAgent)
        factory.register_template(AgentSpecialization.PERFORMANCE, PerformanceAgent)
        factory.register_template(AgentSpecialization.DEPLOYMENT, DeploymentAgent)

        agents = [
            factory.create_agent(AgentSpecialization.SECURITY, {}),
            factory.create_agent(AgentSpecialization.PERFORMANCE, {}),
            factory.create_agent(AgentSpecialization.DEPLOYMENT, {}),
        ]

        results = await asyncio.gather(
            agents[0].execute("vulnerability scan"),
            agents[1].execute("profile performance"),
            agents[2].execute("validate deployment"),
        )

        assert all(r.status == AgentStatus.COMPLETED for r in results)
        assert all(r.latency_ms > 0 for r in results)

    async def test_global_factory_singleton(self) -> None:
        """Global factory maintains singleton pattern."""
        factory1 = get_global_factory()
        factory2 = get_global_factory()

        assert factory1 is factory2
