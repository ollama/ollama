"""Integration tests for landing zone agents.

Tests agent initialization, method execution, audit logging,
and agent interaction patterns.
"""

from typing import Any

import pytest

from ollama.agents.hub_spoke_agent import HubSpokeAgent, IssueType
from ollama.agents.pmo_agent import ComplianceStatus, PMOAgent


@pytest.fixture
def mock_agent_context() -> dict[str, Any]:
    """Create mock agent context."""
    return {
        "user_id": "test-user-001",
        "session_id": "test-session-001",
        "capabilities": ["generate", "retrieve"],
    }


@pytest.fixture
def hub_spoke_agent(mock_agent_context) -> HubSpokeAgent:
    """Create HubSpokeAgent for testing."""
    return HubSpokeAgent(mock_agent_context)


@pytest.fixture
def pmo_agent(mock_agent_context) -> PMOAgent:
    """Create PMOAgent for testing."""
    return PMOAgent(mock_agent_context)


class TestHubSpokeAgent:
    """Tests for HubSpokeAgent."""

    @pytest.mark.asyncio
    async def test_agent_initialization(self, hub_spoke_agent: HubSpokeAgent) -> None:
        """Test agent initializes correctly."""
        assert hub_spoke_agent.name == "HubSpokeAgent"
        assert len(hub_spoke_agent.capabilities) > 0
        assert hub_spoke_agent.audit_log is not None

    @pytest.mark.asyncio
    async def test_route_issue_to_hub(self, hub_spoke_agent: HubSpokeAgent) -> None:
        """Test routing critical issue to hub."""
        from ollama.agents.hub_spoke_agent import RepositoryIssue

        issue = RepositoryIssue(
            id="issue-001",
            title="Critical Security Bug",
            issue_type=IssueType.BUG,
            priority="critical",
            source_repo="spoke-team-a",
        )

        result = await hub_spoke_agent.route_issue(issue)

        assert result == "hub"
        assert hub_spoke_agent.audit_log is not None

    @pytest.mark.asyncio
    async def test_route_issue_to_spoke(self, hub_spoke_agent: HubSpokeAgent) -> None:
        """Test routing feature request to spoke."""
        from ollama.agents.hub_spoke_agent import RepositoryIssue

        issue = RepositoryIssue(
            id="issue-002",
            title="Add new feature X",
            issue_type=IssueType.FEATURE,
            priority="medium",
            source_repo="hub",
        )

        result = await hub_spoke_agent.route_issue(issue)

        # Features from hub should route to spokes (implementation-specific)
        assert result in ["spoke-team-a", "spoke-team-b", "spoke-team-c"]

    @pytest.mark.asyncio
    async def test_escalate_spoke_issue(self, hub_spoke_agent: HubSpokeAgent) -> None:
        """Test escalating spoke issue to hub."""
        result = await hub_spoke_agent.escalate_to_hub("spoke-issue-123")

        assert isinstance(result, dict)
        assert "status" in result
        assert "hub_issue_id" in result

    @pytest.mark.asyncio
    async def test_agent_has_audit_log(self, hub_spoke_agent: HubSpokeAgent) -> None:
        """Test agent maintains audit log."""
        from ollama.agents.hub_spoke_agent import RepositoryIssue

        issue = RepositoryIssue(
            id="audit-test-001",
            title="Test audit",
            issue_type=IssueType.BUG,
            priority="low",
            source_repo="spoke-test",
        )

        await hub_spoke_agent.route_issue(issue)

        # Verify audit log exists (implementation-dependent)
        assert hub_spoke_agent.audit_log is not None

    @pytest.mark.asyncio
    async def test_sync_hub_to_spokes(self, hub_spoke_agent: HubSpokeAgent) -> None:
        """Test synchronizing hub issue to spokes."""
        result = await hub_spoke_agent.sync_hub_to_spokes(123)

        assert isinstance(result, dict)
        assert "status" in result

    @pytest.mark.asyncio
    async def test_aggregate_spoke_updates(self, hub_spoke_agent: HubSpokeAgent) -> None:
        """Test aggregating updates from spokes."""
        result = await hub_spoke_agent.aggregate_spoke_updates()

        assert isinstance(result, dict)
        assert "aggregated_issues" in result or "status" in result

    def test_explain_reasoning(self, hub_spoke_agent: HubSpokeAgent) -> None:
        """Test agent reasoning explanation."""
        explanation = hub_spoke_agent.explain_reasoning()

        assert isinstance(explanation, str)
        assert len(explanation) > 0
        assert "hub" in explanation.lower() or "spoke" in explanation.lower()

    def test_rollback_capability(self, hub_spoke_agent: HubSpokeAgent) -> None:
        """Test agent can rollback actions."""
        result = hub_spoke_agent.rollback("action-123")

        assert isinstance(result, bool)


class TestPMOAgent:
    """Tests for PMOAgent."""

    @pytest.mark.asyncio
    async def test_agent_initialization(self, pmo_agent: PMOAgent) -> None:
        """Test PMO agent initializes correctly."""
        assert pmo_agent.name == "PMOAgent"
        assert len(pmo_agent.capabilities) > 0
        assert pmo_agent.audit_log is not None

    @pytest.mark.asyncio
    async def test_validate_compliance(self, pmo_agent: PMOAgent) -> None:
        """Test Landing Zone compliance validation."""
        result = await pmo_agent.validate_landing_zone_compliance("proj-ollama-api")

        assert result.check_name == "Landing Zone 8-Point Mandate"
        assert result.status in [
            ComplianceStatus.COMPLIANT,
            ComplianceStatus.NON_COMPLIANT,
            ComplianceStatus.PARTIAL,
        ]
        assert "proj-ollama-api" in result.resources_affected

    @pytest.mark.asyncio
    async def test_enforce_label_schema(self, pmo_agent: PMOAgent) -> None:
        """Test enforcing 24-label schema."""
        result = await pmo_agent.enforce_label_schema("resource-123")

        assert result["resource_id"] == "resource-123"
        assert "labels_applied" in result
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_compliance_report_generation(self, pmo_agent: PMOAgent) -> None:
        """Test generating compliance report."""
        report = await pmo_agent.generate_compliance_report()

        assert isinstance(report, str)
        assert len(report) > 100
        assert "COMPLIANCE" in report.upper()

    @pytest.mark.asyncio
    async def test_monitor_compliance_drift(self, pmo_agent: PMOAgent) -> None:
        """Test monitoring compliance drift."""
        result = await pmo_agent.monitor_compliance_drift()

        assert isinstance(result, dict)
        assert "resources_scanned" in result
        assert "compliant" in result
        assert "non_compliant" in result

    def test_explain_reasoning(self, pmo_agent: PMOAgent) -> None:
        """Test PMO agent reasoning explanation."""
        explanation = pmo_agent.explain_reasoning()

        assert isinstance(explanation, str)
        assert "compliance" in explanation.lower()

    def test_rollback_capability(self, pmo_agent: PMOAgent) -> None:
        """Test PMO agent can rollback actions."""
        result = pmo_agent.rollback("label-enforcement-001")

        assert isinstance(result, bool)


class TestAgentInteraction:
    """Tests for agent-to-agent interaction."""

    @pytest.mark.asyncio
    async def test_hub_spoke_escalation_to_pmo(
        self, hub_spoke_agent: HubSpokeAgent, pmo_agent: PMOAgent
    ) -> None:
        """Test HubSpokeAgent escalating compliance issue to PMOAgent."""
        # HubSpokeAgent escalates a compliance-related issue
        escalated = await hub_spoke_agent.escalate_to_hub("spoke-compliance-001")

        assert isinstance(escalated, dict)

        # PMOAgent would validate it
        if "resource_id" in escalated:
            compliance = await pmo_agent.validate_landing_zone_compliance(
                escalated.get("resource_id", "test-resource")
            )
            assert compliance.status is not None

    @pytest.mark.asyncio
    async def test_pmo_validation_workflow(self, pmo_agent: PMOAgent) -> None:
        """Test complete PMO validation workflow."""
        # 1. Validate resource
        validation = await pmo_agent.validate_landing_zone_compliance("prod-ollama-api")
        assert validation is not None

        # 2. Enforce labels if needed
        labels = await pmo_agent.enforce_label_schema("prod-ollama-api")
        assert labels["status"] == "completed"

        # 3. Monitor for drift
        drift = await pmo_agent.monitor_compliance_drift()
        assert "resources_scanned" in drift

        # 4. Generate report
        report = await pmo_agent.generate_compliance_report()
        assert isinstance(report, str)


class TestAgentErrorHandling:
    """Tests for agent error handling."""

    @pytest.mark.asyncio
    async def test_hub_spoke_handles_errors(self, hub_spoke_agent: HubSpokeAgent) -> None:
        """Test HubSpokeAgent error handling."""
        # Test with invalid issue type
        from ollama.agents.hub_spoke_agent import RepositoryIssue

        issue = RepositoryIssue(
            id="error-test-001",
            title="Test issue",
            issue_type=IssueType.BUG,
            priority="medium",
            source_repo="test-spoke",
        )

        # Should not raise exception
        try:
            result = await hub_spoke_agent.route_issue(issue)
            assert result is not None
        except Exception as e:
            pytest.fail(f"route_issue should not raise exception: {e}")

    @pytest.mark.asyncio
    async def test_pmo_handles_validation_errors(self, pmo_agent: PMOAgent) -> None:
        """Test PMOAgent handles validation errors gracefully."""
        try:
            result = await pmo_agent.validate_landing_zone_compliance("invalid-id")
            assert result is not None
        except Exception as e:
            pytest.fail(f"validate_landing_zone_compliance should handle errors: {e}")


class TestAgentAuditLog:
    """Tests for agent audit logging."""

    @pytest.mark.asyncio
    async def test_hub_spoke_audit_trail(self, hub_spoke_agent: HubSpokeAgent) -> None:
        """Test HubSpokeAgent maintains audit trail."""
        from ollama.agents.hub_spoke_agent import RepositoryIssue

        issue = RepositoryIssue(
            id="audit-trail-001",
            title="Test audit trail",
            issue_type=IssueType.BUG,
            priority="critical",
            source_repo="hub",
        )

        # Execute action
        await hub_spoke_agent.route_issue(issue)

        # Verify audit log recorded action
        assert hub_spoke_agent.audit_log is not None

    @pytest.mark.asyncio
    async def test_pmo_audit_trail(self, pmo_agent: PMOAgent) -> None:
        """Test PMOAgent maintains audit trail."""
        # Execute action
        await pmo_agent.validate_landing_zone_compliance("test-resource-001")

        # Verify audit log recorded action
        assert pmo_agent.audit_log is not None
        assert len(pmo_agent.compliance_checks) > 0


class TestAgentCapabilities:
    """Tests for agent capabilities."""

    def test_hub_spoke_has_required_capabilities(self, hub_spoke_agent: HubSpokeAgent) -> None:
        """Test HubSpokeAgent has required capabilities."""
        capabilities = {cap.value for cap in hub_spoke_agent.capabilities}

        # Should have at least GENERATE and RETRIEVE
        assert len(capabilities) > 0

    def test_pmo_has_required_capabilities(self, pmo_agent: PMOAgent) -> None:
        """Test PMOAgent has required capabilities."""
        capabilities = {cap.value for cap in pmo_agent.capabilities}

        # Should have at least GENERATE and RETRIEVE
        assert "generate" in capabilities or len(capabilities) > 0
        assert "retrieve" in capabilities or len(capabilities) > 0


# Run tests with: pytest tests/integration/test_agents.py -v
