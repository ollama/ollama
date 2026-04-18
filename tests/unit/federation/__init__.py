"""
Unit Tests for Federation Manager - Issue #42

Tests for hub registration, routing, health checking, and topology management.

Status: PRODUCTION-READY
Coverage: 20+ tests
Target: ✅ 100% pass rate for Week 1
"""

import pytest
from datetime import datetime, timedelta
from ollama.federation.manager import (
    FederationManager,
    Hub,
    HubCapacity,
    HubStatus,
    RoutingStrategy,
)


@pytest.fixture
def federation_manager():
    """Create federation manager for testing."""
    return FederationManager(
        primary_hub_id="control-plane-1",
        region="us-central1"
    )


@pytest.fixture
def sample_hub_capacity():
    """Create sample hub capacity metrics."""
    return HubCapacity(
        max_concurrent_requests=100,
        current_requests=25,
        max_memory_gb=256,
        current_memory_gb=128,
        available_models=["llama3.2", "mixtral-8x7b"],
        cpu_utilization_percent=45,
        memory_utilization_percent=50,
        network_utilization_percent=30,
    )


@pytest.fixture
def sample_hub(sample_hub_capacity):
    """Create sample hub for testing."""
    return Hub(
        hub_id="hub-us-central1-1",
        name="US Central 1 Primary",
        region="us-central1",
        zone="us-central1-a",
        grpc_endpoint="hub-1.us-central1.ollama.local:8000",
        https_endpoint="https://hub-1.us-central1.ollama.local",
        capacity=sample_hub_capacity,
        status=HubStatus.HEALTHY,
        version="v1.0.0",
        supported_models=["llama3.2", "mixtral-8x7b"],
        labels={"environment": "production", "tier": "primary"},
    )


# ============================================================================
# Hub Registration Tests
# ============================================================================

def test_register_hub_success(federation_manager, sample_hub):
    """Test successful hub registration."""
    federation_manager.register_hub(sample_hub)
    
    assert sample_hub.hub_id in federation_manager.hubs
    assert federation_manager.hubs[sample_hub.hub_id] == sample_hub
    assert sample_hub.last_heartbeat is not None
    assert federation_manager.topology_version == 2  # Incremented from 1


def test_register_hub_duplicate_raises_error(federation_manager, sample_hub):
    """Test that registering duplicate hub raises error."""
    federation_manager.register_hub(sample_hub)
    
    with pytest.raises(ValueError, match="already registered"):
        federation_manager.register_hub(sample_hub)


def test_register_multiple_hubs(federation_manager, sample_hub, sample_hub_capacity):
    """Test registering multiple hubs."""
    hub2 = Hub(
        hub_id="hub-us-east1-1",
        name="US East 1 Primary",
        region="us-east1",
        zone="us-east1-a",
        grpc_endpoint="hub-2.us-east1.ollama.local:8000",
        https_endpoint="https://hub-2.us-east1.ollama.local",
        capacity=sample_hub_capacity,
        status=HubStatus.HEALTHY,
        version="v1.0.0",
        supported_models=["llama3.2"],
        labels={},
    )
    
    federation_manager.register_hub(sample_hub)
    federation_manager.register_hub(hub2)
    
    assert len(federation_manager.hubs) == 2
    assert federation_manager.topology_version == 3


def test_unregister_hub(federation_manager, sample_hub):
    """Test unregistering a hub."""
    federation_manager.register_hub(sample_hub)
    assert len(federation_manager.hubs) == 1
    
    federation_manager.unregister_hub(sample_hub.hub_id)
    assert len(federation_manager.hubs) == 0
    assert federation_manager.topology_version == 3


def test_unregister_nonexistent_hub(federation_manager):
    """Test unregistering nonexistent hub (should not raise)."""
    # Should not raise, just log warning
    federation_manager.unregister_hub("nonexistent-hub")
    assert len(federation_manager.hubs) == 0


# ============================================================================
# Heartbeat & Health Checking Tests
# ============================================================================

def test_update_heartbeat(federation_manager, sample_hub, sample_hub_capacity):
    """Test updating hub heartbeat."""
    federation_manager.register_hub(sample_hub)
    old_heartbeat = sample_hub.last_heartbeat
    
    # Simulate time passing
    import time
    time.sleep(0.1)
    
    new_capacity = HubCapacity(
        max_concurrent_requests=100,
        current_requests=50,
        max_memory_gb=256,
        current_memory_gb=200,
        available_models=["llama3.2"],
        cpu_utilization_percent=60,
        memory_utilization_percent=78,
        network_utilization_percent=40,
    )
    
    federation_manager.update_hub_heartbeat(
        sample_hub.hub_id,
        new_capacity,
        HubStatus.HEALTHY
    )
    
    assert sample_hub.last_heartbeat > old_heartbeat
    assert sample_hub.capacity == new_capacity


def test_check_hub_health_healthy(federation_manager, sample_hub):
    """Test health check for healthy hub."""
    federation_manager.register_hub(sample_hub)
    
    is_healthy, message = federation_manager.check_hub_health(sample_hub.hub_id)
    assert is_healthy is True
    assert message == "Healthy"


def test_check_hub_health_timeout(federation_manager, sample_hub):
    """Test health check detects heartbeat timeout."""
    federation_manager.register_hub(sample_hub)
    
    # Simulate old heartbeat
    sample_hub.last_heartbeat = datetime.utcnow() - timedelta(seconds=30)
    
    is_healthy, message = federation_manager.check_hub_health(sample_hub.hub_id)
    assert is_healthy is False
    assert "timeout" in message.lower()


def test_check_hub_health_unhealthy_status(federation_manager, sample_hub):
    """Test health check detects unhealthy status."""
    federation_manager.register_hub(sample_hub)
    sample_hub.status = HubStatus.OFFLINE
    
    is_healthy, message = federation_manager.check_hub_health(sample_hub.hub_id)
    assert is_healthy is False
    assert "offline" in message.lower()


def test_check_hub_health_nonexistent(federation_manager):
    """Test health check for nonexistent hub."""
    is_healthy, message = federation_manager.check_hub_health("nonexistent-hub")
    assert is_healthy is False
    assert "not registered" in message.lower()


def test_remove_unhealthy_hubs(federation_manager, sample_hub):
    """Test removing unhealthy hubs."""
    federation_manager.register_hub(sample_hub)
    
    # Mark as unhealthy via old heartbeat
    sample_hub.last_heartbeat = datetime.utcnow() - timedelta(seconds=30)
    
    removed = federation_manager.remove_unhealthy_hubs()
    
    assert sample_hub.hub_id in removed
    assert len(federation_manager.hubs) == 0


# ============================================================================
# Topology Management Tests
# ============================================================================

def test_get_topology(federation_manager, sample_hub):
    """Test getting federation topology."""
    federation_manager.register_hub(sample_hub)
    
    topology = federation_manager.get_topology()
    
    assert len(topology.hubs) == 1
    assert topology.hubs[0] == sample_hub
    assert topology.primary_hub_id == "control-plane-1"
    assert topology.version == 2


def test_topology_snapshot_get_hub(federation_manager, sample_hub):
    """Test getting hub from topology snapshot."""
    federation_manager.register_hub(sample_hub)
    topology = federation_manager.get_topology()
    
    found_hub = topology.get_hub(sample_hub.hub_id)
    assert found_hub == sample_hub
    
    not_found = topology.get_hub("nonexistent-hub")
    assert not_found is None


def test_topology_get_healthy_hubs(federation_manager, sample_hub, sample_hub_capacity):
    """Test getting healthy hubs from topology."""
    federation_manager.register_hub(sample_hub)
    
    unhealthy_hub = Hub(
        hub_id="hub-unhealthy",
        name="Unhealthy Hub",
        region="us-west1",
        zone="us-west1-a",
        grpc_endpoint="unhealthy.ollama.local:8000",
        https_endpoint="https://unhealthy.ollama.local",
        capacity=sample_hub_capacity,
        status=HubStatus.OFFLINE,
        version="v1.0.0",
        supported_models=[],
        labels={},
    )
    federation_manager.register_hub(unhealthy_hub)
    
    topology = federation_manager.get_topology()
    healthy = topology.get_healthy_hubs()
    
    assert len(healthy) == 1
    assert healthy[0].hub_id == sample_hub.hub_id


def test_topology_get_hubs_by_region(federation_manager, sample_hub, sample_hub_capacity):
    """Test getting hubs by region."""
    federation_manager.register_hub(sample_hub)
    
    hub2 = Hub(
        hub_id="hub-us-east1-1",
        name="US East 1 Primary",
        region="us-east1",
        zone="us-east1-a",
        grpc_endpoint="hub2.us-east1.ollama.local:8000",
        https_endpoint="https://hub2.us-east1.ollama.local",
        capacity=sample_hub_capacity,
        status=HubStatus.HEALTHY,
        version="v1.0.0",
        supported_models=[],
        labels={},
    )
    federation_manager.register_hub(hub2)
    
    topology = federation_manager.get_topology()
    central_hubs = topology.get_hubs_by_region("us-central1")
    
    assert len(central_hubs) == 1
    assert central_hubs[0].hub_id == sample_hub.hub_id


# ============================================================================
# Request Routing Tests
# ============================================================================

def test_route_request_same_region(federation_manager, sample_hub):
    """Test routing request to hub in same region."""
    federation_manager.register_hub(sample_hub)
    
    routed_hub = federation_manager.route_request(
        model="llama3.2",
        client_region="us-central1"
    )
    
    assert routed_hub is not None
    assert routed_hub.hub_id == sample_hub.hub_id


def test_route_request_model_not_available(federation_manager, sample_hub):
    """Test routing when model is not available."""
    federation_manager.register_hub(sample_hub)
    
    routed_hub = federation_manager.route_request(model="gpt-4")
    
    assert routed_hub is None


def test_route_request_no_healthy_hubs(federation_manager, sample_hub):
    """Test routing when no healthy hubs available."""
    sample_hub.status = HubStatus.OFFLINE
    federation_manager.register_hub(sample_hub)
    
    routed_hub = federation_manager.route_request(model="llama3.2")
    
    assert routed_hub is None


def test_route_request_selects_least_loaded(federation_manager, sample_hub, sample_hub_capacity):
    """Test routing selects least loaded hub."""
    federation_manager.register_hub(sample_hub)
    
    # Create less loaded hub
    light_capacity = HubCapacity(
        max_concurrent_requests=100,
        current_requests=5,  # Much less loaded
        max_memory_gb=256,
        current_memory_gb=50,
        available_models=["llama3.2"],
        cpu_utilization_percent=10,
        memory_utilization_percent=20,
        network_utilization_percent=10,
    )
    
    light_hub = Hub(
        hub_id="hub-light",
        name="Light Hub",
        region="us-central1",
        zone="us-central1-b",
        grpc_endpoint="hub-light.ollama.local:8000",
        https_endpoint="https://hub-light.ollama.local",
        capacity=light_capacity,
        status=HubStatus.HEALTHY,
        version="v1.0.0",
        supported_models=["llama3.2"],
        labels={},
    )
    federation_manager.register_hub(light_hub)
    
    routed_hub = federation_manager.route_request(model="llama3.2")
    
    assert routed_hub.hub_id == light_hub.hub_id


# ============================================================================
# Hub Capacity Tests
# ============================================================================

def test_hub_available_capacity(sample_hub):
    """Test calculating available request capacity."""
    assert sample_hub.capacity.available_capacity == 75  # 100 - 25


def test_hub_available_memory(sample_hub):
    """Test calculating available memory."""
    assert sample_hub.capacity.available_memory_gb == 128  # 256 - 128


def test_hub_utilization_ratio(sample_hub):
    """Test calculating utilization ratio."""
    # (45 + 50 + 30) / 300 = 0.417
    assert 0.40 < sample_hub.capacity.utilization_ratio < 0.43


def test_hub_is_healthy(sample_hub):
    """Test hub health status."""
    assert sample_hub.is_healthy() is True
    
    sample_hub.status = HubStatus.OFFLINE
    assert sample_hub.is_healthy() is False


def test_hub_can_serve_model(sample_hub):
    """Test checking if hub can serve model."""
    assert sample_hub.can_serve_model("llama3.2") is True
    assert sample_hub.can_serve_model("gpt-4") is False


# ============================================================================
# Routing Statistics Tests
# ============================================================================

def test_get_routing_stats(federation_manager, sample_hub):
    """Test getting federation routing statistics."""
    federation_manager.register_hub(sample_hub)
    
    stats = federation_manager.get_routing_stats()
    
    assert stats["total_hubs"] == 1
    assert stats["healthy_hubs"] == 1
    assert stats["topology_version"] == 2
    assert stats["total_capacity"] == 100
    assert stats["current_usage"] == 25
    assert stats["average_utilization"] == 0.25
    assert stats["hubs_by_region"]["us-central1"] == 1


def test_routing_stats_multiple_regions(federation_manager, sample_hub, sample_hub_capacity):
    """Test routing stats with multiple regions."""
    federation_manager.register_hub(sample_hub)
    
    hub2 = Hub(
        hub_id="hub-us-east1-1",
        name="US East 1 Primary",
        region="us-east1",
        zone="us-east1-a",
        grpc_endpoint="hub2.us-east1.ollama.local:8000",
        https_endpoint="https://hub2.us-east1.ollama.local",
        capacity=sample_hub_capacity,
        status=HubStatus.HEALTHY,
        version="v1.0.0",
        supported_models=[],
        labels={},
    )
    federation_manager.register_hub(hub2)
    
    stats = federation_manager.get_routing_stats()
    
    assert stats["total_hubs"] == 2
    assert stats["hubs_by_region"]["us-central1"] == 1
    assert stats["hubs_by_region"]["us-east1"] == 1
