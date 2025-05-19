package cluster

import (
"fmt"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// mockNodeRegistry is a simplified version of NodeRegistry for testing
type mockNodeRegistry struct {
	nodes        map[string]NodeInfo
	healthChecker HealthChecker
}

func newMockNodeRegistry() *mockNodeRegistry {
	return &mockNodeRegistry{
		nodes: make(map[string]NodeInfo),
	}
}

func (m *mockNodeRegistry) RegisterNode(node NodeInfo) {
	m.nodes[node.ID] = node
}

func (m *mockNodeRegistry) GetNode(nodeID string) (NodeInfo, bool) {
	node, exists := m.nodes[nodeID]
	return node, exists
}

func (m *mockNodeRegistry) GetAllNodes() []NodeInfo {
	nodes := make([]NodeInfo, 0, len(m.nodes))
	for _, node := range m.nodes {
		nodes = append(nodes, node)
	}
	return nodes
}

func (m *mockNodeRegistry) UpdateNode(node NodeInfo) bool {
	_, exists := m.nodes[node.ID]
	if exists {
		m.nodes[node.ID] = node
		return true
	}
	return false
}

func (m *mockNodeRegistry) SetHealthChecker(checker HealthChecker) {
	m.healthChecker = checker
}

func TestNewHealthMonitor(t *testing.T) {
	registry := newMockNodeRegistry()
	checkInterval := 2 * time.Second
	timeoutThreshold := 10 * time.Second
	
	health := NewHealthMonitor(registry, checkInterval, timeoutThreshold)
	
	if health == nil {
		t.Fatal("Expected non-nil health monitor")
	}
	
	if health.checkInterval != checkInterval {
		t.Errorf("Expected check interval %v, got %v", checkInterval, health.checkInterval)
	}
	
	if health.nodeTimeoutThreshold != timeoutThreshold {
		t.Errorf("Expected timeout threshold %v, got %v", timeoutThreshold, health.nodeTimeoutThreshold)
	}
	
	if health.client == nil {
		t.Error("Expected health monitor to have an HTTP client")
	}
	
	if health.healthStatusCache == nil {
		t.Error("Expected health monitor to have a status cache")
	}
}

func TestStartStop(t *testing.T) {
	registry := newMockNodeRegistry()
	health := NewHealthMonitor(registry, 100*time.Millisecond, 1*time.Second)
	
	// Start the health monitor
	err := health.Start()
	if err != nil {
		t.Fatalf("Failed to start health monitor: %v", err)
	}
	
	// Verify the health monitor is registered
	if registry.healthChecker != health {
		t.Error("Health monitor did not register itself as the health checker")
	}
	
	// Stop the health monitor
	err = health.Stop()
	if err != nil {
		t.Fatalf("Failed to stop health monitor: %v", err)
	}
}

func TestCheckNodeHealth(t *testing.T) {
	registry := newMockNodeRegistry()
	health := NewHealthMonitor(registry, 1*time.Second, 5*time.Second)
	
	// Create test server that returns a health status
	testStatus := NodeHealthStatus{
		NodeID:          "test-node",
		Status:          NodeStatusBusy,
		CPUUsagePercent: 90.5,
		MemoryUsageMB:   2048,
		LatencyMS:       15,
		ActiveRequests:  5,
	}
	
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/health" {
			t.Errorf("Expected request to /api/health, got %s", r.URL.Path)
			http.NotFound(w, r)
			return
		}
		
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(testStatus)
	}))
	defer server.Close()
	
	// Parse server URL to get host and port
	serverURL, err := net.Parse(server.URL)
	if err != nil {
		t.Fatalf("Failed to parse server URL: %v", err)
	}
	
	host, port, err := net.SplitHostPort(serverURL.Host)
	if err != nil {
		t.Fatalf("Failed to split host/port: %v", err)
	}
	
	// Create a node that points to our test server
	node := NodeInfo{
		ID:     "test-node",
		Name:   "Test Node",
		Addr:   net.ParseIP(host),
		ApiPort: atoi(port),
		Status: NodeStatusOnline,
	}
	
	// Check the node's health
	status, err := health.CheckNodeHealth(node)
	
	if err != nil {
		t.Fatalf("Health check failed: %v", err)
	}
	
	if status != NodeStatusBusy {
		t.Errorf("Expected status %s, got %s", NodeStatusBusy, status)
	}
	
	// Verify the status was cached
	cachedStatus, err := health.GetNodeHealthStatus("test-node")
	if err != nil {
		t.Fatalf("Failed to get cached status: %v", err)
	}
	
	if cachedStatus.CPUUsagePercent != 90.5 {
		t.Errorf("Expected CPU usage 90.5%%, got %.1f%%", cachedStatus.CPUUsagePercent)
	}
	
	if cachedStatus.MemoryUsageMB != 2048 {
		t.Errorf("Expected memory usage 2048MB, got %dMB", cachedStatus.MemoryUsageMB)
	}
	
	if cachedStatus.ActiveRequests != 5 {
		t.Errorf("Expected 5 active requests, got %d", cachedStatus.ActiveRequests)
	}
}

// Helper function to convert port string to int
func atoi(s string) int {
	var port int
	fmt.Sscanf(s, "%d", &port)
	return port
}