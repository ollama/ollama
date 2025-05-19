package cluster

import (
	"encoding/json"
	"net"
	"testing"
	"time"
)

// mockRegistry implements a test version of the node registry
type mockRegistry struct {
	nodes            map[string]NodeInfo
	registerCalled   bool
	lastRegisteredID string
}

func newMockRegistry() *mockRegistry {
	return &mockRegistry{
		nodes: make(map[string]NodeInfo),
	}
}

func (m *mockRegistry) RegisterNode(node NodeInfo) {
	m.nodes[node.ID] = node
	m.registerCalled = true
	m.lastRegisteredID = node.ID
}

func (m *mockRegistry) ListNodes() []NodeInfo {
	nodes := make([]NodeInfo, 0, len(m.nodes))
	for _, node := range m.nodes {
		nodes = append(nodes, node)
	}
	return nodes
}

func TestNewDiscoveryService(t *testing.T) {
	// Prepare test data
	registry := newMockRegistry()
	localNode := NodeInfo{
		ID:       "test-node-1",
		Name:     "Test Node 1",
		Role:     NodeRoleMixed,
		Status:   NodeStatusOnline,
		Addr:     net.ParseIP("192.168.1.100"),
		ApiPort:  11434,
		ClusterPort: 12094,
	}

	config := DiscoveryConfig{
		Method:            DiscoveryMethodMulticast,
		MulticastAddress:  DefaultMulticastAddress,
		NodeList:          []string{},
		HeartbeatInterval: 5 * time.Second,
		NodeTimeoutInterval: 15 * time.Second,
	}

	// Create the discovery service
	discovery := NewDiscoveryService(config, registry, localNode)

	// Verify the service was created properly
	if discovery == nil {
		t.Fatal("Expected non-nil discovery service")
	}
	
	if discovery.config.Method != DiscoveryMethodMulticast {
		t.Errorf("Expected method %s, got %s", DiscoveryMethodMulticast, discovery.config.Method)
	}
	
func TestUpdateLocalNodeInfo(t *testing.T) {
	// Prepare test data
	registry := newMockRegistry()
	localNode := NodeInfo{
		ID:       "test-node-1",
		Name:     "Test Node 1",
		Role:     NodeRoleMixed,
		Status:   NodeStatusOnline,
	}

	config := DiscoveryConfig{
		Method:            DiscoveryMethodManual,
		HeartbeatInterval: 1 * time.Second,
	}

	// Create the discovery service
	discovery := NewDiscoveryService(config, registry, localNode)
	
	// Update local node info
	updatedNode := localNode
	updatedNode.Status = NodeStatusBusy
	updatedNode.Name = "Updated Node 1"
	
	discovery.UpdateLocalNodeInfo(updatedNode)
	
	// Verify the local node info was updated
	if discovery.localNode.Status != NodeStatusBusy {
		t.Errorf("Expected status %s, got %s", NodeStatusBusy, discovery.localNode.Status)
	}
	
	if discovery.localNode.Name != "Updated Node 1" {
		t.Errorf("Expected name 'Updated Node 1', got '%s'", discovery.localNode.Name)
	}
	
	// Verify the node was registered
	if !registry.registerCalled {
		t.Error("Expected RegisterNode to be called")
	}
	
	if registry.lastRegisteredID != "test-node-1" {
		t.Errorf("Expected registered node ID 'test-node-1', got '%s'", registry.lastRegisteredID)
	}
}

func TestProcessDiscoveryMessage(t *testing.T) {
	// Prepare test data
	registry := newMockRegistry()
	localNode := NodeInfo{
		ID:       "local-node",
		Name:     "Local Node",
		Role:     NodeRoleMixed,
		Status:   NodeStatusOnline,
	}

	config := DiscoveryConfig{
		Method:            DiscoveryMethodMulticast,
		HeartbeatInterval: 1 * time.Second,
	}

	// Create the discovery service
	discovery := NewDiscoveryService(config, registry, localNode)
	
	// Create a test heartbeat message
	remoteNode := NodeInfo{
		ID:       "remote-node",
		Name:     "Remote Node",
		Role:     NodeRoleWorker,
		Status:   NodeStatusOnline,
		Addr:     net.ParseIP("192.168.1.200"),
		ApiPort:  11434,
		ClusterPort: 12094,
	}
	
	heartbeat := HeartbeatMessage{
		MessageType: "heartbeat",
		NodeInfo:    remoteNode,
	}
	
	// Marshal to JSON
	data, err := json.Marshal(heartbeat)
	if err != nil {
		t.Fatalf("Failed to marshal heartbeat: %v", err)
	}
	
	// Process the message
	discovery.processDiscoveryMessage(data)
	
	// Verify the node was registered
	if !registry.registerCalled {
		t.Error("Expected RegisterNode to be called")
	}
	
	if registry.lastRegisteredID != "remote-node" {
		t.Errorf("Expected registered node ID 'remote-node', got '%s'", registry.lastRegisteredID)
	}
	
// This test only checks start/stop lifecycle, not actual network activity
func TestDiscoveryStartStop(t *testing.T) {
	// Prepare test data
	registry := newMockRegistry()
	localNode := NodeInfo{
		ID:       "test-node-1",
		Name:     "Test Node 1",
		Role:     NodeRoleMixed,
		Status:   NodeStatusOnline,
		Addr:     net.ParseIP("127.0.0.1"), // Use loopback to avoid network
	}

	config := DiscoveryConfig{
		Method:            DiscoveryMethodManual, // Use manual to avoid actual multicast
		NodeList:          []string{},
		HeartbeatInterval: 1 * time.Second,
	}

	// Create the discovery service
	discovery := NewDiscoveryService(config, registry, localNode)
	
	// Start the service
	err := discovery.Start()
	if err != nil {
		t.Fatalf("Failed to start discovery service: %v", err)
	}
	
	// Verify the service is running
	if !discovery.isRunning {
		t.Error("Expected discovery service to be running")
	}
	
	// Verify local node was registered
	if !registry.registerCalled {
		t.Error("Expected RegisterNode to be called on start")
	}
	
	if registry.lastRegisteredID != "test-node-1" {
		t.Errorf("Expected registered node ID 'test-node-1', got '%s'", registry.lastRegisteredID)
	}
	
	// Stop the service
	err = discovery.Stop()
	if err != nil {
		t.Fatalf("Failed to stop discovery service: %v", err)
	}
	
	// Verify the service is stopped
	if discovery.isRunning {
		t.Error("Expected discovery service to not be running after stop")
	}
	
	// Test double start (should fail)
	discovery = NewDiscoveryService(config, registry, localNode)
	
	err = discovery.Start()
	if err != nil {
		t.Fatalf("Failed to start discovery service: %v", err)
	}
	
	err = discovery.Start()
	if err == nil {
		t.Error("Expected error when starting already running service")
	}
	
	// Test double stop (should fail)
	discovery.Stop()
	
	err = discovery.Stop()
	if err == nil {
		t.Error("Expected error when stopping already stopped service")
	}
}

func TestManualDiscovery(t *testing.T) {
	// Prepare test data
	registry := newMockRegistry()
	localNode := NodeInfo{
		ID:       "local-node",
		Name:     "Local Node",
		Role:     NodeRoleMixed,
		Status:   NodeStatusOnline,
	}

	// Setup with manual discovery method
	config := DiscoveryConfig{
		Method:            DiscoveryMethodManual,
		NodeList:          []string{"192.168.1.200:12094", "192.168.1.201:12094"},
		HeartbeatInterval: 1 * time.Second,
	}

	// Create the discovery service
	discovery := NewDiscoveryService(config, registry, localNode)
	
	// Start the service (this should trigger connection attempts to nodes in the list)
	err := discovery.Start()
	if err != nil {
		t.Fatalf("Failed to start manual discovery service: %v", err)
	}
	
	// We can't easily test the actual node connections since that would involve network
	// Instead, verify the service started successfully and registers the local node
	if !discovery.isRunning {
		t.Error("Expected discovery service to be running")
	}
	
	if !registry.registerCalled {
		t.Error("Expected RegisterNode to be called for local node")
	}
	
	if registry.lastRegisteredID != "local-node" {
		t.Errorf("Expected registered node ID to be 'local-node', got '%s'", registry.lastRegisteredID)
	}
	
	// Cleanup
	discovery.Stop()
}

func TestGetLocalNodeInfo(t *testing.T) {
	// Prepare test data
	registry := newMockRegistry()
	localNode := NodeInfo{
		ID:       "test-node-1",
		Name:     "Test Node 1",
		Role:     NodeRoleMixed,
		Status:   NodeStatusOnline,
	}

	config := DiscoveryConfig{
		Method: DiscoveryMethodManual,
	}

	// Create the discovery service
	discovery := NewDiscoveryService(config, registry, localNode)
	
	// Get local node info
	info := discovery.GetLocalNodeInfo()
	
	// Verify the info matches
	if info.ID != "test-node-1" {
		t.Errorf("Expected ID 'test-node-1', got '%s'", info.ID)
	}
	
	if info.Name != "Test Node 1" {
		t.Errorf("Expected name 'Test Node 1', got '%s'", info.Name)
	}
	
	if info.Role != NodeRoleMixed {
		t.Errorf("Expected role '%s', got '%s'", NodeRoleMixed, info.Role)
	}
	
	if info.Status != NodeStatusOnline {
		t.Errorf("Expected status '%s', got '%s'", NodeStatusOnline, info.Status)
	}
}
	// Test ignoring our own heartbeat
	ownHeartbeat := HeartbeatMessage{
		MessageType: "heartbeat",
		NodeInfo:    localNode,
	}
	
	data, err = json.Marshal(ownHeartbeat)
	if err != nil {
		t.Fatalf("Failed to marshal own heartbeat: %v", err)
	}
	
	// Reset registry state
	registry.registerCalled = false
	registry.lastRegisteredID = ""
	
	// Process own message
	discovery.processDiscoveryMessage(data)
	
	// Verify no registration occurred
	if registry.registerCalled {
		t.Error("Expected RegisterNode not to be called for own heartbeat")
	}
}
	if discovery.config.MulticastAddress != DefaultMulticastAddress {
		t.Errorf("Expected multicast address %s, got %s", DefaultMulticastAddress, discovery.config.MulticastAddress)
	}
	
	if discovery.isRunning {
		t.Error("New discovery service should not be running")
	}
	
	if discovery.localNode.ID != "test-node-1" {
		t.Errorf("Expected local node ID 'test-node-1', got '%s'", discovery.localNode.ID)
	}
}