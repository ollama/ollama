package cluster

import (
	"testing"
	"time"
	"net"
	"sync"
)

func TestNewNodeRegistry(t *testing.T) {
	checkInterval := 1 * time.Second
	timeoutThreshold := 5 * time.Second
	
	registry := NewNodeRegistry(checkInterval, timeoutThreshold)
	
	if registry == nil {
		t.Fatal("Expected non-nil registry")
	}
	
	if registry.healthCheckInterval != checkInterval {
		t.Errorf("Expected health check interval %v, got %v", 
			checkInterval, registry.healthCheckInterval)
	}
	
	if registry.nodeTimeoutInterval != timeoutThreshold {
		t.Errorf("Expected node timeout threshold %v, got %v", 
			timeoutThreshold, registry.nodeTimeoutInterval)
	}
	
	if registry.nodes == nil {
		t.Error("Expected non-nil nodes map")
	}
	
	if len(registry.nodes) != 0 {
		t.Errorf("Expected empty registry, got %d nodes", len(registry.nodes))
	}
}

func TestRegisterNode(t *testing.T) {
	registry := NewNodeRegistry(1*time.Second, 5*time.Second)
	
	// Create a test node
	node := NodeInfo{
		ID:       "test-node-1",
		Name:     "Test Node 1",
		Role:     NodeRoleMixed,
		Status:   NodeStatusOnline,
		Addr:     net.ParseIP("192.168.1.100"),
		ApiPort:  11434,
		ClusterPort: 12094,
		LastHeartbeat: time.Now(),
	}
	
	// Register the node
	registry.RegisterNode(node)
	
	// Check that the node was registered
	if len(registry.nodes) != 1 {
		t.Errorf("Expected 1 node in registry, got %d", len(registry.nodes))
	}
	
	// Verify node data was stored correctly
	storedNode, exists := registry.nodes[node.ID]
	if !exists {
		t.Fatalf("Node not found in registry")
	}
	
	if storedNode.Name != "Test Node 1" {
		t.Errorf("Expected node name 'Test Node 1', got '%s'", storedNode.Name)
	}
	
	if storedNode.Role != NodeRoleMixed {
		t.Errorf("Expected node role %s, got %s", NodeRoleMixed, storedNode.Role)
	}
	
	// Register a second node
	node2 := node
	node2.ID = "test-node-2"
	node2.Name = "Test Node 2"
	
	registry.RegisterNode(node2)
	
	// Check that we now have two nodes
	if len(registry.nodes) != 2 {
		t.Errorf("Expected 2 nodes in registry, got %d", len(registry.nodes))
	}
	
	// Update an existing node
	updatedNode := node
func TestUnregisterNode(t *testing.T) {
	registry := NewNodeRegistry(1*time.Second, 5*time.Second)
	
	// Create and register two test nodes
	node1 := NodeInfo{
		ID:   "test-node-1",
		Name: "Test Node 1",
	}
	
	node2 := NodeInfo{
		ID:   "test-node-2",
		Name: "Test Node 2",
	}
	
	registry.RegisterNode(node1)
	registry.RegisterNode(node2)
	
	// Verify both nodes are in the registry
	if len(registry.nodes) != 2 {
		t.Fatalf("Expected 2 nodes in registry, got %d", len(registry.nodes))
	}
	
	// Unregister one node
	registry.UnregisterNode("test-node-1")
	
	// Check that only one node remains
	if len(registry.nodes) != 1 {
		t.Errorf("Expected 1 node in registry after unregistering, got %d", len(registry.nodes))
	}
	
	// Check that the right node was removed
	_, exists := registry.nodes["test-node-1"]
	if exists {
		t.Error("Node should have been removed but is still in registry")
	}
	
	_, exists = registry.nodes["test-node-2"]
	if !exists {
		t.Error("Wrong node was removed from registry")
	}
	
	// Unregister non-existent node (should not cause error)
	registry.UnregisterNode("non-existent-node")
	
	// Check that registry state hasn't changed
	if len(registry.nodes) != 1 {
		t.Errorf("Expected registry to still have 1 node, got %d", len(registry.nodes))
	}
}

func TestGetNode(t *testing.T) {
	registry := NewNodeRegistry(1*time.Second, 5*time.Second)
	
	// Create and register a test node
	node := NodeInfo{
		ID:     "test-node",
		Name:   "Test Node",
		Role:   NodeRoleWorker,
		Status: NodeStatusOnline,
	}
	
	registry.RegisterNode(node)
	
	// Get the node
	fetchedNode, exists := registry.GetNode("test-node")
	if !exists {
		t.Fatal("Expected node to exist in registry")
	}
	
	// Verify node data
	if fetchedNode.Name != "Test Node" {
		t.Errorf("Expected node name 'Test Node', got '%s'", fetchedNode.Name)
	}
	
	if fetchedNode.Role != NodeRoleWorker {
		t.Errorf("Expected node role %s, got %s", NodeRoleWorker, fetchedNode.Role)
	}
	
	// Try getting a non-existent node
	_, exists = registry.GetNode("non-existent-node")
	if exists {
		t.Error("Non-existent node should not be found")
	}
}

func TestGetAllNodes(t *testing.T) {
	registry := NewNodeRegistry(1*time.Second, 5*time.Second)
	
	// Create and register test nodes
	node1 := NodeInfo{
		ID:   "test-node-1",
		Name: "Test Node 1",
	}
	
	node2 := NodeInfo{
		ID:   "test-node-2",
		Name: "Test Node 2",
	}
	
	registry.RegisterNode(node1)
	registry.RegisterNode(node2)
	
	// Get all nodes
	nodes := registry.GetAllNodes()
	
	// Check count
func TestUpdateNode(t *testing.T) {
	registry := NewNodeRegistry(1*time.Second, 5*time.Second)
	
	// Create and register a node
	node := NodeInfo{
		ID:     "test-node",
		Name:   "Test Node",
		Status: NodeStatusOnline,
	}
	
	registry.RegisterNode(node)
	
	// Update existing node
	updatedNode := node
	updatedNode.Name = "Updated Node"
	updatedNode.Status = NodeStatusBusy
	
	success := registry.UpdateNode(updatedNode)
	
	if !success {
		t.Error("UpdateNode should return true for existing node")
	}
	
	// Verify node was updated
	storedNode, exists := registry.GetNode("test-node")
	if !exists {
		t.Fatal("Node should exist after update")
	}
	
	if storedNode.Name != "Updated Node" {
		t.Errorf("Expected name 'Updated Node', got '%s'", storedNode.Name)
	}
	
	if storedNode.Status != NodeStatusBusy {
		t.Errorf("Expected status %s, got %s", NodeStatusBusy, storedNode.Status)
	}
	
	// Try updating non-existent node
	nonExistentNode := NodeInfo{
		ID:   "non-existent",
		Name: "Non-existent Node",
	}
	
	success = registry.UpdateNode(nonExistentNode)
	
	if success {
		t.Error("UpdateNode should return false for non-existent node")
	}
}

func TestGetAvailableWorkers(t *testing.T) {
	registry := NewNodeRegistry(1*time.Second, 5*time.Second)
	
	// Create test nodes with different roles and statuses
	worker1 := NodeInfo{
		ID:     "worker1",
		Name:   "Worker 1",
		Role:   NodeRoleWorker,
		Status: NodeStatusOnline,
	}
	
	worker2 := NodeInfo{
		ID:     "worker2",
		Name:   "Worker 2",
		Role:   NodeRoleMixed,  // Mixed roles can also be workers
		Status: NodeStatusOnline,
	}
	
	worker3 := NodeInfo{
		ID:     "worker3",
		Name:   "Worker 3", 
		Role:   NodeRoleWorker,
		Status: NodeStatusOffline,  // Offline worker
	}
	
	coordinator := NodeInfo{
		ID:     "coord",
		Name:   "Coordinator",
		Role:   NodeRoleCoordinator,
		Status: NodeStatusOnline,
	}
	
	// Register all nodes
	registry.RegisterNode(worker1)
	registry.RegisterNode(worker2)
	registry.RegisterNode(worker3)
	registry.RegisterNode(coordinator)
	
	// Get available workers
	workers := registry.GetAvailableWorkers()
	
	// Should only include online workers (worker1, worker2)
	if len(workers) != 2 {
		t.Errorf("Expected 2 available workers, got %d", len(workers))
	}
	
	// Check for the correct workers
	foundWorker1 := false
	foundWorker2 := false
	
	for _, node := range workers {
		if node.ID == "worker1" {
			foundWorker1 = true
		}
		if node.ID == "worker2" {
			foundWorker2 = true
		}
		if node.ID == "worker3" || node.ID == "coord" {
			t.Errorf("Unexpected node in available workers: %s", node.ID)
		}
	}
	
	if !foundWorker1 {
		t.Error("Worker1 not found in available workers")
	}
	
	if !foundWorker2 {
		t.Error("Worker2 not found in available workers")
	}
}

func TestGetCoordinators(t *testing.T) {
	registry := NewNodeRegistry(1*time.Second, 5*time.Second)
	
	// Create test nodes with different roles and statuses
	worker := NodeInfo{
		ID:     "worker",
		Name:   "Worker",
		Role:   NodeRoleWorker,
		Status: NodeStatusOnline,
	}
	
	coord1 := NodeInfo{
		ID:     "coord1",
		Name:   "Coordinator 1",
		Role:   NodeRoleCoordinator,
		Status: NodeStatusOnline,
	}
	
	coord2 := NodeInfo{
		ID:     "coord2",
		Name:   "Coordinator 2",
		Role:   NodeRoleMixed,  // Mixed roles can also be coordinators
		Status: NodeStatusOnline,
	}
	
	coord3 := NodeInfo{
		ID:     "coord3",
		Name:   "Coordinator 3",
		Role:   NodeRoleCoordinator,
		Status: NodeStatusOffline,  // Offline coordinator
	}
	
	// Register all nodes
	registry.RegisterNode(worker)
func TestAddEventHandler(t *testing.T) {
	registry := NewNodeRegistry(1*time.Second, 5*time.Second)
	
	// Create channels to track event handler calls
	eventReceived := make(chan ClusterEvent, 5)
	
	// Register event handler
	registry.AddEventHandler(func(event ClusterEvent) {
		eventReceived <- event
	})
	
	// Create and register a node (should trigger an event)
	node := NodeInfo{
		ID:     "test-node",
		Name:   "Test Node",
		Status: NodeStatusOnline,
	}
	
	registry.RegisterNode(node)
	
	// Wait for event or timeout
	var receivedEvent ClusterEvent
	select {
	case receivedEvent = <-eventReceived:
		// Event received
	case <-time.After(100 * time.Millisecond):
		t.Fatal("Timed out waiting for event")
	}
	
	// Verify event details
	if receivedEvent.Type != "node_joined" {
		t.Errorf("Expected event type 'node_joined', got '%s'", receivedEvent.Type)
	}
	
	if receivedEvent.NodeID != "test-node" {
		t.Errorf("Expected node ID 'test-node', got '%s'", receivedEvent.NodeID)
	}
	
	// Update node status (should trigger another event)
	updatedNode := node
	updatedNode.Status = NodeStatusBusy
	
	registry.RegisterNode(updatedNode)
	
	// Wait for second event or timeout
	select {
	case receivedEvent = <-eventReceived:
		// Event received
	case <-time.After(100 * time.Millisecond):
		t.Fatal("Timed out waiting for status change event")
	}
	
	// Verify event details
	if receivedEvent.Type != "node_status_changed" {
		t.Errorf("Expected event type 'node_status_changed', got '%s'", receivedEvent.Type)
	}
	
	// Unregister node (should trigger another event)
	registry.UnregisterNode("test-node")
	
	// Wait for third event or timeout
	select {
	case receivedEvent = <-eventReceived:
		// Event received
	case <-time.After(100 * time.Millisecond):
		t.Fatal("Timed out waiting for node_left event")
	}
	
	// Verify event details
	if receivedEvent.Type != "node_left" {
		t.Errorf("Expected event type 'node_left', got '%s'", receivedEvent.Type)
	}
}

// mockHealthChecker implements the HealthChecker interface for testing
type mockHealthChecker struct {
	statusToReturn NodeStatus
	errorToReturn  error
	checkCalled    bool
	lastCheckedID  string
}

func (m *mockHealthChecker) CheckNodeHealth(node NodeInfo) (NodeStatus, error) {
	m.checkCalled = true
	m.lastCheckedID = node.ID
	return m.statusToReturn, m.errorToReturn
}

func TestCheckNodesHealth(t *testing.T) {
	// Create a registry with a short timeout to test timeout detection
	registry := NewNodeRegistry(10*time.Millisecond, 50*time.Millisecond)
	
	// Create two nodes, one with recent heartbeat, one with old heartbeat
	recentNode := NodeInfo{
		ID:            "recent-node",
		Name:          "Recent Node",
		Status:        NodeStatusOnline,
		LastHeartbeat: time.Now(),
	}
	
	oldNode := NodeInfo{
		ID:            "old-node",
		Name:          "Old Node",
		Status:        NodeStatusOnline,
		LastHeartbeat: time.Now().Add(-100 * time.Millisecond), // Older than timeout
	}
	
	// Register nodes
	registry.RegisterNode(recentNode)
	registry.RegisterNode(oldNode)
	
	// Run a health check cycle manually
	registry.checkNodesHealth()
	
	// Verify the old node was marked offline
	oldNodeInfo, _ := registry.GetNode("old-node")
	if oldNodeInfo.Status != NodeStatusOffline {
		t.Errorf("Expected old node status to be %s, got %s", NodeStatusOffline, oldNodeInfo.Status)
	}
	
	// Recent node should still be online
	recentNodeInfo, _ := registry.GetNode("recent-node")
	if recentNodeInfo.Status != NodeStatusOnline {
		t.Errorf("Expected recent node status to remain %s, got %s", NodeStatusOnline, recentNodeInfo.Status)
	}
	
	// Test with a custom health checker
	mockChecker := &mockHealthChecker{
		statusToReturn: NodeStatusBusy,
	}
	
	registry.SetHealthChecker(mockChecker)
	
	// Reset recent node status to online
	recentNode.Status = NodeStatusOnline
	registry.UpdateNode(recentNode)
	
	// Run another health check cycle
	registry.checkNodesHealth()
	
	// Verify the mock checker was called for the recent node
	if !mockChecker.checkCalled {
		t.Error("Expected health checker to be called")
	}
	
	if mockChecker.lastCheckedID != "recent-node" {
		t.Errorf("Expected health check on recent node, but got '%s'", mockChecker.lastCheckedID)
	}
	
	// Verify the recent node status was updated based on health checker
	recentNodeInfo, _ = registry.GetNode("recent-node")
	if recentNodeInfo.Status != NodeStatusBusy {
		t.Errorf("Expected node status to be updated to %s, got %s", NodeStatusBusy, recentNodeInfo.Status)
	}
}
	registry.RegisterNode(coord1)
	registry.RegisterNode(coord2)
	registry.RegisterNode(coord3)
	
	// Get active coordinators
	coordinators := registry.GetCoordinators()
	
	// Should only include online coordinators (coord1, coord2)
	if len(coordinators) != 2 {
		t.Errorf("Expected 2 available coordinators, got %d", len(coordinators))
	}
	
	// Check for the correct coordinators
	foundCoord1 := false
	foundCoord2 := false
	
	for _, node := range coordinators {
		if node.ID == "coord1" {
			foundCoord1 = true
		}
		if node.ID == "coord2" {
			foundCoord2 = true
		}
		if node.ID == "coord3" || node.ID == "worker" {
			t.Errorf("Unexpected node in active coordinators: %s", node.ID)
		}
	}
	
	if !foundCoord1 {
		t.Error("Coordinator1 not found in active coordinators")
	}
	
	if !foundCoord2 {
		t.Error("Coordinator2 not found in active coordinators")
	}
}
	if len(nodes) != 2 {
		t.Errorf("Expected 2 nodes, got %d", len(nodes))
	}
	
	// Check that both nodes are present
	foundNode1 := false
	foundNode2 := false
	
	for _, node := range nodes {
		if node.ID == "test-node-1" {
			foundNode1 = true
		}
		if node.ID == "test-node-2" {
			foundNode2 = true
		}
	}
	
	if !foundNode1 {
		t.Error("Node 1 not found in GetAllNodes result")
	}
	
	if !foundNode2 {
		t.Error("Node 2 not found in GetAllNodes result")
	}
	
	// Test with empty registry
	emptyRegistry := NewNodeRegistry(1*time.Second, 5*time.Second)
	emptyNodes := emptyRegistry.GetAllNodes()
	
	if len(emptyNodes) != 0 {
		t.Errorf("Expected empty result from GetAllNodes, got %d nodes", len(emptyNodes))
	}
}
	updatedNode.Status = NodeStatusBusy
	
	registry.RegisterNode(updatedNode)
	
	// Check that the node was updated
	if len(registry.nodes) != 2 {
		t.Errorf("Expected still 2 nodes in registry, got %d", len(registry.nodes))
	}
	
	storedNode, exists = registry.nodes[node.ID]
	if !exists {
		t.Fatalf("Node not found in registry after update")
	}
	
	if storedNode.Status != NodeStatusBusy {
		t.Errorf("Expected node status %s, got %s", NodeStatusBusy, storedNode.Status)
	}
}