package cluster

import (
	"context"
	"fmt"
	"net"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/ollama/ollama/cluster"
)

// TestClusterConfig defines the configuration for a test cluster
type TestClusterConfig struct {
	NodeCount        int
	CoordinatorCount int
	WorkerCount      int
	BasePort         int
	NetworkLatency   time.Duration
	PacketLossRate   float64
	ModelSize        int // Size in GB
}

// DefaultTestClusterConfig returns a default configuration for test clusters
func DefaultTestClusterConfig() TestClusterConfig {
	return TestClusterConfig{
		NodeCount:        3,
		CoordinatorCount: 1,
		WorkerCount:      2,
		BasePort:         20000,
		NetworkLatency:   10 * time.Millisecond,
		PacketLossRate:   0.0,
		ModelSize:        7, // 7GB model by default
	}
}

// TestNode represents a node in the test cluster
type TestNode struct {
	ID             string
	Role           cluster.NodeRole
	Status         cluster.NodeStatus
	ClusterPort    int
	APIPort        int
	Config         cluster.ClusterConfig
	Registry       *cluster.NodeRegistry
	Discovery      *cluster.DiscoveryService
	HealthMonitor  *cluster.HealthMonitor
	ctx            context.Context
	cancel         context.CancelFunc
	readyWaitGroup sync.WaitGroup
}

// TestCluster represents a complete test cluster environment
type TestCluster struct {
	Nodes            []*TestNode
	Config           TestClusterConfig
	NetworkSimulator *NetworkSimulator
	ctx              context.Context
	cancel           context.CancelFunc
}
// NetworkSimulator simulates network conditions for testing
type NetworkSimulator struct {
	latency     time.Duration
	packetLoss  float64
	failureMap  map[string]bool // Map of node IDs that should simulate failures
	failureLock sync.RWMutex
}

// NewNetworkSimulator creates a new network simulator
func NewNetworkSimulator(latency time.Duration, packetLoss float64) *NetworkSimulator {
	return &NetworkSimulator{
		latency:    latency,
		packetLoss: packetLoss,
		failureMap: make(map[string]bool),
	}
}

// SimulateLatency adds artificial delay to simulate network latency
func (ns *NetworkSimulator) SimulateLatency() {
	if ns.latency > 0 {
		time.Sleep(ns.latency)
	}
}

// ShouldDropPacket determines if a packet should be dropped based on packet loss rate
func (ns *NetworkSimulator) ShouldDropPacket() bool {
	// Simple random-based packet loss simulation
	if ns.packetLoss <= 0 {
		return false
	}
	
	// Generate random number between 0 and 1
	// For simplicity in tests, we use a deterministic approach based on time
	randVal := float64(time.Now().UnixNano()%1000) / 1000.0
	return randVal < ns.packetLoss
}

// SimulateNodeFailure marks a node as failed
func (ns *NetworkSimulator) SimulateNodeFailure(nodeID string) {
	ns.failureLock.Lock()
	defer ns.failureLock.Unlock()
	ns.failureMap[nodeID] = true
}

// RecoverNode recovers a previously failed node
func (ns *NetworkSimulator) RecoverNode(nodeID string) {
	ns.failureLock.Lock()
	defer ns.failureLock.Unlock()
	delete(ns.failureMap, nodeID)
}

// IsNodeFailed checks if a node is currently simulated as failed
func (ns *NetworkSimulator) IsNodeFailed(nodeID string) bool {
	ns.failureLock.RLock()
	defer ns.failureLock.RUnlock()
	return ns.failureMap[nodeID]
}

// NewTestCluster creates a new test cluster with the specified configuration
func NewTestCluster(config TestClusterConfig) *TestCluster {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &TestCluster{
		Nodes:            make([]*TestNode, 0, config.NodeCount),
// Setup initializes all nodes in the test cluster
func (tc *TestCluster) Setup(t *testing.T) error {
	// Create coordinator nodes
	for i := 0; i < tc.Config.CoordinatorCount; i++ {
		node := tc.createNode(fmt.Sprintf("coord-%d", i), cluster.NodeRoleCoordinator, i)
		tc.Nodes = append(tc.Nodes, node)
	}
	
	// Create worker nodes
	for i := 0; i < tc.Config.WorkerCount; i++ {
		node := tc.createNode(fmt.Sprintf("worker-%d", i), cluster.NodeRoleWorker, 
			i+tc.Config.CoordinatorCount)
		tc.Nodes = append(tc.Nodes, node)
	}
	
	// Start all nodes
	for _, node := range tc.Nodes {
		if err := tc.startNode(node); err != nil {
			return fmt.Errorf("failed to start node %s: %w", node.ID, err)
		}
	}
	
	// Wait for all nodes to be ready
	for _, node := range tc.Nodes {
		node.readyWaitGroup.Wait()
	}
	
	t.Logf("Test cluster with %d nodes set up successfully", len(tc.Nodes))
	return nil
}

// Teardown stops all nodes in the test cluster
func (tc *TestCluster) Teardown() error {
	tc.cancel() // Cancel the context to stop all nodes
	
	// Stop all nodes explicitly
	for _, node := range tc.Nodes {
		if node.Discovery != nil {
			node.Discovery.Stop()
		}
		if node.HealthMonitor != nil {
			node.HealthMonitor.Stop()
		}
	}
	
	return nil
}

// createNode initializes a new test node
func (tc *TestCluster) createNode(id string, role cluster.NodeRole, index int) *TestNode {
	// Calculate ports based on index
	clusterPort := tc.Config.BasePort + (index * 2)
	apiPort := clusterPort + 1
	
	// Create node config
	config := cluster.DefaultClusterConfig()
	config.Enabled = true
	config.NodeName = id
	config.NodeRole = role
	config.ClusterPort = clusterPort
	config.APIPort = apiPort
	config.ClusterHost = "127.0.0.1" // Use loopback for tests
	
	// For discovery, use manual method and provide list of all nodes
	config.Discovery.Method = cluster.DiscoveryMethodManual
	for i := 0; i < tc.Config.NodeCount; i++ {
		nodePort := tc.Config.BasePort + (i * 2)
		config.Discovery.NodeList = append(
			config.Discovery.NodeList,
			fmt.Sprintf("127.0.0.1:%d", nodePort),
		)
	}
	
	// Create context for this node
	ctx, cancel := context.WithCancel(tc.ctx)
	
	node := &TestNode{
		ID:          id,
		Role:        role,
		Status:      cluster.NodeStatusOffline,
		ClusterPort: clusterPort,
		APIPort:     apiPort,
		Config:      config,
		ctx:         ctx,
		cancel:      cancel,
	}
	
	// Add one to the wait group to track when the node is ready
	node.readyWaitGroup.Add(1)
	
	return node
}

// startNode initializes and starts a node's services
func (tc *TestCluster) startNode(node *TestNode) error {
	// Create registry
	registry := cluster.NewNodeRegistry(1*time.Second, 5*time.Second)
	node.Registry = registry
	
	// Create local node info
	localNode := cluster.NodeInfo{
		ID:         node.ID,
		Name:       node.ID,
		Role:       node.Role,
		Status:     cluster.NodeStatusStarting,
		Addr:       net.ParseIP("127.0.0.1"),
		ClusterPort: node.ClusterPort,
		ApiPort:    node.APIPort,
	}
	
	// Create discovery service
	discovery := cluster.NewDiscoveryService(node.Config.Discovery, registry, localNode)
	node.Discovery = discovery
	
	// Create health monitor
	healthMonitor := cluster.NewHealthMonitor(registry, 1*time.Second, 5*time.Second)
	node.HealthMonitor = healthMonitor
	
	// Start services
	if err := discovery.Start(); err != nil {
		return fmt.Errorf("failed to start discovery service: %w", err)
	}
	
	if err := healthMonitor.Start(); err != nil {
		discovery.Stop()
		return fmt.Errorf("failed to start health monitor: %w", err)
	}
	
	// Update node status to online
	localNode.Status = cluster.NodeStatusOnline
	discovery.UpdateLocalNodeInfo(localNode)
	
	// Mark node as ready
	node.Status = cluster.NodeStatusOnline
	node.readyWaitGroup.Done()
	
	return nil
}

// SimulateNodeFailure causes a node to simulate failure
func (tc *TestCluster) SimulateNodeFailure(nodeID string) error {
	var targetNode *TestNode
	
	// Find the node
	for _, node := range tc.Nodes {
		if node.ID == nodeID {
			targetNode = node
			break
		}
	}
	
	if targetNode == nil {
		return fmt.Errorf("node %s not found", nodeID)
	}
	
	// Mark the node as failed in the simulator
	tc.NetworkSimulator.SimulateNodeFailure(nodeID)
	
	// Update node status
	targetNode.Status = cluster.NodeStatusOffline
	
	return nil
}

// RecoverNode recovers a previously failed node
func (tc *TestCluster) RecoverNode(nodeID string) error {
	var targetNode *TestNode
	
	// Find the node
	for _, node := range tc.Nodes {
		if node.ID == nodeID {
			targetNode = node
			break
		}
	}
	
	if targetNode == nil {
		return fmt.Errorf("node %s not found", nodeID)
	}
	
	// Mark the node as recovered in the simulator
	tc.NetworkSimulator.RecoverNode(nodeID)
	
	// Update node status
	targetNode.Status = cluster.NodeStatusOnline
	
	// Update node info in the registry
	localNode := cluster.NodeInfo{
		ID:         targetNode.ID,
		Name:       targetNode.ID,
		Role:       targetNode.Role,
		Status:     cluster.NodeStatusOnline,
		Addr:       net.ParseIP("127.0.0.1"),
		ClusterPort: targetNode.ClusterPort,
		ApiPort:    targetNode.APIPort,
	}
	
	targetNode.Discovery.UpdateLocalNodeInfo(localNode)
	
	return nil
}

// CollectResults gathers test results from all nodes
func (tc *TestCluster) CollectResults() map[string]interface{} {
	results := make(map[string]interface{})
	
	// Collect active nodes
	activeNodes := []string{}
	for _, node := range tc.Nodes {
		if node.Status == cluster.NodeStatusOnline {
			activeNodes = append(activeNodes, node.ID)
		}
	}
	results["active_nodes"] = activeNodes
	
	// Collect registry info from coordinator
	if len(tc.Nodes) > 0 && tc.Nodes[0].Role == cluster.NodeRoleCoordinator {
		coordinator := tc.Nodes[0]
		if coordinator.Registry != nil {
			allNodes := coordinator.Registry.GetAllNodes()
			nodeInfos := make([]map[string]interface{}, 0, len(allNodes))
			
			for _, node := range allNodes {
				nodeInfo := map[string]interface{}{
					"id":     node.ID,
					"name":   node.Name,
					"role":   string(node.Role),
					"status": string(node.Status),
				}
				nodeInfos = append(nodeInfos, nodeInfo)
			}
			
			results["registry_nodes"] = nodeInfos
		}
	}
	
	return results
}
		Config:           config,
		NetworkSimulator: NewNetworkSimulator(config.NetworkLatency, config.PacketLossRate),
		ctx:              ctx,
		cancel:           cancel,
	}
}