package cluster

import (
	"testing"
	"time"

	"github.com/ollama/ollama/cluster"
)

// TestBasicClusterFormation verifies that a multi-node cluster can successfully form
// and all nodes discover each other.
func TestBasicClusterFormation(t *testing.T) {
	// Create test cluster configuration
	config := DefaultTestClusterConfig()
	config.CoordinatorCount = 1
	config.WorkerCount = 2
	config.NodeCount = 3 // Total of 3 nodes
	
	// Create the test cluster
	testCluster := NewTestCluster(config)
	
	// Initialize the cluster
	err := testCluster.Setup(t)
	if err != nil {
		t.Fatalf("Failed to set up test cluster: %v", err)
	}
	
	// Ensure we tear down the cluster at the end of the test
	defer testCluster.Teardown()
	
	// Wait a bit for discovery to complete
	time.Sleep(2 * time.Second)
	
	// Get the coordinator node (should be the first node)
	if len(testCluster.Nodes) == 0 || testCluster.Nodes[0].Role != cluster.NodeRoleCoordinator {
		t.Fatal("Coordinator node not found")
	}
	
	coordinator := testCluster.Nodes[0]
	
	// Check that the coordinator's registry contains all expected nodes
	allNodes := coordinator.Registry.GetAllNodes()
	if len(allNodes) != config.NodeCount {
		t.Errorf("Expected %d nodes in registry, found %d", config.NodeCount, len(allNodes))
	}
	
	// Verify that all nodes are properly discovered
	foundCoordinators := 0
	foundWorkers := 0
	
	for _, node := range allNodes {
		switch node.Role {
		case cluster.NodeRoleCoordinator:
			foundCoordinators++
		case cluster.NodeRoleWorker:
			foundWorkers++
		}
		
		if node.Status != cluster.NodeStatusOnline {
			t.Errorf("Node %s has status %s, expected %s", 
				node.ID, node.Status, cluster.NodeStatusOnline)
		}
	}
	
	// Verify correct number of each node type
	if foundCoordinators != config.CoordinatorCount {
		t.Errorf("Expected %d coordinator(s), found %d", 
			config.CoordinatorCount, foundCoordinators)
	}
	
	if foundWorkers != config.WorkerCount {
		t.Errorf("Expected %d worker(s), found %d", 
			config.WorkerCount, foundWorkers)
	}
	
	// Check that health monitoring is working
	healthyNodes := coordinator.HealthMonitor.GetHealthyNodes()
	if len(healthyNodes) != config.NodeCount - 1 { // Excludes self
		t.Errorf("Expected %d healthy nodes, found %d", 
			config.NodeCount - 1, len(healthyNodes))
	}
}