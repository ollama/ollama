package cluster

import (
	"testing"
	"time"

	"github.com/ollama/ollama/cluster"
	"github.com/ollama/ollama/cluster/model"
)

// TestDistributedModelLoading verifies that a model can be properly partitioned
// and loaded across multiple nodes in the cluster.
func TestDistributedModelLoading(t *testing.T) {
	// Create test cluster configuration
	config := DefaultTestClusterConfig()
	config.CoordinatorCount = 1
	config.WorkerCount = 3  // Need multiple workers for distribution
	config.NodeCount = 4    // Total of 4 nodes
	config.ModelSize = 16   // 16GB model to distribute
	
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
	
	// Get the coordinator node
	if len(testCluster.Nodes) == 0 || testCluster.Nodes[0].Role != cluster.NodeRoleCoordinator {
		t.Fatal("Coordinator node not found")
	}
	coordinator := testCluster.Nodes[0]
	
	// Define model properties
	modelInfo := model.ModelInfo{
		Name:        "test-distributed-model",
		TotalSizeGB: config.ModelSize,
		LayerCount:  32,
	}
	
	// Get all worker nodes
	var workerNodes []cluster.NodeInfo
	nodeResources := make(map[string]cluster.NodeResourceInfo)
	
	for _, node := range coordinator.Registry.GetAllNodes() {
		if node.Role == cluster.NodeRoleWorker && node.Status == cluster.NodeStatusOnline {
			workerNodes = append(workerNodes, node)
			
			// For testing, assume each worker has 8GB available RAM
			nodeResources[node.ID] = cluster.NodeResourceInfo{
				NodeID:       node.ID,
				AvailableRAM: 8 * 1024, // 8GB in MB
			}
		}
	}
	
	// We need at least 2 workers for the test
	if len(workerNodes) < 2 {
		t.Fatalf("Not enough worker nodes for test, found %d", len(workerNodes))
	}
	
	// Create model partitioner
	partitioner := model.NewShardedModelPartitioner()
	
	// Calculate partitions
	partitions, err := partitioner.CalculatePartitions(modelInfo, nodeResources)
	if err != nil {
		t.Fatalf("Failed to calculate model partitions: %v", err)
	}
	
	// Verify partitioning results
	if len(partitions) == 0 {
		t.Fatal("No partitions created")
	}
	
	// Multiple partitions should be created for a model this large
	if len(partitions) < 2 {
		t.Errorf("Expected multiple partitions, got %d", len(partitions))
	}
	
	// Verify total memory allocation matches model size
	totalMemoryMB := int64(0)
	for _, partition := range partitions {
		totalMemoryMB += partition.MemoryMB
	}
	
	expectedTotalMB := modelInfo.TotalSizeGB * 1024
	if totalMemoryMB != expectedTotalMB {
		t.Errorf("Expected total memory %d MB, got %d MB", 
			expectedTotalMB, totalMemoryMB)
	}
	
	// Verify all layers are assigned
	layerMap := make(map[int]bool)
	for _, partition := range partitions {
		for _, layerIdx := range partition.LayerIndices {
			if layerMap[layerIdx] {
				t.Errorf("Layer %d assigned to multiple partitions", layerIdx)
			}
			layerMap[layerIdx] = true
		}
	}
	
	// Check that all layers are accounted for
	for i := 0; i < modelInfo.LayerCount; i++ {
		if !layerMap[i] {
			t.Errorf("Layer %d not assigned to any partition", i)
		}
	}
	
	// Simulate model loading by creating a loader and assigning partitions
	loader := model.NewDistributedModelLoader()
	
	// Register partitions with the loader
	for _, partition := range partitions {
		err = loader.RegisterPartition(modelInfo.Name, partition)
		if err != nil {
			t.Errorf("Failed to register partition: %v", err)
		}
	}
	
	// Verify loader has all partitions
	partitionCount := loader.GetPartitionCount(modelInfo.Name)
	if partitionCount != len(partitions) {
		t.Errorf("Loader has %d partitions, expected %d", partitionCount, len(partitions))
	}
	
	// Check loader reports model as ready
	isReady := loader.IsModelReady(modelInfo.Name)
	if !isReady {
		t.Error("Model should be reported as ready but isn't")
	}
}