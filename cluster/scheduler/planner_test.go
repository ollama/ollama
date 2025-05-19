package scheduler

import (
	"testing"
	"time"
	
	"github.com/ollama/ollama/cluster"
)

// Mock node for testing
type mockNode struct {
	ID        string
	Score     float32
	ModelSize int64
	Available bool
}

// Mock model for testing
type mockModel struct {
	Name       string
	Size       int64
	PartsCount int
}

func TestNewExecutionPlanner(t *testing.T) {
	planner := NewExecutionPlanner()
	
	if planner == nil {
		t.Fatal("Expected non-nil planner")
	}
	
	if planner.strategy != DefaultPlanningStrategy {
		t.Errorf("Expected default strategy %s, got %s", 
			DefaultPlanningStrategy, planner.strategy)
	}
	
	if planner.lastPlan != nil {
		t.Error("New planner should not have a last plan")
	}
}

func TestCreateExecutionPlan(t *testing.T) {
	planner := NewExecutionPlanner()
	
	// Define test nodes
	nodes := []cluster.NodeInfo{
		{
			ID:     "node-1",
			Name:   "Node 1",
			Role:   cluster.NodeRoleWorker,
			Status: cluster.NodeStatusOnline,
		},
		{
			ID:     "node-2",
			Name:   "Node 2",
			Role:   cluster.NodeRoleWorker,
			Status: cluster.NodeStatusOnline,
		},
		{
			ID:     "node-3",
			Name:   "Node 3",
			Role:   cluster.NodeRoleWorker,
			Status: cluster.NodeStatusOnline,
		},
	}
	
	// Define node resources
	resources := map[string]NodeResources{
		"node-1": {
			AvailableRAM: 16 * 1024, // 16GB in MB
			AvailableGPU: 1,
			GPUMemoryMB:  8 * 1024,  // 8GB in MB
		},
		"node-2": {
			AvailableRAM: 32 * 1024, // 32GB in MB
			AvailableGPU: 2,
			GPUMemoryMB:  16 * 1024, // 16GB in MB
		},
		"node-3": {
			AvailableRAM: 16 * 1024, // 16GB in MB
			AvailableGPU: 1,
			GPUMemoryMB:  8 * 1024,  // 8GB in MB
		},
	}
	
	// Define model that needs to be distributed
	modelInfo := cluster.ModelInfo{
		Name:        "test-model",
		TotalSizeGB: 24, // 24GB model
		LayerCount:  32,
	}
	
	// Create execution plan
	plan, err := planner.CreateExecutionPlan(modelInfo, nodes, resources)
	
	// Should succeed
	if err != nil {
		t.Fatalf("Failed to create execution plan: %v", err)
	}
	
	// Plan should be valid
	if plan == nil {
		t.Fatal("Expected non-nil execution plan")
	}
	
	// Should use all nodes for a model this large
	if len(plan.NodeAssignments) != 3 {
		t.Errorf("Expected 3 node assignments, got %d", len(plan.NodeAssignments))
	}
	
	// All nodes should be assigned
	assignedNodeIDs := make(map[string]bool)
	for _, assignment := range plan.NodeAssignments {
		assignedNodeIDs[assignment.NodeID] = true
	}
	
	for _, node := range nodes {
		if !assignedNodeIDs[node.ID] {
			t.Errorf("Node %s was not assigned in the plan", node.ID)
		}
	}
	
	// Total allocated memory should match model size
	totalAllocatedMB := int64(0)
	for _, assignment := range plan.NodeAssignments {
		totalAllocatedMB += assignment.MemoryMB
	}
	
	expectedMemoryMB := modelInfo.TotalSizeGB * 1024
	if totalAllocatedMB != expectedMemoryMB {
		t.Errorf("Expected total memory %d MB, got %d MB", expectedMemoryMB, totalAllocatedMB)
	}
	
	// Larger nodes should be assigned more work
	nodeAssignmentMap := make(map[string]NodeAssignment)
	for _, assignment := range plan.NodeAssignments {
		nodeAssignmentMap[assignment.NodeID] = assignment
	}
	
	node1Memory := nodeAssignmentMap["node-1"].MemoryMB
	node2Memory := nodeAssignmentMap["node-2"].MemoryMB
	
	if node2Memory <= node1Memory {
		t.Errorf("Expected node-2 (larger) to be assigned more memory than node-1, but got %d <= %d",
			node2Memory, node1Memory)
	}
}