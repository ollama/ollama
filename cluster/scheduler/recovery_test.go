package scheduler

import (
	"errors"
	"testing"
	"time"

	"github.com/ollama/ollama/cluster"
)

// Mock execution state for testing
func createMockExecutionState() *ExecutionState {
	return &ExecutionState{
		ModelID: "test-model",
		NodeAssignments: []NodeAssignment{
			{
				NodeID:      "node-1",
				Status:      NodeAssignmentStatusActive,
				Layers:      []int{0, 1, 2, 3, 4, 5},
				MemoryMB:    8 * 1024,  // 8GB
				StartedAt:   time.Now(),
				LastHealthy: time.Now(),
			},
			{
				NodeID:      "node-2",
				Status:      NodeAssignmentStatusActive,
				Layers:      []int{6, 7, 8, 9, 10, 11},
				MemoryMB:    8 * 1024,  // 8GB
				StartedAt:   time.Now(),
				LastHealthy: time.Now(),
			},
			{
				NodeID:      "node-3",
				Status:      NodeAssignmentStatusActive,
				Layers:      []int{12, 13, 14, 15, 16, 17},
				MemoryMB:    8 * 1024,  // 8GB
				StartedAt:   time.Now(),
				LastHealthy: time.Now(),
			},
		},
		Status:    ExecutionStatusRunning,
		StartedAt: time.Now(),
	}
}

func TestNewRecoveryManager(t *testing.T) {
	manager := NewRecoveryManager()
	
	if manager == nil {
		t.Fatal("Expected non-nil recovery manager")
	}
	
	if manager.recoveryStrategy != DefaultRecoveryStrategy {
		t.Errorf("Expected default strategy %s, got %s", 
			DefaultRecoveryStrategy, manager.recoveryStrategy)
	}
}

func TestDetectNodeFailure(t *testing.T) {
	manager := NewRecoveryManager()
	state := createMockExecutionState()
	
	// All nodes are healthy initially
	failures := manager.DetectNodeFailures(state)
	if len(failures) != 0 {
		t.Errorf("Expected no failures in healthy state, found %d", len(failures))
	}
	
	// Make node-2 unhealthy (last healthy time too old)
	for i := range state.NodeAssignments {
		if state.NodeAssignments[i].NodeID == "node-2" {
			state.NodeAssignments[i].LastHealthy = time.Now().Add(-2 * NodeHealthTimeout)
			break
		}
	}
	
	// Should detect one failure
	failures = manager.DetectNodeFailures(state)
	if len(failures) != 1 {
		t.Errorf("Expected 1 failure, found %d", len(failures))
	}
	
	if len(failures) > 0 && failures[0].NodeID != "node-2" {
		t.Errorf("Expected node-2 to be detected as failed, got %s", failures[0].NodeID)
	}
}
// Mark node-3 as failed explicitly
	for i := range state.NodeAssignments {
		if state.NodeAssignments[i].NodeID == "node-3" {
			state.NodeAssignments[i].Status = NodeAssignmentStatusFailed
			break
		}
	}
	
	// Should detect two failures now
	failures = manager.DetectNodeFailures(state)
	if len(failures) != 2 {
		t.Errorf("Expected 2 failures, found %d", len(failures))
	}
	
	// Check that both nodes are included
	failedNodes := make(map[string]bool)
	for _, failure := range failures {
		failedNodes[failure.NodeID] = true
	}
	
	if !failedNodes["node-2"] || !failedNodes["node-3"] {
		t.Error("Expected both node-2 and node-3 to be detected as failed")
	}
}

func TestCreateRecoveryPlan(t *testing.T) {
	manager := NewRecoveryManager()
	state := createMockExecutionState()
	
	// Make node-2 fail
	failedNodeID := "node-2"
	var failedAssignment NodeAssignment
	
	for i := range state.NodeAssignments {
		if state.NodeAssignments[i].NodeID == failedNodeID {
			state.NodeAssignments[i].Status = NodeAssignmentStatusFailed
			failedAssignment = state.NodeAssignments[i]
			break
		}
	}
	
	// Available backup nodes
	backupNodes := []cluster.NodeInfo{
		{
			ID:     "backup-1",
			Name:   "Backup Node 1",
			Role:   cluster.NodeRoleWorker,
			Status: cluster.NodeStatusOnline,
		},
	}
	
	// Resource info for backup nodes
	resources := map[string]NodeResources{
		"backup-1": {
			AvailableRAM: 16 * 1024, // 16GB
			AvailableGPU: 1,
			GPUMemoryMB:  12 * 1024,
		},
	}
	
	// Create recovery plan
	plan, err := manager.CreateRecoveryPlan(state, []NodeAssignment{failedAssignment}, backupNodes, resources)
	
	// Should succeed
	if err != nil {
		t.Fatalf("Failed to create recovery plan: %v", err)
	}
	
	// Plan should be valid
	if plan == nil {
		t.Fatal("Expected non-nil recovery plan")
	}
	
	// Should reassign the failed node's work to the backup node
	if plan.ReassignedNodes["node-2"] != "backup-1" {
		t.Errorf("Expected node-2's work to be reassigned to backup-1, got %s", 
			plan.ReassignedNodes["node-2"])
	}
	
	// Recovery plan should include the layers from the failed node
	if !containsAllLayers(plan.RecoveredLayers, failedAssignment.Layers) {
		t.Errorf("Recovery plan doesn't include all layers from failed node")
	}
}

// Helper function to check if all layers are contained in a slice
func containsAllLayers(container, contained []int) bool {
	if len(contained) == 0 {
		return true
	}
	
	layerSet := make(map[int]bool)
	for _, layer := range container {
		layerSet[layer] = true
	}
	
	for _, layer := range contained {
		if !layerSet[layer] {
			return false
		}
	}
	
	return true
}

func TestExecuteRecovery(t *testing.T) {
	manager := NewRecoveryManager()
	state := createMockExecutionState()
	
	// Make node-2 fail
	for i := range state.NodeAssignments {
		if state.NodeAssignments[i].NodeID == "node-2" {
			state.NodeAssignments[i].Status = NodeAssignmentStatusFailed
			break
		}
	}
	
	// Create a mock recovery plan
	recoveryPlan := &RecoveryPlan{
		ReassignedNodes: map[string]string{
			"node-2": "backup-1",
		},
		RecoveredLayers: []int{6, 7, 8, 9, 10, 11},
		BackupAssignments: []NodeAssignment{
			{
				NodeID:  "backup-1",
				Status:  NodeAssignmentStatusPending,
				Layers:  []int{6, 7, 8, 9, 10, 11},
				MemoryMB: 8 * 1024,
			},
		},
	}
	
	// Execute recovery
	err := manager.ExecuteRecovery(state, recoveryPlan)
	
	// Should succeed
	if err != nil {
		t.Fatalf("Failed to execute recovery: %v", err)
	}
	
	// State should be updated with the recovered assignment
	foundBackup := false
	for _, assignment := range state.NodeAssignments {
		if assignment.NodeID == "backup-1" {
			foundBackup = true
			if assignment.Status != NodeAssignmentStatusPending {
				t.Errorf("Expected backup node status to be Pending, got %s", assignment.Status)
			}
			if !containsAllLayers(assignment.Layers, recoveryPlan.RecoveredLayers) {
				t.Error("Backup assignment doesn't contain all recovered layers")
			}
		}
	}
	
	if !foundBackup {
		t.Error("Backup node assignment not added to execution state")
	}
	
	// Execution status should be recovering
	if state.Status != ExecutionStatusRecovering {
		t.Errorf("Expected execution status Recovering, got %s", state.Status)
	}
}