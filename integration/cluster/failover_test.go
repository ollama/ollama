package cluster

import (
	"testing"
	"time"

	"github.com/ollama/ollama/cluster"
	"github.com/ollama/ollama/cluster/scheduler"
)

// TestNodeFailureRecovery verifies that the cluster can recover from node failures
// and redistribute workloads as needed.
func TestNodeFailureRecovery(t *testing.T) {
	// Create test cluster configuration with extra nodes for recovery
	config := DefaultTestClusterConfig()
	config.CoordinatorCount = 1
	config.WorkerCount = 4  // Multiple workers for better failover testing
	config.NodeCount = 5    // Total of 5 nodes
	config.ModelSize = 20   // 20GB model to distribute
	
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
	
	// Find all worker nodes
	var workerNodes []cluster.NodeInfo
	allNodeResources := make(map[string]scheduler.NodeResources)
	
	for _, node := range coordinator.Registry.GetAllNodes() {
		if node.Role == cluster.NodeRoleWorker && node.Status == cluster.NodeStatusOnline {
			workerNodes = append(workerNodes, node)
			
			// For testing, assume each worker has 8GB available RAM
			allNodeResources[node.ID] = scheduler.NodeResources{
				AvailableRAM: 8 * 1024, // 8GB in MB
				AvailableGPU: 1,
				GPUMemoryMB:  6 * 1024, // 6GB VRAM
			}
		}
	}
	
	// Need at least 3 workers for proper failover testing
	if len(workerNodes) < 3 {
		t.Fatalf("Not enough worker nodes for failover test, found %d", len(workerNodes))
	}
	
	// Create execution planner
	planner := scheduler.NewExecutionPlanner()
	
	// Define model properties
	modelInfo := cluster.ModelInfo{
		Name:        "test-recovery-model",
// Create initial execution plan
	executionPlan, err := planner.CreateExecutionPlan(modelInfo, workerNodes, allNodeResources)
	if err != nil {
		t.Fatalf("Failed to create execution plan: %v", err)
	}
	
	// Make sure we have a valid plan with nodes assigned
	if len(executionPlan.NodeAssignments) < 3 {
		t.Fatalf("Expected at least 3 node assignments, got %d", len(executionPlan.NodeAssignments))
	}
	
	// Keep track of initial node assignments
	initialAssignmentByNode := make(map[string]scheduler.NodeAssignment)
	for _, assignment := range executionPlan.NodeAssignments {
		initialAssignmentByNode[assignment.NodeID] = assignment
	}
	
	// Select a node to fail (pick the second one to avoid the first one which might be special)
	var failedNodeID string
	var failedNodeAssignment scheduler.NodeAssignment
	
	i := 0
	for nodeID, assignment := range initialAssignmentByNode {
		if i == 1 { // Pick the second node
			failedNodeID = nodeID
			failedNodeAssignment = assignment
			break
		}
		i++
	}
	
	if failedNodeID == "" {
		t.Fatal("Failed to select a node to simulate failure")
	}
	
	t.Logf("Simulating failure of node %s", failedNodeID)
	
	// Create an execution state
	execState := &scheduler.ExecutionState{
		ModelID:         modelInfo.Name,
		NodeAssignments: make([]scheduler.NodeAssignment, 0, len(executionPlan.NodeAssignments)),
		Status:          scheduler.ExecutionStatusRunning,
		StartedAt:       time.Now(),
	}
	
	// Copy assignments to state
	for _, assignment := range executionPlan.NodeAssignments {
		execState.NodeAssignments = append(execState.NodeAssignments, assignment)
	}
	
	// Create recovery manager
	recoveryManager := scheduler.NewRecoveryManager()
	
	// Simulate node failure
	err = testCluster.SimulateNodeFailure(failedNodeID)
	if err != nil {
		t.Fatalf("Failed to simulate node failure: %v", err)
	}
	
	// Update execution state to reflect the failure
	for i := range execState.NodeAssignments {
		if execState.NodeAssignments[i].NodeID == failedNodeID {
			execState.NodeAssignments[i].Status = scheduler.NodeAssignmentStatusFailed
			execState.NodeAssignments[i].LastHealthy = time.Now().Add(-5 * time.Minute)
			break
		}
	}
	
	// Detect node failures
	failures := recoveryManager.DetectNodeFailures(execState)
	
	// Should detect our simulated failure
	if len(failures) != 1 {
// Find backup nodes (workers that aren't part of the initial assignment)
	var backupNodes []cluster.NodeInfo
	
	for _, node := range workerNodes {
		if _, used := initialAssignmentByNode[node.ID]; !used {
			backupNodes = append(backupNodes, node)
		}
	}
	
	if len(backupNodes) == 0 {
		t.Fatal("No backup nodes available for recovery")
	}
	
	// Create recovery plan
	recoveryPlan, err := recoveryManager.CreateRecoveryPlan(
		execState, 
		[]scheduler.NodeAssignment{failedNodeAssignment}, 
		backupNodes,
		allNodeResources,
	)
	
	if err != nil {
		t.Fatalf("Failed to create recovery plan: %v", err)
	}
	
	// Recovery plan should reassign the failed node's work
	if recoveryPlan == nil {
		t.Fatal("No recovery plan created")
	}
	
	backupNodeID, ok := recoveryPlan.ReassignedNodes[failedNodeID]
	if !ok {
		t.Errorf("Recovery plan doesn't reassign failed node %s", failedNodeID)
	} else {
		t.Logf("Work from node %s reassigned to backup node %s", failedNodeID, backupNodeID)
	}
	
	// Execute recovery
	err = recoveryManager.ExecuteRecovery(execState, recoveryPlan)
	if err != nil {
		t.Fatalf("Failed to execute recovery: %v", err)
	}
	
	// Execution state should now be recovering
	if execState.Status != scheduler.ExecutionStatusRecovering {
		t.Errorf("Expected execution status to be Recovering, got %s", execState.Status)
	}
	
	// Verify backup node is added to the assignments
	foundBackup := false
	for _, assignment := range execState.NodeAssignments {
		if assignment.NodeID == backupNodeID {
			foundBackup = true
			if assignment.Status != scheduler.NodeAssignmentStatusPending {
				t.Errorf("Expected backup node status to be Pending, got %s", assignment.Status)
			}
			break
		}
	}
	
	if !foundBackup {
		t.Error("Backup node not added to execution state")
	}
	
	// Verify all layers from failed node are reassigned
	allLayersReassigned := true
	failedNodeLayers := make(map[int]bool)
	for _, layer := range failedNodeAssignment.Layers {
		failedNodeLayers[layer] = true
	}
	
	// Check if all failed node layers are in the recovery plan
	for _, layer := range recoveryPlan.RecoveredLayers {
		delete(failedNodeLayers, layer)
	}
	
	if len(failedNodeLayers) > 0 {
		t.Errorf("%d layers from failed node weren't reassigned", len(failedNodeLayers))
		allLayersReassigned = false
	}
	
	// Mark recovery as complete
	if allLayersReassigned {
		execState.Status = scheduler.ExecutionStatusRunning
		t.Log("Recovery completed successfully")
	}
}
		t.Fatalf("Expected to detect 1 failure, got %d", len(failures))
	}
	
	if failures[0].NodeID != failedNodeID {
		t.Errorf("Expected failure of node %s, got %s", failedNodeID, failures[0].NodeID)
	}
		TotalSizeGB: config.ModelSize,
		LayerCount:  40,
	}