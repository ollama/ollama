package scheduler

import (
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/ollama/ollama/cluster"
	"github.com/ollama/ollama/cluster/model"
)

// RecoveryStrategy defines how to handle node failures
type RecoveryStrategy string

const (
	// RecoveryReschedule moves tasks to healthy nodes
	RecoveryReschedule RecoveryStrategy = "reschedule"
	
	// RecoveryWaitAndRetry waits for nodes to recover
	RecoveryWaitAndRetry RecoveryStrategy = "wait_and_retry"
	
	// RecoveryContinuePartial continues with remaining nodes
	RecoveryContinuePartial RecoveryStrategy = "continue_partial"
	
	// RecoveryAbort aborts the execution on failure
	RecoveryAbort RecoveryStrategy = "abort"
)

// RecoveryOptions configures how failures are handled
type RecoveryOptions struct {
	// Strategy determines the approach to node failures
	Strategy RecoveryStrategy
	
	// MaxRetries limits the number of retry attempts
	MaxRetries int
	
	// RetryDelay is the time to wait between retries
	RetryDelay time.Duration
	
	// HealthCheckInterval is how often to check node health
	HealthCheckInterval time.Duration
	
	// EnableCheckpointing saves state for recovery
	EnableCheckpointing bool
}

// DefaultRecoveryOptions provides sensible defaults
var DefaultRecoveryOptions = RecoveryOptions{
	Strategy:            RecoveryReschedule,
	MaxRetries:          3,
	RetryDelay:          5 * time.Second,
	HealthCheckInterval: 10 * time.Second,
	EnableCheckpointing: true,
}

// FailureEvent represents a node or task failure
type FailureEvent struct {
	// Type indicates what failed
	Type string
	
	// NodeID identifies the failed node
	NodeID string
	
	// TaskID identifies the failed task (if applicable)
	TaskID string
	
	// Time when the failure occurred
	Time time.Time
	
	// Error details the failure reason
	Error error
}

// RecoveryManager handles fault tolerance during execution
type RecoveryManager struct {
	// options configures the recovery behavior
	options RecoveryOptions
	
	// registry provides access to available nodes
	registry *cluster.NodeRegistry
	
	// partitioner provides access to model partitioning
	partitioner *model.ModelPartitioner
	
	// executor executes tasks
	executor *Executor
	
	// activeRecoveries tracks ongoing recovery operations
	activeRecoveries map[string]*RecoveryOperation
	
	// mu protects the activeRecoveries map
	mu sync.RWMutex
	
	// healthChecker monitors node health
	healthChecker *cluster.HealthMonitor
	
	// stopCh signals the background monitoring to stop
	stopCh chan struct{}
	
	// eventCh receives failure notifications
	eventCh chan FailureEvent
}

// RecoveryOperation tracks the state of a recovery process
type RecoveryOperation struct {
	// ID uniquely identifies this recovery operation
	ID string
	
	// ModelID is the model being recovered
	ModelID string
	
	// FailedNodeID is the node that failed
	FailedNodeID string
	
	// AffectedTasks are tasks impacted by the failure
	AffectedTasks []string
	
	// StartTime when recovery began
	StartTime time.Time
	
	// CompletionTime when recovery finished (if complete)
	CompletionTime time.Time
	
	// Status of this recovery operation
	Status string
	
	// ReassignedPartitions tracks partition reassignments
	ReassignedPartitions map[string]string // partitionID → new nodeID
}

// NewRecoveryManager creates a new failure recovery manager
func NewRecoveryManager(registry *cluster.NodeRegistry, partitioner *model.ModelPartitioner, 
	executor *Executor, healthChecker *cluster.HealthMonitor, options RecoveryOptions) *RecoveryManager {
	
	rm := &RecoveryManager{
		options:          options,
		registry:         registry,
		partitioner:      partitioner,
		executor:         executor,
		activeRecoveries: make(map[string]*RecoveryOperation),
		healthChecker:    healthChecker,
		stopCh:           make(chan struct{}),
		eventCh:          make(chan FailureEvent, 100),
	}
	
	// Register for task failure notifications
	executor.SetTaskCompletionCallback(func(result *TaskResult) {
		if !result.Success {
			rm.eventCh <- FailureEvent{
				Type:   "task",
				NodeID: result.Task.NodeID,
				TaskID: result.Task.ID,
				Time:   time.Now(),
				Error:  result.Error,
			}
		}
	})
	
	// Start monitoring node health
	go rm.monitorHealth()
	
	// Start handling failure events
	go rm.handleFailures()
	
	return rm
}
// monitorHealth periodically checks node health
func (rm *RecoveryManager) monitorHealth() {
	ticker := time.NewTicker(rm.options.HealthCheckInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-rm.stopCh:
			return
		case <-ticker.C:
			// Get all nodes and check their status
			nodes := rm.registry.GetAllNodes()
			
			for _, node := range nodes {
				if node.Status == cluster.NodeStatusOffline {
					// Report node failure
					rm.eventCh <- FailureEvent{
						Type:   "node",
						NodeID: node.ID,
						Time:   time.Now(),
						Error:  fmt.Errorf("node %s is offline", node.ID),
					}
				}
			}
		}
	}
}

// handleFailures processes incoming failure events
func (rm *RecoveryManager) handleFailures() {
	for event := range rm.eventCh {
		switch event.Type {
		case "node":
			rm.handleNodeFailure(event)
		case "task":
			rm.handleTaskFailure(event)
		}
	}
}

// handleNodeFailure addresses a failed node
func (rm *RecoveryManager) handleNodeFailure(event FailureEvent) {
	fmt.Printf("Node failure detected: %s (%v)\n", event.NodeID, event.Error)
	
	// Check if we're already handling this node failure
	rm.mu.RLock()
	for _, recovery := range rm.activeRecoveries {
		if recovery.FailedNodeID == event.NodeID {
			rm.mu.RUnlock()
			return // Already handling this failure
		}
	}
	rm.mu.RUnlock()
	
	// Create new recovery operation
	recoveryID := fmt.Sprintf("recovery-%s-%d", event.NodeID, time.Now().UnixNano())
	recovery := &RecoveryOperation{
		ID:                  recoveryID,
		FailedNodeID:        event.NodeID,
		StartTime:           time.Now(),
		Status:              "in_progress",
		ReassignedPartitions: make(map[string]string),
	}
	
	rm.mu.Lock()
	rm.activeRecoveries[recoveryID] = recovery
	rm.mu.Unlock()
	
	// Start recovery process based on strategy
	go func() {
		var err error
		
		switch rm.options.Strategy {
		case RecoveryReschedule:
			err = rm.reschedulePartitions(recovery)
		case RecoveryWaitAndRetry:
			err = rm.waitAndRetry(recovery)
		case RecoveryContinuePartial:
			err = rm.continuePartial(recovery)
		case RecoveryAbort:
			err = rm.abortExecution(recovery)
		default:
			err = fmt.Errorf("unsupported recovery strategy: %s", rm.options.Strategy)
		}
		
		// Update recovery status
		rm.mu.Lock()
		defer rm.mu.Unlock()
		
		if err != nil {
			recovery.Status = "failed"
			fmt.Printf("Recovery operation %s failed: %v\n", recoveryID, err)
		} else {
			recovery.Status = "completed"
			recovery.CompletionTime = time.Now()
			fmt.Printf("Recovery operation %s completed successfully\n", recoveryID)
		}
	}()
}

// handleTaskFailure addresses a failed task
func (rm *RecoveryManager) handleTaskFailure(event FailureEvent) {
	fmt.Printf("Task failure detected: %s on node %s (%v)\n",
		event.TaskID, event.NodeID, event.Error)
	
	// For now, we just retry the task a limited number of times
	// This is a simplified implementation and would be more robust in production
	
	// Retrieve task result to check retry count
	result, exists := rm.executor.GetTaskResult(event.TaskID)
	if !exists {
		fmt.Printf("Cannot find result for failed task: %s\n", event.TaskID)
		return
	}
	
	// Check if retry count is in the output
	retryCount := 0
	if countVal, exists := result.Output["retry_count"]; exists {
		if count, ok := countVal.(int); ok {
			retryCount = count
		}
	}
	
	if retryCount >= rm.options.MaxRetries {
		fmt.Printf("Task %s exceeded maximum retry count (%d), giving up\n",
			event.TaskID, rm.options.MaxRetries)
		return
	}
	
	// Wait before retrying
	time.Sleep(rm.options.RetryDelay)
	
	// Retry the task
	fmt.Printf("Retrying task %s (attempt %d of %d)\n",
		event.TaskID, retryCount+1, rm.options.MaxRetries)
	
	// In a real implementation, this would re-execute the task
	// For now just simulate task retry logic
}
// reschedulePartitions moves partitions from a failed node to healthy nodes
func (rm *RecoveryManager) reschedulePartitions(recovery *RecoveryOperation) error {
	// Get all model partitions assigned to the failed node
	affectedPartitions := make(map[string][]model.ModelPartition)
	
	// For each model being managed
	for modelID := range rm.partitioner.ListModels() {
		partitions, exists := rm.partitioner.GetModelPartitions(modelID)
		if !exists {
			continue
		}
		
		// Find partitions on the failed node
		for _, partition := range partitions {
			if partition.NodeID == recovery.FailedNodeID {
				recovery.ModelID = modelID // Track model ID in recovery
				affectedPartitions[modelID] = append(affectedPartitions[modelID], partition)
			}
		}
	}
	
	if len(affectedPartitions) == 0 {
		return fmt.Errorf("no partitions found on failed node %s", recovery.FailedNodeID)
	}
	
	// Get healthy nodes for reassignment
	healthyNodes := make([]cluster.NodeInfo, 0)
	allNodes := rm.registry.GetAllNodes()
	
	for _, node := range allNodes {
		if node.ID != recovery.FailedNodeID && node.Status == cluster.NodeStatusOnline {
			healthyNodes = append(healthyNodes, node)
		}
	}
	
	if len(healthyNodes) == 0 {
		return errors.New("no healthy nodes available for rescheduling")
	}
	
	// Reassign partitions to healthy nodes
	nodeIndex := 0
	for modelID, partitions := range affectedPartitions {
		for _, partition := range partitions {
			// Round-robin assignment to distribute load
			targetNode := healthyNodes[nodeIndex%len(healthyNodes)]
			nodeIndex++
			
			fmt.Printf("Reassigning partition %s from node %s to node %s\n",
				partition.PartitionID, recovery.FailedNodeID, targetNode.ID)
			
			// Update partitioning information
			err := rm.partitioner.ReassignPartition(modelID, partition.PartitionID, targetNode.ID)
			if err != nil {
				return fmt.Errorf("failed to reassign partition %s: %w", partition.PartitionID, err)
			}
			
			recovery.ReassignedPartitions[partition.PartitionID] = targetNode.ID
		}
	}
	
	fmt.Printf("Successfully reassigned %d partitions from failed node %s\n",
		len(recovery.ReassignedPartitions), recovery.FailedNodeID)
	
	return nil
}

// waitAndRetry waits for a node to recover and then retries failed tasks
func (rm *RecoveryManager) waitAndRetry(recovery *RecoveryOperation) error {
	maxWaitTime := rm.options.RetryDelay * time.Duration(rm.options.MaxRetries)
	waitIncrement := rm.options.RetryDelay
	
	fmt.Printf("Waiting up to %v for node %s to recover\n", maxWaitTime, recovery.FailedNodeID)
	
	// Wait and periodically check if the node has recovered
	for waited := time.Duration(0); waited < maxWaitTime; waited += waitIncrement {
		// Check if node is healthy again
		node, exists := rm.registry.GetNode(recovery.FailedNodeID)
		if !exists {
			// Node not found in registry
			continue
		}
		
		if node.Status == cluster.NodeStatusOnline {
			fmt.Printf("Node %s has recovered after %v\n", recovery.FailedNodeID, waited)
			
			// Node is back, continue with normal operation
			return nil
		}
		
		// Wait before checking again
		time.Sleep(waitIncrement)
	}
	
	// Node didn't recover in time, switch to reschedule strategy
	fmt.Printf("Node %s didn't recover after %v, switching to rescheduling\n",
		recovery.FailedNodeID, maxWaitTime)
	
	return rm.reschedulePartitions(recovery)
}

// continuePartial continues execution with available nodes
func (rm *RecoveryManager) continuePartial(recovery *RecoveryOperation) error {
	fmt.Printf("Continuing with partial execution after node %s failure\n",
		recovery.FailedNodeID)
	
	// Mark the node as permanently down
	node, exists := rm.registry.GetNode(recovery.FailedNodeID)
	if !exists {
		return fmt.Errorf("failed to get node information for %s", recovery.FailedNodeID)
	}
	
	node.Status = cluster.NodeStatusOffline
	if !rm.registry.UpdateNode(node) {
		return fmt.Errorf("failed to update node status to offline")
	}
	
	// We'll continue without the partitions on this node
	// This might lead to incomplete results but allows the system to continue operating
	
	// In a real implementation, this would notify users that results will be incomplete
	
	return nil
}

// abortExecution halts execution due to node failure
func (rm *RecoveryManager) abortExecution(recovery *RecoveryOperation) error {
	fmt.Printf("Aborting execution due to node %s failure\n", recovery.FailedNodeID)
	
	// Nothing much to do here except log the event and update status
	// In a real implementation, this would gracefully shut down running tasks
	
	return errors.New("execution aborted due to node failure")
}

// GetRecoveryStatus returns the status of an active recovery operation
func (rm *RecoveryManager) GetRecoveryStatus(recoveryID string) (*RecoveryOperation, bool) {
	rm.mu.RLock()
	defer rm.mu.RUnlock()
	
	recovery, exists := rm.activeRecoveries[recoveryID]
	return recovery, exists
}

// ListActiveRecoveries returns all ongoing recovery operations
func (rm *RecoveryManager) ListActiveRecoveries() []*RecoveryOperation {
	rm.mu.RLock()
	defer rm.mu.RUnlock()
	
	result := make([]*RecoveryOperation, 0, len(rm.activeRecoveries))
	for _, recovery := range rm.activeRecoveries {
		if recovery.Status == "in_progress" {
			result = append(result, recovery)
		}
	}
	
	return result
}

// Close stops the recovery manager and releases resources
func (rm *RecoveryManager) Close() error {
	close(rm.stopCh)  // Stop background monitoring
	close(rm.eventCh) // Stop event processing
	return nil
}