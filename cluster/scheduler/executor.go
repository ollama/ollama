package scheduler

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/ollama/ollama/cluster"
)

// ExecutorStats tracks the performance of the executor
type ExecutorStats struct {
	// TotalTasksExecuted is the number of tasks run
	TotalTasksExecuted int
	
	// SuccessfulTasks is the number of tasks that completed successfully
	SuccessfulTasks int
	
	// FailedTasks is the number of tasks that failed
	FailedTasks int
	
	// TotalExecutionTime is the cumulative execution time
	TotalExecutionTime time.Duration
	
	// LastExecutionTime is when the last task completed
	LastExecutionTime time.Time
}

// TaskResult contains the outcome of task execution
type TaskResult struct {
	// Task is the task that was executed
	Task *Task
	
	// Success indicates whether execution was successful
	Success bool
	
	// StartTime is when execution began
	StartTime time.Time
	
	// EndTime is when execution completed
	EndTime time.Time
	
	// Error holds any error that occurred
	Error error
	
	// Output contains task-specific result data
	Output map[string]interface{}
}

// Executor handles the execution of tasks across cluster nodes
type Executor struct {
	// registry provides access to available nodes
	registry *cluster.NodeRegistry
	
	// stats tracks performance metrics
	stats ExecutorStats
	
	// taskResults maps task IDs to their execution results
	taskResults map[string]*TaskResult
	
	// taskStatusMu protects the taskResults map
	taskStatusMu sync.RWMutex
	
	// nodeConnections pools connections to cluster nodes
	nodeConnections map[string]*NodeConnection
	
	// connMu protects the nodeConnections map
	connMu sync.RWMutex
	
	// callbackFn is called when a task completes
	callbackFn func(result *TaskResult)
	
	// callbackMu protects the callbackFn
	callbackMu sync.RWMutex
}

// NodeConnection manages communication with a cluster node
type NodeConnection struct {
	// NodeID identifies the node
	NodeID string
	
	// Conn holds the connection to the node
	// In a real implementation, this would be an actual connection or client
	Conn interface{}
	
	// LastActive is when this connection was last used
	LastActive time.Time
	
	// IsConnected indicates if the connection is active
	IsConnected bool
	
	// mu protects this struct from concurrent access
	mu sync.RWMutex
}

// NewExecutor creates a new task executor
func NewExecutor(registry *cluster.NodeRegistry) *Executor {
	return &Executor{
		registry:        registry,
		taskResults:     make(map[string]*TaskResult),
		nodeConnections: make(map[string]*NodeConnection),
	}
}

// ExecutePlan runs all tasks in an execution plan
func (e *Executor) ExecutePlan(ctx context.Context, plan *ExecutionPlan) error {
	if plan == nil {
		return errors.New("cannot execute nil plan")
	}
	
	startTime := time.Now()
	fmt.Printf("Executing plan for model %s with %d tasks\n",
		plan.ModelID, len(plan.Tasks))
	
	// Reset task results for this execution
	e.taskStatusMu.Lock()
	e.taskResults = make(map[string]*TaskResult)
	e.taskStatusMu.Unlock()
	
	// Build a task dependency graph
	dependencyMap := make(map[string][]string)
	taskMap := make(map[string]*Task)
	for _, task := range plan.Tasks {
		taskMap[task.ID] = task
		for _, depID := range task.Dependencies {
			dependencyMap[depID] = append(dependencyMap[depID], task.ID)
		}
	}
	
	// Find tasks with no dependencies (roots)
	var rootTasks []*Task
	for _, task := range plan.Tasks {
		if len(task.Dependencies) == 0 {
			rootTasks = append(rootTasks, task)
		}
	}
	
	// Execute based on plan mode
	var err error
	switch plan.Mode {
	case ModeSequential:
		err = e.executeSequentially(ctx, plan.Tasks, taskMap, dependencyMap)
	case ModeParallel:
		err = e.executeInParallel(ctx, rootTasks, taskMap, dependencyMap)
	case ModeHybrid:
		err = e.executeHybrid(ctx, plan.Tasks, taskMap, dependencyMap)
	default:
		err = fmt.Errorf("unsupported execution mode: %s", plan.Mode)
	}
	
	// Update executor stats
	endTime := time.Now()
	executionTime := endTime.Sub(startTime)
	
	e.taskStatusMu.Lock()
	e.stats.LastExecutionTime = endTime
	e.stats.TotalExecutionTime += executionTime
	e.stats.TotalTasksExecuted += len(plan.Tasks)
	
	// Count successes and failures
	successCount := 0
	failureCount := 0
	for _, result := range e.taskResults {
		if result.Success {
			successCount++
		} else {
			failureCount++
		}
	}
	
	e.stats.SuccessfulTasks += successCount
	e.stats.FailedTasks += failureCount
	e.taskStatusMu.Unlock()
	
	fmt.Printf("Plan execution completed in %v: %d succeeded, %d failed\n",
		executionTime, successCount, failureCount)
	
	if err != nil {
		return fmt.Errorf("plan execution error: %w", err)
	}
	
	return nil
}

// executeSequentially runs tasks one at a time in dependency order
func (e *Executor) executeSequentially(ctx context.Context, tasks []*Task, taskMap map[string]*Task, 
	dependencyMap map[string][]string) error {
	
	// Create a copy of tasks to track which ones have been executed
	pendingTasks := make(map[string]*Task)
	for _, task := range tasks {
		pendingTasks[task.ID] = task
	}
	
	// Track completed tasks
	completedTasks := make(map[string]bool)
	
	// Execute until all tasks are complete or an error occurs
	for len(pendingTasks) > 0 {
		executed := false
		
		// Find tasks whose dependencies are satisfied
		for taskID, task := range pendingTasks {
			allDependenciesMet := true
			
			// Check each dependency
			for _, depID := range task.Dependencies {
				if !completedTasks[depID] {
					allDependenciesMet = false
					break
				}
			}
			
			if allDependenciesMet {
				// Execute the task
				result, err := e.executeTask(ctx, task)
				if err != nil {
					return fmt.Errorf("failed to execute task %s: %w", taskID, err)
				}
				
				// Update results and mark as complete
				e.taskStatusMu.Lock()
				e.taskResults[taskID] = result
				e.taskStatusMu.Unlock()
				
				completedTasks[taskID] = true
				delete(pendingTasks, taskID)
				executed = true
				
				// Notify callback if registered
				e.notifyTaskCompletion(result)
				
				break // Start looking for the next task
			}
		}
		
		// If no tasks could be executed, we have a dependency cycle or missing dependency
		if !executed && len(pendingTasks) > 0 {
			return fmt.Errorf("cannot proceed with execution: possible dependency cycle or missing dependency")
		}
	}
	
	return nil
}

// executeInParallel runs independent tasks concurrently
func (e *Executor) executeInParallel(ctx context.Context, rootTasks []*Task, taskMap map[string]*Task,
	dependencyMap map[string][]string) error {
	
	var wg sync.WaitGroup
	errCh := make(chan error, len(rootTasks))
	
	// First execute all root tasks in parallel
	for _, task := range rootTasks {
		wg.Add(1)
		go func(t *Task) {
			defer wg.Done()
			
			result, err := e.executeTask(ctx, t)
			if err != nil {
				errCh <- fmt.Errorf("failed to execute task %s: %w", t.ID, err)
				return
			}
			
			// Update results
			e.taskStatusMu.Lock()
			e.taskResults[t.ID] = result
			e.taskStatusMu.Unlock()
			
			// Notify callback if registered
			e.notifyTaskCompletion(result)
			
			// Handle dependent tasks
			if depTasks, exists := dependencyMap[t.ID]; exists && len(depTasks) > 0 {
				depErr := e.executeDependentTasks(ctx, depTasks, taskMap, dependencyMap)
				if depErr != nil {
					errCh <- depErr
				}
			}
		}(task)
	}
	
	// Wait for all goroutines to complete
	wg.Wait()
	close(errCh)
	
	// Check for errors
	var errs []error
	for err := range errCh {
		errs = append(errs, err)
	}
	
	if len(errs) > 0 {
		// Return the first error (in a real implementation, we might aggregate errors)
		return errs[0]
	}
	
	return nil
}

// executeDependentTasks handles tasks that depend on a completed task
func (e *Executor) executeDependentTasks(ctx context.Context, depTaskIDs []string, taskMap map[string]*Task,
	dependencyMap map[string][]string) error {
	
	for _, depID := range depTaskIDs {
		depTask := taskMap[depID]
		
		// Check if all dependencies of this task are satisfied
		allDependenciesMet := true
		for _, depDepID := range depTask.Dependencies {
			e.taskStatusMu.RLock()
			_, completed := e.taskResults[depDepID]
			e.taskStatusMu.RUnlock()
			
			if !completed {
				allDependenciesMet = false
				break
			}
		}
		
		if allDependenciesMet {
			// Execute this task
			result, err := e.executeTask(ctx, depTask)
			if err != nil {
				return fmt.Errorf("failed to execute dependent task %s: %w", depID, err)
			}
			
			// Update results
			e.taskStatusMu.Lock()
			e.taskResults[depID] = result
			e.taskStatusMu.Unlock()
			
			// Notify callback if registered
			e.notifyTaskCompletion(result)
			
			// Recursively handle tasks that depend on this one
			if nextDepTasks, exists := dependencyMap[depID]; exists && len(nextDepTasks) > 0 {
				err := e.executeDependentTasks(ctx, nextDepTasks, taskMap, dependencyMap)
				if err != nil {
					return err
				}
			}
		}
	}
	
	return nil
}

// executeHybrid combines sequential and parallel execution
func (e *Executor) executeHybrid(ctx context.Context, tasks []*Task, taskMap map[string]*Task,
	dependencyMap map[string][]string) error {
	
	// Group tasks by phase
	phaseMap := make(map[ExecutionPhase][]*Task)
	for _, task := range tasks {
		phaseMap[task.Phase] = append(phaseMap[task.Phase], task)
	}
	
	// Execute phases in sequence (phase ordering is important)
	phases := []ExecutionPhase{
		PhaseInitialize,
		PhaseForward,
		PhaseBackward,
		PhaseUpdate,
		PhaseFinalize,
	}
	
	for _, phase := range phases {
		phaseTasks, exists := phaseMap[phase]
		if !exists || len(phaseTasks) == 0 {
			// Skip phases with no tasks
			continue
		}
		
		fmt.Printf("Executing phase %s with %d tasks\n", phase, len(phaseTasks))
		
		// Find root tasks for this phase
		var phaseRoots []*Task
		for _, task := range phaseTasks {
			isRoot := true
			for _, depID := range task.Dependencies {
				// If dependency is in the same phase, this isn't a root task
				depTask, exists := taskMap[depID]
				if exists && depTask.Phase == phase {
					isRoot = false
					break
				}
			}
			
			if isRoot {
				phaseRoots = append(phaseRoots, task)
			}
		}
		
		// Execute roots in parallel, handling dependencies appropriately
		err := e.executeInParallel(ctx, phaseRoots, taskMap, dependencyMap)
		if err != nil {
			return fmt.Errorf("failed to execute phase %s: %w", phase, err)
		}
	}
	
	return nil
}

// executeTask runs a single task on its assigned node
func (e *Executor) executeTask(ctx context.Context, task *Task) (*TaskResult, error) {
	startTime := time.Now()
	fmt.Printf("Executing task %s on node %s\n", task.ID, task.NodeID)
	
	// Get or establish connection to the target node
	_, err := e.getNodeConnection(task.NodeID)
	if err != nil {
		return &TaskResult{
			Task:      task,
			Success:   false,
			StartTime: startTime,
			EndTime:   time.Now(),
			Error:     err,
		}, err
	}
	
	// In a real implementation, this would send commands to the node
	// For now, we'll simulate execution with a delay based on EstimatedDuration
	select {
	case <-ctx.Done():
		return &TaskResult{
			Task:      task,
			Success:   false,
			StartTime: startTime,
			EndTime:   time.Now(),
			Error:     ctx.Err(),
		}, ctx.Err()
	case <-time.After(task.EstimatedDuration):
		// Task completed successfully
	}
	
	// Simulate execution results
	result := &TaskResult{
		Task:      task,
		Success:   true,
		StartTime: startTime,
		EndTime:   time.Now(),
		Output:    make(map[string]interface{}),
	}
	
	// Simulate some output data
	for _, cmd := range task.Commands {
		// Add command-specific output
		result.Output[cmd.Operation] = fmt.Sprintf("Executed %s on node %s", cmd.Operation, task.NodeID)
	}
	
	fmt.Printf("Task %s completed successfully in %v\n",
		task.ID, result.EndTime.Sub(result.StartTime))
	
	return result, nil
}

// getNodeConnection establishes a connection to a node
func (e *Executor) getNodeConnection(nodeID string) (*NodeConnection, error) {
	e.connMu.RLock()
	conn, exists := e.nodeConnections[nodeID]
	e.connMu.RUnlock()
	
	if exists && conn.IsConnected {
		// Update last active time
		conn.mu.Lock()
		conn.LastActive = time.Now()
		conn.mu.Unlock()
		
		return conn, nil
	}
	
	// Need to create a new connection or reconnect
	e.connMu.Lock()
	defer e.connMu.Unlock()
	
	// Check again under write lock
	if conn, exists := e.nodeConnections[nodeID]; exists && conn.IsConnected {
		conn.LastActive = time.Now()
		return conn, nil
	}
	
	// Get node information
	node, exists := e.registry.GetNode(nodeID)
	if !exists {
		return nil, fmt.Errorf("node %s not found in registry", nodeID)
	}
	
	if node.Status != cluster.NodeStatusOnline {
		return nil, fmt.Errorf("node %s is not online (status: %s)", nodeID, node.Status)
	}
	
	// In a real implementation, this would establish an actual connection
	// For now, just create a placeholder connection
	conn = &NodeConnection{
		NodeID:      nodeID,
		Conn:        nil, // This would be a real connection in production
		LastActive:  time.Now(),
		IsConnected: true,
	}
	
	e.nodeConnections[nodeID] = conn
	
	return conn, nil
}

// closeNodeConnection closes a connection to a node
func (e *Executor) closeNodeConnection(nodeID string) error {
	e.connMu.Lock()
	defer e.connMu.Unlock()
	
	if conn, exists := e.nodeConnections[nodeID]; exists {
		conn.mu.Lock()
		conn.IsConnected = false
		// In a real implementation, this would close the actual connection
		conn.mu.Unlock()
		
		delete(e.nodeConnections, nodeID)
	}
	
	return nil
}

// GetTaskResult retrieves the result of a specific task
func (e *Executor) GetTaskResult(taskID string) (*TaskResult, bool) {
	e.taskStatusMu.RLock()
	defer e.taskStatusMu.RUnlock()
	
	result, exists := e.taskResults[taskID]
	return result, exists
}

// GetStats returns the current executor statistics
func (e *Executor) GetStats() ExecutorStats {
	e.taskStatusMu.RLock()
	defer e.taskStatusMu.RUnlock()
	
	return e.stats
}

// SetTaskCompletionCallback sets a function to be called when tasks complete
func (e *Executor) SetTaskCompletionCallback(cb func(result *TaskResult)) {
	e.callbackMu.Lock()
	defer e.callbackMu.Unlock()
	
	e.callbackFn = cb
}

// notifyTaskCompletion calls the callback function if set
func (e *Executor) notifyTaskCompletion(result *TaskResult) {
	e.callbackMu.RLock()
	cb := e.callbackFn
	e.callbackMu.RUnlock()
	
	if cb != nil {
		// Call in a goroutine to avoid blocking
		go cb(result)
	}
}

// Close releases resources used by the executor
func (e *Executor) Close() error {
	e.connMu.Lock()
	defer e.connMu.Unlock()
	
	// Close all connections
	for nodeID, conn := range e.nodeConnections {
		conn.mu.Lock()
		conn.IsConnected = false
		// In a real implementation, this would close the actual connection
		conn.mu.Unlock()
		
		delete(e.nodeConnections, nodeID)
	}
	
	return nil
}