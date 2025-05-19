package scheduler

import (
	"errors"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/ollama/ollama/cluster"
	"github.com/ollama/ollama/cluster/model"
)

// ExecutionPhase represents a stage in distributed execution
type ExecutionPhase string

const (
	// PhaseInitialize prepares nodes for execution
	PhaseInitialize ExecutionPhase = "initialize"
	
	// PhaseForward runs the forward pass (inference)
	PhaseForward ExecutionPhase = "forward"
	
	// PhaseBackward runs the backward pass (gradient calculation)
	PhaseBackward ExecutionPhase = "backward"
	
	// PhaseUpdate applies model updates
	PhaseUpdate ExecutionPhase = "update"
	
	// PhaseFinalize handles post-execution cleanup
	PhaseFinalize ExecutionPhase = "finalize"
)

// TaskPriority defines the importance level of a task
type TaskPriority int

const (
	// PriorityLow for background tasks
	PriorityLow TaskPriority = 1
	
	// PriorityMedium for normal tasks
	PriorityMedium TaskPriority = 5
	
	// PriorityHigh for urgent tasks
	PriorityHigh TaskPriority = 10
)

// ExecutionMode specifies how tasks are executed
type ExecutionMode string

const (
	// ModeSequential runs tasks in order
	ModeSequential ExecutionMode = "sequential"
	
	// ModeParallel runs independent tasks simultaneously
	ModeParallel ExecutionMode = "parallel"
	
	// ModeHybrid combines sequential and parallel execution
	ModeHybrid ExecutionMode = "hybrid"
)

// Task represents a unit of work in the execution plan
type Task struct {
	// ID uniquely identifies this task
	ID string
	
	// NodeID where this task should run
	NodeID string
	
	// Phase indicates which execution stage this task belongs to
	Phase ExecutionPhase
	
	// Dependencies are IDs of tasks that must complete before this one
	Dependencies []string
	
	// Priority sets the importance of this task
	Priority TaskPriority
	
	// EstimatedDuration is the expected runtime of this task
	EstimatedDuration time.Duration
	
	// Commands holds the operations to perform
	Commands []TaskCommand
}

// TaskCommand represents an operation within a task
type TaskCommand struct {
	// Operation is the type of command
	Operation string
	
	// Params holds command-specific parameters
	Params map[string]interface{}
}

// TaskStatus tracks the state of a task
type TaskStatus struct {
	// Task is the task being tracked
	Task *Task
	
	// IsComplete indicates whether the task has finished
	IsComplete bool
	
	// StartTime is when execution began
	StartTime time.Time
	
	// EndTime is when execution completed
	EndTime time.Time
	
	// Error holds any error that occurred
	Error error
}

// ExecutionPlan organizes tasks for efficient distributed execution
type ExecutionPlan struct {
	// ModelID identifies which model this plan is for
	ModelID string
	
	// Tasks are the operations to perform
	Tasks []*Task
	
	// Mode determines how tasks are executed
	Mode ExecutionMode
	
	// StartTime is when this plan was created
	StartTime time.Time
}

// Planner creates execution plans for distributed model operations
type Planner struct {
	// registry provides access to available nodes
	registry *cluster.NodeRegistry
	
	// partitioner provides access to model partitioning information
	partitioner *model.ModelPartitioner
	
	// router provides access to inter-node communication paths
	router *model.RoutingTable
	
	// plans tracks current and historical execution plans
	plans map[string]*ExecutionPlan
	
	// mu protects the plans map
	mu sync.RWMutex
}

// NewPlanner creates a new execution planner
func NewPlanner(registry *cluster.NodeRegistry, partitioner *model.ModelPartitioner, router *model.RoutingTable) *Planner {
	return &Planner{
		registry:    registry,
		partitioner: partitioner,
		router:      router,
		plans:       make(map[string]*ExecutionPlan),
	}
}

// CreateExecutionPlan builds a new plan for model execution
func (p *Planner) CreateExecutionPlan(modelID string, mode ExecutionMode) (*ExecutionPlan, error) {
	// Get model partitioning
	partitions, exists := p.partitioner.GetModelPartitions(modelID)
	if !exists {
		return nil, fmt.Errorf("no partitioning found for model %s", modelID)
	}
	
	fmt.Printf("Creating %s execution plan for model %s with %d partitions\n",
		mode, modelID, len(partitions))
	
	plan := &ExecutionPlan{
		ModelID:   modelID,
		Tasks:     make([]*Task, 0),
		Mode:      mode,
		StartTime: time.Now(),
	}
	
	// Add initialization tasks for each partition
	for i, partition := range partitions {
		initTask := &Task{
			ID:                fmt.Sprintf("init-%s-%d", modelID, i),
			NodeID:            partition.NodeID,
			Phase:             PhaseInitialize,
			Dependencies:      []string{},
			Priority:          PriorityHigh,
			EstimatedDuration: 500 * time.Millisecond,
			Commands: []TaskCommand{
				{
					Operation: "initialize_partition",
					Params: map[string]interface{}{
						"partition_id": partition.PartitionID,
						"model_id":     modelID,
					},
				},
			},
		}
		
		plan.Tasks = append(plan.Tasks, initTask)
	}
	
	// Add forward pass tasks with dependencies
	var prevForwardTasks []string
	
	for i, partition := range partitions {
		deps := make([]string, 0)
		
		// Depend on initialization
		initTaskID := fmt.Sprintf("init-%s-%d", modelID, i)
		deps = append(deps, initTaskID)
		
		// Forward pass tasks depend on previous partition
		if i > 0 {
			deps = append(deps, prevForwardTasks...)
		}
		
		taskID := fmt.Sprintf("forward-%s-%d", modelID, i)
		forwardTask := &Task{
			ID:                taskID,
			NodeID:            partition.NodeID,
			Phase:             PhaseForward,
			Dependencies:      deps,
			Priority:          PriorityMedium,
			EstimatedDuration: 2 * time.Second,
			Commands: []TaskCommand{
				{
					Operation: "run_forward_pass",
					Params: map[string]interface{}{
						"partition_id": partition.PartitionID,
						"model_id":     modelID,
					},
				},
			},
		}
		
		plan.Tasks = append(plan.Tasks, forwardTask)
		
		// Update for next iteration
		prevForwardTasks = []string{taskID}
	}
	
	// Add finalization tasks
	for i, partition := range partitions {
		deps := []string{fmt.Sprintf("forward-%s-%d", modelID, i)}
		
		finalizeTask := &Task{
			ID:                fmt.Sprintf("finalize-%s-%d", modelID, i),
			NodeID:            partition.NodeID,
			Phase:             PhaseFinalize,
			Dependencies:      deps,
			Priority:          PriorityLow,
			EstimatedDuration: 500 * time.Millisecond,
			Commands: []TaskCommand{
				{
					Operation: "finalize_execution",
					Params: map[string]interface{}{
						"partition_id": partition.PartitionID,
						"model_id":     modelID,
					},
				},
			},
		}
		
		plan.Tasks = append(plan.Tasks, finalizeTask)
	}
	
	// Store the plan
	planID := fmt.Sprintf("%s-%d", modelID, time.Now().UnixNano())
	p.mu.Lock()
	p.plans[planID] = plan
	p.mu.Unlock()
	
	return plan, nil
}

// OptimizePlan analyzes and improves an execution plan
func (p *Planner) OptimizePlan(plan *ExecutionPlan) error {
	if plan == nil {
		return errors.New("cannot optimize nil plan")
	}
	
	fmt.Printf("Optimizing execution plan for model %s\n", plan.ModelID)
	
	// Sort tasks by phase and priority
	sort.Slice(plan.Tasks, func(i, j int) bool {
		// First sort by phase
		if plan.Tasks[i].Phase != plan.Tasks[j].Phase {
			return string(plan.Tasks[i].Phase) < string(plan.Tasks[j].Phase)
		}
		// Then by priority (higher priority first)
		return plan.Tasks[i].Priority > plan.Tasks[j].Priority
	})
	
	// Analyze for potential parallelization
	if plan.Mode == ModeHybrid || plan.Mode == ModeParallel {
		dependencyMap := make(map[string]map[string]bool)
		
		// Build dependency graph
		for _, task := range plan.Tasks {
			if _, exists := dependencyMap[task.ID]; !exists {
				dependencyMap[task.ID] = make(map[string]bool)
			}
			
			for _, depID := range task.Dependencies {
				dependencyMap[task.ID][depID] = true
			}
		}
		
		// Identify tasks that can run in parallel (no dependencies between them)
		for i := 0; i < len(plan.Tasks); i++ {
			for j := i + 1; j < len(plan.Tasks); j++ {
				taskA := plan.Tasks[i]
				taskB := plan.Tasks[j]
				
				// Skip tasks in different phases - phases must remain sequential
				if taskA.Phase != taskB.Phase {
					continue
				}
				
				// Check if A depends on B
				if deps, exists := dependencyMap[taskA.ID]; exists {
					if _, hasDep := deps[taskB.ID]; hasDep {
						continue
					}
				}
				
				// Check if B depends on A
				if deps, exists := dependencyMap[taskB.ID]; exists {
					if _, hasDep := deps[taskA.ID]; hasDep {
						continue
					}
				}
				
				// Tasks can potentially run in parallel
				// This is just marking them for the executor to handle
				fmt.Printf("Tasks %s and %s can run in parallel\n", taskA.ID, taskB.ID)
			}
		}
	}
	
	// Estimate overall plan duration
	var maxDuration time.Duration
	for _, task := range plan.Tasks {
		if len(task.Dependencies) == 0 {
			maxDuration += task.EstimatedDuration
		}
	}
	
	fmt.Printf("Optimized plan estimated duration: %v\n", maxDuration)
	
	return nil
}

// GetExecutionPlan retrieves a stored plan
func (p *Planner) GetExecutionPlan(planID string) (*ExecutionPlan, bool) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	
	plan, exists := p.plans[planID]
	return plan, exists
}

// ValidatePlan checks if a plan is valid and can be executed
func (p *Planner) ValidatePlan(plan *ExecutionPlan) error {
	if plan == nil {
		return errors.New("cannot validate nil plan")
	}
	
	if len(plan.Tasks) == 0 {
		return errors.New("plan contains no tasks")
	}
	
	// Check for circular dependencies
	taskMap := make(map[string]*Task)
	for _, task := range plan.Tasks {
		taskMap[task.ID] = task
	}
	
	// For each task, do a DFS to check for cycles
	visited := make(map[string]bool)
	recursionStack := make(map[string]bool)
	
	var checkCycle func(taskID string) bool
	checkCycle = func(taskID string) bool {
		if !visited[taskID] {
			visited[taskID] = true
			recursionStack[taskID] = true
			
			task, exists := taskMap[taskID]
			if !exists {
				return false // Task not found
			}
			
			for _, depID := range task.Dependencies {
				if !visited[depID] && checkCycle(depID) {
					return true
				} else if recursionStack[depID] {
					return true // Found a cycle
				}
			}
		}
		
		recursionStack[taskID] = false
		return false
	}
	
	for _, task := range plan.Tasks {
		if !visited[task.ID] {
			if checkCycle(task.ID) {
				return fmt.Errorf("circular dependency detected in task %s", task.ID)
			}
		}
	}
	
	// Check that all nodes exist and are active
	nodeSet := make(map[string]bool)
	for _, task := range plan.Tasks {
		if _, exists := nodeSet[task.NodeID]; !exists {
			node, exists := p.registry.GetNode(task.NodeID)
			if !exists {
				return fmt.Errorf("node %s not found in registry for task %s", task.NodeID, task.ID)
			}
			
			if node.Status != cluster.NodeStatusOnline {
				return fmt.Errorf("node %s is not online (status: %s) for task %s",
					task.NodeID, node.Status, task.ID)
			}
			
			nodeSet[task.NodeID] = true
		}
	}
	
	return nil
}