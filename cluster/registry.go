package cluster

import (
	"fmt"
	"sync"
	"time"
)

// NodeRegistry maintains the registry of nodes in the cluster
type NodeRegistry struct {
	// nodes maps node IDs to their NodeInfo
	nodes map[string]NodeInfo
	
	// mu protects the nodes map from concurrent access
	mu sync.RWMutex
	
	// eventHandlers contains registered callbacks for node events
	eventHandlers []func(event ClusterEvent)
	
	// eventsMu protects the eventHandlers from concurrent access
	eventsMu sync.RWMutex
	
	// nodeTimeoutInterval determines how long before a node is considered offline
	nodeTimeoutInterval time.Duration
	
	// healthCheckInterval determines how often to check node health
	healthCheckInterval time.Duration
	
	// healthChecker is an optional interface that can be implemented to provide custom health checks
	healthChecker HealthChecker
}

// HealthChecker interface defines methods for checking node health
type HealthChecker interface {
	// CheckNodeHealth checks if a node is healthy and returns its current status
	CheckNodeHealth(node NodeInfo) (NodeStatus, error)
}

// NewNodeRegistry creates a new node registry
func NewNodeRegistry(healthCheckInterval, nodeTimeoutInterval time.Duration) *NodeRegistry {
	r := &NodeRegistry{
		nodes:               make(map[string]NodeInfo),
		nodeTimeoutInterval: nodeTimeoutInterval,
		healthCheckInterval: healthCheckInterval,
	}
	
	// Start the health check routine
	go r.runHealthChecks()
	
	return r
}
// RegisterNode adds or updates a node in the registry
func (r *NodeRegistry) RegisterNode(node NodeInfo) {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	existing, exists := r.nodes[node.ID]
	r.nodes[node.ID] = node
	
	if !exists {
		fmt.Printf("New node registered: %s (%s)\n", node.Name, node.ID)
		r.emitEvent("node_joined", node.ID, map[string]interface{}{
			"name": node.Name,
			"addr": node.Addr.String(),
			"role": string(node.Role),
		})
	} else if existing.Status != node.Status {
		fmt.Printf("Node %s (%s) changed status from %s to %s\n",
			node.Name, node.ID, existing.Status, node.Status)
		
		r.emitEvent("node_status_changed", node.ID, map[string]interface{}{
			"name":        node.Name,
			"old_status":  string(existing.Status),
			"new_status":  string(node.Status),
		})
	}
}

// UnregisterNode removes a node from the registry
func (r *NodeRegistry) UnregisterNode(nodeID string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	if node, exists := r.nodes[nodeID]; exists {
		fmt.Printf("Node unregistered: %s (%s)\n", node.Name, node.ID)
		
		r.emitEvent("node_left", node.ID, map[string]interface{}{
			"name": node.Name,
		})
		
		delete(r.nodes, nodeID)
	}
}

// GetNode retrieves information about a specific node
func (r *NodeRegistry) GetNode(nodeID string) (NodeInfo, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	node, exists := r.nodes[nodeID]
	return node, exists
}

// GetAllNodes returns a list of all registered nodes
func (r *NodeRegistry) GetAllNodes() []NodeInfo {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	nodes := make([]NodeInfo, 0, len(r.nodes))
	for _, node := range r.nodes {
		nodes = append(nodes, node)
	}
	
	return nodes
}

// UpdateNode updates an existing node's information
func (r *NodeRegistry) UpdateNode(node NodeInfo) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	if _, exists := r.nodes[node.ID]; exists {
		r.nodes[node.ID] = node
		return true
	}
	
	return false
}
// GetAvailableWorkers returns a list of available worker nodes
func (r *NodeRegistry) GetAvailableWorkers() []NodeInfo {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	workers := make([]NodeInfo, 0)
	for _, node := range r.nodes {
		if (node.Role == NodeRoleWorker || node.Role == NodeRoleMixed) &&
			node.Status == NodeStatusOnline {
			workers = append(workers, node)
		}
	}
	
	return workers
}

// GetCoordinators returns a list of coordinator nodes
func (r *NodeRegistry) GetCoordinators() []NodeInfo {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	coordinators := make([]NodeInfo, 0)
	for _, node := range r.nodes {
		if (node.Role == NodeRoleCoordinator || node.Role == NodeRoleMixed) &&
			node.Status == NodeStatusOnline {
			coordinators = append(coordinators, node)
		}
	}
	
	return coordinators
}

// AddEventHandler registers a callback function for node events
func (r *NodeRegistry) AddEventHandler(handler func(event ClusterEvent)) {
	r.eventsMu.Lock()
	defer r.eventsMu.Unlock()
	
	r.eventHandlers = append(r.eventHandlers, handler)
}

// emitEvent emits a cluster event to all registered handlers
func (r *NodeRegistry) emitEvent(eventType, nodeID string, data map[string]interface{}) {
	event := ClusterEvent{
		Type:      eventType,
		NodeID:    nodeID,
		Timestamp: time.Now(),
		Data:      data,
	}
	
	r.eventsMu.RLock()
	defer r.eventsMu.RUnlock()
	
	for _, handler := range r.eventHandlers {
		go handler(event)
	}
}

// SetHealthChecker sets a custom health checker implementation
func (r *NodeRegistry) SetHealthChecker(checker HealthChecker) {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	r.healthChecker = checker
}
// runHealthChecks periodically checks the health of all registered nodes
func (r *NodeRegistry) runHealthChecks() {
	ticker := time.NewTicker(r.healthCheckInterval)
	defer ticker.Stop()
	
	for range ticker.C {
		r.checkNodesHealth()
	}
}

// checkNodesHealth verifies that all nodes are still responsive
func (r *NodeRegistry) checkNodesHealth() {
	// Get a snapshot of the current node registry
	nodes := r.GetAllNodes()
	
	for _, node := range nodes {
		// Skip nodes that are already marked as offline
		if node.Status == NodeStatusOffline {
			continue
		}
		
		// Check if node has timed out
		if time.Since(node.LastHeartbeat) > r.nodeTimeoutInterval {
			fmt.Printf("Node %s (%s) timed out after %v of inactivity\n",
				node.Name, node.ID, r.nodeTimeoutInterval)
			
			node.Status = NodeStatusOffline
			r.UpdateNode(node)
			
			r.emitEvent("node_timeout", node.ID, map[string]interface{}{
				"name":              node.Name,
				"last_heartbeat":    node.LastHeartbeat,
				"timeout_interval":  r.nodeTimeoutInterval.String(),
			})
			
			continue
		}
		
		// If we have a custom health checker, use it for more advanced health checks
		if r.healthChecker != nil {
			status, err := r.healthChecker.CheckNodeHealth(node)
			
			if err != nil {
				fmt.Printf("Health check failed for node %s (%s): %v\n",
					node.Name, node.ID, err)
				
				// Don't immediately mark as offline, let the timeout happen
				continue
			}
			
			if status != node.Status {
				oldStatus := node.Status
				node.Status = status
				r.UpdateNode(node)
				
				fmt.Printf("Node %s (%s) health status changed: %s -> %s\n",
					node.Name, node.ID, oldStatus, status)
				
				r.emitEvent("node_health_changed", node.ID, map[string]interface{}{
					"name":       node.Name,
					"old_status": string(oldStatus),
					"new_status": string(status),
				})
			}
		}
	}
}