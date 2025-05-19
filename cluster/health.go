package cluster

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"
)

// HealthMonitor provides health monitoring services for the cluster
type HealthMonitor struct {
	// registry is a reference to the node registry
	registry *NodeRegistry
	
	// checkInterval is how often health checks are performed
	checkInterval time.Duration
	
	// nodeTimeoutThreshold is how long before considering a node offline
	nodeTimeoutThreshold time.Duration
	
	// client is the HTTP client used for health checks
	client *http.Client
	
	// ctx is the context for managing the health monitor lifecycle
	ctx context.Context
	
	// cancel is the function to stop the health monitor
	cancel context.CancelFunc
	
	// mu protects the health monitor state
	mu sync.Mutex
	
	// healthStatusCache caches the results of node health checks
	healthStatusCache map[string]*NodeHealthStatus
}

// NodeHealthStatus represents detailed health information for a node
type NodeHealthStatus struct {
	// NodeID is the identifier of the node
	NodeID string `json:"node_id"`
	
	// Status is the current operational status
	Status NodeStatus `json:"status"`
	
	// LastChecked is when the health was last verified
	LastChecked time.Time `json:"last_checked"`
	
	// CPUUsagePercent represents current CPU usage as a percentage
	CPUUsagePercent float64 `json:"cpu_usage_percent"`
	
	// MemoryUsageMB represents current memory usage in megabytes
	MemoryUsageMB uint64 `json:"memory_usage_mb"`
	
	// GPUUtilization represents current GPU utilization as a percentage
	GPUUtilization []float64 `json:"gpu_utilization,omitempty"`
	
	// GPUMemoryUsageMB represents current GPU memory usage in megabytes
	GPUMemoryUsageMB []uint64 `json:"gpu_memory_usage_mb,omitempty"`
	
	// ActiveRequests is the number of active requests being processed
	ActiveRequests int `json:"active_requests"`
	
	// QueuedRequests is the number of requests waiting to be processed
	QueuedRequests int `json:"queued_requests"`
	
	// LatencyMS is the average response latency in milliseconds
	LatencyMS int `json:"latency_ms"`
	
	// ModelLoadCount is the number of models loaded
	ModelLoadCount int `json:"model_load_count"`
}
// NewHealthMonitor creates a new health monitoring service
func NewHealthMonitor(registry *NodeRegistry, checkInterval, nodeTimeoutThreshold time.Duration) *HealthMonitor {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &HealthMonitor{
		registry:            registry,
		checkInterval:       checkInterval,
		nodeTimeoutThreshold: nodeTimeoutThreshold,
		client: &http.Client{
			Timeout: 5 * time.Second,
		},
		ctx:              ctx,
		cancel:           cancel,
		healthStatusCache: make(map[string]*NodeHealthStatus),
	}
}

// Start begins the health monitoring service
func (h *HealthMonitor) Start() error {
	h.mu.Lock()
	defer h.mu.Unlock()
	
	fmt.Println("Starting cluster health monitoring service")
	
	// Start the background health check routine
	go h.runHealthChecks()
	
	// Register as the registry's health checker
	h.registry.SetHealthChecker(h)
	
	return nil
}

// Stop halts the health monitoring service
func (h *HealthMonitor) Stop() error {
	h.mu.Lock()
	defer h.mu.Unlock()
	
	fmt.Println("Stopping cluster health monitoring service")
	
	h.cancel()
	
	return nil
}

// runHealthChecks periodically performs health checks on all nodes
func (h *HealthMonitor) runHealthChecks() {
	ticker := time.NewTicker(h.checkInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-h.ctx.Done():
			return
		case <-ticker.C:
			h.checkAllNodesHealth()
		}
	}
}
// checkAllNodesHealth performs health checks on all registered nodes
func (h *HealthMonitor) checkAllNodesHealth() {
	nodes := h.registry.GetAllNodes()
	
	for _, node := range nodes {
		// Skip our own node, we know our health directly
		if node.ID == h.getLocalNodeID() {
			continue
		}
		
		status, err := h.CheckNodeHealth(node)
		if err != nil {
			fmt.Printf("Health check failed for node %s (%s): %v\n",
				node.Name, node.ID, err)
			continue
		}
		
		// Update node status if it has changed
		if status != node.Status {
			updatedNode := node
			updatedNode.Status = status
			updatedNode.LastHeartbeat = time.Now() // Also update heartbeat time
			h.registry.UpdateNode(updatedNode)
		}
	}
}

// getLocalNodeID retrieves the local node ID from the registry
func (h *HealthMonitor) getLocalNodeID() string {
	// In a real implementation, this would get the local node ID from the registry
	// For now, we'll just return an empty string which won't match any node ID
	return ""
}

// CheckNodeHealth implements the HealthChecker interface
func (h *HealthMonitor) CheckNodeHealth(node NodeInfo) (NodeStatus, error) {
	// Build the health check URL
	healthURL := fmt.Sprintf("http://%s:%d/api/health", node.Addr.String(), node.ApiPort)
	
	// Create a context with timeout for the request
	ctx, cancel := context.WithTimeout(h.ctx, 3*time.Second)
	defer cancel()
	
	req, err := http.NewRequestWithContext(ctx, "GET", healthURL, nil)
	if err != nil {
		return NodeStatusOffline, fmt.Errorf("error creating health check request: %w", err)
	}
	
	resp, err := h.client.Do(req)
	if err != nil {
		return NodeStatusOffline, fmt.Errorf("error performing health check: %w", err)
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return NodeStatusOffline, fmt.Errorf("health check returned status: %s", resp.Status)
	}
	
	var healthStatus NodeHealthStatus
	err = json.NewDecoder(resp.Body).Decode(&healthStatus)
	if err != nil {
		return NodeStatusOffline, fmt.Errorf("error decoding health response: %w", err)
	}
	
	// Cache the health status
	h.mu.Lock()
	h.healthStatusCache[node.ID] = &healthStatus
	h.mu.Unlock()
	
	return healthStatus.Status, nil
}
// GetNodeHealthStatus retrieves detailed health information for a node
func (h *HealthMonitor) GetNodeHealthStatus(nodeID string) (*NodeHealthStatus, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	
	health, found := h.healthStatusCache[nodeID]
	if !found {
		return nil, fmt.Errorf("no health data available for node %s", nodeID)
	}
	
	return health, nil
}

// GetClusterHealthSummary provides an overall health summary for the cluster
func (h *HealthMonitor) GetClusterHealthSummary() map[string]interface{} {
	h.mu.Lock()
	defer h.mu.Unlock()
	
	nodes := h.registry.GetAllNodes()
	
	summary := map[string]interface{}{
		"node_count": len(nodes),
		"online_nodes": 0,
		"offline_nodes": 0,
		"busy_nodes": 0,
		"total_active_requests": 0,
		"total_queued_requests": 0,
		"average_latency_ms": 0,
		"nodes": make([]map[string]interface{}, 0),
	}
	
	var totalLatency int
	var nodesWithLatency int
	
	for _, node := range nodes {
		nodeSummary := map[string]interface{}{
			"id":     node.ID,
			"name":   node.Name,
			"status": string(node.Status),
			"role":   string(node.Role),
		}
		
		// Update status counts
		switch node.Status {
		case NodeStatusOnline:
			summary["online_nodes"] = summary["online_nodes"].(int) + 1
		case NodeStatusOffline:
			summary["offline_nodes"] = summary["offline_nodes"].(int) + 1
		case NodeStatusBusy:
			summary["busy_nodes"] = summary["busy_nodes"].(int) + 1
		}
		
		// Add health details if available
		if health, ok := h.healthStatusCache[node.ID]; ok {
			nodeSummary["cpu_usage"] = health.CPUUsagePercent
			nodeSummary["memory_usage_mb"] = health.MemoryUsageMB
			nodeSummary["active_requests"] = health.ActiveRequests
			nodeSummary["queued_requests"] = health.QueuedRequests
			
			summary["total_active_requests"] = summary["total_active_requests"].(int) + health.ActiveRequests
			summary["total_queued_requests"] = summary["total_queued_requests"].(int) + health.QueuedRequests
			
			if health.LatencyMS > 0 {
				totalLatency += health.LatencyMS
				nodesWithLatency++
			}
		}
		
		summary["nodes"] = append(summary["nodes"].([]map[string]interface{}), nodeSummary)
	}
	
	// Calculate average latency
	if nodesWithLatency > 0 {
		summary["average_latency_ms"] = totalLatency / nodesWithLatency
	}
	
	return summary
}