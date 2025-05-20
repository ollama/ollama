package cluster

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"net/http"
	"sync"
	"time"
	
	cerrors "github.com/ollama/ollama/cluster/errors"
)

// HealthMonitor provides health monitoring services for the cluster
type HealthMonitor struct {
	// registry is a reference to the node registry
	registry *NodeRegistry
	
	// statusController is a reference to the node status controller
	statusController *NodeStatusController
	
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
	
	// clusterMode is a reference back to the cluster mode
	clusterMode *ClusterMode
	
	// communicationMetrics tracks communication success/failure stats
	communicationMetrics map[string]*NodeCommunicationMetrics
	
	// reconnectionManager manages automatic reconnection of failed nodes
	reconnectionManager *NodeReconnectionManager
	
	// degradationManager handles graceful degradation when nodes are temporarily unavailable
	degradationManager *cerrors.GracefulDegradation
}

// NodeCommunicationMetrics tracks communication statistics for a node
type NodeCommunicationMetrics struct {
	// NodeID is the identifier for the node
	NodeID string
	
	// SuccessCount tracks successful communications
	SuccessCount int
	
	// FailureCount tracks failed communications
	FailureCount int
	
	// LastFailure records when the last failure occurred
	LastFailure time.Time
	
	// ConsecutiveFailures counts failures without an intervening success
	ConsecutiveFailures int
	
	// LatencyHistory tracks recent latency measurements
	LatencyHistory []time.Duration
	
	// AvgLatency is the rolling average latency
	AvgLatency time.Duration
	
	// MaxLatency is the maximum observed latency
	MaxLatency time.Duration
	
	// MinLatency is the minimum observed latency
	MinLatency time.Duration
	
	// LastChecked records when metrics were last updated
	LastChecked time.Time
}

// NodeReconnectionManager handles automatic reconnection of failed nodes
type NodeReconnectionManager struct {
	// parentMonitor is a reference to the owning health monitor
	parentMonitor *HealthMonitor
	
	// retryPolicy defines how to handle retries for temporary failures
	retryPolicy cerrors.RetryPolicy
	
	// nodeRetries tracks retry attempts per node
	nodeRetries map[string]int
	
	// nextRetryTime tracks when to next attempt reconnection
	nextRetryTime map[string]time.Time
	
	// mu protects the reconnection state
	mu sync.Mutex
	
	// reconnectionQueue holds nodes scheduled for reconnection
	reconnectionQueue map[string]time.Time
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
	
	// Create HTTP client with enhanced settings
	client := &http.Client{
		Timeout: 5 * time.Second,
		Transport: &http.Transport{
			MaxIdleConns:          100,
			MaxIdleConnsPerHost:   10,
			IdleConnTimeout:       90 * time.Second,
			TLSHandshakeTimeout:   3 * time.Second,
			ResponseHeaderTimeout: 5 * time.Second,
			ExpectContinueTimeout: 1 * time.Second,
			DisableKeepAlives:     false,
		},
	}
	
	hm := &HealthMonitor{
		registry:            registry,
		checkInterval:       checkInterval,
		nodeTimeoutThreshold: nodeTimeoutThreshold,
		client:               client,
		ctx:                  ctx,
		cancel:               cancel,
		healthStatusCache:    make(map[string]*NodeHealthStatus),
		communicationMetrics: make(map[string]*NodeCommunicationMetrics),
	}
	
	// Initialize reconnection manager with improved retry policy
	hm.reconnectionManager = &NodeReconnectionManager{
		parentMonitor:    hm,
		retryPolicy:      cerrors.NewDefaultRetryPolicy(),
		nodeRetries:      make(map[string]int),
		nextRetryTime:    make(map[string]time.Time),
		reconnectionQueue: make(map[string]time.Time),
	}
	
	// Initialize graceful degradation manager
	hm.degradationManager = cerrors.NewGracefulDegradation(cerrors.DegradationModeAdaptive)
	
	return hm
}

// InitNodeMetrics initializes communication metrics tracking for a node
func (h *HealthMonitor) InitNodeMetrics(nodeID string) *NodeCommunicationMetrics {
	h.mu.Lock()
	defer h.mu.Unlock()
	
	if _, exists := h.communicationMetrics[nodeID]; !exists {
		h.communicationMetrics[nodeID] = &NodeCommunicationMetrics{
			NodeID:             nodeID,
			SuccessCount:       0,
			FailureCount:       0,
			ConsecutiveFailures: 0,
			LatencyHistory:     make([]time.Duration, 0, 10), // Track last 10 measurements
			AvgLatency:         0,
			MaxLatency:         0,
			MinLatency:         time.Hour, // Initialize to a high value
			LastChecked:        time.Now(),
		}
	}
	
	return h.communicationMetrics[nodeID]
}

// SetClusterMode sets the reference to the cluster mode
// This must be called after the ClusterMode is fully initialized
func (h *HealthMonitor) SetClusterMode(cm *ClusterMode) {
	h.clusterMode = cm
	h.statusController = cm.StatusController
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
	
	// Start the reconnection manager in a separate goroutine
	go h.reconnectionManager.RunReconnectionTasks()
	
	for {
		select {
		case <-h.ctx.Done():
			return
		case <-ticker.C:
			h.checkAllNodesHealth()
		}
	}
}

// RecordCommunicationSuccess records a successful communication with a node
func (h *HealthMonitor) RecordCommunicationSuccess(nodeID string, latency time.Duration) {
	h.mu.Lock()
	defer h.mu.Unlock()
	
	metrics, exists := h.communicationMetrics[nodeID]
	if !exists {
		metrics = h.InitNodeMetrics(nodeID)
	}
	
	// Update success metrics
	metrics.SuccessCount++
	metrics.ConsecutiveFailures = 0
	metrics.LastChecked = time.Now()
	
	// Update latency metrics
	metrics.LatencyHistory = append(metrics.LatencyHistory, latency)
	if len(metrics.LatencyHistory) > 10 {
		// Keep only the last 10 measurements
		metrics.LatencyHistory = metrics.LatencyHistory[1:]
	}
	
	// Calculate average latency
	var totalLatency time.Duration
	for _, l := range metrics.LatencyHistory {
		totalLatency += l
	}
	metrics.AvgLatency = totalLatency / time.Duration(len(metrics.LatencyHistory))
	
	// Update min/max latency
	if latency < metrics.MinLatency {
		metrics.MinLatency = latency
	}
	if latency > metrics.MaxLatency {
		metrics.MaxLatency = latency
	}
	
	// If node was pending reconnection, remove it
	h.reconnectionManager.mu.Lock()
	defer h.reconnectionManager.mu.Unlock()
	
	delete(h.reconnectionManager.nodeRetries, nodeID)
	delete(h.reconnectionManager.nextRetryTime, nodeID)
	delete(h.reconnectionManager.reconnectionQueue, nodeID)
}

// RecordCommunicationFailure records a failed communication with a node
func (h *HealthMonitor) RecordCommunicationFailure(nodeID string, err error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	
	metrics, exists := h.communicationMetrics[nodeID]
	if !exists {
		metrics = h.InitNodeMetrics(nodeID)
	}
	
	// Update failure metrics
	metrics.FailureCount++
	metrics.ConsecutiveFailures++
	metrics.LastFailure = time.Now()
	metrics.LastChecked = time.Now()
	
	// Determine error type and create a more detailed cluster error
	var severity cerrors.ErrorSeverity
	var errorCategory cerrors.ErrorCategory
	var errorType string
	var metadata map[string]string = make(map[string]string)
	
	// Add common metadata
	metadata["consecutive_failures"] = fmt.Sprintf("%d", metrics.ConsecutiveFailures)
	metadata["failure_count"] = fmt.Sprintf("%d", metrics.FailureCount)
	metadata["last_success"] = metrics.LastChecked.Format(time.RFC3339)
	
	// Categorize the error by type for appropriate handling
	switch {
	case cerrors.IsTimeoutError(err):
		severity = cerrors.TemporaryError
		errorCategory = cerrors.TimeoutError
		errorType = "timeout error"
		metadata["recovery_strategy"] = "retry with increased timeout"
	
	case cerrors.IsConnectionRefusedError(err):
		severity = cerrors.TemporaryError
		errorCategory = cerrors.ConnectionRefused
		errorType = "connection refused"
		metadata["recovery_strategy"] = "check if node is running"
	
	case cerrors.IsNameResolutionError(err):
		severity = cerrors.TemporaryError
		errorCategory = cerrors.NameResolution
		errorType = "name resolution error"
		metadata["recovery_strategy"] = "check DNS configuration"
	
	case cerrors.IsNetworkUnreachable(err):
		severity = cerrors.TemporaryError
		errorCategory = cerrors.NetworkTemporary
		errorType = "network unreachable"
		metadata["recovery_strategy"] = "check network connectivity"
	
	case cerrors.IsConnectionResetError(err):
		severity = cerrors.TemporaryError
		errorCategory = cerrors.NetworkTemporary
		errorType = "connection reset"
		metadata["recovery_strategy"] = "retry connection"
	
	case cerrors.IsPermissionError(err):
		severity = cerrors.PersistentError
		errorCategory = cerrors.PermissionDenied
		errorType = "permission denied"
		metadata["recovery_strategy"] = "check access permissions"
	
	case cerrors.IsTemporaryNetworkError(err):
		severity = cerrors.TemporaryError
		errorCategory = cerrors.NetworkTemporary
		errorType = "temporary network error"
		metadata["recovery_strategy"] = "automatic retry with backoff"
	
	default:
		if metrics.ConsecutiveFailures > 10 {
			severity = cerrors.PersistentError
			errorCategory = cerrors.UnknownError
			errorType = "persistent unknown error"
			metadata["recovery_strategy"] = "manual intervention may be required"
		} else {
			severity = cerrors.TemporaryError
			errorCategory = cerrors.UnknownError
			errorType = "unknown error"
			metadata["recovery_strategy"] = "automatic retry"
		}
	}
	
	// Create detailed error with metadata and retry information
	clusterErr := cerrors.NewDetailedCommunicationError6(
		nodeID,
		fmt.Sprintf("Communication failure with node %s: %s", nodeID, errorType),
		severity,
		errorCategory,
		err,
		metadata,
	)
	
	// If it's a temporary error, add retry information
	if clusterErr.IsTemporary() {
		// Calculate backoff using retry policy
		backoff := h.reconnectionManager.retryPolicy.CalculateBackoff(metrics.ConsecutiveFailures - 1)
		nextRetry := time.Now().Add(backoff)
		clusterErr = clusterErr.WithRetry(metrics.ConsecutiveFailures, nextRetry)
	}
	
	cerrors.LogErrorf(
		clusterErr,
		"Communication failure with node %s after %d consecutive failures",
		nodeID,
		metrics.ConsecutiveFailures,
	)
	
	// Handle node status change based on consecutive failures
	node, exists := h.registry.GetNode(nodeID)
	if exists {
		if metrics.ConsecutiveFailures >= 3 {
			// After 3 consecutive failures, mark node as degraded
			if node.Status == NodeStatusOnline || node.Status == NodeStatusBusy {
				newStatus, _ := h.determineStatusChange(node.Status, NodeStatusDegraded,
					fmt.Sprintf("consecutive failures: %d", metrics.ConsecutiveFailures))
				
				if newStatus == NodeStatusDegraded {
					updatedNode := node
					updatedNode.Status = NodeStatusDegraded
					h.registry.UpdateNode(updatedNode)
					
					fmt.Printf("Node %s marked as degraded after %d consecutive failures\n",
						nodeID, metrics.ConsecutiveFailures)
				}
			}
		}
		
		if metrics.ConsecutiveFailures >= 5 {
			// After 5 consecutive failures, mark node as offline and schedule reconnection
			if node.Status != NodeStatusOffline {
				newStatus, _ := h.determineStatusChange(node.Status, NodeStatusOffline,
					fmt.Sprintf("consecutive failures: %d", metrics.ConsecutiveFailures))
				
				if newStatus == NodeStatusOffline {
					updatedNode := node
					updatedNode.Status = NodeStatusOffline
					h.registry.UpdateNode(updatedNode)
					
					fmt.Printf("Node %s marked as offline after %d consecutive failures\n",
						nodeID, metrics.ConsecutiveFailures)
						
					// Schedule reconnection attempt
					h.reconnectionManager.ScheduleReconnection(nodeID)
				}
			}
		}
	}
}
// checkAllNodesHealth enhanced implementation
func (h *HealthMonitor) checkAllNodesHealth() {
	nodes := h.registry.GetAllNodes()
	
	// Track degradation to determine overall system status
	degradedCount := 0
	offlineCount := 0
	
	for _, node := range nodes {
		// Skip our own node, we know our health directly
		if node.ID == h.getLocalNodeID() {
			continue
		}
		
		// Check node health with enhanced error handling
		status, err := h.CheckNodeHealth(node)
		
		// Track degraded and offline nodes
		if status == NodeStatusDegraded {
			degradedCount++
		} else if status == NodeStatusOffline {
			offlineCount++
		}
		
		// Skip status update if health check failed
		if err != nil {
			fmt.Printf("Health check failed for node %s (%s): %v\n",
				node.Name, node.ID, err)
			continue
		}
		
		// Update node status if it has changed, but only if transition is valid
		if status != node.Status && h.statusController != nil {
			// Validate the transition using the status controller
			if !h.statusController.IsValidTransition(node.Status, status) {
				fmt.Printf("Invalid status transition attempted for node %s (%s): %s -> %s\n",
					node.Name, node.ID, node.Status, status)
				continue
			}
			
			fmt.Printf("Node %s (%s) status changed: %s -> %s\n",
				node.Name, node.ID, node.Status, status)
			
			updatedNode := node
			updatedNode.Status = status
			updatedNode.LastHeartbeat = time.Now() // Also update heartbeat time
			h.registry.UpdateNode(updatedNode)
		} else if status != node.Status {
			// If we don't have a status controller yet (during initialization), allow the update
			updatedNode := node
			updatedNode.Status = status
			updatedNode.LastHeartbeat = time.Now()
			h.registry.UpdateNode(updatedNode)
		}
	}
	
	// Log overall system status if there are degraded or offline nodes
	if degradedCount > 0 || offlineCount > 0 {
		totalNodes := len(nodes) - 1 // Exclude self
		healthyNodes := totalNodes - degradedCount - offlineCount
		fmt.Printf("Cluster health: %d total nodes, %d healthy, %d degraded, %d offline\n",
			totalNodes, healthyNodes, degradedCount, offlineCount)
	}
}

// getLocalNodeID retrieves the local node ID from the registry
func (h *HealthMonitor) getLocalNodeID() string {
	// In a real implementation, this would get the local node ID from the registry
	// For now, we'll just return an empty string which won't match any node ID
	return ""
}

// determineStatusChange decides what status a node should have based on current conditions
// and validates the transition using the status controller if available
func (h *HealthMonitor) determineStatusChange(currentStatus, proposedStatus NodeStatus, reason string) (NodeStatus, error) {
	// Log the status change attempt
	fmt.Printf("Attempting status change: %s -> %s (reason: %s)\n",
		currentStatus, proposedStatus, reason)
	
	// If we have a status controller, validate the transition
	if h.statusController != nil {
		if !h.statusController.IsValidTransition(currentStatus, proposedStatus) {
			return currentStatus, fmt.Errorf("invalid status transition from %s to %s: %s",
				currentStatus, proposedStatus, reason)
		}
	}
	
	return proposedStatus, nil
}

// CheckNodeHealth implements the HealthChecker interface with enhanced error handling
func (h *HealthMonitor) CheckNodeHealth(node NodeInfo) (NodeStatus, error) {
	// Build the health check URL
	healthURL := fmt.Sprintf("http://%s:%d/api/health", node.Addr.String(), node.ApiPort)
	
	// Start tracking latency and metrics
	startTime := time.Now()
	
	// Get metrics for this node (initialize if not exists)
	metrics, exists := h.communicationMetrics[node.ID]
	if !exists {
		metrics = h.InitNodeMetrics(node.ID)
	}
	
	// Determine appropriate timeout based on historical latency
	var timeout time.Duration = 3 * time.Second
	if exists && metrics.AvgLatency > 0 {
		// Scale timeout based on average latency with limits
		adaptiveTimeout := metrics.AvgLatency * 3
		if adaptiveTimeout > 500*time.Millisecond && adaptiveTimeout < 10*time.Second {
			timeout = adaptiveTimeout
		}
	}
	
	// Create a context with timeout for the request
	ctx, cancel := context.WithTimeout(h.ctx, timeout)
	defer cancel()
	
	// Prepare request
	req, err := http.NewRequestWithContext(ctx, "GET", healthURL, nil)
	if err != nil {
		// Error creating request
		commErr := cerrors.NewNodeCommunicationError(
			node.ID,
			fmt.Sprintf("%s:%d", node.Addr.String(), node.ApiPort),
			"health-check-request",
			fmt.Sprintf("Failed to create health check request for node %s", node.Name),
			cerrors.TemporaryError,
			cerrors.InternalError,
			err,
		).WithMetadata(map[string]string{
			"node_name": node.Name,
			"operation": "health-check",
			"timeout_ms": fmt.Sprintf("%d", timeout.Milliseconds()),
		})
		
		// Log and record failure
		cerrors.LogError(commErr)
		h.RecordCommunicationFailure(node.ID, commErr)
		
		// If we can't create the request, consider the node offline
		return h.determineStatusChange(node.Status, NodeStatusOffline, "error creating health check request")
	}
	
	// Add diagnostic headers
	req.Header.Set("X-Cluster-Health-Check", "true")
	req.Header.Set("X-Source-Node", h.getLocalNodeID())
	req.Header.Set("X-Request-Timeout", fmt.Sprintf("%d", timeout.Milliseconds()))
	
	// Execute the request with error handling
	resp, err := h.client.Do(req)
	
	// Calculate and record latency
	latency := time.Since(startTime)
	
	// If the request failed, handle the error
	if err != nil {
		// Get proper error category and severity
		errorCategory := cerrors.ErrorCategoryFromError(err)
		errorSeverity := cerrors.ErrorSeverityFromError(err)
		
		// Create detailed error with context
		var errorMessage string
		
		// Customize error message based on error type
		switch errorCategory {
		case cerrors.TimeoutError:
			errorMessage = fmt.Sprintf("Health check timed out for node %s after %v",
				node.Name, timeout)
		case cerrors.ConnectionRefused:
			errorMessage = fmt.Sprintf("Connection refused by node %s", node.Name)
		case cerrors.NameResolution:
			errorMessage = fmt.Sprintf("Failed to resolve hostname for node %s", node.Name)
		case cerrors.NetworkTemporary:
			errorMessage = fmt.Sprintf("Temporary network error reaching node %s", node.Name)
		default:
			errorMessage = fmt.Sprintf("Error checking health of node %s: %v", node.Name, err)
		}
		
		// Create communication error with full context
		commErr := cerrors.NewNodeCommunicationError(
			node.ID,
			fmt.Sprintf("%s:%d", node.Addr.String(), node.ApiPort),
			"health-check-execute",
			errorMessage,
			errorSeverity,
			errorCategory,
			err,
		).WithMetadata(map[string]string{
			"node_name": node.Name,
			"timeout_ms": fmt.Sprintf("%d", timeout.Milliseconds()),
			"latency_ms": fmt.Sprintf("%d", latency.Milliseconds()),
		})
		
		// Record and log the failure
		h.RecordCommunicationFailure(node.ID, commErr)
		
		// Update degradation manager
		consecutiveFailures := metrics.ConsecutiveFailures
		if h.degradationManager != nil {
			h.degradationManager.RecordConsecutiveFailure(node.ID, consecutiveFailures, commErr)
		}
		
		// Determine appropriate status based on failure history
		// For first few failures, maintain current status
		const tempFailureThreshold = 2
		if metrics.ConsecutiveFailures <= tempFailureThreshold && node.Status == NodeStatusOnline {
			// Allow a couple of temporary failures before changing status
			return node.Status, commErr
		}
		
		// For persistent failures or multiple consecutive temporary failures
		const degradedThreshold = 3
		const offlineThreshold = 5
		
		if metrics.ConsecutiveFailures >= offlineThreshold {
			// After 5+ failures, mark as offline
			return h.determineStatusChange(node.Status, NodeStatusOffline, errorMessage)
		} else if metrics.ConsecutiveFailures >= degradedThreshold {
			// After 3+ failures but less than 5, mark as degraded
			return h.determineStatusChange(node.Status, NodeStatusDegraded, errorMessage)
		}
		
		// For other cases, don't change status yet
		return node.Status, commErr
	}
	
	// Success - process the response
	defer resp.Body.Close()
	
	// Record the successful communication with latency data
	h.RecordCommunicationSuccess(node.ID, latency)
	
	// Check status code
	if resp.StatusCode != http.StatusOK {
		// Non-200 response indicates a problem
		statusErr := fmt.Errorf("health check returned non-OK status: %s", resp.Status)
		h.RecordCommunicationFailure(node.ID, statusErr)
		return h.determineStatusChange(node.Status, NodeStatusOffline,
			fmt.Sprintf("health check returned status: %s", resp.Status))
	}
	
	// Attempt to decode the health response
	var healthStatus NodeHealthStatus
	err = json.NewDecoder(resp.Body).Decode(&healthStatus)
	if err != nil {
		// Error decoding the health data
		decodeErr := cerrors.NewNodeCommunicationError(
			node.ID,
			fmt.Sprintf("%s:%d", node.Addr.String(), node.ApiPort),
			"health-check-decode",
			"Error decoding health response",
			cerrors.TemporaryError,
			cerrors.InternalError,
			err,
		)
		
		h.RecordCommunicationFailure(node.ID, decodeErr)
		return h.determineStatusChange(node.Status, NodeStatusOffline, "error decoding health response")
	}
	
	// Update health status cache
	h.mu.Lock()
	h.healthStatusCache[node.ID] = &healthStatus
	h.mu.Unlock()
	
	// If latency has increased significantly, record it for degradation tracking
	if metrics.AvgLatency > 0 && h.degradationManager != nil {
		baselineLatency := metrics.AvgLatency
		
		// If current latency is 2x or more than average, record as increased latency
		if latency > baselineLatency*2 {
			h.degradationManager.RecordLatencyIncrease(node.ID, latency, baselineLatency)
		}
	}
	
	// Determine if node should be marked as busy based on load metrics
	if healthStatus.CPUUsagePercent > 80.0 || healthStatus.ActiveRequests > 10 {
		if node.Status == NodeStatusOnline {
			return h.determineStatusChange(node.Status, NodeStatusBusy, "high load detected")
		}
	} else if node.Status == NodeStatusBusy {
		// If the node was busy but load has decreased, mark it as online
		return h.determineStatusChange(node.Status, NodeStatusOnline, "load returned to normal")
	} else if node.Status == NodeStatusDegraded {
		// If the node was degraded but now showing good health, mark it as online
		return h.determineStatusChange(node.Status, NodeStatusOnline, "node recovered from degraded state")
	}
	
	// Return the status from the health data
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

// ScheduleReconnection adds a node to the reconnection queue with enhanced tracking
func (mgr *NodeReconnectionManager) ScheduleReconnection(nodeID string) {
	mgr.mu.Lock()
	defer mgr.mu.Unlock()

	// Retrieve or initialize retry count
	retryCount := mgr.nodeRetries[nodeID]
	mgr.nodeRetries[nodeID] = retryCount + 1

	// Calculate next retry time with exponential backoff
	backoffDuration := mgr.retryPolicy.CalculateBackoff(retryCount)
	
	// Add jitter to prevent thundering herd problems (all nodes reconnecting simultaneously)
	jitterFactor := float64(0.1 + (0.2 * rand.Float64())) // 10-30% jitter
	jitterDuration := time.Duration(float64(backoffDuration) * jitterFactor)
	adjustedBackoff := backoffDuration + jitterDuration
	
	nextRetry := time.Now().Add(adjustedBackoff)
	mgr.nextRetryTime[nodeID] = nextRetry
	mgr.reconnectionQueue[nodeID] = nextRetry

	// Get node info if available for better logging
	var nodeName string
	if node, exists := mgr.parentMonitor.registry.GetNode(nodeID); exists {
		nodeName = fmt.Sprintf("%s (%s)", node.Name, node.ID)
	} else {
		nodeName = nodeID
	}

	// Log warning if approaching retry limit
	if retryCount+1 >= mgr.retryPolicy.MaxRetries-1 {
		fmt.Printf("WARNING: Node %s approaching max reconnection attempts (%d of %d)\n",
			nodeName, retryCount+1, mgr.retryPolicy.MaxRetries)
	}

	fmt.Printf("Scheduled reconnection for node %s in %s (attempt #%d of max %d)\n",
		nodeName, adjustedBackoff, retryCount+1, mgr.retryPolicy.MaxRetries)
}

// RunReconnectionTasks periodically attempts to reconnect to failed nodes
func (mgr *NodeReconnectionManager) RunReconnectionTasks() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-mgr.parentMonitor.ctx.Done():
			return
		case <-ticker.C:
			mgr.processReconnectionQueue()
		}
	}
}

// processReconnectionQueue attempts reconnection for nodes that are due
func (mgr *NodeReconnectionManager) processReconnectionQueue() {
	now := time.Now()
	
	mgr.mu.Lock()
	// Find nodes that are due for reconnection
	nodesToTry := make([]string, 0)
	for nodeID, nextRetry := range mgr.nextRetryTime {
		if nextRetry.Before(now) {
			nodesToTry = append(nodesToTry, nodeID)
		}
	}
	mgr.mu.Unlock()

	// Attempt reconnection for each node
	for _, nodeID := range nodesToTry {
		success := mgr.attemptReconnection(nodeID)
		
		mgr.mu.Lock()
		if success {
			// On success, remove node from retry tracking
			delete(mgr.nodeRetries, nodeID)
			delete(mgr.nextRetryTime, nodeID)
			delete(mgr.reconnectionQueue, nodeID)
			fmt.Printf("Successfully reconnected to node %s\n", nodeID)
		} else {
			// Failed attempt, check if we should keep trying
			retryCount := mgr.nodeRetries[nodeID]
			if retryCount >= mgr.retryPolicy.MaxRetries {
				fmt.Printf("Reached maximum retry attempts (%d) for node %s\n",
					mgr.retryPolicy.MaxRetries, nodeID)
				delete(mgr.nodeRetries, nodeID)
				delete(mgr.nextRetryTime, nodeID)
				delete(mgr.reconnectionQueue, nodeID)
			} else {
				// Schedule next attempt with exponential backoff
				backoff := mgr.retryPolicy.CalculateBackoff(retryCount)
				mgr.nextRetryTime[nodeID] = time.Now().Add(backoff)
				fmt.Printf("Reconnection attempt failed for node %s, retrying in %s\n",
					nodeID, backoff)
			}
		}
		mgr.mu.Unlock()
	}
}

// attemptReconnection tries to reconnect to a specific node with enhanced error handling
func (mgr *NodeReconnectionManager) attemptReconnection(nodeID string) bool {
	// Get node information from registry
	node, exists := mgr.parentMonitor.registry.GetNode(nodeID)
	if !exists {
		fmt.Printf("Cannot reconnect to node %s: node not found in registry\n", nodeID)
		return false
	}
	
	var nodeName string = fmt.Sprintf("%s (%s)", node.Name, node.ID)
	fmt.Printf("Attempting to reconnect to node %s at %s:%d\n", nodeName, node.Addr.String(), node.ApiPort)

	// Build the health check URL
	healthURL := fmt.Sprintf("http://%s:%d/api/health", node.Addr.String(), node.ApiPort)
	
	// Progressive timeout approach - increase timeout with each retry
	retryCount := mgr.nodeRetries[nodeID]
	timeout := time.Duration(3+retryCount/2) * time.Second // Scale timeout with retry count
	if timeout > 15*time.Second {
		timeout = 15 * time.Second // Cap at 15 seconds
	}
	
	// Create a context with calculated timeout for the request with improved error handling
	ctx, cancel, wrapErr := cerrors.CreateTimeoutContext(context.Background(), "node-reconnection", timeout)
	defer cancel()
	
	// Try to establish connection with detailed error tracking
	startTime := time.Now() // Track latency from start of attempt
	
	// Set up request with connection tracing headers
	req, err := http.NewRequestWithContext(ctx, "GET", healthURL, nil)
	if err != nil {
		cerrs := cerrors.NewConnectionError(
			nodeID,
			"failed to create reconnection request",
			cerrors.TemporaryError,
			err,
		)
		cerrors.LogError(cerrs)
		return false
	}
	
	// Add headers for tracking reconnection attempts
	req.Header.Set("X-Reconnection-Attempt", fmt.Sprintf("%d", retryCount+1))
	req.Header.Set("X-Reconnection-Max", fmt.Sprintf("%d", mgr.retryPolicy.MaxRetries))
	
	// Execute request with error tracing
	resp, err := mgr.parentMonitor.client.Do(req)
	reconnectLatency := time.Since(startTime)
	
	if err != nil {
		// Use timeout wrapper for better error context
		err = wrapErr(err)
		
		// Categorize error for better diagnostics using enhanced error utils
		category := cerrors.ErrorCategoryFromError(err)
		severity := cerrors.ErrorSeverityFromError(err)
		
		cerrs := cerrors.NewDetailedCommunicationError6(
			nodeID,
			fmt.Sprintf("Reconnection attempt failed: %v", err),
			severity,
			category,
			err,
			map[string]string{
				"attempt": fmt.Sprintf("%d", retryCount+1),
				"max_attempts": fmt.Sprintf("%d", mgr.retryPolicy.MaxRetries),
				"latency_ms": fmt.Sprintf("%d", reconnectLatency.Milliseconds()),
				"timeout_ms": fmt.Sprintf("%d", timeout.Milliseconds()),
			},
		)
		cerrors.LogErrorf(cerrs, "Reconnection attempt %d failed for node %s after %v: %v",
			retryCount+1, nodeName, reconnectLatency, err)
		return false
	}
	defer resp.Body.Close()
	
	// Check if the response is valid
	if resp.StatusCode != http.StatusOK {
		cerrs := cerrors.NewConnectionError(
			nodeID,
			fmt.Sprintf("node returned non-OK status: %s", resp.Status),
			cerrors.TemporaryError,
			fmt.Errorf("http status: %s", resp.Status),
		)
		cerrors.LogErrorf(cerrs, "Node %s returned status %s during reconnection attempt %d",
			nodeName, resp.Status, retryCount+1)
		return false
	}
	
	// Connection successful - update node status based on current state
	var newStatus NodeStatus
	if node.Status == NodeStatusDegraded {
		// If was degraded, transition to degraded->online
		newStatus, _ = mgr.parentMonitor.determineStatusChange(node.Status, NodeStatusOnline,
			"successful reconnection from degraded state")
	} else {
		// Otherwise transition from offline->online directly
		newStatus, _ = mgr.parentMonitor.determineStatusChange(node.Status, NodeStatusOnline,
			"successful reconnection")
	}
		
	if newStatus == NodeStatusOnline || newStatus == NodeStatusDegraded {
		// Update node status
		updatedNode := node
		updatedNode.Status = newStatus
		updatedNode.LastHeartbeat = time.Now()
		mgr.parentMonitor.registry.UpdateNode(updatedNode)
			
		fmt.Printf("Successfully reconnected to node %s after %s (attempt #%d). Status set to: %s\n",
			nodeName, reconnectLatency, retryCount+1, newStatus)
	}
	
	// Record successful connection metrics
	mgr.parentMonitor.RecordCommunicationSuccess(nodeID, reconnectLatency)
	
	return true
}