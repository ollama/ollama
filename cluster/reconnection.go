package cluster

import (
	"context"
	"fmt"
	"net/http"
	"time"
	
	cerrors "github.com/ollama/ollama/cluster/errors"
)

// ReconnectionSystem handles automatic reconnection to failed nodes
type ReconnectionSystem struct {
	// healthMonitor is the parent health monitor
	healthMonitor *HealthMonitor
	
	// reconnectionManager manages the reconnection queue and retry logic
	reconnectionManager *cerrors.ReconnectionManager
	
	// degradationManager tracks node degradation during reconnection attempts
	degradationManager *cerrors.GracefulDegradation
	
	// metrics tracks reconnection statistics
	metrics *cerrors.MetricsTracker
}

// NewReconnectionSystem creates a new system for handling node reconnections
func NewReconnectionSystem(healthMonitor *HealthMonitor) *ReconnectionSystem {
	return &ReconnectionSystem{
		healthMonitor:       healthMonitor,
		reconnectionManager: cerrors.NewReconnectionManager(),
		degradationManager:  cerrors.NewGracefulDegradation(cerrors.DegradationModeAdaptive),
		metrics:             cerrors.NewMetricsTracker(),
	}
}

// Start begins the reconnection system
func (rs *ReconnectionSystem) Start() {
	// Start a goroutine to process reconnection queue
	go rs.processReconnectionQueue()
}

// Stop halts the reconnection system
func (rs *ReconnectionSystem) Stop() {
	// No explicit stopping needed as it will stop when the context is cancelled
	fmt.Println("Stopping reconnection system")
}

// ScheduleReconnection adds a node to the reconnection queue
func (rs *ReconnectionSystem) ScheduleReconnection(nodeID string, lastError error) {
	fmt.Printf("Scheduling reconnection for node %s\n", nodeID)
	rs.reconnectionManager.ScheduleReconnection(nodeID, lastError)
}

// processReconnectionQueue periodically attempts to reconnect to failed nodes
func (rs *ReconnectionSystem) processReconnectionQueue() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-rs.healthMonitor.ctx.Done():
			return
		case <-ticker.C:
			nodesToReconnect := rs.reconnectionManager.GetNodesForReconnection()
			for _, nodeID := range nodesToReconnect {
				node, exists := rs.healthMonitor.registry.GetNode(nodeID)
				if !exists {
					continue
				}
				
				// Attempt to reconnect to the node
				success := rs.attemptNodeReconnection(node)
				rs.reconnectionManager.RecordReconnectionResult(nodeID, success)
				
				// Update metrics
				if success {
					rs.metrics.RecordSuccess(0, rs.reconnectionManager.GetNodeRetryCount(nodeID))
					rs.degradationManager.MarkNodeRecovered(nodeID)
				} else {
					rs.metrics.RecordFailure(nil, rs.reconnectionManager.GetNodeRetryCount(nodeID))
				}
			}
		}
	}
}

// attemptNodeReconnection makes a reconnection attempt to a node
func (rs *ReconnectionSystem) attemptNodeReconnection(node NodeInfo) bool {
	fmt.Printf("Attempting to reconnect to node %s (%s)\n", node.Name, node.ID)
	
	// Get the appropriate retry configuration based on node history
	retryConfig := cerrors.NewDefaultRetryConfig().
		WithMaxRetries(5).
		WithBaseDelay(200 * time.Millisecond).
		WithMaxDelay(10 * time.Second)
	
	// Create retry function with proper context
	retryFunc := func(ctx context.Context) error {
		return rs.singleReconnectionAttempt(ctx, node)
	}
	
	// Execute with retry logic
	err := cerrors.RunWithRetry(rs.healthMonitor.ctx, node.ID, retryConfig, retryFunc)
	
	// Return success if no error
	return err == nil
}

// singleReconnectionAttempt performs a single connection attempt
func (rs *ReconnectionSystem) singleReconnectionAttempt(ctx context.Context, node NodeInfo) error {
	startTime := time.Now()
	
	// Prepare the health check URL
	healthURL := fmt.Sprintf("http://%s:%d/api/health", node.Addr.String(), node.ApiPort)
	
	// Create the request with reconnection-specific headers
	req, err := http.NewRequestWithContext(ctx, "GET", healthURL, nil)
	if err != nil {
		return cerrors.NewNodeCommunicationError(
			node.ID,
			fmt.Sprintf("%s:%d", node.Addr.String(), node.ApiPort),
			"reconnection-request",
			fmt.Sprintf("Failed to create reconnection request for %s", node.Name),
			cerrors.TemporaryError,
			cerrors.InternalError,
			err,
		)
	}
	
	// Add reconnection headers
	req.Header.Set("X-Reconnection-Attempt", "true")
	req.Header.Set("X-Source-Node", rs.healthMonitor.getLocalNodeID())
	
	// Execute the request
	resp, err := rs.healthMonitor.client.Do(req)
	latency := time.Since(startTime)
	
	// Handle request failure
	if err != nil {
		// Categorize the error properly
		errorCategory := cerrors.ErrorCategoryFromError(err)
		errorSeverity := cerrors.ErrorSeverityFromError(err)
		
		// Create detailed error
		reconnectErr := cerrors.NewNodeCommunicationError(
			node.ID,
			fmt.Sprintf("%s:%d", node.Addr.String(), node.ApiPort),
			"reconnection-execute",
			fmt.Sprintf("Reconnection attempt failed to %s: %v", node.Name, err),
			errorSeverity,
			errorCategory,
			err,
		).WithMetadata(map[string]string{
			"node_name":  node.Name,
			"latency_ms": fmt.Sprintf("%d", latency.Milliseconds()),
		})
		
		return reconnectErr
	}
	defer resp.Body.Close()
	
	// Check response
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("reconnection health check returned non-OK status: %s", resp.Status)
	}
	
	// Successful reconnection - update node status
	rs.healthMonitor.RecordCommunicationSuccess(node.ID, latency)
	
	// Update node status based on previous state
	var updatedStatus NodeStatus
	switch node.Status {
	case NodeStatusOffline:
		// If node was offline, transition through degraded first
		updatedStatus = NodeStatusDegraded
	case NodeStatusDegraded:
		// If node was degraded, return to online
		updatedStatus = NodeStatusOnline
	default:
		// For other states, go to online
		updatedStatus = NodeStatusOnline
	}
	
	// Use status controller to validate transition
	newStatus := updatedStatus
	if rs.healthMonitor.statusController != nil {
		validStatus, err := rs.healthMonitor.statusController.TransitionStatus(
			node.Status, updatedStatus)
		if err == nil {
			newStatus = validStatus
		}
	}
	
	// Update the node
	if newStatus != node.Status {
		updatedNode := node
		updatedNode.Status = newStatus
		updatedNode.LastHeartbeat = time.Now()
		rs.healthMonitor.registry.UpdateNode(updatedNode)
		
		fmt.Printf("Successful reconnection to node %s. Status changed from %s to %s\n",
			node.Name, node.Status, newStatus)
	}
	
	return nil
}

// GetReconnectionStats returns statistics about reconnection attempts
func (rs *ReconnectionSystem) GetReconnectionStats() map[string]interface{} {
	return map[string]interface{}{
		"success_rate":       rs.metrics.GetSuccessRate(),
		"total_attempts":     rs.metrics.TotalOperations,
		"successful":         rs.metrics.SuccessfulOperations,
		"failed":             rs.metrics.FailedOperations,
		"avg_attempts":       rs.metrics.AverageAttemptsNeeded,
		"pending_reconnects": len(rs.reconnectionManager.GetNodesForReconnection()),
	}
}

// IsPendingReconnection checks if a node is scheduled for reconnection
func (rs *ReconnectionSystem) IsPendingReconnection(nodeID string) bool {
	_, exists := rs.reconnectionManager.GetNextRetryTime(nodeID)
	return exists
}

// GetReconnectionSummary returns a human-readable summary of reconnection activity
func (rs *ReconnectionSystem) GetReconnectionSummary() string {
	pending := rs.reconnectionManager.GetNodesForReconnection()
	
	summary := fmt.Sprintf(
		"Reconnection summary:\n"+
			"- Success rate: %.1f%%\n"+
			"- Total reconnection attempts: %d\n"+
			"- Average attempts needed: %.1f\n"+
			"- Pending reconnections: %d\n",
		rs.metrics.GetSuccessRate(),
		rs.metrics.TotalOperations,
		rs.metrics.AverageAttemptsNeeded,
		len(pending),
	)
	
	if len(pending) > 0 {
		summary += "Pending reconnections:\n"
		for _, nodeID := range pending {
			nextRetry, _ := rs.reconnectionManager.GetNextRetryTime(nodeID)
			retryIn := time.Until(nextRetry).Round(time.Second)
			
			nodeName := nodeID
			if node, exists := rs.healthMonitor.registry.GetNode(nodeID); exists {
				nodeName = node.Name
			}
			
			summary += fmt.Sprintf("  - %s: next attempt in %v\n", nodeName, retryIn)
		}
	}
	
	return summary
}