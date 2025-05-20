package cerrors

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"
)

// ReconnectionManager handles automatic reconnection attempts to failed nodes
type ReconnectionManager struct {
	// nodeRetries tracks retry counts per node
	nodeRetries map[string]int
	
	// reconnectionQueue holds nodes scheduled for reconnection
	// The time.Time value indicates when to attempt the next reconnection
	reconnectionQueue map[string]time.Time
	
	// standardPolicy is the default retry policy
	standardPolicy *RetryConfig
	
	// customPolicies are node-specific retry policies
	customPolicies map[string]*RetryConfig
	
	// nodeFailures tracks detailed failure information
	nodeFailures map[string]*NodeFailureInfo
	
	// mu protects concurrent access to the manager's state
	mu sync.RWMutex
	
	// metrics tracks reconnection statistics
	metrics ReconnectionMetrics
}

// ReconnectionMetrics tracks statistics about reconnection attempts
type ReconnectionMetrics struct {
	// TotalAttempts counts all reconnection attempts
	TotalAttempts int
	
	// SuccessfulReconnections counts successful reconnection attempts
	SuccessfulReconnections int
	
	// FailedReconnections counts failed reconnection attempts
	FailedReconnections int
	
	// AverageAttemptsBeforeSuccess tracks average attempts needed for success
	AverageAttemptsBeforeSuccess float64
	
	// ReconnectionsByCategory tracks reconnection attempts by error category
	ReconnectionsByCategory map[ErrorCategory]int
	
	// LastSuccessfulReconnection is when the last successful reconnection occurred
	LastSuccessfulReconnection time.Time
}

// NewReconnectionManager creates a new manager for handling reconnections
func NewReconnectionManager() *ReconnectionManager {
	return &ReconnectionManager{
		nodeRetries:       make(map[string]int),
		reconnectionQueue: make(map[string]time.Time),
		standardPolicy:    NewDefaultRetryConfig(),
		customPolicies:    make(map[string]*RetryConfig),
		nodeFailures:      make(map[string]*NodeFailureInfo),
		metrics: ReconnectionMetrics{
			ReconnectionsByCategory: make(map[ErrorCategory]int),
		},
	}
}

// ScheduleReconnection adds a node to the reconnection queue
func (m *ReconnectionManager) ScheduleReconnection(nodeID string, err error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	// Get current retry count or initialize to 0
	retryCount, exists := m.nodeRetries[nodeID]
	if !exists {
		retryCount = 0
	}
	
	// Get the appropriate policy (custom or standard)
	policy := m.getRetryPolicy(nodeID)
	
	// Check if we've exceeded max retries
	if retryCount >= policy.MaxRetries {
		// Log that we're not scheduling because max retries reached
		fmt.Printf("Not scheduling reconnection for node %s: max retries (%d) reached\n",
			nodeID, policy.MaxRetries)
		return
	}
	
	// Increment retry count for next attempt
	m.nodeRetries[nodeID] = retryCount + 1
	
	// Calculate backoff with jitter for this retry attempt
	backoff := m.calculateBackoffWithJitter(nodeID, retryCount)
	nextRetry := time.Now().Add(backoff)
	
	// Add to reconnection queue
	m.reconnectionQueue[nodeID] = nextRetry
	
	// Get or create node failure info
	failureInfo, exists := m.nodeFailures[nodeID]
	if !exists {
		failureInfo = NewNodeFailureInfo(nodeID)
		m.nodeFailures[nodeID] = failureInfo
	}
	
	// Record the failure and set next retry time
	failureInfo.RecordFailure(err)
	failureInfo.SetNextRetryTime(policy)
	
	// Track reconnection by error category
	if err != nil {
		category := ErrorCategoryFromError(err)
		m.metrics.ReconnectionsByCategory[category]++
	}
	
	m.metrics.TotalAttempts++
	
	fmt.Printf("Scheduled reconnection for node %s in %v (attempt #%d/%d)\n",
		nodeID, backoff, retryCount+1, policy.MaxRetries)
}

// GetNodesForReconnection returns a list of node IDs that are due for reconnection
func (m *ReconnectionManager) GetNodesForReconnection() []string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	now := time.Now()
	nodesToReconnect := []string{}
	
	for nodeID, nextRetry := range m.reconnectionQueue {
		if now.After(nextRetry) {
			nodesToReconnect = append(nodesToReconnect, nodeID)
		}
	}
	
	return nodesToReconnect
}

// RecordReconnectionResult updates metrics based on reconnection attempt result
func (m *ReconnectionManager) RecordReconnectionResult(nodeID string, successful bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	if successful {
		// On success, update metrics and clear retry state
		retryCount := m.nodeRetries[nodeID]
		m.metrics.SuccessfulReconnections++
		m.metrics.LastSuccessfulReconnection = time.Now()
		
		// Update average attempts
		n := float64(m.metrics.SuccessfulReconnections)
		oldAvg := m.metrics.AverageAttemptsBeforeSuccess
		m.metrics.AverageAttemptsBeforeSuccess = oldAvg + (float64(retryCount)-oldAvg)/n
		
		// Clear retry state
		delete(m.nodeRetries, nodeID)
		delete(m.reconnectionQueue, nodeID)
		
		// Reset failure info
		if failureInfo, exists := m.nodeFailures[nodeID]; exists {
			failureInfo.ResetRetries()
			failureInfo.RecordSuccess()
		}
		
		fmt.Printf("Successful reconnection to node %s after %d attempts\n", 
			nodeID, retryCount)
	} else {
		// On failure, update failure metrics
		m.metrics.FailedReconnections++
		fmt.Printf("Failed reconnection attempt for node %s\n", nodeID)
	}
}

// SetCustomRetryPolicy sets a custom retry policy for a specific node
func (m *ReconnectionManager) SetCustomRetryPolicy(nodeID string, policy *RetryConfig) {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	m.customPolicies[nodeID] = policy
}

// getRetryPolicy returns the appropriate retry policy for a node
func (m *ReconnectionManager) getRetryPolicy(nodeID string) *RetryConfig {
	if policy, exists := m.customPolicies[nodeID]; exists {
		return policy
	}
	return m.standardPolicy
}

// calculateBackoffWithJitter calculates backoff duration with jitter
func (m *ReconnectionManager) calculateBackoffWithJitter(nodeID string, retryCount int) time.Duration {
	policy := m.getRetryPolicy(nodeID)
	
	// Calculate base backoff with exponential growth
	baseDelay := float64(policy.BaseDelay)
	maxDelay := float64(policy.MaxDelay)
	
	// Calculate exponential backoff
	backoff := baseDelay * math.Pow(policy.Factor, float64(retryCount))
	
	// Cap at max delay
	if backoff > maxDelay {
		backoff = maxDelay
	}
	
	// Add jitter to prevent thundering herd problem
	// Full jitter: random value between 0 and calculated backoff
	jitter := rand.Float64() * backoff * policy.Jitter
	backoff = backoff + jitter
	
	return time.Duration(backoff)
}

// GetNodeRetryCount returns the current retry count for a node
func (m *ReconnectionManager) GetNodeRetryCount(nodeID string) int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	return m.nodeRetries[nodeID]
}

// GetNextRetryTime returns the scheduled time for the next reconnection attempt
func (m *ReconnectionManager) GetNextRetryTime(nodeID string) (time.Time, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	nextRetry, exists := m.reconnectionQueue[nodeID]
	return nextRetry, exists
}

// GetReconnectionMetrics returns metrics about reconnection attempts
func (m *ReconnectionManager) GetReconnectionMetrics() ReconnectionMetrics {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	// Return a copy to avoid concurrent modification
	metricsCopy := m.metrics
	
	// Copy the map to avoid race conditions
	metricsCopy.ReconnectionsByCategory = make(map[ErrorCategory]int)
	for k, v := range m.metrics.ReconnectionsByCategory {
		metricsCopy.ReconnectionsByCategory[k] = v
	}
	
	return metricsCopy
}

// HasPendingReconnections checks if there are any pending reconnections
func (m *ReconnectionManager) HasPendingReconnections() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	return len(m.reconnectionQueue) > 0
}

// GetNodeFailureInfo returns the failure info for a specific node
func (m *ReconnectionManager) GetNodeFailureInfo(nodeID string) (*NodeFailureInfo, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	info, exists := m.nodeFailures[nodeID]
	return info, exists
}

// RunWithRetry executes a function with automatic retries based on the retry policy
func RunWithRetry(ctx context.Context, nodeID string, policy *RetryConfig, 
	operation func(context.Context) error) error {
	
	var lastErr error
	
	for attempt := 0; attempt <= policy.MaxRetries; attempt++ {
		// For retries (not first attempt), sleep with backoff
		if attempt > 0 {
			backoff := policy.CalculateBackoff(attempt - 1)
			select {
			case <-ctx.Done():
				// Context cancelled/timedout during backoff
				return fmt.Errorf("operation cancelled during backoff: %w", ctx.Err())
			case <-time.After(backoff):
				// Backoff period completed
			}
		}
		
		// Create a context for this attempt
		attemptCtx := ctx
		if attempt > 0 {
			// Log retry attempt
			fmt.Printf("Retry attempt %d/%d for node %s\n", 
				attempt+1, policy.MaxRetries, nodeID)
		}
		
		// Execute the operation
		err := operation(attemptCtx)
		if err == nil {
			// Success!
			return nil
		}
		
		// Record the error for potential return
		lastErr = err
		
		// Check if the error is permanent and we should stop retrying
		severity := ErrorSeverityFromError(err)
		if severity == PersistentError {
			return fmt.Errorf("permanent error, halting retries: %w", err)
		}
		
		// Check for context cancellation
		if ctx.Err() != nil {
			return fmt.Errorf("operation cancelled: %w", ctx.Err())
		}
		
		// Log the error and continue retrying
		fmt.Printf("Attempt %d failed for node %s: %v\n", 
			attempt+1, nodeID, err)
	}
	
	// All retries failed
	return fmt.Errorf("all %d retry attempts failed: %w", 
		policy.MaxRetries, lastErr)
}