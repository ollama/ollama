package cerrors

import (
	"sync"
	"time"
)

// NodeFailureInfo tracks failure information for a single node
type NodeFailureInfo struct {
	// NodeID identifies the node this info belongs to
	NodeID string
	
	// ConsecutiveFailures counts how many failures occurred without a success
	ConsecutiveFailures int
	
	// TotalFailures counts the total number of failures over time
	TotalFailures int
	
	// LastFailureTime records when the last failure occurred
	LastFailureTime time.Time
	
	// FailureCategories tracks failure counts by category
	FailureCategories map[ErrorCategory]int
	
	// LastError stores the most recent error
	LastError error
	
	// NextRetryTime indicates when the next retry should occur
	NextRetryTime time.Time
	
	// RetryCount tracks how many retries have been attempted
	RetryCount int
	
	// mu protects concurrent access to this data
	mu sync.RWMutex
}

// NewNodeFailureInfo creates a new failure tracking object for a node
func NewNodeFailureInfo(nodeID string) *NodeFailureInfo {
	return &NodeFailureInfo{
		NodeID:           nodeID,
		FailureCategories: make(map[ErrorCategory]int),
	}
}

// RecordFailure records a new failure with proper categorization
func (nf *NodeFailureInfo) RecordFailure(err error) {
	if err == nil {
		return
	}
	
	nf.mu.Lock()
	defer nf.mu.Unlock()
	
	nf.ConsecutiveFailures++
	nf.TotalFailures++
	nf.LastFailureTime = time.Now()
	nf.LastError = err
	
	// Categorize the error
	category := ErrorCategoryFromError(err)
	nf.FailureCategories[category]++
}

// RecordSuccess records a successful operation, resetting consecutive failures
func (nf *NodeFailureInfo) RecordSuccess() {
	nf.mu.Lock()
	defer nf.mu.Unlock()
	
	nf.ConsecutiveFailures = 0
}

// GetConsecutiveFailures returns the current consecutive failure count
func (nf *NodeFailureInfo) GetConsecutiveFailures() int {
	nf.mu.RLock()
	defer nf.mu.RUnlock()
	
	return nf.ConsecutiveFailures
}

// SetNextRetryTime schedules the next retry time with backoff
func (nf *NodeFailureInfo) SetNextRetryTime(retryConfig *RetryConfig) {
	nf.mu.Lock()
	defer nf.mu.Unlock()
	
	nf.RetryCount++
	backoff := retryConfig.CalculateBackoff(nf.RetryCount)
	nf.NextRetryTime = time.Now().Add(backoff)
}

// ShouldRetryNow checks if it's time to retry based on the next retry time
func (nf *NodeFailureInfo) ShouldRetryNow() bool {
	nf.mu.RLock()
	defer nf.mu.RUnlock()
	
	return time.Now().After(nf.NextRetryTime)
}

// ResetRetries resets the retry counter and clears the next retry time
func (nf *NodeFailureInfo) ResetRetries() {
	nf.mu.Lock()
	defer nf.mu.Unlock()
	
	nf.RetryCount = 0
	nf.NextRetryTime = time.Time{}
}