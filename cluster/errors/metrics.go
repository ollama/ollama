package cerrors

import (
	"fmt"
	"sync"
	"time"
)

// MetricsTracker tracks various metrics about error handling and recovery
type MetricsTracker struct {
	// TotalOperations is the total number of operations tracked
	TotalOperations int
	
	// SuccessfulOperations is the number of operations that succeeded
	SuccessfulOperations int
	
	// FailedOperations is the number of operations that failed
	FailedOperations int
	
	// SuccessfulRetries is the number of operations that succeeded after retry
	SuccessfulRetries int
	
	// FailedRetries is the number of operations that failed after all retries
	FailedRetries int
	
	// FirstAttemptSuccesses is the number of operations that succeeded on first try
	FirstAttemptSuccesses int
	
	// AverageAttemptsNeeded tracks the average number of attempts needed for success
	AverageAttemptsNeeded float64
	
	// ErrorsByCategory tracks errors by their category
	ErrorsByCategory map[ErrorCategory]int
	
	// LatencyHistory tracks the latency of recent operations
	LatencyHistory []time.Duration
	
	// MaxLatencyHistory is the maximum number of latency samples to keep
	MaxLatencyHistory int
	
	// AverageLatency is the average latency across operations
	AverageLatency time.Duration
	
	// MinLatency is the minimum observed latency
	MinLatency time.Duration
	
	// MaxLatency is the maximum observed latency
	MaxLatency time.Duration
	
	// CreatedAt is when this tracker was created
	CreatedAt time.Time
	
	// LastUpdated is when metrics were last updated
	LastUpdated time.Time
	
	// mu protects concurrent access to metrics
	mu sync.RWMutex
}

// NewMetricsTracker creates a new metrics tracker with default settings
func NewMetricsTracker() *MetricsTracker {
	now := time.Now()
	return &MetricsTracker{
		ErrorsByCategory: make(map[ErrorCategory]int),
		LatencyHistory:   make([]time.Duration, 0, 50),
		MaxLatencyHistory: 50,
		MinLatency:       time.Hour * 24, // Initialize to a high value
		CreatedAt:        now,
		LastUpdated:      now,
	}
}

// RecordSuccess records a successful operation with latency
func (m *MetricsTracker) RecordSuccess(latency time.Duration, attemptCount int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	m.TotalOperations++
	m.SuccessfulOperations++
	m.LastUpdated = time.Now()
	
	// Track if it was a first attempt or retry success
	if attemptCount == 1 {
		m.FirstAttemptSuccesses++
	} else if attemptCount > 1 {
		m.SuccessfulRetries++
	}
	
	// Update latency metrics
	m.recordLatency(latency)
	
	// Update average attempts needed for success
	n := float64(m.SuccessfulOperations)
	oldAvg := m.AverageAttemptsNeeded
	m.AverageAttemptsNeeded = oldAvg + (float64(attemptCount)-oldAvg)/n
}

// RecordFailure records a failed operation with error details
func (m *MetricsTracker) RecordFailure(err error, attemptCount int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	m.TotalOperations++
	m.FailedOperations++
	m.LastUpdated = time.Now()
	
	// Track if it was a regular failure or after retries
	if attemptCount > 1 {
		m.FailedRetries++
	}
	
	// Categorize and track the error
	if err != nil {
		category := ErrorCategoryFromError(err)
		m.ErrorsByCategory[category]++
	}
}

// RecordLatency records operation latency independently
func (m *MetricsTracker) RecordLatency(latency time.Duration) {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	m.recordLatency(latency)
	m.LastUpdated = time.Now()
}

// recordLatency is an internal method for tracking latency metrics
func (m *MetricsTracker) recordLatency(latency time.Duration) {
	// Store in history with circular buffer behavior
	if len(m.LatencyHistory) >= m.MaxLatencyHistory {
		// Remove oldest entry when at capacity
		m.LatencyHistory = m.LatencyHistory[1:]
	}
	m.LatencyHistory = append(m.LatencyHistory, latency)
	
	// Update min/max latency
	if latency < m.MinLatency {
		m.MinLatency = latency
	}
	if latency > m.MaxLatency {
		m.MaxLatency = latency
	}
	
	// Update average using exponential moving average for smooth updates
	if len(m.LatencyHistory) == 1 {
		m.AverageLatency = latency
	} else {
		alpha := 0.2 // Weight factor for new values (0.2 = 20% weight to new sample)
		m.AverageLatency = time.Duration(
			float64(m.AverageLatency)*(1-alpha) + float64(latency)*alpha)
	}
}

// GetSuccessRate returns the percentage of successful operations
func (m *MetricsTracker) GetSuccessRate() float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	if m.TotalOperations == 0 {
		return 0.0
	}
	return float64(m.SuccessfulOperations) / float64(m.TotalOperations) * 100.0
}

// GetErrorRates returns the error rates by category (percentage of total errors)
func (m *MetricsTracker) GetErrorRates() map[ErrorCategory]float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	rates := make(map[ErrorCategory]float64)
	
	if m.FailedOperations == 0 {
		return rates
	}
	
	for category, count := range m.ErrorsByCategory {
		rates[category] = float64(count) / float64(m.FailedOperations) * 100.0
	}
	
	return rates
}

// GetLatencyPercentile returns the latency at a given percentile
func (m *MetricsTracker) GetLatencyPercentile(percentile float64) time.Duration {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	if len(m.LatencyHistory) == 0 {
		return 0
	}
	
	// Create a copy of the latency history and sort it
	sortedLatencies := make([]time.Duration, len(m.LatencyHistory))
	copy(sortedLatencies, m.LatencyHistory)
	
	// Simple insertion sort since the array is typically small
	for i := 1; i < len(sortedLatencies); i++ {
		key := sortedLatencies[i]
		j := i - 1
		for j >= 0 && sortedLatencies[j] > key {
			sortedLatencies[j+1] = sortedLatencies[j]
			j--
		}
		sortedLatencies[j+1] = key
	}
	
	// Calculate the index for the percentile
	idx := int(float64(len(sortedLatencies)-1) * percentile / 100.0)
	if idx >= len(sortedLatencies) {
		idx = len(sortedLatencies) - 1
	}
	
	return sortedLatencies[idx]
}

// GetSummary returns a string summary of the metrics
func (m *MetricsTracker) GetSummary() string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	uptime := time.Since(m.CreatedAt).Round(time.Second)
	
	summary := fmt.Sprintf(
		"Metrics Summary (last %v):\n"+
			"- Operations: %d total, %d success (%.1f%%), %d failed\n"+
			"- Latency: %v avg, %v min, %v max\n"+
			"- Retries: %d successful, %d failed\n"+
			"- Average attempts needed: %.2f\n",
		uptime,
		m.TotalOperations, m.SuccessfulOperations, m.GetSuccessRate(), m.FailedOperations,
		m.AverageLatency.Round(time.Millisecond), m.MinLatency.Round(time.Millisecond), m.MaxLatency.Round(time.Millisecond),
		m.SuccessfulRetries, m.FailedRetries,
		m.AverageAttemptsNeeded)
	
	// Add error category breakdown if there are any errors
	if m.FailedOperations > 0 {
		summary += "Error categories:\n"
		for category, count := range m.ErrorsByCategory {
			percentage := float64(count) / float64(m.FailedOperations) * 100.0
			summary += fmt.Sprintf("  - %s: %d (%.1f%%)\n", category, count, percentage)
		}
	}
	
	return summary
}

// CommunicationMetrics tracks metrics for node-to-node communication
type CommunicationMetrics struct {
	// NodeID identifies the node these metrics belong to
	NodeID string
	
	// SuccessCount tracks successful communications
	SuccessCount int
	
	// FailureCount tracks failed communications
	FailureCount int
	
	// LastSuccess records when the last successful communication occurred
	LastSuccess time.Time
	
	// LastFailure records when the last failure occurred
	LastFailure time.Time
	
	// ConsecutiveFailures counts failures without an intervening success
	ConsecutiveFailures int
	
	// LatencyHistory stores recent latency measurements
	LatencyHistory []time.Duration
	
	// AvgLatency is the rolling average latency
	AvgLatency time.Duration
	
	// MaxLatency is the maximum observed latency
	MaxLatency time.Duration
	
	// MinLatency is the minimum observed latency
	MinLatency time.Duration
	
	// FailureCategories tracks failures by category
	FailureCategories map[ErrorCategory]int
	
	// LastCheck is when metrics were last updated
	LastCheck time.Time
	
	// mu guards concurrent access
	mu sync.RWMutex
}

// NewCommunicationMetrics creates a new metrics tracker for node communication
func NewCommunicationMetrics(nodeID string) *CommunicationMetrics {
	return &CommunicationMetrics{
		NodeID:             nodeID,
		LatencyHistory:     make([]time.Duration, 0, 10),
		MinLatency:         time.Hour, // Initialize to a high value
		FailureCategories:  make(map[ErrorCategory]int),
		LastCheck:          time.Now(),
	}
}

// RecordSuccess records a successful communication with latency
func (m *CommunicationMetrics) RecordSuccess(latency time.Duration) {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	// Update success metrics
	m.SuccessCount++
	m.ConsecutiveFailures = 0
	m.LastSuccess = time.Now()
	m.LastCheck = time.Now()
	
	// Update latency history
	// Keep only the last 10 measurements
	if len(m.LatencyHistory) >= 10 {
		m.LatencyHistory = m.LatencyHistory[1:]
	}
	m.LatencyHistory = append(m.LatencyHistory, latency)
	
	// Update min/max latency
	if latency < m.MinLatency {
		m.MinLatency = latency
	}
	if latency > m.MaxLatency {
		m.MaxLatency = latency
	}
	
	// Update average latency using exponential moving average
	alpha := 0.2 // Weight for new sample
	if m.AvgLatency == 0 {
		m.AvgLatency = latency
	} else {
		m.AvgLatency = time.Duration(
			(1-alpha)*float64(m.AvgLatency) + alpha*float64(latency))
	}
}

// RecordFailure records a failed communication with details
func (m *CommunicationMetrics) RecordFailure(err error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	// Update failure metrics
	m.FailureCount++
	m.ConsecutiveFailures++
	m.LastFailure = time.Now()
	m.LastCheck = time.Now()
	
	// Categorize the error
	if err != nil {
		category := ErrorCategoryFromError(err)
		m.FailureCategories[category]++
	}
}

// GetConsecutiveFailures returns the number of consecutive failures
func (m *CommunicationMetrics) GetConsecutiveFailures() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.ConsecutiveFailures
}

// GetLatency returns the average, min, and max latency
func (m *CommunicationMetrics) GetLatency() (avg, min, max time.Duration) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.AvgLatency, m.MinLatency, m.MaxLatency
}

// GetFailureRate returns the percentage of failures
func (m *CommunicationMetrics) GetFailureRate() float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	total := m.SuccessCount + m.FailureCount
	if total == 0 {
		return 0.0
	}
	return float64(m.FailureCount) / float64(total) * 100.0
}

// GetSummary returns a string summary of communication metrics
func (m *CommunicationMetrics) GetSummary() string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	total := m.SuccessCount + m.FailureCount
	successRate := 100.0
	if total > 0 {
		successRate = float64(m.SuccessCount) / float64(total) * 100.0
	}
	
	summary := fmt.Sprintf(
		"Communication with %s:\n"+
			"- Status: %s\n"+
			"- Success rate: %.1f%% (%d/%d)\n"+
			"- Consecutive failures: %d\n"+
			"- Latency: %v avg, %v min, %v max\n",
		m.NodeID,
		m.getStatusIndicator(),
		successRate, m.SuccessCount, total,
		m.ConsecutiveFailures,
		m.AvgLatency.Round(time.Millisecond),
		m.MinLatency.Round(time.Millisecond),
		m.MaxLatency.Round(time.Millisecond))
	
	// Add error category breakdown if there are any failures
	if m.FailureCount > 0 {
		summary += "Failure categories:\n"
		for category, count := range m.FailureCategories {
			percentage := float64(count) / float64(m.FailureCount) * 100.0
			summary += fmt.Sprintf("  - %s: %d (%.1f%%)\n", category, count, percentage)
		}
	}
	
	return summary
}

// getStatusIndicator returns a human-readable status based on metrics
func (m *CommunicationMetrics) getStatusIndicator() string {
	if m.ConsecutiveFailures == 0 {
		return "Healthy"
	}
	
	if m.ConsecutiveFailures < 3 {
		return "Warning"
	}
	
	if m.ConsecutiveFailures < 5 {
		return "Degraded"
	}
	
	return "Failed"
}