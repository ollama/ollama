package monitoring

import (
	"fmt"
	"sync"
	"time"
)

// MetricType represents the type of metric
type MetricType string

const (
	MetricTypeCounter   MetricType = "counter"
	MetricTypeGauge     MetricType = "gauge"
	MetricTypeHistogram MetricType = "histogram"
)

// Metric represents a single metric data point
type Metric struct {
	Name      string                 `json:"name"`
	Type      MetricType             `json:"type"`
	Value     float64                `json:"value"`
	Tags      map[string]string      `json:"tags"`
	Timestamp time.Time              `json:"timestamp"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// PerformanceMetrics tracks system performance
type PerformanceMetrics struct {
	// Request metrics
	TotalRequests       int64         `json:"total_requests"`
	SuccessfulRequests  int64         `json:"successful_requests"`
	FailedRequests      int64         `json:"failed_requests"`
	AverageResponseTime time.Duration `json:"average_response_time"`

	// Token metrics
	TotalInputTokens  int64   `json:"total_input_tokens"`
	TotalOutputTokens int64   `json:"total_output_tokens"`
	TotalCost         float64 `json:"total_cost"`

	// Provider metrics
	ProviderStats map[string]*ProviderMetrics `json:"provider_stats"`

	// System metrics
	CPUUsage    float64 `json:"cpu_usage"`
	MemoryUsage float64 `json:"memory_usage"`
	DiskUsage   float64 `json:"disk_usage"`

	// Time range
	StartTime time.Time `json:"start_time"`
	EndTime   time.Time `json:"end_time"`
}

// ProviderMetrics tracks metrics per provider
type ProviderMetrics struct {
	ProviderName        string        `json:"provider_name"`
	RequestCount        int64         `json:"request_count"`
	SuccessCount        int64         `json:"success_count"`
	FailureCount        int64         `json:"failure_count"`
	AverageLatency      time.Duration `json:"average_latency"`
	TotalTokens         int64         `json:"total_tokens"`
	TotalCost           float64       `json:"total_cost"`
	LastRequestTime     time.Time     `json:"last_request_time"`
	ErrorRate           float64       `json:"error_rate"`
	TokensPerSecond     float64       `json:"tokens_per_second"`
}

// MetricsCollector collects and aggregates metrics
type MetricsCollector struct {
	metrics    []Metric
	metricsMu  sync.RWMutex

	performance *PerformanceMetrics
	perfMu      sync.RWMutex

	startTime time.Time
}

// NewMetricsCollector creates a new metrics collector
func NewMetricsCollector() *MetricsCollector {
	return &MetricsCollector{
		metrics:    make([]Metric, 0, 1000),
		performance: &PerformanceMetrics{
			ProviderStats: make(map[string]*ProviderMetrics),
			StartTime:     time.Now(),
		},
		startTime: time.Now(),
	}
}

// RecordMetric records a new metric
func (mc *MetricsCollector) RecordMetric(metric Metric) {
	metric.Timestamp = time.Now()

	mc.metricsMu.Lock()
	mc.metrics = append(mc.metrics, metric)

	// Keep only last 10000 metrics
	if len(mc.metrics) > 10000 {
		mc.metrics = mc.metrics[len(mc.metrics)-10000:]
	}
	mc.metricsMu.Unlock()
}

// RecordRequest records a request metric
func (mc *MetricsCollector) RecordRequest(provider string, success bool, latency time.Duration, inputTokens, outputTokens int64, cost float64) {
	mc.perfMu.Lock()
	defer mc.perfMu.Unlock()

	// Update total metrics
	mc.performance.TotalRequests++
	if success {
		mc.performance.SuccessfulRequests++
	} else {
		mc.performance.FailedRequests++
	}

	// Update average response time
	mc.performance.AverageResponseTime = time.Duration(
		(int64(mc.performance.AverageResponseTime)*mc.performance.TotalRequests + int64(latency)) /
		(mc.performance.TotalRequests + 1),
	)

	// Update token metrics
	mc.performance.TotalInputTokens += inputTokens
	mc.performance.TotalOutputTokens += outputTokens
	mc.performance.TotalCost += cost

	// Update provider metrics
	if _, exists := mc.performance.ProviderStats[provider]; !exists {
		mc.performance.ProviderStats[provider] = &ProviderMetrics{
			ProviderName: provider,
		}
	}

	providerMetrics := mc.performance.ProviderStats[provider]
	providerMetrics.RequestCount++
	if success {
		providerMetrics.SuccessCount++
	} else {
		providerMetrics.FailureCount++
	}

	// Update average latency
	providerMetrics.AverageLatency = time.Duration(
		(int64(providerMetrics.AverageLatency)*providerMetrics.RequestCount + int64(latency)) /
		(providerMetrics.RequestCount + 1),
	)

	providerMetrics.TotalTokens += (inputTokens + outputTokens)
	providerMetrics.TotalCost += cost
	providerMetrics.LastRequestTime = time.Now()

	// Calculate error rate
	if providerMetrics.RequestCount > 0 {
		providerMetrics.ErrorRate = float64(providerMetrics.FailureCount) / float64(providerMetrics.RequestCount) * 100
	}

	// Calculate tokens per second
	if providerMetrics.AverageLatency > 0 {
		providerMetrics.TokensPerSecond = float64(inputTokens+outputTokens) / providerMetrics.AverageLatency.Seconds()
	}
}

// GetMetrics returns metrics within a time range
func (mc *MetricsCollector) GetMetrics(startTime, endTime time.Time) []Metric {
	mc.metricsMu.RLock()
	defer mc.metricsMu.RUnlock()

	filtered := make([]Metric, 0)
	for _, metric := range mc.metrics {
		if metric.Timestamp.After(startTime) && metric.Timestamp.Before(endTime) {
			filtered = append(filtered, metric)
		}
	}

	return filtered
}

// GetPerformanceMetrics returns current performance metrics
func (mc *MetricsCollector) GetPerformanceMetrics() *PerformanceMetrics {
	mc.perfMu.RLock()
	defer mc.perfMu.RUnlock()

	// Create a copy
	perfCopy := *mc.performance
	perfCopy.EndTime = time.Now()

	// Copy provider stats
	perfCopy.ProviderStats = make(map[string]*ProviderMetrics)
	for k, v := range mc.performance.ProviderStats {
		providerCopy := *v
		perfCopy.ProviderStats[k] = &providerCopy
	}

	return &perfCopy
}

// GetProviderMetrics returns metrics for a specific provider
func (mc *MetricsCollector) GetProviderMetrics(provider string) *ProviderMetrics {
	mc.perfMu.RLock()
	defer mc.perfMu.RUnlock()

	if metrics, exists := mc.performance.ProviderStats[provider]; exists {
		metricsCopy := *metrics
		return &metricsCopy
	}

	return nil
}

// Reset resets all metrics
func (mc *MetricsCollector) Reset() {
	mc.metricsMu.Lock()
	mc.perfMu.Lock()
	defer mc.metricsMu.Unlock()
	defer mc.perfMu.Unlock()

	mc.metrics = make([]Metric, 0, 1000)
	mc.performance = &PerformanceMetrics{
		ProviderStats: make(map[string]*ProviderMetrics),
		StartTime:     time.Now(),
	}
	mc.startTime = time.Now()
}

// GetSystemMetrics returns current system metrics
func (mc *MetricsCollector) GetSystemMetrics() map[string]interface{} {
	mc.perfMu.RLock()
	defer mc.perfMu.RUnlock()

	uptime := time.Since(mc.startTime)

	return map[string]interface{}{
		"uptime_seconds":    uptime.Seconds(),
		"cpu_usage":         mc.performance.CPUUsage,
		"memory_usage":      mc.performance.MemoryUsage,
		"disk_usage":        mc.performance.DiskUsage,
		"total_requests":    mc.performance.TotalRequests,
		"success_rate":      mc.calculateSuccessRate(),
		"requests_per_min":  mc.calculateRequestsPerMinute(),
		"avg_response_time": mc.performance.AverageResponseTime.Milliseconds(),
	}
}

// calculateSuccessRate calculates the success rate
func (mc *MetricsCollector) calculateSuccessRate() float64 {
	if mc.performance.TotalRequests == 0 {
		return 0
	}
	return float64(mc.performance.SuccessfulRequests) / float64(mc.performance.TotalRequests) * 100
}

// calculateRequestsPerMinute calculates requests per minute
func (mc *MetricsCollector) calculateRequestsPerMinute() float64 {
	uptime := time.Since(mc.startTime)
	if uptime.Minutes() == 0 {
		return 0
	}
	return float64(mc.performance.TotalRequests) / uptime.Minutes()
}

// GetTopProviders returns top N providers by usage
func (mc *MetricsCollector) GetTopProviders(n int) []*ProviderMetrics {
	mc.perfMu.RLock()
	defer mc.perfMu.RUnlock()

	providers := make([]*ProviderMetrics, 0, len(mc.performance.ProviderStats))
	for _, metrics := range mc.performance.ProviderStats {
		providerCopy := *metrics
		providers = append(providers, &providerCopy)
	}

	// Sort by request count (simple bubble sort for small datasets)
	for i := 0; i < len(providers); i++ {
		for j := i + 1; j < len(providers); j++ {
			if providers[j].RequestCount > providers[i].RequestCount {
				providers[i], providers[j] = providers[j], providers[i]
			}
		}
	}

	// Return top N
	if n > len(providers) {
		n = len(providers)
	}
	return providers[:n]
}

// GetCostBreakdown returns cost breakdown by provider
func (mc *MetricsCollector) GetCostBreakdown() map[string]float64 {
	mc.perfMu.RLock()
	defer mc.perfMu.RUnlock()

	breakdown := make(map[string]float64)
	for provider, metrics := range mc.performance.ProviderStats {
		breakdown[provider] = metrics.TotalCost
	}

	return breakdown
}

// GetAlerts returns current alerts based on thresholds
func (mc *MetricsCollector) GetAlerts() []Alert {
	mc.perfMu.RLock()
	defer mc.perfMu.RUnlock()

	alerts := make([]Alert, 0)

	// High error rate alert
	for provider, metrics := range mc.performance.ProviderStats {
		if metrics.ErrorRate > 10 {
			alerts = append(alerts, Alert{
				Level:     "warning",
				Type:      "high_error_rate",
				Message:   fmt.Sprintf("High error rate for %s: %.2f%%", provider, metrics.ErrorRate),
				Timestamp: time.Now(),
				Metadata: map[string]interface{}{
					"provider":   provider,
					"error_rate": metrics.ErrorRate,
				},
			})
		}
	}

	// High response time alert
	if mc.performance.AverageResponseTime > 10*time.Second {
		alerts = append(alerts, Alert{
			Level:     "warning",
			Type:      "high_latency",
			Message:   fmt.Sprintf("High average response time: %v", mc.performance.AverageResponseTime),
			Timestamp: time.Now(),
			Metadata: map[string]interface{}{
				"avg_response_time": mc.performance.AverageResponseTime.Milliseconds(),
			},
		})
	}

	// High cost alert
	if mc.performance.TotalCost > 100 {
		alerts = append(alerts, Alert{
			Level:     "info",
			Type:      "high_cost",
			Message:   fmt.Sprintf("Total cost exceeds $100: $%.2f", mc.performance.TotalCost),
			Timestamp: time.Now(),
			Metadata: map[string]interface{}{
				"total_cost": mc.performance.TotalCost,
			},
		})
	}

	return alerts
}

// Alert represents a system alert
type Alert struct {
	Level     string                 `json:"level"` // info, warning, error
	Type      string                 `json:"type"`
	Message   string                 `json:"message"`
	Timestamp time.Time              `json:"timestamp"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}
