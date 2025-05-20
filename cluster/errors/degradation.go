package cerrors

import (
	"fmt"
	"sync"
	"time"
)

// DegradationMode represents how the system should behave when nodes are degraded
type DegradationMode string

const (
	// DegradationModeStrict fails operations immediately when nodes are degraded
	DegradationModeStrict DegradationMode = "strict"
	
	// DegradationModeTolerant attempts operations despite node degradation
	DegradationModeTolerant DegradationMode = "tolerant"
	
	// DegradationModeAdaptive adjusts behavior based on the severity of degradation
	DegradationModeAdaptive DegradationMode = "adaptive"
)

// DegradationLevel represents the severity of system degradation
type DegradationLevel int

const (
	// DegradationLevelNone indicates normal operation
	DegradationLevelNone DegradationLevel = 0
	
	// DegradationLevelMinor indicates slight performance impact
	DegradationLevelMinor DegradationLevel = 1
	
	// DegradationLevelModerate indicates noticeable performance impact
	DegradationLevelModerate DegradationLevel = 2
	
	// DegradationLevelSevere indicates significant performance impact
	DegradationLevelSevere DegradationLevel = 3
	
	// DegradationLevelCritical indicates system is barely operational
	DegradationLevelCritical DegradationLevel = 4
)

// GracefulDegradation manages system behavior during degraded states
type GracefulDegradation struct {
	// mode defines how the system handles degraded states
	mode DegradationMode
	
	// currentLevel is the current degradation level
	currentLevel DegradationLevel
	
	// degradedNodes tracks nodes in a degraded state
	degradedNodes map[string]NodeDegradationInfo
	
	// toleranceThresholds defines thresholds for adaptive behavior
	toleranceThresholds struct {
		// maxDegradedNodePercentage is the maximum percentage of nodes
		// that can be degraded before switching to strict mode
		maxDegradedNodePercentage float64
		
		// maxConsecutiveFailures is the maximum number of consecutive failures
		// tolerated for a single node before marking it as severely degraded
		maxConsecutiveFailures int
		
		// maxResponseTimeMultiplier is the maximum multiplier for normal response times
		// tolerated before marking a node as degraded
		maxResponseTimeMultiplier float64
	}
	
	// mu protects concurrent access
	mu sync.RWMutex
}

// NodeDegradationInfo tracks the degradation status of a single node
type NodeDegradationInfo struct {
	// NodeID is the identifier for the node
	NodeID string
	
	// Level is the current degradation level
	Level DegradationLevel
	
	// LastUpdated is when the degradation info was last updated
	LastUpdated time.Time
	
	// ConsecutiveFailures is the count of consecutive failures
	ConsecutiveFailures int
	
	// LatencyRatio is the ratio between current and normal latency
	// (1.0 = normal, >1.0 = degraded)
	LatencyRatio float64
	
	// InitialDetectionTime is when degradation was first detected
	InitialDetectionTime time.Time
	
	// Reason describes why the node is degraded
	Reason string
}

// NewGracefulDegradation creates a new manager for graceful degradation
func NewGracefulDegradation(mode DegradationMode) *GracefulDegradation {
	gd := &GracefulDegradation{
		mode:           mode,
		currentLevel:   DegradationLevelNone,
		degradedNodes:  make(map[string]NodeDegradationInfo),
	}
	
	// Set default tolerance thresholds
	gd.toleranceThresholds.maxDegradedNodePercentage = 30.0    // 30% of nodes can be degraded
	gd.toleranceThresholds.maxConsecutiveFailures = 5         // 5 consecutive failures
	gd.toleranceThresholds.maxResponseTimeMultiplier = 3.0    // 3x normal response time
	
	return gd
}

// MarkNodeDegraded records that a node is in a degraded state with a reason
func (gd *GracefulDegradation) MarkNodeDegraded(nodeID string, level DegradationLevel, reason string) {
	gd.mu.Lock()
	defer gd.mu.Unlock()
	
	now := time.Now()
	
	// Check if this is a new degradation or an update
	info, exists := gd.degradedNodes[nodeID]
	if !exists {
		// New degradation
		info = NodeDegradationInfo{
			NodeID:               nodeID,
			Level:                level,
			LastUpdated:          now,
			InitialDetectionTime: now,
			Reason:               reason,
		}
	} else {
		// Update existing degradation
		info.Level = level
		info.LastUpdated = now
		info.Reason = reason
	}
	
	// Store the updated info
	gd.degradedNodes[nodeID] = info
	
	// Recalculate overall degradation level
	gd.recalculateDegradationLevel()
	
	// Log the degradation
	fmt.Printf("Node %s marked as degraded (level: %d): %s\n",
		nodeID, level, reason)
}

// RecordConsecutiveFailure records a consecutive failure for a node
func (gd *GracefulDegradation) RecordConsecutiveFailure(nodeID string, consecutiveFailures int, err error) {
	gd.mu.Lock()
	defer gd.mu.Unlock()
	
	// Get existing info or create new
	info, exists := gd.degradedNodes[nodeID]
	if !exists {
		info = NodeDegradationInfo{
			NodeID:               nodeID,
			LastUpdated:          time.Now(),
			InitialDetectionTime: time.Now(),
			Reason:               fmt.Sprintf("Consecutive failures: %d", consecutiveFailures),
		}
	}
	
	// Update failure count
	info.ConsecutiveFailures = consecutiveFailures
	info.LastUpdated = time.Now()
	
	// Determine degradation level based on consecutive failures
	switch {
	case consecutiveFailures >= gd.toleranceThresholds.maxConsecutiveFailures:
		info.Level = DegradationLevelSevere
	case consecutiveFailures >= 3:
		info.Level = DegradationLevelModerate
	case consecutiveFailures >= 1:
		info.Level = DegradationLevelMinor
	default:
		info.Level = DegradationLevelNone
	}
	
	// Update reason with error information
	if err != nil {
		errCategory := ErrorCategoryFromError(err)
		errSeverity := ErrorSeverityFromError(err)
		info.Reason = fmt.Sprintf("Consecutive failures: %d, Error: %s (%s severity)",
			consecutiveFailures, errCategory, errSeverity)
	} else {
		info.Reason = fmt.Sprintf("Consecutive failures: %d", consecutiveFailures)
	}
	
	// Store updated info
	gd.degradedNodes[nodeID] = info
	
	// Recalculate overall degradation level
	gd.recalculateDegradationLevel()
	
	// Log the degradation
	if info.Level > DegradationLevelNone {
		fmt.Printf("Node %s degraded to level %d after %d consecutive failures\n",
			nodeID, info.Level, consecutiveFailures)
	}
}

// RecordLatencyIncrease records increased latency for a node
func (gd *GracefulDegradation) RecordLatencyIncrease(nodeID string, currentLatency, baselineLatency time.Duration) {
	gd.mu.Lock()
	defer gd.mu.Unlock()
	
	// Calculate latency ratio
	var ratio float64
	if baselineLatency > 0 {
		ratio = float64(currentLatency) / float64(baselineLatency)
	} else {
		ratio = 1.0 // Default to normal if no baseline
	}
	
	// Get existing info or create new
	info, exists := gd.degradedNodes[nodeID]
	if !exists {
		info = NodeDegradationInfo{
			NodeID:               nodeID,
			LastUpdated:          time.Now(),
			InitialDetectionTime: time.Now(),
		}
	}
	
	// Update latency ratio
	info.LatencyRatio = ratio
	info.LastUpdated = time.Now()
	
	// Determine degradation level based on latency ratio
	switch {
	case ratio >= gd.toleranceThresholds.maxResponseTimeMultiplier * 2:
		info.Level = DegradationLevelSevere
	case ratio >= gd.toleranceThresholds.maxResponseTimeMultiplier:
		info.Level = DegradationLevelModerate
	case ratio >= 1.5: // 50% increase in latency
		info.Level = DegradationLevelMinor
	default:
		info.Level = DegradationLevelNone
	}
	
	// Update reason
	info.Reason = fmt.Sprintf("Latency increase: %.1fx normal (%.2fms vs %.2fms baseline)",
		ratio, currentLatency.Seconds()*1000, baselineLatency.Seconds()*1000)
	
	// Store updated info
	gd.degradedNodes[nodeID] = info
	
	// Recalculate overall degradation level
	gd.recalculateDegradationLevel()
	
	// Log significant latency increases
	if ratio >= 1.5 {
		fmt.Printf("Node %s showing increased latency: %.1fx normal (level: %d)\n",
			nodeID, ratio, info.Level)
	}
}

// MarkNodeRecovered marks a node as recovered from degradation
func (gd *GracefulDegradation) MarkNodeRecovered(nodeID string) {
	gd.mu.Lock()
	defer gd.mu.Unlock()
	
	// Check if the node was previously degraded
	info, exists := gd.degradedNodes[nodeID]
	if !exists {
		return // Node wasn't degraded
	}
	
	// Calculate how long the node was degraded
	degradedDuration := time.Since(info.InitialDetectionTime)
	
	// Log recovery
	fmt.Printf("Node %s recovered from degradation level %d after %v\n",
		nodeID, info.Level, degradedDuration.Round(time.Second))
	
	// Remove from degraded nodes map
	delete(gd.degradedNodes, nodeID)
	
	// Recalculate overall degradation level
	gd.recalculateDegradationLevel()
}

// recalculateDegradationLevel updates the overall system degradation level
func (gd *GracefulDegradation) recalculateDegradationLevel() {
	if len(gd.degradedNodes) == 0 {
		gd.currentLevel = DegradationLevelNone
		return
	}
	
	// Count nodes at each degradation level
	levelCounts := map[DegradationLevel]int{
		DegradationLevelMinor:    0,
		DegradationLevelModerate: 0,
		DegradationLevelSevere:   0,
		DegradationLevelCritical: 0,
	}
	
	for _, info := range gd.degradedNodes {
		levelCounts[info.Level]++
	}
	
	// Determine overall level based on counts
	if levelCounts[DegradationLevelCritical] > 0 {
		gd.currentLevel = DegradationLevelCritical
	} else if levelCounts[DegradationLevelSevere] > 1 {
		gd.currentLevel = DegradationLevelSevere
	} else if levelCounts[DegradationLevelSevere] == 1 || levelCounts[DegradationLevelModerate] > 2 {
		gd.currentLevel = DegradationLevelModerate
	} else if levelCounts[DegradationLevelModerate] > 0 || levelCounts[DegradationLevelMinor] > 3 {
		gd.currentLevel = DegradationLevelMinor
	} else {
		gd.currentLevel = DegradationLevelNone
	}
}

// GetDegradationLevel returns the current system degradation level
func (gd *GracefulDegradation) GetDegradationLevel() DegradationLevel {
	gd.mu.RLock()
	defer gd.mu.RUnlock()
	return gd.currentLevel
}

// GetDegradedNodes returns a copy of the current degraded nodes map
func (gd *GracefulDegradation) GetDegradedNodes() map[string]NodeDegradationInfo {
	gd.mu.RLock()
	defer gd.mu.RUnlock()
	
	// Create a copy to avoid concurrent modification
	result := make(map[string]NodeDegradationInfo)
	for id, info := range gd.degradedNodes {
		result[id] = info
	}
	
	return result
}

// GetDegradedNodeCount returns the number of degraded nodes
func (gd *GracefulDegradation) GetDegradedNodeCount() int {
	gd.mu.RLock()
	defer gd.mu.RUnlock()
	return len(gd.degradedNodes)
}

// ShouldTolerateError determines whether an error should be tolerated
// based on the current degradation mode and level
func (gd *GracefulDegradation) ShouldTolerateError(err error, nodeID string) bool {
	if err == nil {
		return true
	}
	
	gd.mu.RLock()
	defer gd.mu.RUnlock()
	
	// Get error severity
	severity := ErrorSeverityFromError(err)
	
	// In strict mode, never tolerate errors
	if gd.mode == DegradationModeStrict {
		return false
	}
	
	// In tolerant mode, always tolerate temporary errors
	if gd.mode == DegradationModeTolerant {
		return severity == TemporaryError
	}
	
	// In adaptive mode, behavior depends on current degradation level
	if gd.mode == DegradationModeAdaptive {
		// For temporary errors, tolerance depends on degradation level
		if severity == TemporaryError {
			switch gd.currentLevel {
			case DegradationLevelNone, DegradationLevelMinor:
				return true // Tolerate temporary errors in normal or minor degradation
			case DegradationLevelModerate:
				// Check if this specific node is severely degraded
				info, exists := gd.degradedNodes[nodeID]
				if exists && info.Level >= DegradationLevelSevere {
					return false // Don't tolerate errors from severely degraded nodes
				}
				return true // Otherwise tolerate temporary errors
			case DegradationLevelSevere, DegradationLevelCritical:
				return false // Don't tolerate more errors in severe degradation
			}
		}
		
		// Never tolerate persistent errors
		return false
	}
	
	// Default behavior
	return false
}

// GetDegradationStatus returns a human-readable status message
func (gd *GracefulDegradation) GetDegradationStatus() string {
	gd.mu.RLock()
	defer gd.mu.RUnlock()
	
	switch gd.currentLevel {
	case DegradationLevelNone:
		return "System operating normally"
	case DegradationLevelMinor:
		return fmt.Sprintf("Minor degradation: %d node(s) showing slight performance impact", 
			len(gd.degradedNodes))
	case DegradationLevelModerate:
		return fmt.Sprintf("Moderate degradation: %d node(s) with reduced performance", 
			len(gd.degradedNodes))
	case DegradationLevelSevere:
		return fmt.Sprintf("Severe degradation: %d node(s) with significant issues", 
			len(gd.degradedNodes))
	case DegradationLevelCritical:
		return fmt.Sprintf("Critical degradation: System barely operational, %d node(s) affected", 
			len(gd.degradedNodes))
	default:
		return "Unknown degradation state"
	}
}