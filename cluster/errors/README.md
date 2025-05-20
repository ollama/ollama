# Enhanced Error Handling for Cluster Communications

## Overview

This implementation enhances error handling for node communication in the Ollama cluster system. The improvements focus on several key areas:

1. **Categorized Error Types**: Errors are now classified based on their nature (timeout, connection refused, network issues, etc.)
2. **Severity Classification**: Distinguishes between temporary and permanent failures
3. **Retry Mechanisms**: Implements exponential backoff with jitter for transient failures
4. **Graceful Degradation**: Handles temporary node unavailability with adaptive responses
5. **Detailed Metrics**: Tracks communication success/failure rates and latency

## Components

### Error Package (`cluster/errors/`)

- **`types.go`**: Defines error categories, severity levels, and interfaces
- **`errors.go`**: Core error types and constructors
- **`utils.go`**: Utility functions for error categorization and handling
- **`communication.go`**: Specialized error types for node communication
- **`context.go`**: Context enrichment for error handling
- **`failures.go`**: Tracks node failures over time
- **`metrics.go`**: Communication metrics collection
- **`reconnect.go`**: Reconnection management with retry logic
- **`degradation.go`**: Graceful degradation during node failures

### Cluster Health (`cluster/health_enhanced.go`)

Enhanced node health monitoring with:
- Intelligent status transitions using the node status controller
- Detailed error categorization with context
- Adaptive timeouts based on communication history
- Graceful degradation when nodes are temporarily unavailable
- Comprehensive metrics collection

### Node Reconnection (`cluster/reconnection.go`)

A dedicated system for reconnecting to failed nodes with:
- Exponential backoff retry mechanism
- Success/failure tracking
- Automatic status transitions through the state machine

### Node Discovery (`cluster/discovery_enhanced.go`)

Improved node discovery with:
- Enhanced error handling for connection attempts
- Detailed metrics collection
- Exponential backoff with jitter for retries
- Intelligent handling of different error types

## Integration Guide

### 1. Update HealthMonitor Initialization

```go
// In cluster/health.go

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
        registry:             registry,
        checkInterval:        checkInterval,
        nodeTimeoutThreshold: nodeTimeoutThreshold,
        client:               client,
        ctx:                  ctx,
        cancel:               cancel,
        healthStatusCache:    make(map[string]*NodeHealthStatus),
        communicationMetrics: make(map[string]*cerrors.CommunicationMetrics),
        lastCheckResults:     make(map[string]checkResult),
    }
    
    // Initialize reconnection manager
    hm.reconnectionManager = cerrors.NewReconnectionManager()
    
    // Initialize graceful degradation manager
    hm.degradationManager = cerrors.NewGracefulDegradation(cerrors.DegradationModeAdaptive)
    
    return hm
}
```

### 2. Update DiscoveryService Initialization

```go
// In cluster/discovery.go

// NewDiscoveryService creates a new discovery service
func NewDiscoveryService(config DiscoveryConfig, registry *NodeRegistry, localNode NodeInfo) *DiscoveryService {
    ctx, cancel := context.WithCancel(context.Background())
    
    return &DiscoveryService{
        config:             config,
        registry:           registry,
        localNode:          localNode,
        ctx:                ctx,
        cancel:             cancel,
        isRunning:          false,
        retryPolicy:        cerrors.NewDefaultRetryConfig(),
        nodeRetries:        make(map[string]int),
        mdnsEntries:        make(map[string]*mdns.ServiceEntry),
        nodeFailureTracker: make(map[string]*cerrors.NodeFailureInfo),
        statusController:   nil, // Will be set after ClusterMode is initialized
        communicationMetrics: struct {
            successCount   int
            failureCount   int
            lastFailure    time.Time
            avgLatency     time.Duration
            minLatency     time.Duration
            maxLatency     time.Duration
            latencySamples int
            errorRates     map[cerrors.ErrorCategory]int
        }{
            errorRates: make(map[cerrors.ErrorCategory]int),
        },
    }
}
```

### 3. Replace CheckNodeHealth Implementation

Replace the `CheckNodeHealth` method in `health.go` with the version from `health_enhanced.go` to benefit from the improved error handling.

### 4. Update Connect Method

Replace the `connectToNode` method in `discovery.go` with the `ConnectToNodeWithRetry` method from `discovery_enhanced.go`.

### 5. Initialize StatusController

Ensure that each component has access to the NodeStatusController for proper status transitions:

```go
// In ClusterMode.Start() method
func (c *ClusterMode) Start() error {
    // Create the status controller first
    c.StatusController = NewNodeStatusController()
    
    // Set status controller in health monitor and discovery service
    c.Health.statusController = c.StatusController
    c.Discovery.statusController = c.StatusController
    
    // Start health monitoring
    if err := c.Health.Start(); err != nil {
        return err
    }
    
    // Start discovery service
    if err := c.Discovery.Start(); err != nil {
        // Stop health monitor if discovery fails
        c.Health.Stop()
        return err
    }
    
    // Update node status to online using the status controller
    if err := c.SetNodeToOnline(); err != nil {
        c.Health.Stop()
        c.Discovery.Stop()
        return fmt.Errorf("failed to transition to online status: %w", err)
    }
    
    return nil
}
```

## Key Benefits

1. **Resilience**: The system can better handle temporary network issues
2. **Clarity**: Errors are now categorized and include detailed context
3. **Performance**: Adaptive timeouts based on node communication patterns
4. **Availability**: Graceful degradation during partial outages
5. **Observability**: Detailed metrics for debugging and monitoring
6. **Consistency**: Status transitions are validated through the state machine

## Testing Recommendations

1. Test with simulated network failures (delays, packet loss, DNS failures)
2. Test with node restarts to verify reconnection behavior
3. Test with high latency scenarios to verify timeout handling
4. Test with multiple simultaneous node failures to verify degradation behavior
5. Test recovery scenarios to verify status transitions