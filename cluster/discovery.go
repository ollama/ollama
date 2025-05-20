package cluster

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/hashicorp/mdns"
	
	cerrors "github.com/ollama/ollama/cluster/errors"
)

// min returns the smaller of x or y
func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}

// DiscoveryMethod defines the method used for node discovery
type DiscoveryMethod string

const (
	// DiscoveryMethodMDNS uses multicast DNS for automatic service discovery
	DiscoveryMethodMDNS DiscoveryMethod = "mdns"
	
	// DiscoveryMethodMulticast uses multicast UDP for automatic node discovery (legacy)
	DiscoveryMethodMulticast DiscoveryMethod = "multicast"
	
	// DiscoveryMethodManual uses a manually specified list of nodes
	DiscoveryMethodManual DiscoveryMethod = "manual"
	
	// DefaultMulticastAddress is the default multicast address used for discovery
	DefaultMulticastAddress = "239.255.10.10:12095"
	
	// OllamaServiceName is the mDNS service name for Ollama clusters
	OllamaServiceName = "_ollama._tcp"
	
	// DefaultMDNSPort is the default port for mDNS discovery
	DefaultMDNSPort = 12095
)

// DiscoveryConfig holds configuration for the discovery service
type DiscoveryConfig struct {
	// Method determines how nodes discover each other
	Method DiscoveryMethod `json:"method"`
	
	// MulticastAddress is the address used for multicast discovery
	MulticastAddress string `json:"multicast_address"`
	
	// NodeList is a list of known node addresses for manual discovery
	NodeList []string `json:"node_list"`
	
	// HeartbeatInterval defines how often nodes announce themselves
	HeartbeatInterval time.Duration `json:"heartbeat_interval"`
	
	// NodeTimeoutInterval defines how long before a node is considered offline
	NodeTimeoutInterval time.Duration `json:"node_timeout_interval"`
	
	// ServiceName is the name used for mDNS service discovery
	ServiceName string `json:"service_name"`
	
	// DiscoveryPort is the port used for discovery
	DiscoveryPort int `json:"discovery_port"`
}

// DiscoveryService handles node discovery and heartbeats
type DiscoveryService struct {
	// config contains the discovery configuration
	config DiscoveryConfig
	
	// registry is a reference to the node registry
	registry *NodeRegistry
	
	// localNode contains information about this node
	localNode NodeInfo
	
	// mu protects the discovery service state
	mu sync.Mutex
	
	// ctx is the context for managing the discovery service lifecycle
	ctx context.Context
	
	// cancel is the function to stop the discovery service
	cancel context.CancelFunc
	
	// multicastConn is the UDP connection for multicast discovery
	multicastConn *net.UDPConn
	
	// mdnsServer is the mDNS server for service advertisement
	mdnsServer *mdns.Server
	
	// mdnsEntries tracks discovered nodes via mDNS
	mdnsEntries map[string]*mdns.ServiceEntry
	
	// isRunning indicates if the discovery service is active
	isRunning bool
	
	// retryPolicy defines how to handle retries for temporary failures
	retryPolicy cerrors.RetryPolicy
	
	// nodeRetries tracks retry attempts per node
	nodeRetries map[string]int
	
	// communicationMetrics tracks communication success/failure stats
	communicationMetrics struct {
		successCount   int
		failureCount   int
		lastFailure    time.Time
		avgLatency     time.Duration
		minLatency     time.Duration
		maxLatency     time.Duration
		latencySamples int
		errorRates     map[cerrors.ErrorCategory]int
	}
	
	// nodeFailureTracker tracks failure history for each node
	nodeFailureTracker map[string]*cerrors.NodeFailureInfo
	
	// statusController is a reference to the node status controller
	statusController *NodeStatusController
}

// NewDiscoveryService creates a new discovery service
func NewDiscoveryService(config DiscoveryConfig, registry *NodeRegistry, localNode NodeInfo) *DiscoveryService {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &DiscoveryService{
		config:      config,
		registry:    registry,
		localNode:   localNode,
		ctx:         ctx,
		cancel:      cancel,
		isRunning:   false,
		retryPolicy: cerrors.NewDefaultRetryPolicy(),
		nodeRetries: make(map[string]int),
		mdnsEntries: make(map[string]*mdns.ServiceEntry),
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

// Start begins the discovery service
func (d *DiscoveryService) Start() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	
	if d.isRunning {
		return fmt.Errorf("discovery service is already running")
	}
	
	fmt.Printf("Starting cluster discovery service using %s method\n", d.config.Method)
	
	// Initialize mdnsEntries map
	d.mdnsEntries = make(map[string]*mdns.ServiceEntry)
	
	var err error
	switch d.config.Method {
	case DiscoveryMethodMDNS:
		err = d.startMDNSDiscovery()
	case DiscoveryMethodMulticast:
		err = d.startMulticastDiscovery()
	case DiscoveryMethodManual:
		err = d.startManualDiscovery()
	default:
		err = fmt.Errorf("unsupported discovery method: %s", d.config.Method)
	}
	
	if err != nil {
		return fmt.Errorf("failed to start discovery service: %w", err)
	}
	
	d.isRunning = true
	return nil
}

// Stop halts the discovery service
func (d *DiscoveryService) Stop() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	
	if !d.isRunning {
		return fmt.Errorf("discovery service is not running")
	}
	
	fmt.Println("Stopping cluster discovery service")
	
	d.cancel()
	
	if d.multicastConn != nil {
		d.multicastConn.Close()
		d.multicastConn = nil
	}
	
	// Shutdown mDNS server if running
	if d.mdnsServer != nil {
		d.mdnsServer.Shutdown()
		d.mdnsServer = nil
	}
	
	d.isRunning = false
	return nil
}

// startMDNSDiscovery initializes and starts mDNS-based service discovery
func (d *DiscoveryService) startMDNSDiscovery() error {
	// Register local node first
	d.registry.RegisterNode(d.localNode)
	
	// Create mDNS service registration with extended information
	host, _ := os.Hostname()
	info := []string{
		fmt.Sprintf("id=%s", d.localNode.ID),
		fmt.Sprintf("name=%s", d.localNode.Name),
		fmt.Sprintf("role=%s", d.localNode.Role),
		fmt.Sprintf("api_port=%d", d.localNode.ApiPort),
		fmt.Sprintf("cluster_port=%d", d.localNode.ClusterPort),
		fmt.Sprintf("version=%s", "v0.2.0"),
		fmt.Sprintf("auto_discovery=true"),
	}
	
	// Add resource information if available
	if d.localNode.Resources.CPUCores > 0 {
		info = append(info, fmt.Sprintf("cpu_cores=%d", d.localNode.Resources.CPUCores))
		info = append(info, fmt.Sprintf("memory_mb=%d", d.localNode.Resources.MemoryMB))
	}
	
	if d.localNode.Resources.GPUCount > 0 {
		info = append(info, fmt.Sprintf("gpu_count=%d", d.localNode.Resources.GPUCount))
	}
	
	// Create service
	service, err := mdns.NewMDNSService(
		host,                       // Instance name (hostname)
		d.config.ServiceName,       // Service name (_ollama._tcp)
		"",                         // Domain (default local)
		"",                         // Host name (default to host)
		d.config.DiscoveryPort,     // Port
		[]net.IP{d.localNode.Addr}, // IP addresses
		info,                       // TXT records with node metadata
	)
	
	if err != nil {
		clusterErr := cerrors.NewDiscoveryError4(
			d.localNode.ID,
			"Failed to create mDNS service",
			cerrors.PersistentError,
			err,
		)
		cerrors.LogError(clusterErr)
		return clusterErr
	}
	
	// Create server
	server, err := mdns.NewServer(&mdns.Config{Zone: service})
	if err != nil {
		clusterErr := cerrors.NewDiscoveryError4(
			d.localNode.ID,
			"Failed to create mDNS server",
			cerrors.PersistentError,
			err,
		)
		cerrors.LogError(clusterErr)
		return clusterErr
	}
	
	d.mdnsServer = server
	
	// Start discovery of other nodes
	go d.discoverMDNSNodes()
	
	fmt.Printf("Registered node %s as %s.%s.local, advertising on mDNS\n",
		d.localNode.ID, host, d.config.ServiceName)
	
	return nil
}
// startMulticastDiscovery initializes and starts multicast-based discovery
func (d *DiscoveryService) startMulticastDiscovery() error {
	// Parse the multicast address
	addr, err := net.ResolveUDPAddr("udp", d.config.MulticastAddress)
	if err != nil {
		clusterErr := cerrors.NewConfigurationError4(
			d.localNode.ID,
			"Invalid multicast address configuration",
			cerrors.PersistentError,
			err,
		)
		cerrors.LogError(clusterErr)
		return clusterErr
	}
	
	// Create UDP connection for sending
	conn, err := net.ListenUDP("udp", &net.UDPAddr{IP: net.IPv4zero, Port: 0})
	if err != nil {
		clusterErr := cerrors.NewCommunicationError(
			d.localNode.ID,
			"Failed to create UDP connection for multicast discovery",
			cerrors.TemporaryError,
			err,
		)
		cerrors.LogError(clusterErr)
		return clusterErr
	}
	d.multicastConn = conn
	
	// Start the multicast receiver in a goroutine
	go d.receiveMulticastMessages(addr)
	
	// Start the heartbeat sender in a goroutine
	go d.sendHeartbeats(addr)
	
	// Register local node
	d.registry.RegisterNode(d.localNode)
	
	return nil
}

// startManualDiscovery initializes and starts manual discovery
func (d *DiscoveryService) startManualDiscovery() error {
	// Register local node first
	d.registry.RegisterNode(d.localNode)
	
	// For each node in the manual node list, attempt to connect and register
	for _, nodeAddr := range d.config.NodeList {
		go d.ConnectToNodeWithRetry(nodeAddr)
	}
	
	// Start a goroutine to periodically refresh connections to manual nodes
	go d.refreshManualConnections()
	
	return nil
}

// refreshManualConnections periodically attempts to connect to manually configured nodes
func (d *DiscoveryService) refreshManualConnections() {
	ticker := time.NewTicker(d.config.HeartbeatInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-d.ctx.Done():
			return
		case <-ticker.C:
			for _, nodeAddr := range d.config.NodeList {
				go d.ConnectToNodeWithRetry(nodeAddr)
			}
		}
	}
}

// ConnectToNodeWithRetry attempts to connect to a node with built-in retry and error tracking
func (d *DiscoveryService) ConnectToNodeWithRetry(nodeAddr string) {
	// Extract node ID from address (temporary ID until we get real one)
	tempNodeID := "node-" + nodeAddr
	
	// Get existing node failure info or create one if it doesn't exist
	var nodeFailure *cerrors.NodeFailureInfo
	d.mu.Lock()
	existing, exists := d.nodeFailureTracker[tempNodeID]
	if !exists {
		nodeFailure = cerrors.NewNodeFailureInfo(tempNodeID)
		d.nodeFailureTracker[tempNodeID] = nodeFailure
	} else {
		nodeFailure = existing
	}
	d.mu.Unlock()
	
	// Create a metrics tracker for this connection
	metrics := cerrors.NewMetricsTracker()
	
	// Create retry config with exponential backoff
	retryConfig := cerrors.NewDefaultRetryConfig().
		WithMaxRetries(5).
		WithBaseDelay(200 * time.Millisecond).
		WithMaxDelay(8 * time.Second)
	
	// Get the current retry count
	retryCount := d.nodeRetries[tempNodeID]
	timeout := time.Duration(min(30, 10+retryCount)) * time.Second
	
	// Create timeout context
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	
	fmt.Printf("Attempting to connect to node at %s (timeout: %v, retry attempt: %d)\n",
		nodeAddr, timeout, retryCount)
	
	// Define connection function with proper error handling
	connectFunc := func(attemptCtx context.Context) (NodeInfo, error) {
		startTime := time.Now()
		
		// Check for context cancellation
		if attemptCtx.Err() != nil {
			return NodeInfo{}, cerrors.NewTimeoutError(
				tempNodeID,
				"node-connection",
				timeout,
				attemptCtx.Err(),
			)
		}
		
		// In a real implementation, make an HTTP request here
		// For demonstration, we'll create a simulated connection attempt
		
		// ---- Begin connection simulation code (replace with actual connection) ----
		// This would make an HTTP request to the node endpoint to get node information
		// e.g., client.Get(fmt.Sprintf("http://%s/api/node_info", nodeAddr))
		
		// Simulated delay for connection attempt (remove in real code)
		time.Sleep(100 * time.Millisecond)
		
		// For debugging/testing, randomly return error cases
		errorSim := false // Set to true to simulate errors
		if errorSim && (retryCount % 3 == 0) {
			// Simulate various error types
			var err error
			switch retryCount % 4 {
			case 0:
				// Timeout error
				err = context.DeadlineExceeded
			case 1:
				// Connection refused
				err = &net.OpError{
					Op:  "dial",
					Err: fmt.Errorf("connection refused"),
				}
			case 2:
				// Name resolution error
				err = &net.DNSError{
					Err:       "no such host",
					Name:      nodeAddr,
					IsTimeout: false,
				}
			case 3:
				// Network error
				err = fmt.Errorf("network is unreachable")
			}
			
			// Create proper error category and return
			errorCategory := cerrors.ErrorCategoryFromError(err)
			errorSeverity := cerrors.ErrorSeverityFromError(err)
			
			return NodeInfo{}, cerrors.NewDetailedCommunicationError6(
				tempNodeID,
				fmt.Sprintf("Connection to %s failed", nodeAddr),
				errorSeverity,
				errorCategory,
				err,
				map[string]string{
					"node_addr": nodeAddr,
					"attempt":   fmt.Sprintf("%d", retryCount),
				},
			)
		}
		
		// No error - create simulated node info
		// In a real implementation, this would parse the HTTP response data
		nodeInfo := NodeInfo{
			ID:            tempNodeID,
			Name:          "node-" + nodeAddr,
			Addr:          net.ParseIP(strings.Split(nodeAddr, ":")[0]),
			ClusterPort:   12095,
			ApiPort:       11434,
			Role:          NodeRoleWorker,
			Status:        NodeStatusOnline,
			LastHeartbeat: time.Now(),
		}
		// ---- End simulation code ----
		
		// Track latency for this attempt
		latency := time.Since(startTime)
		metrics.RecordLatency(latency)
		
		return nodeInfo, nil
	}
	
	// Execute the function with retry logic
	startTime := time.Now()
	nodeInfo, err := d.executeWithRetryAndMetrics(ctx, connectFunc, retryConfig, metrics)
	connectionDuration := time.Since(startTime)
	
	// Handle result based on success or failure
	if err != nil {
		// Create appropriate error with detailed category and context
		errorCategory := cerrors.ErrorCategoryFromError(err)
		errorSeverity := cerrors.ErrorSeverityFromError(err)
		
		// Construct a detailed error message with context
		var errorMessage string
		var recoverySuggestion string
		
		switch errorCategory {
		case cerrors.ConnectionRefused:
			errorMessage = fmt.Sprintf("Connection refused by node at %s", nodeAddr)
			recoverySuggestion = "Check if the Ollama service is running on the target node"
		case cerrors.TimeoutError:
			errorMessage = fmt.Sprintf("Connection timed out to node at %s after %v", nodeAddr, timeout)
			recoverySuggestion = "Check network latency or increase connection timeout"
		case cerrors.NameResolution:
			errorMessage = fmt.Sprintf("Failed to resolve hostname for node at %s", nodeAddr)
			recoverySuggestion = "Verify DNS configuration and node hostname"
		case cerrors.NetworkTemporary:
			errorMessage = fmt.Sprintf("Temporary network error connecting to %s", nodeAddr)
			recoverySuggestion = "Retry after checking network connectivity"
		default:
			errorMessage = fmt.Sprintf("Failed to connect to node at %s", nodeAddr)
			recoverySuggestion = "Check node logs for more details"
		}
		
		// Create enhanced error with detailed metadata
		clusterErr := cerrors.NewNodeCommunicationError(
			tempNodeID,
			nodeAddr,
			"node-discovery",
			errorMessage,
			errorSeverity,
			errorCategory,
			err,
		).WithMetadata(map[string]string{
			"recovery_strategy": recoverySuggestion,
			"attempts":          fmt.Sprintf("%d", retryCount+1),
			"duration_ms":       fmt.Sprintf("%d", connectionDuration.Milliseconds()),
		})
		
		// Log the error with context
		cerrors.LogErrorf(clusterErr,
			"Connection failed to %s after %d %s: %v (recovery: %s)",
			nodeAddr,
			retryCount+1,
			pluralize("attempt", retryCount+1),
			err,
			recoverySuggestion,
		)
		
		// Update node status in registry if it exists
		if node, exists := d.registry.GetNode(tempNodeID); exists {
			if d.statusController != nil {
				// Use state machine for proper transition
				newStatus, err := d.statusController.TransitionStatus(node.Status, NodeStatusOffline)
				if err == nil {
					node.Status = newStatus
					d.registry.RegisterNode(node)
				}
			} else {
				node.Status = NodeStatusOffline
				d.registry.RegisterNode(node)
			}
		}
		
		// Record the failure in node failure info
		nodeFailure.RecordFailure(clusterErr)
		
		// Update connection metrics
		d.mu.Lock()
		d.communicationMetrics.failureCount++
		d.communicationMetrics.lastFailure = time.Now()
		d.communicationMetrics.errorRates[errorCategory]++
		
		// Increment retry counter for next attempt
		d.nodeRetries[tempNodeID] = retryCount + 1
		d.mu.Unlock()
		
		return
	}
	
	// Connection was successful
	nodeFailure.RecordSuccess()
	
	// Log success with metrics
	fmt.Printf("Successfully connected to node %s at %s (duration: %v, attempts: %d)\n",
		nodeInfo.Name, nodeAddr, connectionDuration, retryCount+1)
	
	// Update connection metrics
	d.mu.Lock()
	delete(d.nodeRetries, tempNodeID) // Reset retry counter on success
	d.communicationMetrics.successCount++
	
	// Update latency metrics with exponential moving average
	alpha := 0.2 // Weight for new samples
	if d.communicationMetrics.latencySamples == 0 {
		d.communicationMetrics.avgLatency = connectionDuration
		d.communicationMetrics.minLatency = connectionDuration
		d.communicationMetrics.maxLatency = connectionDuration
	} else {
		// Update average with weighted average
		d.communicationMetrics.avgLatency = time.Duration(
			float64(d.communicationMetrics.avgLatency)*(1-alpha) +
			float64(connectionDuration)*alpha)
		
		// Update min/max tracking
		if connectionDuration < d.communicationMetrics.minLatency {
			d.communicationMetrics.minLatency = connectionDuration
		}
		if connectionDuration > d.communicationMetrics.maxLatency {
			d.communicationMetrics.maxLatency = connectionDuration
		}
	}
	d.communicationMetrics.latencySamples++
	d.mu.Unlock()
	
	// Register the node with proper status
	status := NodeStatusOnline
	if d.statusController != nil {
		// Use state machine for proper transition
		if oldNode, exists := d.registry.GetNode(tempNodeID); exists {
			newStatus, err := d.statusController.TransitionStatus(oldNode.Status, NodeStatusOnline)
			if err == nil {
				status = newStatus
			}
		}
	}
	
	nodeInfo.Status = status
	d.registry.RegisterNode(nodeInfo)
}

// executeWithRetryAndMetrics executes a function with retry logic and tracks metrics
func (d *DiscoveryService) executeWithRetryAndMetrics(
	ctx context.Context,
	fn func(context.Context) (NodeInfo, error),
	config *cerrors.RetryConfig,
	metrics *cerrors.MetricsTracker,
) (NodeInfo, error) {
	var result NodeInfo
	var lastErr error
	
	for attempt := 0; attempt <= config.MaxRetries; attempt++ {
		// For retries (not first attempt), calculate and apply backoff
		if attempt > 0 {
			backoff := config.CalculateBackoff(attempt - 1)
			
			// Log retry information
			fmt.Printf("Retrying connection (attempt %d/%d) after %v\n",
				attempt+1, config.MaxRetries, backoff.Round(time.Millisecond))
			
			// Sleep for backoff duration, but respect context cancellation
			select {
			case <-ctx.Done():
				// Context was cancelled during backoff
				return result, fmt.Errorf("connection cancelled during backoff: %w", ctx.Err())
			case <-time.After(backoff):
				// Backoff completed, proceed with retry
			}
		}
		
		// Check if context is already cancelled before attempting
		if ctx.Err() != nil {
			return result, ctx.Err()
		}
		
		// Create sub-context for this attempt
		attemptCtx := ctx
		
		// Execute the function
		startTime := time.Now()
		nodeInfo, err := fn(attemptCtx)
		attemptDuration := time.Since(startTime)
		
		if err == nil {
			// Success!
			metrics.RecordSuccess(attemptDuration, attempt+1)
			return nodeInfo, nil
		}
		
		// Record the failure with metrics
		metrics.RecordFailure(err, attempt+1)
		lastErr = err
		
		// Check if this is a permanent error
		severity := cerrors.ErrorSeverityFromError(err)
		if severity == cerrors.PersistentError {
			return result, fmt.Errorf("permanent error, halting retries: %w", err)
		}
		
		// Check for context cancellation
		if ctx.Err() != nil {
			return result, fmt.Errorf("operation cancelled: %w", ctx.Err())
		}
		
		// Log the error and continue retrying
		fmt.Printf("Connection attempt %d failed: %v\n", attempt+1, err)
	}
	
	// All retries failed
	return result, fmt.Errorf("all %d retry attempts failed: %w", config.MaxRetries, lastErr)
}

// pluralize returns the singular or plural form of a word based on count
func pluralize(word string, count int) string {
	if count == 1 {
		return word
	}
	return word + "s"
}

// sendHeartbeats periodically sends multicast heartbeat messages
func (d *DiscoveryService) sendHeartbeats(addr *net.UDPAddr) {
	ticker := time.NewTicker(d.config.HeartbeatInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-d.ctx.Done():
			return
		case <-ticker.C:
			d.sendHeartbeat(addr)
		}
	}
}

// sendHeartbeat sends a single heartbeat message
func (d *DiscoveryService) sendHeartbeat(addr *net.UDPAddr) {
	d.mu.Lock()
	defer d.mu.Unlock()
	
	// Update timestamp
	d.localNode.LastHeartbeat = time.Now()
	
	// Create heartbeat message
	heartbeat := HeartbeatMessage{
		MessageType: "heartbeat",
		NodeInfo:    d.localNode,
	}
	
	// Marshal to JSON
	data, err := json.Marshal(heartbeat)
	if err != nil {
		clusterErr := cerrors.NewSerializationError4(
			d.localNode.ID,
			"Failed to marshal heartbeat message",
			cerrors.TemporaryError,
			err,
		)
		cerrors.LogError(clusterErr)
		return
	}
	
	// Send the message
	if d.multicastConn != nil {
		startTime := time.Now()
		_, err = d.multicastConn.WriteToUDP(data, addr)
		latency := time.Since(startTime)
		
		if err != nil {
			// Create appropriate error based on type
			var clusterErr cerrors.ClusterError
			
			if cerrors.IsNetworkUnreachable(err) {
				clusterErr = cerrors.NewCommunicationError(
					d.localNode.ID,
					"Network unreachable when sending heartbeat",
					cerrors.TemporaryError,
					err,
				)
			} else if cerrors.IsPermissionError(err) {
				clusterErr = cerrors.NewCommunicationError(
					d.localNode.ID,
					"Permission denied when sending heartbeat",
					cerrors.PersistentError,
					err,
				)
			} else {
				clusterErr = cerrors.NewCommunicationError(
					d.localNode.ID,
					"Error sending heartbeat",
					cerrors.TemporaryError,
					err,
				)
			}
			
			cerrors.LogError(clusterErr)
		} else {
			// Record successful communication for metrics
			d.mu.Lock()
			d.communicationMetrics.successCount++
			
			// Update average latency
			totalLatency := d.communicationMetrics.avgLatency * time.Duration(d.communicationMetrics.latencySamples)
			newLatency := totalLatency + latency
			d.communicationMetrics.latencySamples++
			d.communicationMetrics.avgLatency = newLatency / time.Duration(d.communicationMetrics.latencySamples)
			d.mu.Unlock()
		}
	}
}
// HeartbeatMessage represents a node heartbeat announcement
type HeartbeatMessage struct {
	// MessageType identifies the type of message
	MessageType string `json:"message_type"`
	
	// NodeInfo contains information about the sending node
	NodeInfo NodeInfo `json:"node_info"`
}

// receiveMulticastMessages listens for and processes multicast discovery messages
func (d *DiscoveryService) receiveMulticastMessages(addr *net.UDPAddr) {
	// Create a UDP connection for receiving multicast messages
	conn, err := net.ListenMulticastUDP("udp", nil, addr)
	if err != nil {
		fmt.Printf("Error setting up multicast listener: %v\n", err)
		return
	}
	defer conn.Close()
	
	// Set read buffer size
	conn.SetReadBuffer(1024)
	
	// Buffer to read into
	buffer := make([]byte, 1024)
	
	for {
		select {
		case <-d.ctx.Done():
			return
		default:
			// Set a read timeout so we can check the context periodically
			conn.SetReadDeadline(time.Now().Add(1 * time.Second))
			
			// Read from the connection
			n, src, err := conn.ReadFromUDP(buffer)
			if err != nil {
				// Check if it's a timeout - not a real error
				if netErr, ok := err.(net.Error); ok && netErr.Timeout() {
					// Just a timeout, continue the loop
					continue
				}
				
				// Categorize the error properly
				var clusterErr cerrors.ClusterError
				
				if cerrors.IsIOError(err) {
					clusterErr = cerrors.NewCommunicationError(
						d.localNode.ID,
						"I/O error reading from multicast",
						cerrors.TemporaryError,
						err,
					)
				} else if cerrors.IsBufferError(err) {
					clusterErr = cerrors.NewCommunicationError(
						d.localNode.ID,
						"Buffer overflow when reading from multicast",
						cerrors.TemporaryError,
						err,
					)
				} else if cerrors.IsConnectionResetError(err) {
					clusterErr = cerrors.NewCommunicationError(
						d.localNode.ID,
						"Connection reset while reading from multicast",
						cerrors.TemporaryError,
						err,
					)
				} else {
					clusterErr = cerrors.NewCommunicationError(
						d.localNode.ID,
						"Error reading from multicast",
						cerrors.TemporaryError,
						err,
					)
				}
				
				// Log the error with source address if available
				if src != nil {
					cerrors.LogErrorf(clusterErr, "Error reading from %s: %v", src.String(), err)
				} else {
					cerrors.LogError(clusterErr)
				}
				continue
			}
			
			// Process the received message
			d.processDiscoveryMessage(buffer[:n])
		}
	}
}

// discoverMDNSNodes continuously looks for other Ollama nodes via mDNS
func (d *DiscoveryService) discoverMDNSNodes() {
	// Create mDNS discovery parameters with optimized settings
	params := mdns.DefaultParams(d.config.ServiceName)
	params.DisableIPv6 = false // Enable IPv6 for better network coverage
	params.Timeout = time.Second * 2 // Longer timeout for better discovery
	params.Entries = make(chan *mdns.ServiceEntry, 20) // Larger buffer
	
	// Use shorter interval for faster discovery in the beginning
	initialDiscoveryTicker := time.NewTicker(time.Second * 2)
	defer initialDiscoveryTicker.Stop()
	
	// After initial discovery, use normal heartbeat interval
	regularTicker := time.NewTicker(d.config.HeartbeatInterval)
	defer regularTicker.Stop()
	
	// Track number of discovery attempts for switching to regular interval
	discoveryAttempts := 0
	
	fmt.Println("Starting zero-configuration mDNS discovery for Ollama nodes...")
	
	for {
		select {
		case <-d.ctx.Done():
			return
		case <-initialDiscoveryTicker.C:
			discoveryAttempts++
			
			// After 5 rapid discovery attempts, stop the initial ticker
			if discoveryAttempts >= 5 {
				initialDiscoveryTicker.Stop()
			}
			
			// Perform discovery
			go d.performMDNSDiscovery(params)
		case <-regularTicker.C:
			// Perform discovery in separate goroutine
			go d.performMDNSDiscovery(params)
		}
	}
}

// performMDNSDiscovery executes a single mDNS discovery operation
func (d *DiscoveryService) performMDNSDiscovery(params *mdns.QueryParam) {
	// Create our own bidirectional channel that we can receive from
	entryChan := make(chan *mdns.ServiceEntry, 20)
	
	// Set up params to use our channel for results
	params.Entries = entryChan
	
	// Execute the query
	if err := mdns.Query(params); err != nil {
		fmt.Printf("mDNS lookup error: %v\n", err)
		return
	}
	
	// Close the channel after timeout
	go func() {
		time.Sleep(params.Timeout)
		close(entryChan)
	}()
	
	// Track discovered nodes in this operation
	discoveredNodes := 0
	
	// Process discovered entries from our channel
	for entry := range entryChan {
		d.processMDNSEntry(entry)
		discoveredNodes++
	}
	
	// Log discovery results if any nodes found
	if discoveredNodes > 0 {
		fmt.Printf("Zero-config discovery found %d Ollama node(s)\n", discoveredNodes)
	}
}

// processMDNSEntry handles a discovered mDNS service entry
func (d *DiscoveryService) processMDNSEntry(entry *mdns.ServiceEntry) {
	d.mu.Lock()
	defer d.mu.Unlock()
	
	// Store entry for future reference
	d.mdnsEntries[entry.Name] = entry
	
	// Extract node information from TXT records
	nodeInfo := extractNodeInfoFromMDNS(entry)
	
	// Skip our own node
	if nodeInfo.ID == d.localNode.ID {
		return
	}
	
	// Update last heartbeat time
	nodeInfo.LastHeartbeat = time.Now()
	
	// Register or update the node in registry
	d.registry.RegisterNode(nodeInfo)
	
	fmt.Printf("Discovered Ollama node: %s (%s) at %s:%d\n",
		nodeInfo.Name, nodeInfo.ID, nodeInfo.Addr, nodeInfo.ClusterPort)
}

// extractNodeInfoFromMDNS extracts NodeInfo from mDNS service entry
func extractNodeInfoFromMDNS(entry *mdns.ServiceEntry) NodeInfo {
	nodeInfo := NodeInfo{
		Status: NodeStatusOnline,
	}
	
	// Set IP address
	if len(entry.AddrV4) > 0 {
		nodeInfo.Addr = entry.AddrV4
	} else if len(entry.AddrV6) > 0 {
		nodeInfo.Addr = entry.AddrV6
	}
	
	// Parse TXT records
	for _, txt := range entry.InfoFields {
		parts := strings.SplitN(txt, "=", 2)
		if len(parts) != 2 {
			continue
		}
		
		key, value := parts[0], parts[1]
		
		switch key {
		case "id":
			nodeInfo.ID = value
		case "name":
			nodeInfo.Name = value
		case "role":
			nodeInfo.Role = NodeRole(value)
		case "api_port":
			port, _ := strconv.Atoi(value)
			nodeInfo.ApiPort = port
		case "cluster_port":
			port, _ := strconv.Atoi(value)
			nodeInfo.ClusterPort = port
		}
	}
	
	// If no ID was found, create one from the mDNS name
	if nodeInfo.ID == "" {
		nodeInfo.ID = "mdns-" + entry.Name
	}
	
	// If no name was found, use the hostname part
	if nodeInfo.Name == "" {
		parts := strings.Split(entry.Name, ".")
		if len(parts) > 0 {
			nodeInfo.Name = parts[0]
		}
	}
	
	return nodeInfo
}

// processDiscoveryMessage handles an incoming discovery message
func (d *DiscoveryService) processDiscoveryMessage(data []byte) {
	startTime := time.Now() // Track processing time for metrics
	var message HeartbeatMessage
	
	// Unmarshal the message
	err := json.Unmarshal(data, &message)
	if err != nil {
		// Create serialization error
		clusterErr := cerrors.NewSerializationError4(
			d.localNode.ID,
			"Failed to parse discovery message",
			cerrors.TemporaryError,
			err,
		)
		
		// Add metadata about the message
		clusterErr = clusterErr.WithMetadata(map[string]string{
			"data_size_bytes": fmt.Sprintf("%d", len(data)),
			"first_bytes": fmt.Sprintf("%x", data[:min(10, len(data))]),
			"time_utc": time.Now().UTC().Format(time.RFC3339),
		})
		
		// Log with detailed context
		cerrors.LogErrorf(clusterErr, "Malformed message received (likely from another service): %v", err)
		return
	}
	
	// Validate the message
	if message.MessageType != "heartbeat" {
		clusterErr := cerrors.NewProtocolError4(
			d.localNode.ID,
			fmt.Sprintf("Unexpected message type: %s", message.MessageType),
			cerrors.TemporaryError,
			nil,
		)
		cerrors.LogError(clusterErr)
		return
	}
	
	// Skip our own messages
	if message.NodeInfo.ID == d.localNode.ID {
		return
	}
	
	// Ensure node ID is valid
	if message.NodeInfo.ID == "" {
		clusterErr := cerrors.NewValidationError4(
			d.localNode.ID,
			"Received heartbeat with empty node ID",
			cerrors.TemporaryError,
			nil,
		)
		cerrors.LogError(clusterErr)
		return
	}
	
	// Update the last heartbeat time
	message.NodeInfo.LastHeartbeat = time.Now()
	
	// Update metrics
	d.mu.Lock()
	processingTime := time.Since(startTime)
	d.communicationMetrics.avgLatency = (d.communicationMetrics.avgLatency + processingTime) / 2
	d.mu.Unlock()
	
	// Register or update the node
	d.registry.RegisterNode(message.NodeInfo)
}

// SendNodeListRequest sends a request for the complete node list to a specific node
func (d *DiscoveryService) SendNodeListRequest(targetAddr *net.UDPAddr) error {
	startTime := time.Now()
	request := map[string]string{
		"message_type": "node_list_request",
		"source_node":  d.localNode.ID,
		"timestamp":    time.Now().UTC().Format(time.RFC3339),
		"request_id":   fmt.Sprintf("%s-%d", d.localNode.ID, time.Now().UnixNano()),
	}
	
	// Marshal the request to JSON
	data, err := json.Marshal(request)
	if err != nil {
		clusterErr := cerrors.NewSerializationError4(
			d.localNode.ID,
			"Failed to marshal node list request",
			cerrors.TemporaryError,
			err,
		)
		cerrors.LogError(clusterErr)
		return clusterErr
	}
	
	// Check for valid connection
	if d.multicastConn == nil {
		clusterErr := cerrors.NewCommunicationError(
			d.localNode.ID,
			"No active multicast connection for sending node list request",
			cerrors.PersistentError,
			nil,
		)
		cerrors.LogError(clusterErr)
		return clusterErr
	}
	
	// Send the request
	_, err = d.multicastConn.WriteToUDP(data, targetAddr)
	if err != nil {
		var severity cerrors.ErrorSeverity
		var msg string
		
		if cerrors.IsNetworkUnreachable(err) {
			severity = cerrors.TemporaryError
			msg = "Network unreachable when sending node list request"
		} else if cerrors.IsConnectionRefusedError(err) {
			severity = cerrors.TemporaryError
			msg = "Connection refused when sending node list request"
		} else {
			severity = cerrors.TemporaryError
			msg = "Failed to send node list request"
		}
		
		clusterErr := cerrors.NewDetailedCommunicationError6(
			d.localNode.ID,
			msg,
			severity,
			cerrors.NetworkSendError,
			err,
			map[string]string{
				"target_addr": targetAddr.String(),
				"request_id":  request["request_id"],
			},
		)
		
		cerrors.LogError(clusterErr)
		return clusterErr
	}
	
	// Record metrics for successful send
	latency := time.Since(startTime)
	d.mu.Lock()
	defer d.mu.Unlock()
	d.communicationMetrics.successCount++
	totalLatency := d.communicationMetrics.avgLatency * time.Duration(d.communicationMetrics.latencySamples)
	d.communicationMetrics.latencySamples++
	d.communicationMetrics.avgLatency = (totalLatency + latency) / time.Duration(d.communicationMetrics.latencySamples)
	
	return nil
}

// GetLocalNodeInfo returns information about the local node
func (d *DiscoveryService) GetLocalNodeInfo() NodeInfo {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.localNode
}

// UpdateLocalNodeInfo updates information about the local node
func (d *DiscoveryService) UpdateLocalNodeInfo(info NodeInfo) {
	d.mu.Lock()
	d.localNode = info
	d.mu.Unlock()
	
	// Register the updated info in the registry
	d.registry.RegisterNode(info)
}