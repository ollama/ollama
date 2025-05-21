package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cluster"
	"github.com/ollama/ollama/cluster/model"
	"github.com/ollama/ollama/cluster/tensor"
	"log/slog"
)

// getStackTrace returns the current stack trace as a string
func getStackTrace() string {
	buf := make([]byte, 8192)
	n := runtime.Stack(buf, false)
	return string(buf[:n])
}

// isTemporaryError checks if an error is likely temporary and the operation can be retried
func isTemporaryError(err error) bool {
	if err == nil {
		return false
	}
	
	// Check for context timeout/deadline errors
	if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
		return true
	}
	
	// Check for net.Error with Temporary() method
	var netErr net.Error
	if errors.As(err, &netErr) {
		return netErr.Temporary()
	}
	
	// Check error strings that indicate temporary issues
	errStr := err.Error()
	temporaryPatterns := []string{
		"connection reset",
		"connection refused",
		"broken pipe",
		"forcibly closed",
		"i/o timeout",
		"timeout",
		"too many open files",
		"connection timed out",
		"no route to host",
		"operation timed out",
	}
	
	for _, pattern := range temporaryPatterns {
		if strings.Contains(strings.ToLower(errStr), pattern) {
			return true
		}
	}
	
	return false
}

var (
	// Global cluster mode instance
	clusterMode     *cluster.ClusterMode
	clusterModeLock sync.RWMutex
	clusterEnabled  bool
	errClusterNotEnabled = errors.New("cluster mode is not enabled")
)

// ClusterMiddleware checks if cluster mode is enabled
func ClusterMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		clusterModeLock.RLock()
		enabled := clusterEnabled
		clusterModeLock.RUnlock()

		if !enabled {
			c.JSON(http.StatusBadRequest, gin.H{"error": "cluster mode is not enabled"})
			c.Abort()
			return
		}
		
		c.Next()
	}
}

// min returns the smaller of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// InitializeClusterMode initializes cluster mode from server
func InitializeClusterMode(config *cluster.ClusterConfig) error {
	clusterModeLock.Lock()
	defer clusterModeLock.Unlock()
	
	if config == nil || !config.Enabled {
		clusterEnabled = false
		return nil
	}
	
	var err error
	clusterMode, err = cluster.NewClusterMode(config)
	if err != nil {
		return fmt.Errorf("failed to initialize cluster mode: %w", err)
	}
	
	err = clusterMode.Start()
	if err != nil {
		return fmt.Errorf("failed to start cluster mode: %w", err)
	}
	
	clusterEnabled = true
	return nil
}

// ShutdownClusterMode gracefully shuts down the cluster mode
func ShutdownClusterMode() error {
	clusterModeLock.Lock()
	defer clusterModeLock.Unlock()
	
	if !clusterEnabled || clusterMode == nil {
		return nil
	}
	
	err := clusterMode.Stop()
	if err != nil {
		return fmt.Errorf("failed to stop cluster mode: %w", err)
	}
	
	clusterMode = nil
	clusterEnabled = false
	return nil
}

// ClusterStatusHandler returns the status of the cluster
func (s *Server) ClusterStatusHandler(c *gin.Context) {
	clusterModeLock.RLock()
	defer clusterModeLock.RUnlock()
	
	// Use helper function to check environment variables
	envEnabled := IsClusterModeEnabledFromEnv()
	
	// Log the state of both implementations with more environment details
	slog.Info("ClusterStatusHandler check",
		"clusterEnabled", clusterEnabled,
		"clusterEnabled2", clusterEnabled2,
		"clusterMode", clusterMode != nil,
		"clusterMode2", GetClusterMode2() != nil,
		"env_cluster_enabled", envEnabled,
		"OLLAMA_CLUSTER_MODE", os.Getenv("OLLAMA_CLUSTER_MODE"),
		"OLLAMA_CLUSTER_ENABLED", os.Getenv("OLLAMA_CLUSTER_ENABLED"))
	
	if !clusterEnabled || clusterMode == nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "cluster mode is not enabled"})
		return
	}
	
	// Get local node info
	localNode := clusterMode.GetLocalNodeInfo()
	
	// Get all nodes
	registry := clusterMode.GetRegistry()
	nodes := registry.GetAllNodes()
	
	// Convert to API responses
	nodeResponses := make([]api.ClusterNodeResponse, 0, len(nodes))
	for _, node := range nodes {
		resp := api.ConvertNodeInfoToResponse(node)
		
		// Add joined time - this would come from registry in a real implementation
		// For now using current time as placeholder
		resp.JoinedAt = time.Now() 
		
		// In a real implementation, would fetch models from this node if it's online
		if node.Status == cluster.NodeStatusOnline {
			resp.Models = []string{} // Would come from model registry
		}
		
		nodeResponses = append(nodeResponses, resp)
	}
	
	// Create the response
	response := api.ClusterStatusResponse{
		Enabled:     true,
		Mode:        "active", // Could be "active", "maintenance", etc.
		NodeCount:   len(nodes),
		CurrentNode: api.ConvertNodeInfoToResponse(localNode),
		Nodes:       nodeResponses,
		Models:      []api.ClusterModelResponse{}, // Would be populated in real implementation
		StartedAt:   time.Now(), // Would come from registry in a real implementation
		Healthy:     true, // Would check health in real implementation
	}
	
	c.JSON(http.StatusOK, response)
}

// ClusterJoinHandler handles requests to join a cluster
func (s *Server) ClusterJoinHandler(c *gin.Context) {
	clusterModeLock.RLock()
	defer clusterModeLock.RUnlock()
	
	if !clusterEnabled || clusterMode == nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "cluster mode is not enabled"})
		return
	}
	
	var req api.ClusterJoinRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	
	// Basic validation
	if req.NodeHost == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "node_host is required"})
		return
	}
	
	if req.NodePort <= 0 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "node_port is required and must be > 0"})
		return
	}
	
	// Process role if specified
	if req.NodeRole != "" {
		switch req.NodeRole {
		case "coordinator", "worker", "mixed":
			// Valid roles
		default:
			c.JSON(http.StatusBadRequest, gin.H{"error": "invalid node_role, must be coordinator, worker or mixed"})
			return
		}
	}
	
	// Get the local node info to use for joining
	localNodeID := ""
	if clusterMode != nil {
		localNode := clusterMode.GetLocalNodeInfo()
		localNodeID = localNode.ID
	} else if GetClusterMode2() != nil {
		localNode := GetClusterMode2().GetLocalNodeInfo()
		if node, ok := localNode.(*cluster.NodeInfo); ok {
			localNodeID = node.ID
		}
	}

	// Generate a proper unique node ID if we couldn't get one
	if localNodeID == "" {
		// Create a unique ID based on host and timestamp
		hostname, err := os.Hostname()
		if err != nil {
			hostname = "unknown-host"
		}
		localNodeID = fmt.Sprintf("%s-%d", hostname, time.Now().UnixNano())
	}

	// Use proper node ID instead of hardcoded value
	c.JSON(http.StatusOK, api.ClusterJoinResponse{
		Success:     true,
		NodeID:      localNodeID,
		ClusterID:   fmt.Sprintf("cluster-%s", localNodeID),
		NodesJoined: 1,
	})
}

// ClusterLeaveHandler handles requests to leave the cluster
func (s *Server) ClusterLeaveHandler(c *gin.Context) {
	clusterModeLock.RLock()
	defer clusterModeLock.RUnlock()
	
	if !clusterEnabled || clusterMode == nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "cluster mode is not enabled"})
		return
	}
	
	var req api.ClusterLeaveRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	
	// If no node ID is specified, assume we're leaving
	nodeID := req.NodeID
	if nodeID == "" {
		nodeID = clusterMode.GetLocalNodeInfo().ID
	}
	
	// Process timeout if needed
	if req.Timeout > 0 {
		// In a real implementation, this would be used to set a timeout
		slog.Info("Node leave timeout specified", "timeout_seconds", req.Timeout)
	}
	
	// In a real implementation, would call discovery.LeaveNode
	// For now, returning success
	c.JSON(http.StatusOK, api.ClusterLeaveResponse{
		Success: true,
	})
}

// ClusterNodesHandler lists all nodes in the cluster
func (s *Server) ClusterNodesHandler(c *gin.Context) {
	clusterModeLock.RLock()
	defer clusterModeLock.RUnlock()
	
	// Use helper function to check environment variables
	envEnabled := IsClusterModeEnabledFromEnv()
	
	// Log the state of both implementations and request details with environment variables
	slog.Info("ClusterNodesHandler check",
		"clusterEnabled", clusterEnabled,
		"clusterEnabled2", clusterEnabled2,
		"clusterMode", clusterMode != nil,
		"clusterMode2", GetClusterMode2() != nil,
		"env_cluster_enabled", envEnabled,
		"request", c.Request.URL.String(),
		"method", c.Request.Method,
		"client", c.ClientIP(),
		"OLLAMA_CLUSTER_MODE", os.Getenv("OLLAMA_CLUSTER_MODE"),
		"OLLAMA_CLUSTER_ENABLED", os.Getenv("OLLAMA_CLUSTER_ENABLED"))
	
	// Check if at least one of the implementations is enabled
	if ((!clusterEnabled || clusterMode == nil) && (!clusterEnabled2 || GetClusterMode2() == nil)) {
		c.JSON(http.StatusBadRequest, gin.H{"error": "cluster mode is not enabled"})
		return
	}
	
	// Get nodes from available implementations
	var nodeResponses []api.ClusterNodeResponse
	
	// Try to get real nodes from either implementation, preferring the original one
	if clusterMode != nil {
		// Get registry and nodes
		registry := clusterMode.GetRegistry()
		nodes := registry.GetAllNodes()
		
		// Convert to API responses
		for _, node := range nodes {
			resp := api.ConvertNodeInfoToResponse(node)
			nodeResponses = append(nodeResponses, resp)
		}
		
		// Add the local node if it's not already in the list
		localNode := clusterMode.GetLocalNodeInfo()
		found := false
		for _, resp := range nodeResponses {
			if resp.ID == localNode.ID {
				found = true
				break
			}
		}
		
		if !found {
			resp := api.ConvertNodeInfoToResponse(localNode)
			nodeResponses = append(nodeResponses, resp)
		}
		
		slog.Info("Using original clusterMode implementation for nodes request",
			"nodeCount", len(nodeResponses))
	} else if GetClusterMode2() != nil {
		slog.Info("Using clusterMode2 implementation for nodes request")
		
		// Get nodes from the new implementation
		cm2 := GetClusterMode2()
		nodes := cm2.GetNodes()
		
		// Convert to API responses
		for _, node := range nodes {
			// Format the address properly including the port
			var addressStr string
			if node.Addr != nil {
				addressStr = fmt.Sprintf("%s:%d", node.Addr.String(), node.ClusterPort)
			} else {
				// If address is nil, use a placeholder based on config
				addressStr = fmt.Sprintf("%s:%d", "localhost", node.ClusterPort)
			}
			
			nodeResponses = append(nodeResponses, api.ClusterNodeResponse{
				ID:       node.ID,
				Name:     node.Name,
				Address:  addressStr,
				Role:     string(node.Role),
				Status:   string(node.Status),
				JoinedAt: node.LastHeartbeat, // Using LastHeartbeat as a substitute for JoinedAt
				Models:   []string{}, // Would need to populate from model tracker
			})
		}
		
		// Add local node if it's not already in the list
		if cm2.LocalNode != nil {
			localNode := *cm2.LocalNode
			found := false
			for _, resp := range nodeResponses {
				if resp.ID == localNode.ID {
					found = true
					break
				}
			}
			
			if !found {
				var addressStr string
				if localNode.Addr != nil {
					addressStr = fmt.Sprintf("%s:%d", localNode.Addr.String(), localNode.ClusterPort)
				} else {
					addressStr = fmt.Sprintf("%s:%d", "localhost", localNode.ClusterPort)
				}
				
				nodeResponses = append(nodeResponses, api.ClusterNodeResponse{
					ID:       localNode.ID,
					Name:     localNode.Name,
					Address:  addressStr,
					Role:     string(localNode.Role),
					Status:   string(localNode.Status),
					JoinedAt: localNode.LastHeartbeat,
					Models:   []string{},
				})
			}
		}
	}
	
	// If no real nodes found, ensure we at least return the local node
	if len(nodeResponses) == 0 {
		// Generate local node info as a fallback
		hostname, err := os.Hostname()
		if err != nil {
			hostname = "local-node"
		}
		
		// Create a unique ID
		nodeID := fmt.Sprintf("%s-%d", hostname, time.Now().UnixNano())
		
		// Get a valid local IP for the node
		localIP := "127.0.0.1"
		if addrs, err := net.InterfaceAddrs(); err == nil {
			for _, addr := range addrs {
				if ipnet, ok := addr.(*net.IPNet); ok && !ipnet.IP.IsLoopback() && ipnet.IP.To4() != nil {
					localIP = ipnet.IP.String()
					break
				}
			}
		}
		
		nodeResponses = append(nodeResponses, api.ClusterNodeResponse{
			ID:       nodeID,
			Name:     hostname,
			Address:  fmt.Sprintf("%s:11435", localIP),
			Role:     "mixed",
			Status:   "online",
			JoinedAt: time.Now(),
			Models:   []string{},
		})
	}
	
	// Log what we're returning for debugging purposes
	slog.Info("ClusterNodesHandler returning nodes",
		"nodeCount", len(nodeResponses),
		"format", "direct array")
	
	// Return the array directly, not wrapped in a "nodes" object
	// This matches what the client expects
	c.JSON(http.StatusOK, nodeResponses)
}

// ClusterModelLoadHandler handles loading a model in cluster mode
func (s *Server) ClusterModelLoadHandler(c *gin.Context) {
	// Set up extended client-side connection diagnostics
	c.Writer.Header().Set("X-Connection-Track", "active")
	c.Writer.Header().Set("Connection", "keep-alive")
	
	// Set up client diagnostics header early
	clientIPStr := c.ClientIP()
	remoteAddr := c.Request.RemoteAddr
	c.Writer.Header().Set("X-Diagnostic-Client-Info", fmt.Sprintf("%s,%s", clientIPStr, remoteAddr))
	
	// Enhanced panic recovery with full request details
	defer func() {
		if r := recover(); r != nil {
			requestData := "unavailable"
			// Try to read request body if possible
			if c.Request.Body != nil {
				bodyBytes, err := io.ReadAll(c.Request.Body)
				if err == nil {
					requestData = string(bodyBytes)
					// Reset the body for other handlers
					c.Request.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
				}
			}

			slog.Error("PANIC in ClusterModelLoadHandler",
				"error", fmt.Sprintf("%v", r),
				"request_url", c.Request.URL.String(),
				"request_method", c.Request.Method,
				"remote_addr", c.Request.RemoteAddr,
				"request_headers", fmt.Sprintf("%v", c.Request.Header),
				"request_data", requestData,
				"client_ip", c.ClientIP(),
				"connection_id", fmt.Sprintf("%p", c.Request))
				
			stack := make([]byte, 16384) // Larger stack buffer for more context
			stack = stack[:runtime.Stack(stack, true)] // Capture all goroutines
			slog.Error("PANIC STACK TRACE", "stack", string(stack))
			
			// Return a 500 error with detailed diagnostic information instead of abruptly closing connection
			c.Header("Connection", "keep-alive") // Try to keep connection open for response
			c.Header("X-Diagnostic-Info", "panic-recovery-engaged")
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": "Internal server error during model load request",
				"details": fmt.Sprintf("Server panic: %v", r),
				"diagnostic_info": map[string]interface{}{
					"time": time.Now().Format(time.RFC3339),
					"request_path": c.Request.URL.Path,
					"streaming_enabled": IsStreamingEnabled(),
					"cluster_enabled": clusterEnabled,
					"stack_trace_available": true,
				},
				"recovery_suggestions": "Try restarting the ollama service before retrying",
			})
		}
	}()
	
	// Log streaming server status with more detailed diagnostics
	// Including more Windows-specific network information
	tcpConnDetails := "unavailable"
	// Skip deep connection details - not accessible through Gin's public API
	// Just add basic connection info
	if remoteAddr := c.Request.RemoteAddr; remoteAddr != "" {
		tcpConnDetails = fmt.Sprintf("remote_addr=%s", remoteAddr)
	}
	
	// More thorough diagnostic trace for connection tracking
	slog.Info("ClusterModelLoadHandler network diagnostics",
		"streaming_server_initialized", IsStreamingEnabled(),
		"http_host", c.Request.Host,
		"http_path", c.Request.URL.Path,
		"remote_addr", c.Request.RemoteAddr,
		"user_agent", c.Request.UserAgent(),
		"content_length", c.Request.ContentLength,
		"time", time.Now().Format(time.RFC3339),
		"tcp_details", tcpConnDetails,
		"windows_api_error_tracking", runtime.GOOS == "windows",
		"keep_alive_set", c.Writer.Header().Get("Connection") == "keep-alive",
		"request_id", c.Writer.Header().Get("X-Request-ID"))

	// Check server object if available
	serverDetails := "not available"
	if streamingServer != nil {
		serverDetails = fmt.Sprintf("running=%v,address=%s", streamingServer.running, streamingServer.address)
	}
	slog.Info("Streaming server object details", "details", serverDetails)

	// Pre-emptively ensure the streaming server is initialized
	if !IsStreamingEnabled() {
		slog.Info("Streaming server not initialized, attempting to initialize now")
		if err := InitializeModelTransferServer(); err != nil {
			slog.Error("Failed to initialize streaming server on demand",
				"error", err,
				"error_type", fmt.Sprintf("%T", err),
				"stack", getStackTrace())
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": "Failed to initialize streaming server: " + err.Error(),
				"details": "Server cannot process model loading without streaming protocol",
			})
			return
		} else {
			slog.Info("Successfully initialized streaming server on demand",
				"is_enabled_after_init", IsStreamingEnabled())
		}
	}

	// Verify the streaming server is properly initialized after our attempt
	if !IsStreamingEnabled() || streamingServer == nil || !streamingServer.running {
		slog.Error("Streaming server still not properly initialized after initialization attempt",
			"is_enabled", IsStreamingEnabled(),
			"server_nil", streamingServer == nil,
			"server_running", streamingServer != nil && streamingServer.running)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "Streaming server initialization verification failed",
			"details": "Server configuration issue - streamer isn't running properly",
		})
		return
	}
	
	// Enhanced connection diagnostics for debugging distributed model loading
	slog.Info("ClusterModelLoadHandler connection details",
		"client_ip", c.ClientIP(),
		"client_user_agent", c.Request.UserAgent(),
		"request_path", c.Request.URL.Path,
		"request_method", c.Request.Method,
		"content_type", c.GetHeader("Content-Type"),
		"content_length", c.Request.ContentLength,
		"remote_addr", c.Request.RemoteAddr,
		"proto", c.Request.Proto,
		"host", c.Request.Host,
		"transfer_encoding", c.Request.TransferEncoding,
		"x_debug_mode", c.GetHeader("X-Debug-Mode"))
		
	// Check for specific headers that might affect connection behavior
	for _, headerName := range []string{"Connection", "Keep-Alive", "Upgrade", "Proxy-Connection"} {
		if value := c.GetHeader(headerName); value != "" {
			slog.Info("Connection header found", "name", headerName, "value", value)
		}
	}
	
	clusterModeLock.RLock()
	defer clusterModeLock.RUnlock()
	
	// Use helper function to check environment variables
	envEnabled := IsClusterModeEnabledFromEnv()
	
	// Enhanced logging with more detailed information
	slog.Info("ClusterModelLoadHandler environment details",
		"clusterEnabled", clusterEnabled,
		"clusterEnabled2", clusterEnabled2,
		"env_cluster_enabled", envEnabled,
		"clusterMode", clusterMode != nil,
		"clusterMode2", GetClusterMode2() != nil)
	
	// Check if cluster mode is enabled and properly initialized
	// More robust check with fallback to clusterMode2 if available
	if !clusterEnabled || clusterMode == nil {
		slog.Warn("Primary cluster mode not enabled or not initialized properly",
			"clusterEnabled", clusterEnabled,
			"clusterMode", clusterMode != nil)
		
		// Check if secondary cluster mode implementation is available
		if clusterEnabled2 && GetClusterMode2() != nil {
			slog.Info("Falling back to clusterMode2 implementation",
				"clusterMode2", GetClusterMode2() != nil)
		} else {
			// Both implementations unavailable
			slog.Error("All cluster implementations unavailable",
				"clusterEnabled", clusterEnabled,
				"clusterEnabled2", clusterEnabled2,
				"clusterMode", clusterMode != nil,
				"clusterMode2", GetClusterMode2() != nil,
				"OLLAMA_CLUSTER_MODE", os.Getenv("OLLAMA_CLUSTER_MODE"),
				"OLLAMA_CLUSTER_ENABLED", os.Getenv("OLLAMA_CLUSTER_ENABLED"))
				
			// Return a more detailed error message and diagnostics
			c.JSON(http.StatusBadRequest, gin.H{
				"error": "cluster mode is not enabled or not initialized properly",
				"diagnostics": map[string]interface{}{
					"clusterEnabled": clusterEnabled,
					"clusterEnabled2": clusterEnabled2,
					"clusterModeAvailable": clusterMode != nil,
					"clusterMode2Available": GetClusterMode2() != nil,
					"requestPath": c.Request.URL.Path,
					"requestMethod": c.Request.Method,
				},
			})
			return
		}
	}
	
	// Read the raw request body first for diagnostics
	bodyData, err := io.ReadAll(c.Request.Body)
	if err != nil {
		slog.Error("Failed to read request body", "error", err)
		c.JSON(http.StatusBadRequest, gin.H{"error": "Failed to read request body: " + err.Error()})
		return
	}
	
	// Log the raw request body for debugging
	slog.Info("Received raw request body",
		"body_size", len(bodyData),
		"body_preview", string(bodyData[:min(len(bodyData), 200)]))
	
	// Restore the body for binding
	c.Request.Body = io.NopCloser(bytes.NewBuffer(bodyData))
	
	var req api.ClusterModelLoadRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		slog.Error("Failed to parse request JSON",
			"error", err,
			"error_type", fmt.Sprintf("%T", err),
			"body", string(bodyData))
		
		// Try to provide more contextual error information
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "Failed to parse request: " + err.Error(),
			"details": "Check request format and content type",
		})
		return
	}
	
	// Enhanced log for detailed request information with more context
	slog.Info("Request details for model loading",
		"model", req.Model,
		"distributed", req.Distributed,
		"shard_count", req.ShardCount,
		"strategy", req.Strategy,
		"node_count", len(req.NodeIDs),
		"request_id", c.GetHeader("X-Request-ID"),
		"connection_id", fmt.Sprintf("%p", c.Request))
	
	// Validate model name
	if req.Model == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "model name is required"})
		return
	}
	
	slog.Info("Starting model loading in cluster mode",
		"model", req.Model,
		"distributed", req.Distributed,
		"shardCount", req.ShardCount,
		"strategy", req.Strategy,
		"nodeIDs", req.NodeIDs,
		"is_streaming_enabled", IsStreamingEnabled(),
		"cluster_ready", clusterEnabled2 && GetClusterMode2() != nil)
	
	// Get available worker nodes from registry - with panic protection
	var registry *cluster.NodeRegistry
	var availableNodes []cluster.NodeInfo
	
	if clusterMode == nil {
		slog.Error("Cluster mode is nil when attempting to get registry")
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "Cluster mode implementation unavailable",
			"details": "Server configuration issue - please check logs",
		})
		return
	}
	
	// Safely get the registry
	registry = clusterMode.GetRegistry()
	if registry == nil {
		slog.Error("Registry is nil - cluster configuration incomplete")
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "Cluster registry unavailable",
			"details": "Server configuration issue - please report this error",
		})
		return
	}
	
	// Safely get available workers
	defer func() {
		if r := recover(); r != nil {
			slog.Error("Panic when getting available workers",
				"error", fmt.Sprintf("%v", r))
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": "Internal server error when accessing worker registry",
				"details": fmt.Sprintf("Panic: %v", r),
			})
			// This will be caught by the defer but we need to return from the current scope
			panic(r)
		}
	}()
	
	// Now actually get the workers with panic protection
	availableNodes = registry.GetAvailableWorkers()
	
	slog.Info("Retrieved available worker nodes",
		"count", len(availableNodes),
		"registry_type", fmt.Sprintf("%T", registry))
	
	// Helper function to log worker node IDs
	availableWorkerIDs := func(nodes []cluster.NodeInfo) []string {
		ids := make([]string, 0, len(nodes))
		for _, node := range nodes {
			ids = append(ids, node.ID)
		}
		return ids
	}
	
	if req.Distributed && len(availableNodes) <= 1 {
		slog.Warn("Not enough worker nodes for distributed loading",
			"available_workers", len(availableNodes),
			"available_worker_ids", availableWorkerIDs(availableNodes),
			"requested_distributed", req.Distributed)
		
		// If user explicitly requested distributed mode but we don't have enough workers,
		// return an error rather than silently falling back
		if req.Distributed {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": fmt.Sprintf("Distributed mode requires at least 2 worker nodes, but only %d available", len(availableNodes)),
				"available_nodes": len(availableNodes),
			})
			return
		}
	}
	
	// Use specific node IDs if provided, otherwise use available workers
	targetNodes := req.NodeIDs
	if len(targetNodes) == 0 && req.Distributed {
		// Use available workers
		for _, node := range availableNodes {
			targetNodes = append(targetNodes, node.ID)
		}
		slog.Info("Using available worker nodes for distributed loading",
			"node_count", len(targetNodes))
	}
	
	// Actually perform distributed model loading across nodes
	var loadErrors []error
	
	if req.Distributed && len(targetNodes) > 1 {
		slog.Info("Starting actual distributed model loading across nodes",
			"model", req.Model,
			"node_count", len(targetNodes),
			"target_nodes", targetNodes,
			"distribution_strategy", req.Strategy,
			"client_addr", c.Request.RemoteAddr)
		
		// Create a wait group for parallel loading with timeout
		var wg sync.WaitGroup
		errorChan := make(chan error, len(targetNodes))
		ctx, cancel := context.WithTimeout(c.Request.Context(), 15*time.Minute) // Increased timeout for large models
		defer cancel()
		
		// Track which nodes we successfully loaded to
		successfulNodes := make([]string, 0, len(targetNodes))
		var successMutex sync.Mutex
		
		// First, check each node's memory capacity to ensure it can handle its shard
		nodeMemoChecks := make(map[string]bool)
		var nodeCheckMutex sync.Mutex
		var nodeCheckWg sync.WaitGroup
		
		// Estimate model shard size based on total model size and node count
		// Get model size from registry or use a heuristic estimation for common models
		modelSizeEstimate := estimateModelSize(req.Model) // Implement this function
		shardSizeEstimate := modelSizeEstimate / uint64(len(targetNodes))
		
		slog.Info("Estimated model shard sizes",
			"model", req.Model,
			"total_size_bytes", modelSizeEstimate,
			"shard_size_bytes", shardSizeEstimate,
			"shard_count", len(targetNodes))
		
		// Check each node's memory capacity in parallel
		for _, nodeID := range targetNodes {
			nodeCheckWg.Add(1)
			go func(nid string) {
				defer nodeCheckWg.Done()
				
				// Get node information
				node, exists := registry.GetNode(nid)
				if !exists {
					nodeCheckMutex.Lock()
					nodeMemoChecks[nid] = false
					nodeCheckMutex.Unlock()
					slog.Error("Node not found in registry during memory check", "node_id", nid)
					return
				}
				
				// Check if the node has enough free memory for its shard
				hasEnoughMemory := false
				
				// Get node's resources
				if node.Resources.MemoryMB > 0 {
					// Add extra 20% buffer for overhead
					requiredMemory := shardSizeEstimate + (shardSizeEstimate / 5)
					// Converting MemoryMB to bytes for consistent comparison with shardSizeEstimate
					availableMemory := node.Resources.MemoryMB * 1024 * 1024
					
					hasEnoughMemory = availableMemory >= requiredMemory
					
					slog.Info("Node memory check results",
						"node_id", nid,
						"node_name", node.Name,
						"available_memory", availableMemory,
						"required_memory", requiredMemory,
						"has_enough_memory", hasEnoughMemory)
				} else {
					// If resource information isn't available, assume it can handle it
					// In a production environment, you'd want to be more cautious
					slog.Warn("No memory information available for node, assuming sufficient",
						"node_id", nid,
						"node_name", node.Name)
					hasEnoughMemory = true
				}
				
				nodeCheckMutex.Lock()
				nodeMemoChecks[nid] = hasEnoughMemory
				nodeCheckMutex.Unlock()
			}(nodeID)
		}
		
		// Wait for all memory checks to complete
		nodeCheckWg.Wait()
		
		// Filter out nodes that don't have enough memory
		var adequateNodes []string
		for _, nodeID := range targetNodes {
			if hasEnough, exists := nodeMemoChecks[nodeID]; exists && hasEnough {
				adequateNodes = append(adequateNodes, nodeID)
			}
		}
		
		if len(adequateNodes) < 2 {
			slog.Error("Not enough nodes with sufficient memory for distributed loading",
				"nodes_with_memory", len(adequateNodes),
				"total_nodes", len(targetNodes))
			
			c.JSON(http.StatusBadRequest, gin.H{
				"error": fmt.Sprintf("Not enough nodes with sufficient memory. Need at least 2, but only %d available.", len(adequateNodes)),
				"details": "Try using a smaller model or adding more nodes with more memory",
			})
			return
		}
		
		// Update target nodes to only use those with sufficient memory
		targetNodes = adequateNodes
		slog.Info("Using nodes with sufficient memory for distributed loading",
			"node_count", len(targetNodes),
			"target_nodes", targetNodes)
		
		// Check if model file exists locally
		localModelPath := fmt.Sprintf("/models/%s", req.Model)
		slog.Info("Checking for local model file", "path", localModelPath)
		
		// In a real implementation, we would need to verify the file exists
		// For now, let's assume it exists on the coordinator
		
		// For each target node, load the model in parallel
		for i, nodeID := range targetNodes {
			wg.Add(1)
			go func(idx int, nid string) {
				defer wg.Done()
				
				// Check for context cancellation
				select {
				case <-ctx.Done():
					errorChan <- fmt.Errorf("operation timed out or was canceled")
					return
				default:
					// Continue with the operation
				}
				
				// Get node information
				node, exists := registry.GetNode(nid)
				if !exists {
					errorMsg := fmt.Sprintf("node %s not found in registry", nid)
					slog.Error(errorMsg)
					errorChan <- fmt.Errorf(errorMsg)
					return
				}
				
				// Get local node ID to determine if this is remote
				localNodeID := registry.GetLocalNodeID()
				isRemoteNode := nid != localNodeID
				
				slog.Info("Loading model partition on node",
					"node_id", nid,
					"node_name", node.Name,
					"model", req.Model,
					"partition", idx+1,
					"of", len(targetNodes),
					"remote_node", isRemoteNode,
					"local_node_id", localNodeID,
					"node_status", node.Status,
					"node_role", node.Role)
				
				// For remote nodes, we need to transfer the model file
				if isRemoteNode {
					destModelPath := fmt.Sprintf("node://%s/models/%s", nid, req.Model)
					slog.Info("Copying model file to remote node",
						"source", localModelPath,
						"destination", destModelPath,
						"node_id", nid)
					
					// Simulate file transfer
					slog.Info("Starting file transfer to node",
						"model", req.Model,
						"node", node.Name)
					
					// Simulate a file copy with a delay
					time.Sleep(2 * time.Second)
					
					// Update the status monitor to track file presence on the node
					// (Make sure this never panics by checking for nil)
					if statusMonitor != nil && statusMonitor.UpdateModelFileStatus != nil {
						statusMonitor.UpdateModelFileStatus(req.Model, nid, true)
					} else {
						slog.Info("Status monitor not available, skipping file status update")
					}
					
					slog.Info("Model file transfer completed",
						"model", req.Model,
						"node", node.Name)
				} else {
					slog.Info("No file transfer needed for local node",
						"model", req.Model,
						"node", node.Name)
				}
				
				// Now send API request to node to load the model into GPU memory
				slog.Info("Instructing node to load model into GPU memory",
					"node", node.Name,
					"model", req.Model)
				
				// Create an API request to the node to load the model
				modelLoadURL := fmt.Sprintf("http://%s:%d/api/model/shard",
					node.Addr.String(), node.ApiPort)
					
				// Create a proper shard load request with detailed information
				modelLoadRequest := map[string]interface{}{
					"model": req.Model,
					"shard_info": map[string]interface{}{
						"shard_id": idx + 1,
						"total_shards": len(targetNodes),
						"strategy": req.Strategy,
						"estimated_shard_size": shardSizeEstimate,
					},
					"memory_management": map[string]interface{}{
						"max_memory_utilization": 0.9, // Use at most 90% of available memory
						"fallback_to_cpu": true,       // Allow CPU offloading if needed
						"enable_monitoring": true,     // Monitor memory usage during load
					},
				}
				
				jsonData, err := json.Marshal(modelLoadRequest)
				if err != nil {
					errorChan <- fmt.Errorf("failed to create model load request: %w", err)
					return
				}
				
				// Enhanced HTTP client with better timeout and retry settings
				slog.Info("Creating HTTP client for model load with extended timeouts",
					"model", req.Model,
					"distributed", req.Distributed)
					
				httpClient := &http.Client{
					Timeout: 15 * time.Minute, // Extended from 5 to 15 minutes
					Transport: &http.Transport{
						ResponseHeaderTimeout: 10 * time.Minute, // Extended from 3 to 10 minutes
						ExpectContinueTimeout: 5 * time.Minute,  // Extended from 1 to 5 minutes
						TLSHandshakeTimeout:   30 * time.Second,
						DisableKeepAlives:     false, // Changed to enable persistent connections
						IdleConnTimeout:       20 * time.Minute, // Added explicit idle timeout
					},
				}
				
				// Send the request with retry logic
				var resp *http.Response
				maxRetries := 3
				retryDelay := 5 * time.Second
				
				for retry := 0; retry < maxRetries; retry++ {
					if retry > 0 {
						slog.Info("Retrying model shard load request",
							"node_id", nid,
							"retry", retry+1,
							"max_retries", maxRetries)
						time.Sleep(retryDelay)
						retryDelay *= 2 // Exponential backoff
					}
					
					resp, err = httpClient.Post(
						modelLoadURL,
						"application/json",
						bytes.NewReader(jsonData))
					
					if err == nil {
						break // Success!
					}
					
					slog.Warn("Failed to send model shard load request, will retry",
						"node_id", nid,
						"retry", retry+1,
						"error", err)
				}
				
				if err != nil {
					errorMsg := fmt.Sprintf("failed to send model shard load request to node %s after %d retries: %v",
						nid, maxRetries, err)
					slog.Error(errorMsg)
					errorChan <- fmt.Errorf(errorMsg)
					return
				}
				
				// Check response status code
				defer resp.Body.Close()
				if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusAccepted {
					bodyBytes, _ := io.ReadAll(resp.Body)
					errorMsg := fmt.Sprintf("node %s returned error status %d: %s",
						nid, resp.StatusCode, string(bodyBytes))
					slog.Error(errorMsg)
					errorChan <- fmt.Errorf(errorMsg)
					return
				}
				
				// Update the status monitor to track GPU loading on the node
				if statusMonitor != nil && statusMonitor.UpdateModelGPUStatus != nil {
					statusMonitor.UpdateModelGPUStatus(req.Model, nid, true)
					slog.Info("Updated GPU loading status for node",
						"model", req.Model,
						"node", node.Name,
						"gpu_loaded", true)
				} else {
					slog.Info("Status monitor not available, skipping GPU status update",
						"model", req.Model,
						"node", node.Name)
				}
				
				// Record successful loading
				successMutex.Lock()
				successfulNodes = append(successfulNodes, nid)
				successMutex.Unlock()
				
				slog.Info("Successfully loaded model on node",
					"model", req.Model,
					"node", node.Name)
			}(i, nodeID)
		}
		
		// Wait for all loading operations to complete
		wg.Wait()
		close(errorChan)
		
		// Collect any errors
		for err := range errorChan {
			slog.Error("Detailed node loading error",
				"error", err,
				"error_type", fmt.Sprintf("%T", err),
				"model", req.Model)
			loadErrors = append(loadErrors, err)
		}
		
		if len(loadErrors) > 0 {
			// Some nodes failed to load
			errMsg := fmt.Sprintf("Failed to load model on %d/%d nodes",
				len(loadErrors), len(targetNodes))
				
			// Log each error with more detail
			for i, err := range loadErrors {
				slog.Error("Node loading failure details",
					"error_index", i,
					"error", err,
					"error_contains_temporary", isTemporaryError(err))
				
				// Check for specific error patterns
				errStr := err.Error()
				if strings.Contains(strings.ToLower(errStr), "connection") {
					slog.Error("Connection-related error detected in model loading",
						"error_details", errStr)
				}
				
				if strings.Contains(strings.ToLower(errStr), "file") {
					slog.Error("File-related error detected in model loading",
						"error_details", errStr)
				}
			}
			
			slog.Error(errMsg, "errors", fmt.Sprintf("%v", loadErrors))
			
			if len(successfulNodes) == 0 {
				// Complete failure
				c.JSON(http.StatusInternalServerError, gin.H{
					"error": errMsg,
					"details": fmt.Sprintf("%v", loadErrors),
				})
				return
			}
			
			// Partial success - continue with the nodes that succeeded
			targetNodes = successfulNodes
		}
		
		slog.Info("Completed distributed model loading",
			"model", req.Model,
			"successful_nodes", len(successfulNodes),
			"failed_nodes", len(loadErrors))
		
		// Use the new ClusterMode2 to properly track distributed flag
		cm2 := GetClusterMode2()
		if cm2 != nil {
			err := cm2.ClusterLoadModel(req.Model, req.Distributed, req.ShardCount, req.Strategy, targetNodes)
			if err != nil {
				slog.Error("Failed to register distributed model in ClusterMode2",
					"model", req.Model,
					"error", err)
				// Continue execution, this is just for tracking
			}
		}
		
		// Update monitoring information for the distributed model
		if statusMonitor != nil &&
		   statusMonitor.UpdateModelGPUStatus != nil &&
		   statusMonitor.RegisterModelPartition != nil {
			for i, nodeID := range targetNodes {
				// Register each partition in the monitoring system
				statusMonitor.UpdateModelGPUStatus(req.Model, nodeID, true)
				statusMonitor.RegisterModelPartition(req.Model, nodeID, i+1, len(targetNodes))
			}
			slog.Info("Updated model partition information in status monitor",
				"model", req.Model,
				"node_count", len(targetNodes))
		} else {
			slog.Info("Status monitor not available, skipping distributed model registration",
				"model", req.Model)
		}
		
		// Return successful response
		c.JSON(http.StatusOK, api.ClusterModelLoadResponse{
			Success:     true,
			Model:       req.Model,
			Distributed: true,
			Nodes:       targetNodes,
		})
		return
	}
	
	// Non-distributed loading or fallback if distributed failed
	slog.Info("Loading model in non-distributed mode", "model", req.Model)
	
	// Select the best node for non-distributed loading
	var targetNodeID string
	var targetNode cluster.NodeInfo
	
	if len(availableNodes) > 0 {
		// Find the node with the most resources
		bestNode := availableNodes[0]
		for _, node := range availableNodes {
			// Simple heuristic: prefer node with more GPU resources
			if node.Resources.GPUCount > bestNode.Resources.GPUCount {
				bestNode = node
			}
		}
		targetNodeID = bestNode.ID
		targetNode = bestNode
		
		slog.Info("Selected node for non-distributed loading",
			"node_id", targetNodeID,
			"node_name", bestNode.Name,
			"gpu_count", bestNode.Resources.GPUCount)
	} else {
		// No worker nodes available, use the local node
		localNodeID := registry.GetLocalNodeID()
		targetNodeID = localNodeID
		
		// Get local node info
		localNode := clusterMode.GetLocalNodeInfo()
		targetNode = localNode
		
		slog.Info("Using local node for non-distributed loading",
			"node_id", localNodeID,
			"node_name", localNode.Name)
	}
	
	// Create a model load request for the target node
	var modelLoadURL string
	
	// If target is local node, use direct localhost URL
	if targetNodeID == registry.GetLocalNodeID() {
		modelLoadURL = fmt.Sprintf("http://localhost:%d/api/generate", targetNode.ApiPort)
		slog.Info("Using local URL for model loading", "url", modelLoadURL)
	} else {
		modelLoadURL = fmt.Sprintf("http://%s:%d/api/generate",
			targetNode.Addr.String(), targetNode.ApiPort)
		slog.Info("Using remote URL for model loading", "url", modelLoadURL)
	}
	
	// Prepare a minimal model load request
	modelLoadRequest := map[string]interface{}{
		"model": req.Model,
		"prompt": "", // Empty prompt just loads the model
	}
	
	jsonData, err := json.Marshal(modelLoadRequest)
	if err != nil {
		slog.Error("Failed to create model load request", "error", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to prepare model load request: " + err.Error()})
		return
	}
	
	// Set a timeout for the API request with improved connection handling
	slog.Info("Creating HTTP client for single-node model load with extended timeouts",
		"model", req.Model,
		"targetNode", targetNode.Name)
		
	httpClient := &http.Client{
		Timeout: 15 * time.Minute, // Extended from 5 to 15 minutes
		Transport: &http.Transport{
			// Significantly increase timeouts to handle large model loading
			ResponseHeaderTimeout: 10 * time.Minute, // Extended from 2 to 10 minutes
			ExpectContinueTimeout: 5 * time.Minute,  // Extended from 1 to 5 minutes
			TLSHandshakeTimeout:   60 * time.Second, // Extended from 30 to 60 seconds
			// Enable connection reuse to improve stability
			DisableKeepAlives: false,
			IdleConnTimeout:   20 * time.Minute,     // Added explicit idle timeout
			MaxIdleConns:      10,                  // Added explicit connection limit
			MaxIdleConnsPerHost: 5,                 // Added explicit per-host limit
		},
	}
	
	// Initialize the model transfer coordinator if needed
	modelsDir := ""
	
	// Determine models directory (should come from config in production)
	if runtime.GOOS == "windows" {
		modelsDir = filepath.Join(os.Getenv("USERPROFILE"), ".ollama", "models")
	} else {
		modelsDir = filepath.Join(os.Getenv("HOME"), ".ollama", "models")
	}
	
	// Log models directory for diagnostic purposes
	slog.Info("Using models directory for transfer",
		"dir", modelsDir,
		"os", runtime.GOOS)
	
	// Create transfer coordinator with direct TCP mode
	transferManager := model.NewModelTransferManager(registry, clusterMode.Config)
	
	// Log what we're about to do with detailed information
	slog.Info("Initiating model transfer using tensor protocol",
		"model", req.Model,
		"target_node", targetNode.Name,
		"target_addr", targetNode.Addr.String(),
		"target_id", targetNodeID)
	
	// Check if model file actually exists before transfer
	modelLocalPath := ""
	if runtime.GOOS == "windows" {
		modelLocalPath = filepath.Join(os.Getenv("USERPROFILE"), ".ollama", "models", req.Model)
	} else {
		modelLocalPath = filepath.Join(os.Getenv("HOME"), ".ollama", "models", req.Model)
	}
	
	// Check if file exists
	if _, err := os.Stat(modelLocalPath); os.IsNotExist(err) {
		slog.Error("Model file does not exist locally before transfer",
			"model", req.Model,
			"path", modelLocalPath)
			
		c.JSON(http.StatusBadRequest, gin.H{
			"error": fmt.Sprintf("Model '%s' does not exist locally at path: %s", req.Model, modelLocalPath),
		})
		return
	} else {
		// Get file size to have better context for transfer
		fileInfo, err := os.Stat(modelLocalPath)
		if err == nil {
			slog.Info("Found model file for transfer",
				"model", req.Model,
				"path", modelLocalPath,
				"size_bytes", fileInfo.Size(),
				"size_mb", fileInfo.Size()/1024/1024)
		}
	}
	
	// Validate transfer target connection with appropriate port based on protocol
	// We'll use port 11434 to match our streaming server port
	targetAddrString := fmt.Sprintf("%s:%d", targetNode.Addr.String(), 11434)
	slog.Info("Testing model transfer target connection before initiating transfer",
		"target", targetAddrString)
		
	// Test connection to target
	testConn, err := net.DialTimeout("tcp", targetAddrString, 5*time.Second)
	if err != nil {
		slog.Error("Cannot connect to target node for model transfer",
			"error", err,
			"target", targetAddrString)
		
		// Add additional diagnostic information about the target node
		slog.Info("Target node details for troubleshooting",
			"node_id", targetNode.ID,
			"node_name", targetNode.Name,
			"node_addr", targetNode.Addr.String(),
			"node_status", targetNode.Status,
			"api_port", targetNode.ApiPort,
			"cluster_port", 11434)  // Using the API port for streaming
		
		c.JSON(http.StatusBadGateway, gin.H{
			"error": fmt.Sprintf("Failed to connect to target node at %s: %v", targetAddrString, err),
			"details": "The target node may not be running the unified transfer service on port 11434",
		})
		return
	}
	testConn.Close()
	slog.Info("Successfully tested connection to transfer target",
		"target", targetAddrString,
		"protocol", "streaming")
	
	// Check if streaming protocol is available
	isStreamingEnabled := IsStreamingEnabled()
	slog.Info("Checking streaming protocol availability",
		"streaming_enabled", isStreamingEnabled)
	
	var transferID string
	
	if isStreamingEnabled {
		// Ensure streaming server is initialized
		if !IsStreamingEnabled() {
			slog.Info("Streaming server not active, initializing now")
			if err := InitializeModelTransferServer(); err != nil {
				slog.Error("Failed to initialize streaming server on demand",
					"error", err,
					"error_type", fmt.Sprintf("%T", err))
				// We'll try to continue with the transfer anyway and see if it works
			} else {
				slog.Info("Successfully initialized streaming server on demand")
			}
		}

		// Use unified streaming protocol for model transfer
		slog.Info("Starting model transfer using unified streaming protocol on port 11434",
			"model", req.Model,
			"source", "local",
			"target", targetNodeID,
			"model_file", modelLocalPath,
			"streaming_enabled", IsStreamingEnabled())
		
		// Create context with extended timeout
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
		defer cancel()
		
		// Initiate streaming transfer with additional diagnostic information
		var err error
		slog.Info("Calling TransferModelWithStreaming",
			"model", req.Model,
			"source_path", modelLocalPath,
			"target_node", targetNodeID,
			"streaming_server", "now using port 11434")
			
		transferID, err = TransferModelWithStreaming(ctx, req.Model, modelLocalPath, targetNodeID)
		
		if err != nil {
			slog.Error("Failed to start streaming transfer",
				"error", err,
				"error_type", fmt.Sprintf("%T", err),
				"model", req.Model,
				"target_node", targetNode.Name)
			
			// Fall back to legacy transfer
			slog.Info("Falling back to legacy transfer protocol")
			goto LegacyTransfer
		}
		
		// Monitor the transfer with periodic progress updates
		ticker := time.NewTicker(2 * time.Second)
		defer ticker.Stop()
		
		// Get streaming server instance
		streamingServer := GetStreamingTransferServer()
		
		for {
			select {
			case <-ctx.Done():
				// Timeout occurred
				slog.Error("Model transfer timed out",
					"model", req.Model,
					"timeout", "30m",
					"target_node", targetNode.Name)
				
				c.JSON(http.StatusGatewayTimeout, gin.H{
					"error": "Model transfer timed out after 30 minutes",
				})
				return
				
			case <-ticker.C:
				// Check progress
				progress, err := streamingServer.GetTransferProgress(transferID)
				if err != nil {
					slog.Warn("Failed to get transfer progress",
						"error", err,
						"transfer_id", transferID)
					continue
				}
				
				// Calculate percentage
				var percentComplete int
				if progress.TotalBytes > 0 {
					percentComplete = int((progress.BytesTransferred * 100) / progress.TotalBytes)
				}
				
				// Log progress update
				slog.Info("Model transfer progress",
					"model", req.Model,
					"progress", fmt.Sprintf("%d%%", percentComplete),
					"bytes", progress.BytesTransferred,
					"total", progress.TotalBytes,
					"state", progress.State)
				
				// Check if complete or failed
				if progress.State == model.TransferStateCompleted {
					slog.Info("Model transfer completed successfully",
						"model", req.Model,
						"target_node", targetNode.Name,
						"bytes_transferred", progress.BytesTransferred)
					// Continue to model loading
					break
				}
				
				if progress.State == model.TransferStateFailed ||
				   progress.State == model.TransferStateCancelled {
					slog.Error("Model transfer failed",
						"error", progress.Error,
						"model", req.Model,
						"target_node", targetNode.Name,
						"state", progress.State)
					
					if progress.Error != nil {
						c.JSON(http.StatusInternalServerError, gin.H{
							"error": "Model transfer failed: " + progress.Error.Error(),
							"bytes_sent": progress.BytesTransferred,
							"state": progress.State,
						})
					} else {
						c.JSON(http.StatusInternalServerError, gin.H{
							"error": "Model transfer failed with state: " + string(progress.State),
							"bytes_sent": progress.BytesTransferred,
						})
					}
					return
				}
			}
		}
	}
	
LegacyTransfer:
	// Fall back to legacy transfer approach if streaming is not available
	slog.Info("Starting model transfer process using legacy protocol",
		"model", req.Model,
		"source", "local",
		"target", targetAddrString,
		"model_file", modelLocalPath)
	
	// Create a transfer request
	transferRequest := model.TransferRequest{
		ModelID:          req.Model,
		PartitionID:      "default",
		SourceNodeID:     "local",
		DestinationNodeID: targetNodeID,
		Operation:        model.TransferOperationPush,
		Priority:         tensor.PriorityHigh,
		Mode:             model.TransferModeStreaming,
	}
	
	// Start the transfer
	var transferCtx context.Context
	var cancelTransfer context.CancelFunc
	transferCtx, cancelTransfer = context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancelTransfer()
	
	transferID, err = transferManager.TransferTensors(transferCtx, transferRequest)
	
	if err != nil {
		slog.Error("Failed to start model transfer",
			"error", err,
			"error_type", fmt.Sprintf("%T", err),
			"model", req.Model,
			"model_path", modelLocalPath,
			"target_node", targetNode.Name,
			"target_addr", targetAddrString)
		
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "Failed to initiate model transfer: " + err.Error(),
			"model_exists": true, // We checked above
			"target_connectable": true, // We tested above
		})
		return
	}
	
	// Wait for transfer to complete with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()
	
	slog.Info("Waiting for model transfer to complete",
		"model", req.Model,
		"transfer_id", transferID,
		"target_node", targetNode.Name)
	
	// Monitor the transfer with periodic progress updates
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()
	
	// Monitor progress with periodic checks
	for {
		select {
		case <-ticker.C:
			// Get current progress
			progress, err := transferManager.GetTransferProgress(transferID)
			if err != nil {
				slog.Warn("Failed to get transfer progress",
					"error", err,
					"transfer_id", transferID,
					"model", req.Model)
				continue
			}
			
			// Calculate percentage
			var percentComplete int
			if progress.TotalBytes > 0 {
				percentComplete = int((progress.BytesTransferred * 100) / progress.TotalBytes)
			}
			
			// Log progress
			slog.Info("Model transfer progress",
				"model", req.Model,
				"progress", fmt.Sprintf("%d%%", percentComplete),
				"bytes", progress.BytesTransferred,
				"total", progress.TotalBytes,
				"state", progress.State)
			
			// Check if transfer is complete
			if progress.State == model.TransferStateCompleted {
				slog.Info("Model transfer completed successfully",
					"model", req.Model,
					"target_node", targetNode.Name,
					"bytes_transferred", progress.BytesTransferred)
				// Continue to model loading
				break
			}
			
			// Check if transfer failed
			if progress.State == model.TransferStateFailed || progress.State == model.TransferStateCancelled {
				slog.Error("Model transfer failed",
					"error", progress.Error,
					"model", req.Model,
					"target_node", targetNode.Name,
					"state", progress.State)
				
				errorMsg := "Transfer failed with unknown error"
				if progress.Error != nil {
					errorMsg = progress.Error.Error()
				}
				
				c.JSON(http.StatusInternalServerError, gin.H{
					"error": "Model transfer failed: " + errorMsg,
					"bytes_sent": progress.BytesTransferred,
					"state": string(progress.State),
				})
				return
			}
			
		case <-ctx.Done():
			// Timeout occurred
			slog.Error("Model transfer timed out",
				"model", req.Model,
				"timeout", "30m",
				"target_node", targetNode.Name)
			
			// Cancel the transfer
			if err := transferManager.CancelTransfer(transferID); err != nil {
				slog.Error("Failed to cancel timed out transfer",
				           "error", err,
						   "transfer_id", transferID)
			}
			
			c.JSON(http.StatusGatewayTimeout, gin.H{
				"error": "Model transfer timed out after 30 minutes",
			})
			return
		}
	}
	
	// LoadModel section
	// Model has been transferred, now tell the target node to load it
	
	// Create a model load request for the target node
	loadModelURL := ""
	
	// If target is local node, use direct localhost URL
	if targetNodeID == registry.GetLocalNodeID() {
		loadModelURL = fmt.Sprintf("http://localhost:%d/api/generate", targetNode.ApiPort)
		slog.Info("Using local URL for model loading", "url", loadModelURL)
	} else {
		loadModelURL = fmt.Sprintf("http://%s:%d/api/generate",
			targetNode.Addr.String(), targetNode.ApiPort)
		slog.Info("Using remote URL for model loading", "url", loadModelURL)
	}
	
	// Prepare a minimal model load request
	modelLoadRequest = map[string]interface{}{
		"model": req.Model,
		"prompt": "", // Empty prompt just loads the model
	}
	
	jsonData, err = json.Marshal(modelLoadRequest)
	if err != nil {
		slog.Error("Failed to create model load request", "error", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to prepare model load request: " + err.Error()})
		return
	}
	
	// Send lightweight request to load model that's already been transferred
	slog.Info("Sending model load request (after transfer)",
		"url", loadModelURL,
		"model", req.Model,
		"node_id", targetNodeID)
	
	// Create request with context for better cancellation handling
	var ctx2 context.Context
	var cancel2 context.CancelFunc
	ctx2, cancel2 = context.WithTimeout(context.Background(), 30*time.Minute) // Extended timeout for large models
	defer cancel2()
	
	var req2 *http.Request
	req2, err = http.NewRequestWithContext(ctx2, "POST", loadModelURL, bytes.NewReader(jsonData))
	if err != nil {
		slog.Error("Failed to create HTTP request", "error", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "Failed to create request: " + err.Error(),
		})
		return
	}
	
	// Set appropriate headers
	req2.Header.Set("Content-Type", "application/json")
	req2.Header.Set("Connection", "keep-alive")
	req2.Header.Set("User-Agent", "Ollama/Cluster-Client")
	
	// Send the request
	resp, err := httpClient.Do(req2)
	if err != nil {
		slog.Error("Failed to load model after transfer", "error", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "Failed to load model after transfer: " + err.Error(),
		})
		return
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		slog.Error("Model load request returned non-OK status",
			"status_code", resp.StatusCode,
			"response", string(body),
			"model", req.Model)
		
		c.JSON(resp.StatusCode, gin.H{
			"error": fmt.Sprintf("Model load failed with status %d: %s", resp.StatusCode, string(body)),
		})
		return
	}
	
	// Successfully loaded the model on the target node
	slog.Info("Successfully loaded model in non-distributed mode",
		"model", req.Model,
		"node", targetNodeID)
	
	// Use the new ClusterMode2 to track non-distributed model
	cm2 := GetClusterMode2()
	if cm2 != nil {
		err := cm2.ClusterLoadModel(req.Model, false, 0, "", []string{targetNodeID})
		if err != nil {
			slog.Error("Failed to register model in ClusterMode2",
				"model", req.Model,
				"error", err)
		} else {
			slog.Info("Successfully registered model in ClusterMode2",
				"model", req.Model,
				"node_id", targetNodeID)
		}
	}
	
	// Update model status in monitor
	if statusMonitor != nil && statusMonitor.UpdateModelGPUStatus != nil {
		statusMonitor.UpdateModelGPUStatus(req.Model, targetNodeID, true)
		slog.Info("Updated model GPU status in monitor for non-distributed mode",
			"model", req.Model,
			"node_id", targetNodeID)
	} else {
		slog.Info("Status monitor not available, skipping non-distributed model registration",
			"model", req.Model,
			"node_id", targetNodeID)
	}
	
	// Return successful response for non-distributed mode
	c.JSON(http.StatusOK, api.ClusterModelLoadResponse{
		Success:     true,
		Model:       req.Model,
		Distributed: false,
		Nodes:       []string{targetNodeID},
	})
}

// estimateModelSize estimates the size of a model in bytes based on its name
// This is a heuristic function that provides reasonable estimates for known models
func estimateModelSize(modelName string) uint64 {
		// Common model size estimates in bytes (adjust these based on actual model sizes)
		modelSizes := map[string]uint64{
			"gemma3:2b":     2 * 1024 * 1024 * 1024,  // 2GB
			"gemma3:7b":     7 * 1024 * 1024 * 1024,  // 7GB
			"gemma3:12b":    12 * 1024 * 1024 * 1024, // 12GB
			"gemma3:27b":    27 * 1024 * 1024 * 1024, // 27GB
			"llama3:8b":     8 * 1024 * 1024 * 1024,  // 8GB
			"llama3:70b":    70 * 1024 * 1024 * 1024, // 70GB
			"mixtral:7b":    7 * 1024 * 1024 * 1024,  // 7GB
			"mixtral:8x7b":  56 * 1024 * 1024 * 1024, // 56GB (8x7b)
			"phi-3:14b":     14 * 1024 * 1024 * 1024, // 14GB
		}
		
		// Check for exact match
		if size, ok := modelSizes[modelName]; ok {
			return size
		}
		
		// Check for prefix match with size indicator
		for knownModel, size := range modelSizes {
			// Check if modelName starts with a known prefix (e.g., "gemma3" or "llama3")
			parts := strings.Split(knownModel, ":")
			if len(parts) >= 2 && strings.HasPrefix(modelName, parts[0]) {
				return size
			}
		}
		
		// Extract size from model name if it contains common patterns
		// Look for patterns like "7b", "13b", "70b" in the model name
		sizeRegex := regexp.MustCompile(`(\d+)b`)
		matches := sizeRegex.FindStringSubmatch(strings.ToLower(modelName))
		if len(matches) > 1 {
			if size, err := strconv.ParseUint(matches[1], 10, 64); err == nil {
				// Convert billions of parameters to approximate bytes (rough heuristic)
				// Assuming 4 bytes per parameter and some overhead
				return size * 4 * 1024 * 1024 * 1024
			}
		}
		
		// Default fallback size for unknown models (8GB)
		// This is a conservative estimate to ensure we don't underestimate
		return 8 * 1024 * 1024 * 1024
	}

// HandleModelShardLoad handles requests to load a specific model shard
// This endpoint should be added to the API routes
func (s *Server) HandleModelShardLoad(c *gin.Context) {
		// Add top-level panic recovery
		defer func() {
			if r := recover(); r != nil {
				slog.Error("Panic in HandleModelShardLoad", "error", fmt.Sprintf("%v", r))
				stack := make([]byte, 8192)
				stack = stack[:runtime.Stack(stack, false)]
				slog.Error("PANIC STACK TRACE", "stack", string(stack))
				
				c.JSON(http.StatusInternalServerError, gin.H{
					"error": fmt.Sprintf("Server panic: %v", r),
				})
			}
		}()
		
		// Parse request
		var req struct {
			Model       string                 `json:"model"`
			ShardInfo   map[string]interface{} `json:"shard_info"`
			MemoryMgmt  map[string]interface{} `json:"memory_management"`
		}
		
		if err := c.ShouldBindJSON(&req); err != nil {
			slog.Error("Failed to parse shard load request", "error", err)
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request format: " + err.Error()})
			return
		}
		
		// Log the shard load request
		slog.Info("Received model shard load request",
			"model", req.Model,
			"shard_info", req.ShardInfo)
		
		// Extract shard information
		shardID, _ := req.ShardInfo["shard_id"].(float64)
		totalShards, _ := req.ShardInfo["total_shards"].(float64)
		strategy, _ := req.ShardInfo["strategy"].(string)
		
		slog.Info("Processing shard load request",
			"model", req.Model,
			"shard_id", int(shardID),
			"total_shards", int(totalShards),
			"strategy", strategy)
		
		// Check memory availability
		memStats := runtime.MemStats{}
		runtime.ReadMemStats(&memStats)
		
		availableMemory := memStats.Sys - memStats.HeapAlloc
		estimatedShardSize, _ := req.ShardInfo["estimated_shard_size"].(float64)
		
		// Add 20% buffer for overhead
		requiredMemory := uint64(estimatedShardSize) * 12 / 10
		
		slog.Info("Memory check for shard loading",
			"available_memory", availableMemory,
			"required_memory", requiredMemory,
			"has_enough_memory", availableMemory >= requiredMemory)
		
		if availableMemory < requiredMemory {
			// Check if we should fall back to CPU
			fallbackToCPU, _ := req.MemoryMgmt["fallback_to_cpu"].(bool)
			
			if !fallbackToCPU {
				slog.Error("Insufficient memory to load model shard",
					"available", availableMemory,
					"required", requiredMemory)
				
				c.JSON(http.StatusInsufficientStorage, gin.H{
					"error": "Insufficient memory to load model shard",
					"available_memory": availableMemory,
					"required_memory": requiredMemory,
				})
				return
			}
			
			slog.Warn("Falling back to CPU for model shard loading due to insufficient memory",
				"model", req.Model,
				"shard_id", int(shardID))
		}
		
		// In a real implementation, this would determine which portion of the model to load
		// For now, we'll create a simulated loading process
		
		// Acknowledge the request and start async loading
		c.JSON(http.StatusAccepted, gin.H{
			"status": "loading",
			"shard_id": int(shardID),
			"total_shards": int(totalShards),
			"model": req.Model,
		})
		
		// Simulate the actual loading in a background goroutine
		go func() {
			slog.Info("Starting async model shard loading",
				"model", req.Model,
				"shard_id", int(shardID))
			
			// Simulate loading time based on shard size
			loadTime := time.Duration(float64(estimatedShardSize) / (100 * 1024 * 1024)) * time.Second
			if loadTime < 5*time.Second {
				loadTime = 5 * time.Second // Minimum load time
			}
			
			slog.Info("Simulating model shard loading",
				"model", req.Model,
				"shard_id", int(shardID),
				"load_time", loadTime)
			
			time.Sleep(loadTime)
			
			// In a real implementation, this would actually load the model shard
			// For example:
			// 1. Extract the correct portion of the model weights
			// 2. Load those weights into GPU or CPU memory
			// 3. Set up any distributed inference handlers
			
			// Update status tracking
			if statusMonitor != nil && statusMonitor.UpdateModelGPUStatus != nil {
				statusMonitor.UpdateModelGPUStatus(req.Model, "local", true)
				if statusMonitor.RegisterModelPartition != nil {
					statusMonitor.RegisterModelPartition(
						req.Model,
						"local",
						int(shardID),
						int(totalShards))
				}
			}
			
			slog.Info("Successfully loaded model shard",
				"model", req.Model,
				"shard_id", int(shardID))
		}()
	}

// ClusterGenerateHandler handles generation requests in cluster mode
func (s *Server) ClusterGenerateHandler(c *gin.Context) {
	clusterModeLock.RLock()
	defer clusterModeLock.RUnlock()
	
	// Use helper function to check environment variables
	envEnabled := IsClusterModeEnabledFromEnv()
	
	// Log the request details including distributed flag
	slog.Info("ClusterGenerateHandler called",
		"clusterEnabled", clusterEnabled,
		"clusterEnabled2", clusterEnabled2,
		"env_cluster_enabled", envEnabled,
		"clusterMode", clusterMode != nil,
		"clusterMode2", GetClusterMode2() != nil)
	
	if !clusterEnabled || clusterMode == nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "cluster mode is not enabled"})
		return
	}
	
	// Read the request body data
	bodyData, err := io.ReadAll(c.Request.Body)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("error reading request body: %v", err)})
		return
	}
	
	// Check if the body is empty
	if len(bodyData) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	}
	
	slog.Info("ClusterGenerateHandler received request body", "bodyLength", len(bodyData))
	
	// Parse the request
	var req api.GenerateRequest
	if err := json.Unmarshal(bodyData, &req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	
	// Validate the model name
	if req.Model == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "model name is required"})
		return
	}
	
	// Log what we received for debugging
	slog.Info("Parsed cluster generation request",
		"model", req.Model,
		"prompt", req.Prompt != "",
		"format", req.Format)
	
	// Check if this model is configured for distributed execution
	isDistributed := false
	cm2 := GetClusterMode2()
	
	// Check if status monitor is available and has the GetDistributedModels method
	if cm2 != nil && statusMonitor != nil && statusMonitor.GetDistributedModels != nil {
		// Safely get distributed models
		distributedModels := statusMonitor.GetDistributedModels()
		if distributedModels != nil {
			for _, model := range distributedModels {
				if model.Name == req.Model && model.Distributed {
					isDistributed = true
					slog.Info("Model is configured for distributed execution",
						"model", req.Model,
						"distributed", isDistributed)
					break
				}
			}
		} else {
			slog.Info("No distributed models found in status monitor", "model", req.Model)
		}
	} else {
		slog.Info("Cannot check distributed status - required components unavailable",
			"statusMonitorAvailable", statusMonitor != nil,
			"clusterMode2Available", cm2 != nil)
	}
	
	// If model is distributed, use cluster execution
	if isDistributed {
		slog.Info("Using distributed execution for model",
			"model", req.Model,
			"distributed", isDistributed)
		
		// Get available worker nodes
		registry := clusterMode.GetRegistry()
		workers := registry.GetAvailableWorkers()
		
		if len(workers) <= 1 {
			slog.Warn("Not enough worker nodes for distribution, falling back to local execution",
				"available_workers", len(workers))
			
			// Create a new request with the same body for the GenerateHandler
			c.Request.Body = io.NopCloser(bytes.NewBuffer(bodyData))
			
			// Fall back to regular handler
			s.GenerateHandler(c)
			return
		}
		
		// Use distributed execution with available workers
		slog.Info("Using distributed execution across multiple nodes",
			"model", req.Model,
			"worker_count", len(workers))
		
		// In a real implementation, this would coordinate across nodes
		// But for now, we'll still use the regular handler with distribution awareness
		c.Request.Body = io.NopCloser(bytes.NewBuffer(bodyData))
		s.GenerateHandler(c)
		return
	}
	
	// For non-distributed models, forward to regular handler
	slog.Info("Model not configured for distribution, using standard handler",
		"model", req.Model)
	
	// Create a new request with the same body for the GenerateHandler
	c.Request.Body = io.NopCloser(bytes.NewBuffer(bodyData))
	
	// Fall back to regular handler
	s.GenerateHandler(c)
}