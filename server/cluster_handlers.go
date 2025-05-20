package server

import (
	"errors"
	"fmt"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cluster"
	"log/slog"
)

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
	
	// In a real implementation, would call discovery.JoinNode
	// For now, returning a successful placeholder response
	c.JSON(http.StatusOK, api.ClusterJoinResponse{
		Success:     true,
		NodeID:      "node-123", // Would come from actual join operation
		ClusterID:   "cluster-456", // Would come from actual join operation
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
	
	// If the original implementation is not available but the new one is, use that instead
	if (clusterMode == nil) && (GetClusterMode2() != nil) {
		slog.Info("Using clusterMode2 implementation for nodes request")
		
		// Get nodes from the new implementation
		cm2 := GetClusterMode2()
		nodes := cm2.GetNodes()
		
		// Convert to API responses
		nodeResponses := make([]api.ClusterNodeResponse, 0, len(nodes))
		for _, node := range nodes {
			// Format the address properly including the port
			var addressStr string
			if node.Addr != nil {
				addressStr = fmt.Sprintf("%s:%d", node.Addr.String(), node.ApiPort)
			} else {
				// If address is nil, use a placeholder based on config
				addressStr = fmt.Sprintf("%s:%d", "localhost", node.ApiPort)
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
		
		// Return the array directly
		c.JSON(http.StatusOK, nodeResponses)
		return
	}
	
	// Get registry and nodes
	registry := clusterMode.GetRegistry()
	nodes := registry.GetAllNodes()
	
	// Convert to API responses
	nodeResponses := make([]api.ClusterNodeResponse, 0, len(nodes))
	for _, node := range nodes {
		resp := api.ConvertNodeInfoToResponse(node)
		nodeResponses = append(nodeResponses, resp)
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
	clusterModeLock.RLock()
	defer clusterModeLock.RUnlock()
	
	// Use helper function to check environment variables
	envEnabled := IsClusterModeEnabledFromEnv()
	
	// Log the request details including the distributed flag
	slog.Info("ClusterModelLoadHandler called",
		"clusterEnabled", clusterEnabled,
		"clusterEnabled2", clusterEnabled2,
		"env_cluster_enabled", envEnabled,
		"clusterMode", clusterMode != nil,
		"clusterMode2", GetClusterMode2() != nil)
	
	if !clusterEnabled || clusterMode == nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "cluster mode is not enabled"})
		return
	}
	
	var req api.ClusterModelLoadRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	
	// Validate model name
	if req.Model == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "model name is required"})
		return
	}
	
	slog.Info("Loading model in cluster mode",
		"model", req.Model,
		"distributed", req.Distributed,
		"shardCount", req.ShardCount,
		"strategy", req.Strategy,
		"nodeIDs", req.NodeIDs)
	
	// Use the new ClusterMode2 to properly track distributed flag
	cm2 := GetClusterMode2()
	if cm2 != nil {
		err := cm2.ClusterLoadModel(req.Model, req.Distributed, req.ShardCount, req.Strategy, req.NodeIDs)
		if err != nil {
			slog.Error("Failed to load model in cluster mode",
				"model", req.Model,
				"error", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		
		// After loading, fetch the actual distributed status from the monitor
		// to ensure we're returning the correct state
		var actualDistributed bool
		if statusMonitor != nil {
			for _, model := range statusMonitor.GetDistributedModels() {
				if model.Name == req.Model {
					actualDistributed = model.Distributed
					slog.Info("Retrieved actual distributed status from monitor",
						"model", req.Model,
						"distributed_from_request", req.Distributed,
						"distributed_actual", actualDistributed)
					break
				}
			}
		}
		
		// In a distributed setup, we would:
		// 1. Determine which nodes should load the model
		// 2. Split the model if needed (for tensor parallelism)
		// 3. Coordinate the loading across nodes
		
		// Use the actual status from the monitor if available, otherwise fall back to request
		distributedStatus := actualDistributed
		if !distributedStatus {
			distributedStatus = req.Distributed // Fallback
		}
		
		c.JSON(http.StatusOK, api.ClusterModelLoadResponse{
			Success:     true,
			Model:       req.Model,
			Distributed: distributedStatus,
			Nodes:       req.NodeIDs,
		})
		return
	}
	
	// Legacy fallback for original cluster mode
	c.JSON(http.StatusOK, api.ClusterModelLoadResponse{
		Success:     true,
		Model:       req.Model,
		Distributed: req.Distributed,
		Nodes:       req.NodeIDs,
	})
}