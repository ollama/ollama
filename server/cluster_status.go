package server

import (
	"sync"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cluster"
	"log/slog"
)

// ModelFileStatus tracks the file presence and GPU loading status on nodes
const (
	ModelStatusRegistered  = "registered"
	ModelStatusFilePresent = "files_present"
	ModelStatusGPULoaded   = "gpu_loaded"
	ModelStatusUnloaded    = "unloaded"
	ModelStatusError       = "error"
)

// ClusterStatusMonitor tracks and reports on the health and status of the cluster
type ClusterStatusMonitor struct {
	mu                 sync.RWMutex
	modelDistribution  map[string]*DistributedModelInfo
	nodeModels         map[string][]string
	clusterHealthy     bool
	lastHealthyCheck   time.Time
	healthCheckInterval time.Duration
}

// DistributedModelInfo tracks information about models distributed across the cluster
type DistributedModelInfo struct {
	Name               string
	Size               int64
	Distributed        bool
	Nodes              []string
	NodesWithFiles     map[string]bool // Tracks which nodes have actual model files
	NodesWithGPULoaded map[string]bool // Tracks which nodes have loaded the model into GPU
	Shards             int
	Status             string
	LoadedAt           time.Time
}

// NewClusterStatusMonitor creates a new cluster status monitor
func NewClusterStatusMonitor() *ClusterStatusMonitor {
	return &ClusterStatusMonitor{
		modelDistribution:  make(map[string]*DistributedModelInfo),
		nodeModels:         make(map[string][]string),
		clusterHealthy:     false,
		healthCheckInterval: 30 * time.Second, // Default to checking every 30 seconds
	}
}

// RegisterModelOnNode registers that a model is loaded on a specific node
func (csm *ClusterStatusMonitor) RegisterModelOnNode(modelName string, nodeID string, distributed bool, shardCount int) {
	csm.mu.Lock()
	defer csm.mu.Unlock()
	
	slog.Info("Registering model on node",
		"model", modelName,
		"node_id", nodeID,
		"distributed", distributed,
		"shards", shardCount)
	
	// Update node models mapping
	if _, exists := csm.nodeModels[nodeID]; !exists {
		csm.nodeModels[nodeID] = make([]string, 0)
	}
	
	// Check if this model is already registered for this node
	for _, m := range csm.nodeModels[nodeID] {
		if m == modelName {
			// Even if already registered, we should update the distributed flag
			// to ensure consistency across the cluster
			if info, exists := csm.modelDistribution[modelName]; exists {
				if info.Distributed != distributed {
					slog.Info("Updating distributed flag for existing model",
						"model", modelName,
						"old_distributed", info.Distributed,
						"new_distributed", distributed)
					info.Distributed = distributed || info.Distributed // If true anywhere, should be true
				}
			}
			return
		}
	}
	
	csm.nodeModels[nodeID] = append(csm.nodeModels[nodeID], modelName)
	
	// Update distributed model info
	if _, exists := csm.modelDistribution[modelName]; !exists {
		csm.modelDistribution[modelName] = &DistributedModelInfo{
			Name:               modelName,
			Size:               0, // Would be populated with actual model size
			Distributed:        distributed,
			Nodes:              []string{nodeID},
			NodesWithFiles:     make(map[string]bool),
			NodesWithGPULoaded: make(map[string]bool),
			Shards:             shardCount,
			Status:             "registered", // Changed from "loaded" to more accurately reflect state
			LoadedAt:           time.Now(),
		}
		// Mark the coordinator node as having the file by default
		localNode := nodeID // Default assumption
		csm.modelDistribution[modelName].NodesWithFiles[localNode] = true
		
		slog.Info("Created new model distribution entry",
			"model", modelName,
			"distributed", distributed,
			"has_file", true)
	} else {
		// Add this node to the existing model info if not already present
		found := false
		for _, n := range csm.modelDistribution[modelName].Nodes {
			if n == nodeID {
				found = true
				break
			}
		}
		
		if !found {
			csm.modelDistribution[modelName].Nodes =
				append(csm.modelDistribution[modelName].Nodes, nodeID)
		}
		
		// If distributed flag is true on any node, make it true for all nodes
		// This ensures consistency across the cluster
		if distributed {
			if !csm.modelDistribution[modelName].Distributed {
				slog.Info("Updating existing model to distributed mode",
					"model", modelName,
					"nodes_count", len(csm.modelDistribution[modelName].Nodes))
				csm.modelDistribution[modelName].Distributed = true
			}
		}
		
		// Update shard count if provided
		if shardCount > 0 && csm.modelDistribution[modelName].Shards != shardCount {
			slog.Info("Updating shard count for model",
				"model", modelName,
				"old_count", csm.modelDistribution[modelName].Shards,
				"new_count", shardCount)
			csm.modelDistribution[modelName].Shards = shardCount
		}
	}
}

// UnregisterModelFromNode removes a model from a node
func (csm *ClusterStatusMonitor) UnregisterModelFromNode(modelName string, nodeID string) {
	csm.mu.Lock()
	defer csm.mu.Unlock()
	
	// Remove from node models mapping
	if models, exists := csm.nodeModels[nodeID]; exists {
		updatedModels := make([]string, 0, len(models))
		for _, m := range models {
			if m != modelName {
				updatedModels = append(updatedModels, m)
			}
		}
		csm.nodeModels[nodeID] = updatedModels
	}
	
	// Update distributed model info
	if info, exists := csm.modelDistribution[modelName]; exists {
		updatedNodes := make([]string, 0, len(info.Nodes))
		for _, n := range info.Nodes {
			if n != nodeID {
				updatedNodes = append(updatedNodes, n)
			}
		}
		
		info.Nodes = updatedNodes
		
		// If no nodes have this model loaded, consider it unloaded
		if len(info.Nodes) == 0 {
			info.Status = "unloaded"
		}
	}
}

// GetNodeModels returns the models loaded on a specific node
func (csm *ClusterStatusMonitor) GetNodeModels(nodeID string) []string {
	csm.mu.RLock()
	defer csm.mu.RUnlock()
	
	if models, exists := csm.nodeModels[nodeID]; exists {
		result := make([]string, len(models))
		copy(result, models)
		return result
	}
	
	return []string{}
}

// GetDistributedModels returns information about all models distributed across the cluster
func (csm *ClusterStatusMonitor) GetDistributedModels() []api.ClusterModelResponse {
	csm.mu.RLock()
	defer csm.mu.RUnlock()
	
	result := make([]api.ClusterModelResponse, 0, len(csm.modelDistribution))
	
	for _, info := range csm.modelDistribution {
		// Calculate file presence for each node
		nodesWithFiles := make([]string, 0)
		nodesWithGPULoaded := make([]string, 0)
		
		for nodeID := range info.NodesWithFiles {
			if info.NodesWithFiles[nodeID] {
				nodesWithFiles = append(nodesWithFiles, nodeID)
			}
		}
		
		for nodeID := range info.NodesWithGPULoaded {
			if info.NodesWithGPULoaded[nodeID] {
				nodesWithGPULoaded = append(nodesWithGPULoaded, nodeID)
			}
		}
		
		// Determine real status based on files and GPU loading
		status := info.Status
		if len(nodesWithGPULoaded) > 0 {
			status = "gpu_loaded"
		} else if len(nodesWithFiles) > 0 {
			status = "files_present"
		}
		
		// Create extended response with detailed file and GPU info
		result = append(result, api.ClusterModelResponse{
			Name:              info.Name,
			Size:              info.Size,
			Distributed:       info.Distributed,
			Nodes:             info.Nodes,
			NodesWithFiles:    nodesWithFiles,     // Add this field to API response
			NodesWithGPULoaded: nodesWithGPULoaded, // Add this field to API response
			Shards:            info.Shards,
			Status:            status,
			LoadedAt:          info.LoadedAt,
		})
	}
	
	return result
}

// IsClusterHealthy returns whether the cluster is currently healthy
func (csm *ClusterStatusMonitor) IsClusterHealthy() bool {
	csm.mu.RLock()
	defer csm.mu.RUnlock()
	return csm.clusterHealthy
}

// CheckClusterHealth checks the health of the cluster and updates the health status
func (csm *ClusterStatusMonitor) CheckClusterHealth(registry *cluster.NodeRegistry) {
	if time.Since(csm.lastHealthyCheck) < csm.healthCheckInterval {
		return // Don't check too frequently
	}
	
	csm.mu.Lock()
	defer csm.mu.Unlock()
	
	// In a stub implementation, just consider the cluster healthy
	// In a real implementation, this would check status of actual nodes
	
	hasCoordinator := true
	hasWorker := true
	
	// A healthy cluster needs at least one coordinator and one worker
	// (they could be the same node if it's a "mixed" role)
	csm.clusterHealthy = hasCoordinator && hasWorker
	csm.lastHealthyCheck = time.Now()
	
	if !csm.clusterHealthy {
		slog.Warn("cluster health check failed",
			"coordinators", hasCoordinator,
			"workers", hasWorker)
	} else {
		slog.Info("cluster health check passed")
	}
}

// UpdateModelFileStatus records whether a node has the model file
func (csm *ClusterStatusMonitor) UpdateModelFileStatus(modelName string, nodeID string, hasFile bool) {
	csm.mu.Lock()
	defer csm.mu.Unlock()
	
	if info, exists := csm.modelDistribution[modelName]; exists {
		info.NodesWithFiles[nodeID] = hasFile
		slog.Info("Updated model file status",
			"model", modelName,
			"node", nodeID,
			"has_file", hasFile)
	} else {
		slog.Warn("Attempted to update file status for unknown model",
			"model", modelName,
			"node", nodeID)
	}
}

// UpdateModelGPUStatus records whether a node has loaded the model into GPU
func (csm *ClusterStatusMonitor) UpdateModelGPUStatus(modelName string, nodeID string, gpuLoaded bool) {
	csm.mu.Lock()
	defer csm.mu.Unlock()
	
	if info, exists := csm.modelDistribution[modelName]; exists {
		info.NodesWithGPULoaded[nodeID] = gpuLoaded
		slog.Info("Updated model GPU status",
			"model", modelName,
			"node", nodeID,
			"gpu_loaded", gpuLoaded)
	} else {
		slog.Warn("Attempted to update GPU status for unknown model",
			"model", modelName,
			"node", nodeID)
	}
}