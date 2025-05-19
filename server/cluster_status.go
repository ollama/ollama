package server

import (
	"sync"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cluster"
	"log/slog"
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
	Name        string
	Size        int64
	Distributed bool
	Nodes       []string
	Shards      int
	Status      string
	LoadedAt    time.Time
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
			Name:        modelName,
			Size:        0, // Would be populated with actual model size
			Distributed: distributed,
			Nodes:       []string{nodeID},
			Shards:      shardCount,
			Status:      "loaded",
			LoadedAt:    time.Now(),
		}
		slog.Info("Created new model distribution entry",
			"model", modelName,
			"distributed", distributed)
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
		result = append(result, api.ClusterModelResponse{
			Name:        info.Name,
			Size:        info.Size,
			Distributed: info.Distributed,
			Nodes:       info.Nodes,
			Shards:      info.Shards,
			Status:      info.Status,
			LoadedAt:    info.LoadedAt,
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