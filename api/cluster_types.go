package api

import (
	"time"
)

// ClusterNodeResponse represents information about a node in the cluster
type ClusterNodeResponse struct {
	ID          string   `json:"id"`
	Name        string   `json:"name"`
	Role        string   `json:"role"`
	Status      string   `json:"status"`
	Address     string   `json:"address"`
	JoinedAt    time.Time `json:"joined_at"`
	LastContact time.Time `json:"last_contact,omitempty"`
	Models      []string `json:"models,omitempty"`
	Resources   *NodeResources `json:"resources,omitempty"`
}

// NodeResources represents the resource information for a node
type NodeResources struct {
	CPUCores    int     `json:"cpu_cores"`
	CPUUsage    float64 `json:"cpu_usage"`
	TotalMemory int64   `json:"total_memory"`
	UsedMemory  int64   `json:"used_memory"`
	GPUCount    int     `json:"gpu_count"`
	GPUModels   []string `json:"gpu_models,omitempty"`
	GPUMemory   []int64  `json:"gpu_memory,omitempty"`
}

// ClusterStatusResponse represents the overall status of the cluster
type ClusterStatusResponse struct {
	Enabled     bool                 `json:"enabled"`
	Mode        string               `json:"mode"`
	NodeCount   int                  `json:"node_count"`
	CurrentNode ClusterNodeResponse  `json:"current_node"`
	Nodes       []ClusterNodeResponse `json:"nodes"`
	Models      []ClusterModelResponse `json:"models"`
	StartedAt   time.Time            `json:"started_at"`
	Healthy     bool                 `json:"healthy"`
}

// ClusterModelResponse represents information about a model distributed in the cluster
type ClusterModelResponse struct {
	Name        string    `json:"name"`
	Size        int64     `json:"size"`
	Distributed bool      `json:"distributed"`
	Nodes       []string  `json:"nodes"`
	Shards      int       `json:"shards"`
	Status      string    `json:"status"`
	LoadedAt    time.Time `json:"loaded_at"`
}

// ClusterJoinRequest represents a request to join an existing cluster
type ClusterJoinRequest struct {
	NodeHost     string `json:"node_host"`
	NodePort     int    `json:"node_port"`
	NodeRole     string `json:"node_role,omitempty"`
	JoinToken    string `json:"join_token,omitempty"`
	ForceReplace bool   `json:"force_replace,omitempty"`
}

// ClusterJoinResponse represents the response to a join request
type ClusterJoinResponse struct {
	Success     bool   `json:"success"`
	NodeID      string `json:"node_id,omitempty"`
	ClusterID   string `json:"cluster_id,omitempty"`
	NodesJoined int    `json:"nodes_joined,omitempty"`
	Error       string `json:"error,omitempty"`
}

// ClusterLeaveRequest represents a request to leave the cluster
type ClusterLeaveRequest struct {
	NodeID   string `json:"node_id,omitempty"`
	Graceful bool   `json:"graceful,omitempty"`
	Timeout  int    `json:"timeout_seconds,omitempty"`
}

// ClusterLeaveResponse represents the response to a leave request
type ClusterLeaveResponse struct {
	Success bool   `json:"success"`
	Error   string `json:"error,omitempty"`
}

// ClusterModelLoadRequest represents a request to load a model in cluster mode
type ClusterModelLoadRequest struct {
	Model       string   `json:"model"`
	Distributed bool     `json:"distributed,omitempty"`
	ShardCount  int      `json:"shard_count,omitempty"`
	Strategy    string   `json:"strategy,omitempty"`
	NodeIDs     []string `json:"node_ids,omitempty"`
}

// ClusterModelLoadResponse represents the response to a model load request
type ClusterModelLoadResponse struct {
	Success     bool     `json:"success"`
	Model       string   `json:"model"`
	Distributed bool     `json:"distributed"`
	Nodes       []string `json:"nodes,omitempty"`
	Error       string   `json:"error,omitempty"`
}

// ConvertNodeInfoToResponse converts a NodeInfo struct from the cluster package to an API response
func ConvertNodeInfoToResponse(nodeInfo interface{}) ClusterNodeResponse {
	// In real implementation, this would convert from cluster.NodeInfo to ClusterNodeResponse
	// For this example, we'll just create a dummy response
	return ClusterNodeResponse{
		ID:      "node-123",
		Name:    "worker-1",
		Role:    "worker",
		Status:  "online",
		Address: "192.168.1.101:11435",
		Resources: &NodeResources{
			CPUCores:    8,
			CPUUsage:    25.5,
			TotalMemory: 16 * 1024 * 1024 * 1024, // 16GB
			UsedMemory:  4 * 1024 * 1024 * 1024,  // 4GB
			GPUCount:    1,
			GPUModels:   []string{"NVIDIA A100"},
			GPUMemory:   []int64{40 * 1024 * 1024 * 1024}, // 40GB
		},
	}
}