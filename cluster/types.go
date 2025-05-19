package cluster

import (
	"net"
	"time"
)

// NodeRole defines the role of a node in the cluster
type NodeRole string

const (
	// NodeRoleCoordinator indicates the node manages distribution but doesn't run models
	NodeRoleCoordinator NodeRole = "coordinator"
	
	// NodeRoleWorker indicates the node runs models but doesn't coordinate
	NodeRoleWorker NodeRole = "worker"
	
	// NodeRoleMixed indicates the node can both coordinate and run models
	NodeRoleMixed NodeRole = "mixed"
)

// NodeStatus represents the operational status of a node
type NodeStatus string

const (
	// NodeStatusOnline indicates the node is operational
	NodeStatusOnline NodeStatus = "online"
	
	// NodeStatusOffline indicates the node is unreachable
	NodeStatusOffline NodeStatus = "offline"
	
	// NodeStatusBusy indicates the node is operational but fully loaded
	NodeStatusBusy NodeStatus = "busy"
	
	// NodeStatusStarting indicates the node is starting up
	NodeStatusStarting NodeStatus = "starting"
	
	// NodeStatusStopping indicates the node is shutting down
	NodeStatusStopping NodeStatus = "stopping"
	
	// NodeStatusMaintenance indicates the node is in maintenance mode
	NodeStatusMaintenance NodeStatus = "maintenance"
)

// NodeInfo contains information about a node in the cluster
type NodeInfo struct {
	// ID is the unique identifier for the node
	ID string `json:"id"`
	
	// Name is a human-readable name for the node
	Name string `json:"name"`
	
	// Role defines the node's capabilities
	Role NodeRole `json:"role"`
	
	// Status represents the node's current operational status
	Status NodeStatus `json:"status"`
	
	// Addr is the node's IP address
	Addr net.IP `json:"addr"`
	
	// ApiPort is the port for the API server
	ApiPort int `json:"api_port"`
	
	// ClusterPort is the port for cluster communication
	ClusterPort int `json:"cluster_port"`
	
	// LastHeartbeat is when the node was last seen
	LastHeartbeat time.Time `json:"last_heartbeat"`
	
	// Resources describes the node's available resources
	Resources ResourceInfo `json:"resources"`
	
	// Version is the node's software version
	Version string `json:"version"`
}

// ResourceInfo describes the resources available on a node
type ResourceInfo struct {
	// CPUCores is the number of CPU cores
	CPUCores int `json:"cpu_cores"`
	
	// CPUFrequencyMHz is the CPU frequency in MHz
	CPUFrequencyMHz int `json:"cpu_frequency_mhz"`
	
	// MemoryMB is the total RAM in megabytes
	MemoryMB uint64 `json:"memory_mb"`
	
	// DiskSpaceMB is the available disk space in megabytes
	DiskSpaceMB uint64 `json:"disk_space_mb"`
	
	// GPUModels lists the available GPU models
	GPUModels []string `json:"gpu_models,omitempty"`
	
	// GPUMemoryMB lists the GPU memory in megabytes per GPU
	GPUMemoryMB []uint64 `json:"gpu_memory_mb,omitempty"`
	
	// GPUCount is the number of GPUs
	GPUCount int `json:"gpu_count"`
	
	// NVLinkAvailable indicates if NVLink is available
	NVLinkAvailable bool `json:"nvlink_available,omitempty"`
	
	// NetworkBandwidthMbps is the network bandwidth in Mbps
	NetworkBandwidthMbps int `json:"network_bandwidth_mbps"`
}

// ClusterEvent represents an event in the cluster
type ClusterEvent struct {
	// Type is the event type (e.g., "node_joined", "node_left")
	Type string `json:"type"`
	
	// NodeID is the ID of the node related to the event
	NodeID string `json:"node_id"`
	
	// Timestamp is when the event occurred
	Timestamp time.Time `json:"timestamp"`
	
	// Data contains additional event-specific information
	Data map[string]interface{} `json:"data,omitempty"`
}

// ClusterMode represents the operational mode of the cluster
type ClusterMode struct {
	// Config holds the cluster configuration
	Config *ClusterConfig
	
	// Registry is the node registry
	Registry *NodeRegistry
	
	// Discovery is the discovery service
	Discovery *DiscoveryService
	
	// Health is the health monitor
	Health *HealthMonitor
	
	// localNodeInfo contains information about this node
	localNodeInfo NodeInfo
}

// NewClusterMode creates a new instance of the cluster mode
func NewClusterMode(config *ClusterConfig) (*ClusterMode, error) {
	// Generate a node ID
	nodeID := config.GetNodeID()
	
	// Create local node info
	localNodeInfo := NodeInfo{
		ID:           nodeID,
		Name:         config.NodeName,
		Role:         config.NodeRole,
		Status:       NodeStatusStarting,
		Addr:         net.ParseIP(config.ClusterHost),
		ApiPort:      config.APIPort,
		ClusterPort:  config.ClusterPort,
		LastHeartbeat: time.Now(),
		Version:      "1.0.0", // Replace with actual version
	}
	
	// Populate resource info (in a real implementation, this would detect actual resources)
	localNodeInfo.Resources = ResourceInfo{
		CPUCores:           4,
		CPUFrequencyMHz:    3000,
		MemoryMB:           16384,
		DiskSpaceMB:        100000,
		GPUCount:           1,
		NetworkBandwidthMbps: 1000,
	}
	
	// Create node registry
	registry := NewNodeRegistry(
		config.Health.CheckInterval,
		config.Health.NodeTimeoutThreshold,
	)
	
	// Create discovery service
	discovery := NewDiscoveryService(
		config.Discovery,
		registry,
		localNodeInfo,
	)
	
	// Create health monitor
	health := NewHealthMonitor(
		registry,
		config.Health.CheckInterval,
		config.Health.NodeTimeoutThreshold,
	)
	
	return &ClusterMode{
		Config:       config,
		Registry:     registry,
		Discovery:    discovery,
		Health:       health,
		localNodeInfo: localNodeInfo,
	}, nil
}

// Start initializes and begins the cluster mode components
func (c *ClusterMode) Start() error {
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
	
	// Update node status to online
	c.localNodeInfo.Status = NodeStatusOnline
	c.Discovery.UpdateLocalNodeInfo(c.localNodeInfo)
	
	return nil
}

// Stop halts all cluster mode components
func (c *ClusterMode) Stop() error {
	// Update node status to stopping
	c.localNodeInfo.Status = NodeStatusStopping
	c.Discovery.UpdateLocalNodeInfo(c.localNodeInfo)
	
	// Stop discovery service
	if err := c.Discovery.Stop(); err != nil {
		return err
	}
	
	// Stop health monitoring
	if err := c.Health.Stop(); err != nil {
		return err
	}
	
	return nil
}

// GetLocalNodeInfo returns information about the local node
func (c *ClusterMode) GetLocalNodeInfo() NodeInfo {
	return c.localNodeInfo
}

// GetRegistry returns the node registry
func (c *ClusterMode) GetRegistry() *NodeRegistry {
	return c.Registry
}

// GetDiscovery returns the discovery service
func (c *ClusterMode) GetDiscovery() *DiscoveryService {
	return c.Discovery
}

// GetHealth returns the health monitoring system
func (c *ClusterMode) GetHealth() *HealthMonitor {
	return c.Health
}