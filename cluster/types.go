package cluster

import (
	"fmt"
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
	// Valid transitions: from Starting, Busy, Maintenance, Offline
	NodeStatusOnline NodeStatus = "online"
	
	// NodeStatusOffline indicates the node is unreachable
	// Valid transitions: from Online
	NodeStatusOffline NodeStatus = "offline"
	
	// NodeStatusBusy indicates the node is operational but fully loaded
	// Valid transitions: from Online
	NodeStatusBusy NodeStatus = "busy"
	
	// NodeStatusStarting indicates the node is starting up
	// Valid transitions: initial state
	NodeStatusStarting NodeStatus = "starting"
	
	// NodeStatusStopping indicates the node is shutting down
	// Valid transitions: from Online
	NodeStatusStopping NodeStatus = "stopping"
	
	// NodeStatusMaintenance indicates the node is in maintenance mode
	// Valid transitions: from Online
	NodeStatusMaintenance NodeStatus = "maintenance"
	
	// NodeStatusFailed indicates the node has encountered a critical error
	// Valid transitions: from any state
	NodeStatusFailed NodeStatus = "failed"
)

// NodeStatusTransition represents a valid transition between node statuses
type NodeStatusTransition struct {
	From NodeStatus
	To   NodeStatus
}

// NodeStatusController manages and validates node status transitions
type NodeStatusController struct {
	// validTransitions contains all allowed state transitions
	validTransitions []NodeStatusTransition
}

// NewNodeStatusController creates a new controller with predefined valid transitions
func NewNodeStatusController() *NodeStatusController {
	return &NodeStatusController{
		validTransitions: []NodeStatusTransition{
			{From: NodeStatusStarting, To: NodeStatusOnline},
			{From: NodeStatusStarting, To: NodeStatusFailed},
			
			{From: NodeStatusOnline, To: NodeStatusBusy},
			{From: NodeStatusOnline, To: NodeStatusMaintenance},
			{From: NodeStatusOnline, To: NodeStatusOffline},
			{From: NodeStatusOnline, To: NodeStatusStopping},
			{From: NodeStatusOnline, To: NodeStatusFailed},
			
			{From: NodeStatusBusy, To: NodeStatusOnline},
			{From: NodeStatusBusy, To: NodeStatusFailed},
			
			{From: NodeStatusMaintenance, To: NodeStatusOnline},
			{From: NodeStatusMaintenance, To: NodeStatusFailed},
			
			{From: NodeStatusOffline, To: NodeStatusOnline},
			{From: NodeStatusOffline, To: NodeStatusFailed},
			
			{From: NodeStatusStopping, To: NodeStatusFailed},
		},
	}
}

// IsValidTransition checks if a status transition is valid
func (c *NodeStatusController) IsValidTransition(from, to NodeStatus) bool {
	// Failed state can be reached from any state
	if to == NodeStatusFailed {
		return true
	}
	
	// Check if the transition exists in the valid transitions list
	for _, transition := range c.validTransitions {
		if transition.From == from && transition.To == to {
			return true
		}
	}
	
	return false
}

// TransitionStatus attempts to transition the status and returns error if invalid
func (c *NodeStatusController) TransitionStatus(from, to NodeStatus) (NodeStatus, error) {
	if !c.IsValidTransition(from, to) {
		return from, fmt.Errorf("invalid status transition from '%s' to '%s'", from, to)
	}
	
	return to, nil
}

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
	
	// StatusController manages node status transitions
	StatusController *NodeStatusController
	
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
	
	// Create status controller
	statusController := NewNodeStatusController()
	
	return &ClusterMode{
		Config:          config,
		Registry:        registry,
		Discovery:       discovery,
		Health:          health,
		StatusController: statusController,
		localNodeInfo:   localNodeInfo,
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
	
	// Update node status to online using the status controller
	if err := c.SetNodeToOnline(); err != nil {
		// This should not happen as StartingToOnline is a valid transition
		// But handle it anyway
		c.Health.Stop()
		c.Discovery.Stop()
		return fmt.Errorf("failed to transition to online status: %w", err)
	}
	
	return nil
}

// Stop halts all cluster mode components
func (c *ClusterMode) Stop() error {
	// Update node status to stopping using the status controller
	if err := c.UpdateNodeStatus(NodeStatusStopping); err != nil {
		// Log the error but continue with shutdown
		fmt.Printf("Warning: Failed to transition to stopping status: %v\n", err)
	}
	
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

// GetStatusController returns the node status controller
func (c *ClusterMode) GetStatusController() *NodeStatusController {
	return c.StatusController
}

// UpdateNodeStatus safely updates the status of the local node using the status controller
func (c *ClusterMode) UpdateNodeStatus(newStatus NodeStatus) error {
	currentStatus := c.localNodeInfo.Status
	
	// Validate the transition
	status, err := c.StatusController.TransitionStatus(currentStatus, newStatus)
	if err != nil {
		return fmt.Errorf("invalid node status transition: %w", err)
	}
	
	// Update the status and notify the cluster
	c.localNodeInfo.Status = status
	
	// If discovery service is initialized, notify it of the change
	if c.Discovery != nil {
		c.Discovery.UpdateLocalNodeInfo(c.localNodeInfo)
	}
	
	// Log the transition
	fmt.Printf("Node status changed from %s to %s\n", currentStatus, status)
	
	return nil
}

// SetNodeToOnline transitions the node to Online state
func (c *ClusterMode) SetNodeToOnline() error {
	return c.UpdateNodeStatus(NodeStatusOnline)
}

// SetNodeToBusy transitions the node to Busy state
func (c *ClusterMode) SetNodeToBusy() error {
	return c.UpdateNodeStatus(NodeStatusBusy)
}

// SetNodeToMaintenance transitions the node to Maintenance state
func (c *ClusterMode) SetNodeToMaintenance() error {
	return c.UpdateNodeStatus(NodeStatusMaintenance)
}

// SetNodeToOffline transitions the node to Offline state
// This is typically called when detecting connection issues
func (c *ClusterMode) SetNodeToOffline() error {
	return c.UpdateNodeStatus(NodeStatusOffline)
}

// SetNodeToFailed transitions the node to Failed state
// This represents a critical error condition
func (c *ClusterMode) SetNodeToFailed(reason string) error {
	err := c.UpdateNodeStatus(NodeStatusFailed)
	if err != nil {
		return err
	}
	
	// Log the failure reason
	fmt.Printf("Node marked as failed: %s\n", reason)
	
	return nil
}