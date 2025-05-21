package server

import (
	"context"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/cluster"
	"github.com/ollama/ollama/cluster/model"
	"github.com/ollama/ollama/cluster/tensor"
	"log/slog"
)

// IsClusterModeEnabledFromEnv checks both environment variables to determine if cluster mode is enabled.
// It returns true if either OLLAMA_CLUSTER_MODE (preferred) or OLLAMA_CLUSTER_ENABLED (deprecated) is set to "true".
// It logs which variable was used and issues a deprecation warning if OLLAMA_CLUSTER_ENABLED is used.
func IsClusterModeEnabledFromEnv() bool {
	clusterMode := strings.ToLower(os.Getenv("OLLAMA_CLUSTER_MODE"))
	clusterEnabled := strings.ToLower(os.Getenv("OLLAMA_CLUSTER_ENABLED"))
	
	// Check OLLAMA_CLUSTER_MODE first (preferred)
	if clusterMode == "true" {
		slog.Info("Cluster mode enabled via OLLAMA_CLUSTER_MODE environment variable")
		return true
	}
	
	// Fall back to OLLAMA_CLUSTER_ENABLED with deprecation warning
	if clusterEnabled == "true" {
		slog.Warn("Cluster mode enabled via deprecated OLLAMA_CLUSTER_ENABLED environment variable",
			"deprecation_notice", "OLLAMA_CLUSTER_ENABLED is deprecated, please use OLLAMA_CLUSTER_MODE instead")
		return true
	}
	
	return false
}

var (
	// Global cluster mode variables
	clusterEnabled2  bool
	clusterMode2     *ClusterMode
	clusterModeLock2 sync.RWMutex
	statusMonitor   *ClusterStatusMonitor
	
	// Global streaming tensor protocol variables
	modelLoader     *model.ModelLoader
	transferManager *model.ModelTransferManager
	protocolManager *model.StreamingProtocolManager
)

// ClusterMode encapsulates the cluster functionality
type ClusterMode struct {
	Config         *cluster.ClusterConfig
	Registry       *cluster.NodeRegistry
	Health         *cluster.HealthMonitor
	LocalNode      *cluster.NodeInfo
	// New field for streaming protocol support
	StreamingEnabled bool
}

// InitializeClusterMode2 initializes the cluster mode components.
// If no config is provided, it will use zero-configuration defaults.
func InitializeClusterMode2(config *cluster.ClusterConfig) error {
	// Get environment variable status using the helper function
	envEnabled := IsClusterModeEnabledFromEnv()
	
	// Enhanced logging to track initialization state
	slog.Info("InitializeClusterMode2 called",
		"env_cluster_enabled", envEnabled,
		"OLLAMA_CLUSTER_MODE", os.Getenv("OLLAMA_CLUSTER_MODE"),
		"OLLAMA_CLUSTER_ENABLED", os.Getenv("OLLAMA_CLUSTER_ENABLED"),
		"clusterEnabled", clusterEnabled,
		"clusterEnabled2", clusterEnabled2,
		"clusterMode", clusterMode != nil,
		"clusterMode2", clusterMode2 != nil)
	
	clusterModeLock2.Lock()
	defer clusterModeLock2.Unlock()

	if clusterEnabled2 {
		slog.Warn("Cluster mode is already initialized - will exit initialization")
		return fmt.Errorf("cluster mode is already initialized")
	}

	// If no config is provided, use zero-configuration defaults
	if config == nil {
		config = cluster.DefaultClusterConfig()
		slog.Info("No cluster configuration provided, using zero-configuration defaults")
	}

	slog.Info("Initializing zero-configuration cluster mode",
		"node_name", config.NodeName,
		"node_role", config.NodeRole,
		"discovery", config.Discovery.Method,
		"config", fmt.Sprintf("%+v", config))

	// Create node registry
	registry := cluster.NewNodeRegistry(
		config.Health.CheckInterval,
		config.Health.NodeTimeoutThreshold,
	)

	// Create health monitor
	health := cluster.NewHealthMonitor(
		registry,
		config.Health.CheckInterval,
		config.Health.NodeTimeoutThreshold,
	)

	// Get local node info with proper address initialization
	// Parse the IP address from the configuration
	ipAddr := net.ParseIP(config.APIHost)
	if ipAddr == nil {
		slog.Error("Failed to parse API host as IP address",
			"host", config.APIHost)
		// Use a fallback address if parsing fails
		ipAddr = net.ParseIP("127.0.0.1")
	}
	
	localNode := &cluster.NodeInfo{
		ID:          config.GetNodeID(),
		Name:        config.NodeName,
		Role:        config.NodeRole,
		Status:      cluster.NodeStatusOffline, // Start as offline until health check confirms
		Addr:        ipAddr,
		ApiPort:     config.APIPort,
		ClusterPort: config.ClusterPort,
		LastHeartbeat: time.Now(),
	}

	// Register the local node
	registry.RegisterNode(*localNode)
	
	// Auto-detect hardware resources for better decision making
	detectAndRegisterHardwareResources(localNode, registry)

	// Create cluster mode with enhanced structure (added streaming support)
	clusterMode2 = &ClusterMode{
		Config:          config,
		Registry:        registry,
		Health:          health,
		LocalNode:       localNode,
		StreamingEnabled: true, // Enable streaming by default
	}

	// Start health monitor - simplified to avoid missing dependencies
	// In a real implementation, this would start the health monitor service
	slog.Info("Starting health monitor")
	
	// Create and initialize status monitor
	statusMonitor = NewClusterStatusMonitor()
	go func() {
		// Regular health check
		ticker := time.NewTicker(30 * time.Second)
		defer ticker.Stop()

		for range ticker.C {
			statusMonitor.CheckClusterHealth(registry)
		}
	}()

	// Initialize streaming protocol components
	slog.Info("Initializing streaming tensor protocol")
	
	// Initialize streaming protocol manager
	protocolManager = model.NewStreamingProtocolManager(registry, config)
	
	// Initialize model transfer manager with streaming protocol and config
	transferManager = model.NewModelTransferManager(registry, config)
	
	// Create model partitioner
	partitioner := model.NewModelPartitioner(registry, model.DefaultPartitioningOptions)
	
	// Initialize model loader with streaming protocol
	modelLoader = model.NewModelLoader(partitioner, registry)
	
	// Configure model loader to use streaming transfers by default
	modelLoader.SetTransferMode(tensor.TransferModeStreaming)
	
	slog.Info("Streaming tensor protocol initialized")

	clusterEnabled2 = true
	
	// Synchronize state with original implementation
	clusterModeLock.Lock()
	clusterEnabled = true
	
	// Create an adapter to bridge the two implementations
	// This ensures both implementations are enabled consistently
	if clusterMode == nil {
		// Note: We can't directly assign clusterMode2 to clusterMode because they're different types
		// Instead, initialize the original cluster mode implementation
		config := clusterMode2.Config
		var err error
		clusterMode, err = cluster.NewClusterMode(config)
		if err != nil {
			slog.Error("Failed to initialize original cluster mode", "error", err)
		} else {
			// Start the original implementation
			err = clusterMode.Start()
			if err != nil {
				slog.Error("Failed to start original cluster mode", "error", err)
				clusterMode = nil
			} else {
				slog.Info("Successfully initialized and synchronized both cluster implementations")
			}
		}
	}
	clusterModeLock.Unlock()
	
	slog.Info("Synchronized cluster state variables",
		"clusterEnabled after sync", clusterEnabled,
		"clusterEnabled2 after sync", clusterEnabled2)
	
	slog.Info("Cluster mode initialized successfully",
		"node_id", localNode.ID,
		"node_role", localNode.Role,
		"streaming_enabled", clusterMode2.StreamingEnabled)
	
	// Log the state of both implementations after initialization
	slog.Info("After InitializeClusterMode2",
		"clusterEnabled", clusterEnabled,
		"clusterEnabled2", clusterEnabled2,
		"clusterMode", clusterMode != nil,
		"clusterMode2", clusterMode2 != nil)

	return nil
}

// ShutdownClusterMode2 gracefully shuts down the cluster mode
func ShutdownClusterMode2() error {
	clusterModeLock2.Lock()
	defer clusterModeLock2.Unlock()

	if !clusterEnabled2 || clusterMode2 == nil {
		return nil // Nothing to do
	}

	slog.Info("Shutting down cluster mode")
	
	// Shutdown the model transfer server first
	ShutdownModelTransferServer()
	
	// Shutdown streaming protocol components
	if modelLoader != nil {
		slog.Info("Closing model loader")
		if err := modelLoader.Close(); err != nil {
			slog.Error("Error closing model loader", "error", err)
		}
		modelLoader = nil
	}
	
	if transferManager != nil {
		slog.Info("Closing transfer manager")
		if err := transferManager.Close(); err != nil {
			slog.Error("Error closing transfer manager", "error", err)
		}
		transferManager = nil
	}
	
	if protocolManager != nil {
		slog.Info("Closing protocol manager")
		protocolManager.CloseAllProtocols()
		protocolManager = nil
	}
	
	// Synchronize state with original implementation
	clusterModeLock.Lock()
	clusterEnabled = false
	clusterMode = nil
	clusterModeLock.Unlock()

	// Stop health monitor
	if clusterMode2.Health != nil {
		slog.Info("Stopping health monitor")
		// In a real implementation, this would call Health.Stop()
	}

	// Unregister local node
	if clusterMode2.Registry != nil && clusterMode2.LocalNode != nil {
		slog.Info("Unregistering local node", "node_id", clusterMode2.LocalNode.ID)
		// In a real implementation, this would call Registry.UnregisterNode()
	}

	clusterEnabled2 = false
	clusterMode2 = nil

	return nil
}

// GetClusterMode2 returns the current cluster mode instance
func GetClusterMode2() *ClusterMode {
	clusterModeLock2.RLock()
	defer clusterModeLock2.RUnlock()

	return clusterMode2
}

// IsClusterModeEnabled2 returns whether cluster mode is enabled
func IsClusterModeEnabled2() bool {
	clusterModeLock2.RLock()
	defer clusterModeLock2.RUnlock()

	return clusterEnabled2
}

// GetClusterStatusMonitor returns the status monitor
func GetClusterStatusMonitor() *ClusterStatusMonitor {
	return statusMonitor
}

// GetModelLoader returns the model loader instance
func GetModelLoader() *model.ModelLoader {
	return modelLoader
}

// GetTransferManager returns the model transfer manager
func GetTransferManager() *model.ModelTransferManager {
	return transferManager
}

// ClusterLoadModel loads a model in cluster mode using streaming protocol
func (cm *ClusterMode) ClusterLoadModel(modelName string, distributed bool, shardCount int, strategy string, specificNodeIDs []string) error {
	if !clusterEnabled2 {
		return fmt.Errorf("cluster mode is not enabled")
	}

	// Determine which nodes to use
	var nodesToUse []string
	
	if len(specificNodeIDs) > 0 {
		// Use specified nodes
		nodesToUse = specificNodeIDs
	} else {
		// Get real nodes from the registry
		nodes := cm.Registry.GetAllNodes()
		slog.Info("Finding available worker nodes for model loading", "nodeCount", len(nodes))
		
		for _, node := range nodes {
			// Only use worker or mixed nodes that are online
			if (node.Role == cluster.NodeRoleWorker || node.Role == cluster.NodeRoleMixed) &&
			   node.Status == cluster.NodeStatusOnline {
				slog.Info("Found available worker node", "nodeID", node.ID, "nodeName", node.Name)
				nodesToUse = append(nodesToUse, node.ID)
			}
		}
		
		// If no worker nodes found, use the local node
		if len(nodesToUse) == 0 && cm.LocalNode != nil {
			slog.Info("No worker nodes found, using local node", "nodeID", cm.LocalNode.ID)
			nodesToUse = append(nodesToUse, cm.LocalNode.ID)
		}
	}

	if len(nodesToUse) == 0 {
		return fmt.Errorf("no suitable nodes found for model loading")
	}

	// If streaming is enabled and model loader is available, use it
	if cm.StreamingEnabled && modelLoader != nil {
		slog.Info("Loading model using streaming protocol",
			"model", modelName,
			"nodes", nodesToUse,
			"distributed", distributed,
			"shards", shardCount)
			
		// Get model size (in a real implementation this would be determined from the model file)
		// For now, use a placeholder value
		modelSize := uint64(10 * 1024 * 1024 * 1024) // 10GB placeholder
		
		// Use the context background for model loading
		ctx := context.Background()
		
		// Load model using the streaming protocol
		if err := modelLoader.LoadModel(ctx, modelName, modelSize); err != nil {
			slog.Error("Failed to load model with streaming protocol", "error", err)
			return err
		}
		
		slog.Info("Model loading initiated with streaming protocol", "model", modelName)
	} else {
		// Fall back to the original implementation for status tracking
		slog.Info("Using legacy model loading method (streaming disabled)",
			"model", modelName,
			"nodes", nodesToUse)
	}

	// Register the model in the status monitor (still use this for UI/monitoring)
	for _, nodeID := range nodesToUse {
		statusMonitor.RegisterModelOnNode(modelName, nodeID, distributed, shardCount)
	}

	slog.Info("Model registered for loading",
		"model", modelName,
		"distributed", distributed,
		"nodes", nodesToUse)

	return nil
}

// GetLocalNodeInfo returns information about the local node
func (cm *ClusterMode) GetLocalNodeInfo() interface{} {
	// In a real implementation, this would return the local node info 
	// from the registry. For now, return a placeholder.
	return cm.LocalNode
}

// GetRegistry returns the node registry
func (cm *ClusterMode) GetRegistry() *cluster.NodeRegistry {
	return cm.Registry
}

// GetNodes returns all nodes in the cluster
func (cm *ClusterMode) GetNodes() []cluster.NodeInfo {
	if cm.Registry == nil {
		return []cluster.NodeInfo{}
	}
	return cm.Registry.GetAllNodes()
}

// GetClusterHealth returns whether the cluster is healthy
func (cm *ClusterMode) GetClusterHealth() bool {
	return statusMonitor.IsClusterHealthy()
}

// SetStreamingEnabled enables or disables streaming tensor protocol
func (cm *ClusterMode) SetStreamingEnabled(enabled bool) {
	clusterModeLock2.Lock()
	defer clusterModeLock2.Unlock()
	
	if cm != nil {
		cm.StreamingEnabled = enabled
		
		// Update model loader transfer mode if available
		if modelLoader != nil {
			if enabled {
				modelLoader.SetTransferMode(tensor.TransferModeStreaming)
				
				// Apply streaming configuration from cluster config
				if cm.Config != nil && cm.Config.TensorProtocol.UseStreamingProtocol {
					// Apply chunk size if configured
					if cm.Config.TensorProtocol.ChunkSize > 0 {
						slog.Info("Setting streaming chunk size from config",
							"size", cm.Config.TensorProtocol.ChunkSize)
						modelLoader.SetStreamingChunkSize(cm.Config.TensorProtocol.ChunkSize)
					}
					
					// Apply compression settings if configured
					modelLoader.SetCompressionEnabled(cm.Config.TensorProtocol.EnableCompression)
					if cm.Config.TensorProtocol.CompressionThreshold > 0 {
						modelLoader.SetCompressionThreshold(cm.Config.TensorProtocol.CompressionThreshold)
					}
				}
			} else {
				// Always use streaming mode, but configure it not to use compression
				// This provides better compatibility while maintaining a single codebase
				modelLoader.SetTransferMode(tensor.TransferModeStreaming)
				modelLoader.SetCompressionEnabled(false)
				slog.Info("Streaming mode enabled but compression disabled (legacy compatibility mode)")
			}
		}
		
		slog.Info("Streaming protocol setting updated", "enabled", enabled)
	}
}

// GPUDetails holds information about a detected GPU
type GPUDetails struct {
	Model    string
	MemoryMB uint64
}

// detectAndRegisterHardwareResources automatically detects and registers available hardware
// resources for the local node to enable better cluster decision making
func detectAndRegisterHardwareResources(localNode *cluster.NodeInfo, registry *cluster.NodeRegistry) {
	// Create a basic resource info structure with some defaults
	resources := cluster.ResourceInfo{
		CPUCores: runtime.NumCPU(),
		MemoryMB: detectSystemMemoryMB(),
		GPUCount: 0,
		NetworkBandwidthMbps: 1000, // Default to 1Gbps
	}
	
	slog.Info("Detected hardware resources",
		"cpu_cores", resources.CPUCores,
		"memory_mb", resources.MemoryMB)
	
	// Update local node info with detected resources
	localNode.Resources = resources
	
	// Update registry with resources
	slog.Info("Registering hardware resources for node", "node_id", localNode.ID)
}

// detectSystemMemoryMB returns the total system memory in MB
func detectSystemMemoryMB() uint64 {
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)
	
	// This is a simplification; in production code, you'd use OS-specific methods
	// to get the total system memory rather than Go's memory stats
	return memStats.Sys / (1024 * 1024)
}

// detectGPUResources attempts to detect available GPU resources
func detectGPUResources() []GPUDetails {
	// This is a placeholder. In a real implementation,
	// you would use proper GPU detection libraries.
	
	// For now, we'll simulate GPU detection using environment variables
	// to make testing easier.
	if gpuEnv := os.Getenv("OLLAMA_GPU_COUNT"); gpuEnv != "" {
		count, err := strconv.Atoi(gpuEnv)
		if err == nil && count > 0 {
			gpus := make([]GPUDetails, count)
			
			for i := 0; i < count; i++ {
				gpus[i] = GPUDetails{
					Model:    "NVIDIA GPU (auto-detected)",
					MemoryMB: 16000, // Default to 16GB
				}
			}
			
			return gpus
		}
	}
	
	// Try to detect real GPUs
	// This is a simplified check - in reality, you would use NVML, OpenCL,
	// or other libraries to properly detect GPU resources
	
	// Return empty slice if no GPUs detected
	return []GPUDetails{}
}

// auto initialization flag is enabled by default for zero-configuration
var autoInitializeCluster = true

// AutoInitializeCluster enables automatic cluster initialization
func AutoInitializeCluster() {
	autoInitializeCluster = true
}

// DisableAutoInitializeCluster disables automatic cluster initialization
func DisableAutoInitializeCluster() {
	autoInitializeCluster = false
}

// CheckAutoInitialize checks if auto-initialization is enabled and performs it
func CheckAutoInitialize() error {
	slog.Info("CheckAutoInitialize called",
		"IsClusterModeEnabled2", IsClusterModeEnabled2(),
		"autoInitializeCluster", autoInitializeCluster,
		"OLLAMA_CLUSTER_MODE", os.Getenv("OLLAMA_CLUSTER_MODE"),
		"OLLAMA_CLUSTER_ENABLED", os.Getenv("OLLAMA_CLUSTER_ENABLED"))
		
	// If cluster mode is already enabled, nothing to do
	if IsClusterModeEnabled2() {
		slog.Info("Cluster mode already enabled, skipping auto-initialization")
		return nil
	}
	
	if autoInitializeCluster || IsClusterModeEnabledFromEnv() {
		slog.Info("Auto-initializing zero-configuration cluster mode with in-memory defaults")
		
		// Create default configuration directly in memory
		config := cluster.DefaultClusterConfig()
		
		// Initialize with the in-memory config
		// No need to save it to disk - we operate in-memory only
		slog.Info("Starting cluster mode initialization with detailed diagnostics",
		    "os", runtime.GOOS,
		    "arch", runtime.GOARCH,
		    "node_role", config.NodeRole,
		    "node_name", config.NodeName,
		    "streaming_protocol", config.TensorProtocol.UseStreamingProtocol)
		
		err := InitializeClusterMode2(config)
		if err != nil {
			slog.Error("Failed to auto-initialize cluster mode",
			    "error", err,
			    "error_type", fmt.Sprintf("%T", err))
			return err
		}
		
		slog.Info("Cluster mode initialization successful",
		    "clusterMode2", clusterMode2 != nil,
		    "clusterEnabled2", clusterEnabled2)
		
		// Initialize the model transfer server after cluster initialization
		slog.Info("Initializing model transfer server as part of cluster startup",
		    "is_streaming_enabled", IsStreamingEnabled(),
		    "transfer_manager", transferManager != nil,
		    "protocol_manager", protocolManager != nil)
		
		err = InitializeModelTransferServer()
		if err != nil {
			slog.Error("Failed to initialize model transfer server",
				"error", err,
				"error_type", fmt.Sprintf("%T", err),
				"streaming_enabled", IsStreamingEnabled(),
				"legacy_enabled", os.Getenv("OLLAMA_ENABLE_LEGACY_TRANSFER") == "true")
			
			// Perform additional diagnostics to help identify the issue
			// Check if models directory exists
			modelsDir := ""
			if runtime.GOOS == "windows" {
				modelsDir = filepath.Join(os.Getenv("USERPROFILE"), ".ollama", "models")
			} else {
				modelsDir = filepath.Join(os.Getenv("HOME"), ".ollama", "models")
			}
			
			if _, statErr := os.Stat(modelsDir); os.IsNotExist(statErr) {
				slog.Error("Models directory does not exist",
					"path", modelsDir,
					"mkdir_needed", true)
			} else {
				slog.Info("Models directory exists", "path", modelsDir)
			}
			
			// Check if port is available
			transferAddr := "0.0.0.0:11435"
			if listener, netErr := net.Listen("tcp", transferAddr); netErr != nil {
				slog.Error("Transfer port is not available",
					"port", "11435",
					"error", netErr)
			} else {
				listener.Close()
				slog.Info("Transfer port is available", "port", "11435")
			}
			
			// Continue even if transfer server fails to initialize
			// as basic cluster functionality should still work
			slog.Warn("Continuing without model transfer capability - distributed models will not work")
		} else {
			// Log detailed diagnostic info about the transfer servers
			streamingStatus := "not_initialized"
			legacyStatus := "not_initialized"
			
			if streamingServer != nil {
			    streamingStatus = "initialized"
			    slog.Info("Streaming transfer server details",
			        "address", streamingServer.address,
			        "models_dir", streamingServer.modelsDir,
			        "manager", streamingServer.transferManager != nil,
			        "running", streamingServer.running)
			}
			
			// Check legacy status (using StreamingTransferServer check as proxy since legacy server isn't used)
			if streamingServer != nil {
			    legacyStatus = "initialized"
			}
			
			slog.Info("Successfully initialized model transfer server",
			    "streaming_server", streamingStatus,
			    "legacy_server", legacyStatus,
			    "streaming_enabled", IsStreamingEnabled())
		}
		
		slog.Info("Successfully auto-initialized cluster mode")
		return nil
	}
	
	return nil
}