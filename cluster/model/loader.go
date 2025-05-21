package model

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/ollama/ollama/cluster"
	"github.com/ollama/ollama/cluster/tensor"
)

// LoadStatus represents the current loading state of a model partition
type LoadStatus string

const (
	// LoadStatusPending indicates loading has not yet started
	LoadStatusPending LoadStatus = "pending"
	
	// LoadStatusLoading indicates loading is in progress
	LoadStatusLoading LoadStatus = "loading"
	
	// LoadStatusLoaded indicates loading is complete
	LoadStatusLoaded LoadStatus = "loaded"
	
	// LoadStatusFailed indicates loading failed
	LoadStatusFailed LoadStatus = "failed"
)

// LoadProgress tracks the progress of a model partition load
type LoadProgress struct {
	// Status is the current loading state
	Status LoadStatus
	
	// PercentComplete is the loading progress (0-100)
	PercentComplete int
	
	// BytesLoaded is the number of bytes loaded
	BytesLoaded uint64
	
	// TotalBytes is the total size of the partition
	TotalBytes uint64
	
	// Error contains any error that occurred
	Error error
	
	// StartTime is when loading began
	StartTime time.Time
	
	// CompleteTime is when loading finished (if complete)
	CompleteTime time.Time
	
	// TransferID identifies the current streaming transfer
	TransferID string
}

// ModelLoader manages loading model partitions across nodes
type ModelLoader struct {
	// partitioner provides access to model partitioning information
	partitioner *ModelPartitioner
	
	// registry provides access to available nodes
	registry *cluster.NodeRegistry
	
	// loadStatus tracks the loading status of each partition
	// modelID → partitionID → LoadProgress
	loadStatus map[string]map[string]LoadProgress
	
	// statusMu protects the loadStatus map
	statusMu sync.RWMutex
	
	// loadCallbacks are functions called when loading state changes
	loadCallbacks []func(modelID string, partitionID string, progress LoadProgress)
	
	// callbackMu protects the loadCallbacks slice
	callbackMu sync.RWMutex
	
	// protocolCache caches streaming protocol connections to nodes
	// nodeID → streaming protocol
	protocolCache map[string]*tensor.StreamingProtocol
	
	// protocolMu protects the protocol cache
	protocolMu sync.RWMutex
	
	// transferMode controls how tensors are transferred
	transferMode tensor.TransferMode
}

// LoadError represents errors that can occur during model loading
type LoadError struct {
	Err error
	NodeID string
	PartitionID string
}

func (e LoadError) Error() string {
	return fmt.Sprintf("load error on node %s (partition %s): %v", 
		e.NodeID, e.PartitionID, e.Err)
}

// NewModelLoader creates a new model loader
func NewModelLoader(partitioner *ModelPartitioner, registry *cluster.NodeRegistry) *ModelLoader {
	ml := &ModelLoader{
		partitioner:   partitioner,
		registry:      registry,
		loadStatus:    make(map[string]map[string]LoadProgress),
		protocolCache: make(map[string]*tensor.StreamingProtocol),
		transferMode:  tensor.TransferModeAdaptive, // Default to adaptive mode
	}
	
	// Log model loader initialization
	fmt.Printf("Enhanced ModelLoader initialized with streaming protocol support\n")
	
	return ml
}

// LoadModel initiates loading of a model across the cluster
func (ml *ModelLoader) LoadModel(ctx context.Context, modelID string, modelSize uint64) error {
	// Get the model partitioning
	partitions, exists := ml.partitioner.GetModelPartitions(modelID)
	if !exists {
		// Need to partition the model first
		var err error
		partitions, err = ml.partitioner.PartitionModel(modelID, modelSize)
		if err != nil {
			return fmt.Errorf("failed to partition model: %w", err)
		}
	}
	
	fmt.Printf("Starting distributed loading of model %s across %d partitions using streaming protocol\n",
		modelID, len(partitions))
	
	// Initialize the load status tracking
	ml.statusMu.Lock()
	ml.loadStatus[modelID] = make(map[string]LoadProgress)
	for _, partition := range partitions {
		ml.loadStatus[modelID][partition.PartitionID] = LoadProgress{
			Status:      LoadStatusPending,
			BytesLoaded: 0,
			TotalBytes:  partition.Size,
		}
	}
	ml.statusMu.Unlock()
	
	// Start loading each partition in parallel
	errCh := make(chan error, len(partitions))
	var wg sync.WaitGroup
	
	for _, partition := range partitions {
		wg.Add(1)
		go func(p ModelPartition) {
			defer wg.Done()
			err := ml.loadPartition(ctx, modelID, p)
			if err != nil {
				errCh <- LoadError{
					Err:         err,
					NodeID:      p.NodeID,
					PartitionID: p.PartitionID,
				}
			}
		}(partition)
	}
	
	// Wait for all loading goroutines to complete
	wg.Wait()
	close(errCh)
	
	// Check for errors
	var loadErrors []error
	for err := range errCh {
		loadErrors = append(loadErrors, err)
	}
	
	if len(loadErrors) > 0 {
		// Aggregate errors into a meaningful message
		errMsg := fmt.Sprintf("%d partitions failed to load:\n", len(loadErrors))
		for _, err := range loadErrors {
			errMsg += fmt.Sprintf("  - %v\n", err)
		}
		return errors.New(errMsg)
	}
	
	fmt.Printf("Model %s successfully loaded across all partitions\n", modelID)
	return nil
}

// getOrCreateStreamingProtocol gets a cached protocol or creates a new one
func (ml *ModelLoader) getOrCreateStreamingProtocol(nodeID string) (*tensor.StreamingProtocol, error) {
	ml.protocolMu.RLock()
	protocol, exists := ml.protocolCache[nodeID]
	ml.protocolMu.RUnlock()
	
	if exists {
		return protocol, nil
	}
	
	// Need to create a new connection
	nodeInfo, exists := ml.registry.GetNode(nodeID)
	if !exists {
		return nil, fmt.Errorf("node %s not found in registry", nodeID)
	}
	
	// Convert NodeInfo to Node
	node := &cluster.Node{
		ID: nodeInfo.ID,
		Address: nodeInfo.Addr.String(),
	}
	
	// In a real implementation, this would establish a new connection to the node
	// For this example, we'll simulate creating a connection
	conn, err := ml.connectToNode(node)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to node %s: %w", nodeID, err)
	}
	
	// Create a new streaming protocol handler
	streamingProto := tensor.NewStreamingProtocol(conn)
	
	// Configure protocol options
	streamingProto.SetChunkSize(4 * 1024 * 1024) // 4MB chunks
	streamingProto.SetDefaultPriority(tensor.PriorityHigh) // Prioritize model loading
	streamingProto.StartCleanupRoutine() // Start background cleanup
	
	// Cache the protocol
	ml.protocolMu.Lock()
	ml.protocolCache[nodeID] = streamingProto
	ml.protocolMu.Unlock()
	
	fmt.Printf("Created new streaming protocol connection to node %s\n", nodeID)
	return streamingProto, nil
}

// connectToNode establishes a connection to a node
// In a real implementation, this would use the node's address to create a TCP connection
func (ml *ModelLoader) connectToNode(node *cluster.Node) (cluster.Connection, error) {
	// Simulated connection for this example
	// In a real implementation, this would create a TCP connection
	fmt.Printf("Connecting to node %s at %s...\n", node.ID, node.Address)
	
	// Here we would establish a real connection
	// For now, we'll simulate the connection
	return &simulatedConnection{
		nodeID: node.ID,
		status: "connected",
	}, nil
}

// simulatedConnection is a mock implementation of net.Conn for testing
type simulatedConnection struct {
	nodeID string
	status string
	mutex  sync.Mutex
	buffer []byte
}

// Implement required methods for the cluster.Connection interface
func (c *simulatedConnection) Read(b []byte) (n int, err error) {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	
	if len(c.buffer) == 0 {
		return 0, nil
	}
	
	n = copy(b, c.buffer)
	c.buffer = c.buffer[n:]
	return n, nil
}

func (c *simulatedConnection) Write(b []byte) (n int, err error) {
	return len(b), nil // Simulate successful write
}

func (c *simulatedConnection) Close() error {
	c.status = "closed"
	return nil
}

func (c *simulatedConnection) SetReadDeadline(t time.Time) error {
	// Simulation only - real implementation would set socket deadline
	return nil
}

func (c *simulatedConnection) SetWriteDeadline(t time.Time) error {
	// Simulation only - real implementation would set socket deadline
	return nil
}

// loadPartition loads a single model partition on a specific node using the streaming protocol
func (ml *ModelLoader) loadPartition(ctx context.Context, modelID string, partition ModelPartition) error {
	// Update status to loading
	ml.updateLoadStatus(modelID, partition.PartitionID, LoadProgress{
		Status:      LoadStatusLoading,
		BytesLoaded: 0,
		TotalBytes:  partition.Size,
		StartTime:   time.Now(),
	})
	
	fmt.Printf("Loading partition %s on node %s using streaming protocol\n", 
		partition.PartitionID, partition.NodeID)
	
	// Check if node exists in registry
	node, exists := ml.registry.GetNode(partition.NodeID)
	if !exists {
		return fmt.Errorf("node %s not found in registry", partition.NodeID)
	}
	
	// Get source node (usually the coordinator/local node)
	localNodeID := ml.registry.GetLocalNodeID()
	
	// Check if this is a remote node that needs the model tensors transferred
	isRemoteNode := partition.NodeID != localNodeID
	
	// Log the node relationships
	fmt.Printf("Model %s (%s): Loading on node %s (remote: %v, local node: %s)\n",
		modelID, partition.PartitionID, partition.NodeID, isRemoteNode, localNodeID)
	
	if isRemoteNode {
		// For remote nodes, we need to transfer the model tensors using streaming protocol
		streamingProto, err := ml.getOrCreateStreamingProtocol(partition.NodeID)
		if err != nil {
			ml.updateLoadStatus(modelID, partition.PartitionID, LoadProgress{
				Status:      LoadStatusFailed,
				Error:       err,
				BytesLoaded: 0,
				TotalBytes:  partition.Size,
			})
			return err
		}
		
		// Register transfer progress callback to update load status
		transferID := modelID + "-" + partition.PartitionID
		
		// Use the streaming protocol to transfer the model tensors
		if err := ml.streamModelTensors(ctx, streamingProto, modelID, partition, transferID); err != nil {
			ml.updateLoadStatus(modelID, partition.PartitionID, LoadProgress{
				Status:      LoadStatusFailed,
				Error:       err,
				BytesLoaded: 0,
				TotalBytes:  partition.Size,
				TransferID:  transferID,
			})
			return err
		}
	} else {
		// For local node, just load the model from disk into memory
		fmt.Printf("Loading model %s from local storage on node %s\n", modelID, node.ID)
		// Simulate local loading with a brief delay
		time.Sleep(500 * time.Millisecond)
	}
	
	// Mark as successfully loaded
	ml.updateLoadStatus(modelID, partition.PartitionID, LoadProgress{
		Status:          LoadStatusLoaded,
		PercentComplete: 100,
		BytesLoaded:     partition.Size,
		TotalBytes:      partition.Size,
		StartTime:       time.Now(),
		CompleteTime:    time.Now(),
	})
	
	fmt.Printf("Partition %s successfully loaded on node %s\n",
		partition.PartitionID, partition.NodeID)
	
	return nil
}

// streamModelTensors streams model tensors to a remote node using the streaming protocol
func (ml *ModelLoader) streamModelTensors(ctx context.Context, proto *tensor.StreamingProtocol, modelID string, partition ModelPartition, transferID string) error {
	// In a real implementation, this would:
	// 1. Load the model tensors from disk or memory
	// 2. Determine which tensors belong to this partition
	// 3. Stream each tensor to the remote node
	
	// Get list of tensors for this partition
	tensorIDs, err := ml.getTensorIDsForPartition(modelID, partition.PartitionID)
	if err != nil {
		return fmt.Errorf("failed to get tensor IDs: %w", err)
	}
	
	fmt.Printf("Streaming %d tensors for model %s partition %s to node %s\n", 
		len(tensorIDs), modelID, partition.PartitionID, partition.NodeID)
	
	// Track total bytes and progress
	totalTensors := len(tensorIDs)
	completedTensors := 0
	totalBytes := partition.Size
	transferredBytes := uint64(0)
	
	// Stream each tensor
	for _, tensorID := range tensorIDs {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			// Continue streaming
		}
		
		// Simulate loading tensor data from local storage
		tensorData, tensorSize, err := ml.loadTensorData(modelID, tensorID)
		if err != nil {
			return fmt.Errorf("failed to load tensor data for %s: %w", tensorID, err)
		}
		
		// Stream this tensor to the remote node
		err = proto.StreamTensor(modelID, partition.PartitionID, tensorID, tensorData)
		if err != nil {
			return fmt.Errorf("failed to stream tensor %s: %w", tensorID, err)
		}
		
		// Update progress
		completedTensors++
		transferredBytes += tensorSize
		percentComplete := int((transferredBytes * 100) / totalBytes)
		
		ml.updateLoadStatus(modelID, partition.PartitionID, LoadProgress{
			Status:          LoadStatusLoading,
			PercentComplete: percentComplete,
			BytesLoaded:     transferredBytes,
			TotalBytes:      totalBytes,
			TransferID:      transferID,
		})
		
		fmt.Printf("Tensor %s transferred to node %s (%d/%d, %d%%)\n", 
			tensorID, partition.NodeID, completedTensors, totalTensors, percentComplete)
	}
	
	return nil
}

// getTensorIDsForPartition gets the list of tensor IDs that belong to a partition
// In a real implementation, this would query the model's tensor allocation information
func (ml *ModelLoader) getTensorIDsForPartition(modelID, partitionID string) ([]string, error) {
	// Simulate getting tensor IDs
	// In a real implementation, this would be determined by the model's partitioning scheme
	
	// For this example, we'll simulate 10 tensors per partition
	tensorIDs := make([]string, 10)
	for i := 0; i < 10; i++ {
		tensorIDs[i] = fmt.Sprintf("%s.%s.tensor_%d", modelID, partitionID, i)
	}
	
	return tensorIDs, nil
}

// loadTensorData loads tensor data from local storage
// In a real implementation, this would load the actual tensor data from disk or memory
func (ml *ModelLoader) loadTensorData(modelID, tensorID string) ([]byte, uint64, error) {
	// Simulate loading tensor data
	// In a real implementation, this would load the actual tensor data
	
	// For this example, we'll simulate tensor data with random size between 10KB and 10MB
	size := uint64(10*1024 + (time.Now().UnixNano() % (10*1024*1024)))
	data := make([]byte, size)
	
	// Simulate some delay for loading from disk
	time.Sleep(10 * time.Millisecond)
	
	return data, size, nil
}

// updateLoadStatus updates the loading status and notifies callbacks
func (ml *ModelLoader) updateLoadStatus(modelID, partitionID string, progress LoadProgress) {
	ml.statusMu.Lock()
	if _, exists := ml.loadStatus[modelID]; !exists {
		ml.loadStatus[modelID] = make(map[string]LoadProgress)
	}
	ml.loadStatus[modelID][partitionID] = progress
	ml.statusMu.Unlock()
	
	// Notify callbacks
	ml.callbackMu.RLock()
	defer ml.callbackMu.RUnlock()
	
	for _, callback := range ml.loadCallbacks {
		go callback(modelID, partitionID, progress)
	}
}

// RegisterLoadCallback adds a callback function for load progress updates
func (ml *ModelLoader) RegisterLoadCallback(callback func(modelID string, partitionID string, progress LoadProgress)) {
	ml.callbackMu.Lock()
	defer ml.callbackMu.Unlock()
	
	ml.loadCallbacks = append(ml.loadCallbacks, callback)
}

// GetLoadStatus retrieves the current loading status for a model partition
func (ml *ModelLoader) GetLoadStatus(modelID, partitionID string) (LoadProgress, error) {
	ml.statusMu.RLock()
	defer ml.statusMu.RUnlock()
	
	if modelStatus, exists := ml.loadStatus[modelID]; exists {
		if progress, exists := modelStatus[partitionID]; exists {
			return progress, nil
		}
		return LoadProgress{}, fmt.Errorf("partition %s not found for model %s", partitionID, modelID)
	}
	
	return LoadProgress{}, fmt.Errorf("model %s not found in load status", modelID)
}

// GetModelLoadStatus returns the overall loading status for a model
func (ml *ModelLoader) GetModelLoadStatus(modelID string) (float64, error) {
	ml.statusMu.RLock()
	defer ml.statusMu.RUnlock()
	
	if modelStatus, exists := ml.loadStatus[modelID]; exists {
		if len(modelStatus) == 0 {
			return 0, fmt.Errorf("no partitions found for model %s", modelID)
		}
		
		var totalBytes, loadedBytes uint64
		failedPartitions := 0
		
		for _, status := range modelStatus {
			totalBytes += status.TotalBytes
			loadedBytes += status.BytesLoaded
			
			if status.Status == LoadStatusFailed {
				failedPartitions++
			}
		}
		
		if failedPartitions > 0 {
			return -1, fmt.Errorf("%d partitions failed to load", failedPartitions)
		}
		
		if totalBytes == 0 {
			return 100, nil
		}
		
		return float64(loadedBytes) / float64(totalBytes) * 100, nil
	}
	
	return 0, fmt.Errorf("model %s not found in load status", modelID)
}

// UnloadModel unloads a model from all nodes
func (ml *ModelLoader) UnloadModel(ctx context.Context, modelID string) error {
	// Get the model partitioning
	partitions, exists := ml.partitioner.GetModelPartitions(modelID)
	if !exists {
		return fmt.Errorf("no partitioning found for model %s", modelID)
	}
	
	fmt.Printf("Starting unloading of model %s from %d partitions\n",
		modelID, len(partitions))
	
	// Unload each partition in parallel
	errCh := make(chan error, len(partitions))
	var wg sync.WaitGroup
	
	for _, partition := range partitions {
		wg.Add(1)
		go func(p ModelPartition) {
			defer wg.Done()
			err := ml.unloadPartition(ctx, modelID, p)
			if err != nil {
				errCh <- LoadError{
					Err:         err,
					NodeID:      p.NodeID,
					PartitionID: p.PartitionID,
				}
			}
		}(partition)
	}
	
	// Wait for all unloading goroutines to complete
	wg.Wait()
	close(errCh)
	
	// Check for errors
	var unloadErrors []error
	for err := range errCh {
		unloadErrors = append(unloadErrors, err)
	}
	
	// Clean up load status
	ml.statusMu.Lock()
	delete(ml.loadStatus, modelID)
	ml.statusMu.Unlock()
	
	if len(unloadErrors) > 0 {
		// Aggregate errors into a meaningful message
		errMsg := fmt.Sprintf("%d partitions failed to unload:\n", len(unloadErrors))
		for _, err := range unloadErrors {
			errMsg += fmt.Sprintf("  - %v\n", err)
		}
		return errors.New(errMsg)
	}
	
	fmt.Printf("Model %s successfully unloaded from all partitions\n", modelID)
	return nil
}

// unloadPartition unloads a partition from a specific node
func (ml *ModelLoader) unloadPartition(ctx context.Context, modelID string, partition ModelPartition) error {
	// In a real implementation, this would connect to the node and initiate unloading
	// using the streaming protocol to send unload commands
	
	fmt.Printf("Unloading partition %s from node %s\n", partition.PartitionID, partition.NodeID)
	
	// Get the streaming protocol connection
	_, err := ml.getOrCreateStreamingProtocol(partition.NodeID)
	if err != nil {
		return fmt.Errorf("failed to get streaming protocol: %w", err)
	}
	
	// Send notification to unsubscribe from all tensors in this partition
	tensorIDs, err := ml.getTensorIDsForPartition(modelID, partition.PartitionID)
	if err != nil {
		return fmt.Errorf("failed to get tensor IDs: %w", err)
	}
	
	// Unsubscribe from each tensor
	for _, tensorID := range tensorIDs {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			// Continue unloading
		}
		
		// Notify to unload this tensor (in a real implementation)
		// Here we just simulate the process
		fmt.Printf("Sending unload notification for tensor %s\n", tensorID)
	}
	
	// Simulate unloading time
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(200 * time.Millisecond):
		// Unloading complete
	}
	
	return nil
}

// SetTransferMode configures how tensors are transferred
func (ml *ModelLoader) SetTransferMode(mode tensor.TransferMode) {
	ml.transferMode = mode
}

// SetStreamingChunkSize sets the chunk size for streaming transfers
func (ml *ModelLoader) SetStreamingChunkSize(size int) {
	ml.protocolMu.Lock()
	defer ml.protocolMu.Unlock()

	// Update all existing protocols
	for _, proto := range ml.protocolCache {
		proto.SetChunkSize(size)
	}
	fmt.Printf("Streaming chunk size updated to %d bytes\n", size)
}

// SetCompressionEnabled enables or disables data compression
func (ml *ModelLoader) SetCompressionEnabled(enabled bool) {
	ml.protocolMu.Lock()
	defer ml.protocolMu.Unlock()

	// Update all existing protocols
	for _, proto := range ml.protocolCache {
		proto.SetCompressionEnabled(enabled)
	}
	fmt.Printf("Compression %s for streaming transfers\n",
		map[bool]string{true: "enabled", false: "disabled"}[enabled])
}

// SetCompressionThreshold sets the minimum size for compression to be applied
func (ml *ModelLoader) SetCompressionThreshold(threshold int) {
	ml.protocolMu.Lock()
	defer ml.protocolMu.Unlock()

	// Update all existing protocols
	for _, proto := range ml.protocolCache {
		proto.SetCompressionThreshold(uint64(threshold))
	}
	fmt.Printf("Compression threshold updated to %d bytes\n", threshold)
}

// Close releases all resources used by the model loader
func (ml *ModelLoader) Close() error {
	// Close all protocol connections
	ml.protocolMu.Lock()
	defer ml.protocolMu.Unlock()
	
	var closeErrors []error
	for nodeID, proto := range ml.protocolCache {
		if err := proto.Close(); err != nil {
			closeErrors = append(closeErrors, fmt.Errorf("failed to close connection to node %s: %w", nodeID, err))
		}
	}
	
	// Clear the cache
	ml.protocolCache = make(map[string]*tensor.StreamingProtocol)
	
	if len(closeErrors) > 0 {
		errMsg := fmt.Sprintf("%d connections failed to close:\n", len(closeErrors))
		for _, err := range closeErrors {
			errMsg += fmt.Sprintf("  - %v\n", err)
		}
		return errors.New(errMsg)
	}
	
	return nil
}