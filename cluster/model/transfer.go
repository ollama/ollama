package model

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/ollama/ollama/cluster"
	"github.com/ollama/ollama/cluster/tensor"
)

// TransferState represents the current state of a tensor transfer
type TransferState string

const (
	// TransferStatePending indicates the transfer has not yet started
	TransferStatePending TransferState = "pending"
	
	// TransferStateInProgress indicates the transfer is currently active
	TransferStateInProgress TransferState = "in_progress"
	
	// TransferStateCompleted indicates the transfer has completed successfully
	TransferStateCompleted TransferState = "completed"
	
	// TransferStateFailed indicates the transfer failed
	TransferStateFailed TransferState = "failed"
	
	// TransferStateCancelled indicates the transfer was cancelled
	TransferStateCancelled TransferState = "cancelled"
)

// TransferOperation defines the type of transfer operation
type TransferOperation string

const (
	// TransferOperationPush indicates we're pushing tensors to a remote node
	TransferOperationPush TransferOperation = "push"
	
	// TransferOperationPull indicates we're pulling tensors from a remote node
	TransferOperationPull TransferOperation = "pull"
	
	// TransferOperationSync indicates we're synchronizing tensors bidirectionally
	TransferOperationSync TransferOperation = "sync"
)

// TransferMode options for protocol selection
const (
	// TransferModeStreaming uses the streaming protocol
	TransferModeStreaming = "streaming"
	
	// TransferModeStandard is the legacy non-streaming transfer mode
	// DEPRECATED: This mode is deprecated and will be removed in a future version.
	// All new code should use TransferModeStreaming instead.
	TransferModeStandard = "standard" // Deprecated: Use TransferModeStreaming
)

// TransferRequest represents a request to transfer tensor data
type TransferRequest struct {
	// ModelID identifies the model being transferred
	ModelID string
	
	// PartitionID identifies the partition
	PartitionID string
	
	// SourceNodeID is the node providing the tensor data
	SourceNodeID string
	
	// DestinationNodeID is the node receiving the tensor data
	DestinationNodeID string
	
	// TensorIDs lists the tensors to transfer
	TensorIDs []string
	
	// Operation is the type of transfer (push, pull, sync)
	Operation TransferOperation
	
	// Priority controls the urgency of the transfer
	Priority tensor.PriorityLevel
	
	// Mode selects the transfer protocol to use
	Mode string
}

// TransferProgress tracks the progress of a tensor transfer
type TransferProgress struct {
	// State is the current state of the transfer
	State TransferState
	
	// CompletedTensors is how many tensors have been transferred
	CompletedTensors int
	
	// TotalTensors is the total number of tensors in the transfer
	TotalTensors int
	
	// BytesTransferred is how many bytes have been transferred
	BytesTransferred uint64
	
	// TotalBytes is the total size of all tensors
	TotalBytes uint64
	
	// StartTime is when the transfer began
	StartTime time.Time
	
	// CompleteTime is when the transfer completed (if finished)
	CompleteTime time.Time
	
	// Error contains any error that occurred
	Error error
	
	// TransferIDs maps tensor IDs to their streaming transfer IDs
	TransferIDs map[string]string
}

// ModelTransferManager handles tensor transfers between nodes
type ModelTransferManager struct {
	// registry provides access to nodes
	registry *cluster.NodeRegistry
	
	// protocolManager manages streaming protocol connections
	protocolManager *StreamingProtocolManager
	
	// transfers tracks active transfers
	// transferID → TransferProgress
	transfers map[string]TransferProgress
	
	// transferMu protects the transfers map
	transferMu sync.RWMutex
	
	// transferCallbacks are functions called when transfer state changes
	transferCallbacks []func(transferID string, progress TransferProgress)
	
	// callbackMu protects the transferCallbacks slice
	callbackMu sync.RWMutex
}

// StreamingProtocolManager manages streaming protocol connections to nodes
type StreamingProtocolManager struct {
	// protocolCache caches streaming protocol connections to nodes
	// nodeID → streaming protocol
	protocolCache map[string]*tensor.StreamingProtocol
	
	// protocolMu protects the protocol cache
	protocolMu sync.RWMutex
	
	// registry is used to access node information
	registry *cluster.NodeRegistry
	
	// config provides access to cluster configuration
	config *cluster.ClusterConfig
}

// NewStreamingProtocolManager creates a new protocol manager
func NewStreamingProtocolManager(registry *cluster.NodeRegistry, config *cluster.ClusterConfig) *StreamingProtocolManager {
	return &StreamingProtocolManager{
		protocolCache: make(map[string]*tensor.StreamingProtocol),
		registry:      registry,
		config:        config,
	}
}

// GetOrCreateProtocol gets a cached protocol or creates a new one
func (spm *StreamingProtocolManager) GetOrCreateProtocol(nodeID string) (*tensor.StreamingProtocol, error) {
	spm.protocolMu.RLock()
	protocol, exists := spm.protocolCache[nodeID]
	spm.protocolMu.RUnlock()
	
	if exists {
		return protocol, nil
	}
	
	// Need to create a new connection
	nodeInfo, exists := spm.registry.GetNode(nodeID)
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
	conn, err := connectToNode(node)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to node %s: %w", nodeID, err)
	}
	
	// Create a new streaming protocol handler
	streamingProto := tensor.NewStreamingProtocol(conn)
	
	// Configure protocol options from cluster config
	if spm.config != nil {
		// Set chunk size from config if specified
		if spm.config.TensorProtocol.ChunkSize > 0 {
			streamingProto.SetChunkSize(spm.config.TensorProtocol.ChunkSize)
		}
		
		// Configure compression settings
		streamingProto.SetCompressionEnabled(spm.config.TensorProtocol.EnableCompression)
		if spm.config.TensorProtocol.CompressionThreshold > 0 {
			streamingProto.SetCompressionThreshold(uint64(spm.config.TensorProtocol.CompressionThreshold))
		}
		
		// Set retry policy
		retryPolicy := tensor.RetryPolicy{
			MaxRetries:    spm.config.TensorProtocol.MaxRetries,
			BaseDelay:     time.Duration(spm.config.TensorProtocol.RetryBaseDelay) * time.Millisecond,
			MaxDelay:      time.Duration(spm.config.TensorProtocol.RetryBaseDelay*10) * time.Millisecond,
			BackoffFactor: 2.0,
			JitterFactor:  0.1,
		}
		streamingProto.SetRetryPolicy(retryPolicy)
	} else {
		// Use defaults if no config
		streamingProto.SetChunkSize(1 * 1024 * 1024) // 1MB chunks
	}
	
	// Set default priority
	streamingProto.SetDefaultPriority(tensor.PriorityNormal)
	
	// Start cleanup routine
	streamingProto.StartCleanupRoutine()
	
	// Cache the protocol
	spm.protocolMu.Lock()
	spm.protocolCache[nodeID] = streamingProto
	spm.protocolMu.Unlock()
	
	fmt.Printf("Created new streaming protocol connection to node %s\n", nodeID)
	return streamingProto, nil
}

// connectToNode establishes a connection to a node
// Helper function used by multiple components
func connectToNode(node *cluster.Node) (cluster.Connection, error) {
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

// CloseProtocol closes and removes a protocol connection
func (spm *StreamingProtocolManager) CloseProtocol(nodeID string) error {
	spm.protocolMu.Lock()
	defer spm.protocolMu.Unlock()
	
	if protocol, exists := spm.protocolCache[nodeID]; exists {
		err := protocol.Close()
		delete(spm.protocolCache, nodeID)
		return err
	}
	
	return nil
}

// CloseAllProtocols closes all protocol connections
func (spm *StreamingProtocolManager) CloseAllProtocols() {
	spm.protocolMu.Lock()
	defer spm.protocolMu.Unlock()
	
	for nodeID, protocol := range spm.protocolCache {
		_ = protocol.Close()
		delete(spm.protocolCache, nodeID)
	}
}

// NewModelTransferManager creates a new model transfer manager
func NewModelTransferManager(registry *cluster.NodeRegistry, config *cluster.ClusterConfig) *ModelTransferManager {
	return &ModelTransferManager{
		registry:        registry,
		protocolManager: NewStreamingProtocolManager(registry, config),
		transfers:       make(map[string]TransferProgress),
	}
}

// TransferTensors initiates a tensor transfer between nodes
func (mtm *ModelTransferManager) TransferTensors(ctx context.Context, request TransferRequest) (string, error) {
	// Generate a unique transfer ID
	transferID := fmt.Sprintf("%s-%s-%s-%s-%d",
		request.ModelID, request.PartitionID,
		request.SourceNodeID, request.DestinationNodeID,
		time.Now().UnixNano())
	
	// Initialize transfer progress
	totalBytes := uint64(0)
	progress := TransferProgress{
		State:            TransferStatePending,
		CompletedTensors: 0,
		TotalTensors:     len(request.TensorIDs),
		BytesTransferred: 0,
		TotalBytes:       totalBytes, // Will be updated during transfer
		StartTime:        time.Now(),
		TransferIDs:      make(map[string]string),
	}
	
	// Store in transfers map
	mtm.transferMu.Lock()
	mtm.transfers[transferID] = progress
	mtm.transferMu.Unlock()
	
	// Start transfer in background
	go mtm.processTransfer(ctx, transferID, request)
	
	fmt.Printf("Initiated tensor transfer %s for model %s\n", transferID, request.ModelID)
	return transferID, nil
}

// processTransfer handles the actual tensor transfer
func (mtm *ModelTransferManager) processTransfer(ctx context.Context, transferID string, request TransferRequest) {
	// Update state to in progress
	mtm.updateTransferState(transferID, TransferStateInProgress)
	
	// Get appropriate protocols based on operation type
	var sourceProto, destProto *tensor.StreamingProtocol
	var err error
	
	// For push operations, we need a protocol to the destination
	if request.Operation == TransferOperationPush || request.Operation == TransferOperationSync {
		destProto, err = mtm.protocolManager.GetOrCreateProtocol(request.DestinationNodeID)
		if err != nil {
			mtm.failTransfer(transferID, fmt.Errorf("failed to establish connection to destination node: %w", err))
			return
		}
	}
	
	// For pull operations, we need a protocol to the source
	if request.Operation == TransferOperationPull || request.Operation == TransferOperationSync {
		sourceProto, err = mtm.protocolManager.GetOrCreateProtocol(request.SourceNodeID)
		if err != nil {
			mtm.failTransfer(transferID, fmt.Errorf("failed to establish connection to source node: %w", err))
			return
		}
	}
	
	// Process each tensor based on operation type
	var wg sync.WaitGroup
	errCh := make(chan error, len(request.TensorIDs))
	
	for _, tensorID := range request.TensorIDs {
		select {
		case <-ctx.Done():
			mtm.failTransfer(transferID, ctx.Err())
			return
		default:
			// Continue processing
		}
		
		wg.Add(1)
		go func(tID string) {
			defer wg.Done()
			
			var err error
			
			switch request.Operation {
			case TransferOperationPush:
				err = mtm.pushTensor(ctx, transferID, request, tID, destProto)
			case TransferOperationPull:
				err = mtm.pullTensor(ctx, transferID, request, tID, sourceProto)
			case TransferOperationSync:
				err = mtm.syncTensor(ctx, transferID, request, tID, sourceProto, destProto)
			default:
				err = fmt.Errorf("unsupported transfer operation: %s", request.Operation)
			}
			
			if err != nil {
				errCh <- fmt.Errorf("tensor %s transfer failed: %w", tID, err)
			} else {
				mtm.incrementCompletedTensors(transferID)
			}
		}(tensorID)
	}
	
	// Wait for all tensor transfers to complete
	wg.Wait()
	close(errCh)
	
	// Check for errors
	var transferErrors []error
	for err := range errCh {
		transferErrors = append(transferErrors, err)
	}
	
	if len(transferErrors) > 0 {
		// Aggregate errors
		errMsg := fmt.Sprintf("%d tensors failed to transfer:\n", len(transferErrors))
		for _, err := range transferErrors {
			errMsg += fmt.Sprintf("  - %v\n", err)
		}
		mtm.failTransfer(transferID, fmt.Errorf(errMsg))
		return
	}
	
	// Mark transfer as completed
	mtm.updateTransferState(transferID, TransferStateCompleted)
	
	// Update complete time
	mtm.transferMu.Lock()
	progress := mtm.transfers[transferID]
	progress.CompleteTime = time.Now()
	mtm.transfers[transferID] = progress
	mtm.transferMu.Unlock()
	
	fmt.Printf("Tensor transfer %s completed successfully\n", transferID)
}

// pushTensor pushes a tensor from local node to a remote node
func (mtm *ModelTransferManager) pushTensor(ctx context.Context, transferID string, request TransferRequest, tensorID string, destProto *tensor.StreamingProtocol) error {
	fmt.Printf("Pushing tensor %s to node %s\n", tensorID, request.DestinationNodeID)
	
	// In a real implementation, we would:
	// 1. Load the tensor data from local storage
	// 2. Stream it to the remote node
	
	// Simulate tensor data loading
	tensorData, tensorSize, err := loadTensorData(request.ModelID, tensorID)
	if err != nil {
		return fmt.Errorf("failed to load tensor data: %w", err)
	}
	
	// Use streaming protocol to push the tensor
	err = destProto.StreamTensor(request.ModelID, request.PartitionID, tensorID, tensorData)
	if err != nil {
		return fmt.Errorf("failed to stream tensor: %w", err)
	}
	
	// Update transfer progress
	mtm.updateTransferredBytes(transferID, tensorSize)
	
	return nil
}

// pullTensor pulls a tensor from a remote node to the local node
func (mtm *ModelTransferManager) pullTensor(ctx context.Context, transferID string, request TransferRequest, tensorID string, sourceProto *tensor.StreamingProtocol) error {
	fmt.Printf("Pulling tensor %s from node %s\n", tensorID, request.SourceNodeID)
	
	// In a real implementation, we would:
	// 1. Request the tensor from the remote node
	// 2. Receive and store the tensor locally
	
	// Use streaming protocol to request the tensor
	streamID, err := sourceProto.RequestTensor(
		request.ModelID,
		request.PartitionID,
		tensorID,
		request.Priority,
	)
	
	if err != nil {
		return fmt.Errorf("failed to request tensor: %w", err)
	}
	
	// Store the streamID for potential resumption
	mtm.addTransferStreamID(transferID, tensorID, streamID)
	
	// In a real implementation, we'd wait for the transfer to complete
	// and handle incoming chunks from the remote node
	
	// For this example, simulate a successful transfer with a delay
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(200 * time.Millisecond):
		// Transfer complete
	}
	
	// Simulate tensor size (would be actual received data size)
	tensorSize := uint64(1024 * 1024) // 1MB
	
	// Update transfer progress
	mtm.updateTransferredBytes(transferID, tensorSize)
	
	return nil
}

// syncTensor synchronizes a tensor between two nodes
func (mtm *ModelTransferManager) syncTensor(ctx context.Context, transferID string, request TransferRequest, tensorID string, sourceProto, destProto *tensor.StreamingProtocol) error {
	fmt.Printf("Synchronizing tensor %s between nodes %s and %s\n",
		tensorID, request.SourceNodeID, request.DestinationNodeID)
	
	// For synchronization, we need to:
	// 1. Determine which node has the latest version
	// 2. Transfer from the latest to the other
	
	// For this example, simply push from source to destination
	// In a real implementation, would check timestamps and versions
	
	// First pull from source if needed
	err := mtm.pullTensor(ctx, transferID, request, tensorID, sourceProto)
	if err != nil {
		return fmt.Errorf("failed to pull tensor in sync: %w", err)
	}
	
	// Then push to destination
	err = mtm.pushTensor(ctx, transferID, request, tensorID, destProto)
	if err != nil {
		return fmt.Errorf("failed to push tensor in sync: %w", err)
	}
	
	return nil
}

// loadTensorData loads tensor data from local storage (helper function)
func loadTensorData(modelID, tensorID string) ([]byte, uint64, error) {
	// Simulate loading tensor data
	// In a real implementation, this would load the actual tensor data
	
	// For this example, we'll simulate tensor data with random size
	size := uint64(1024 * 1024) // 1MB
	data := make([]byte, size)
	
	// Simulate some delay for loading from disk
	time.Sleep(10 * time.Millisecond)
	
	return data, size, nil
}

// updateTransferState updates the state of a transfer
func (mtm *ModelTransferManager) updateTransferState(transferID string, state TransferState) {
	mtm.transferMu.Lock()
	if progress, exists := mtm.transfers[transferID]; exists {
		progress.State = state
		mtm.transfers[transferID] = progress
		
		// Make a copy for callbacks
		progressCopy := progress
		mtm.transferMu.Unlock()
		
		// Notify callbacks
		mtm.notifyTransferCallbacks(transferID, progressCopy)
	} else {
		mtm.transferMu.Unlock()
	}
}

// failTransfer marks a transfer as failed with the given error
func (mtm *ModelTransferManager) failTransfer(transferID string, err error) {
	mtm.transferMu.Lock()
	if progress, exists := mtm.transfers[transferID]; exists {
		progress.State = TransferStateFailed
		progress.Error = err
		mtm.transfers[transferID] = progress
		
		// Make a copy for callbacks
		progressCopy := progress
		mtm.transferMu.Unlock()
		
		// Notify callbacks
		mtm.notifyTransferCallbacks(transferID, progressCopy)
	} else {
		mtm.transferMu.Unlock()
	}
	
	fmt.Printf("Transfer %s failed: %v\n", transferID, err)
}

// incrementCompletedTensors updates the completed tensor count
func (mtm *ModelTransferManager) incrementCompletedTensors(transferID string) {
	mtm.transferMu.Lock()
	if progress, exists := mtm.transfers[transferID]; exists {
		progress.CompletedTensors++
		mtm.transfers[transferID] = progress
		
		// Make a copy for callbacks
		progressCopy := progress
		mtm.transferMu.Unlock()
		
		// Notify callbacks
		mtm.notifyTransferCallbacks(transferID, progressCopy)
	} else {
		mtm.transferMu.Unlock()
	}
}

// updateTransferredBytes updates the bytes transferred count
func (mtm *ModelTransferManager) updateTransferredBytes(transferID string, additionalBytes uint64) {
	mtm.transferMu.Lock()
	if progress, exists := mtm.transfers[transferID]; exists {
		progress.BytesTransferred += additionalBytes
		progress.TotalBytes += additionalBytes // May be estimated initially
		mtm.transfers[transferID] = progress
		
		// Make a copy for callbacks
		progressCopy := progress
		mtm.transferMu.Unlock()
		
		// Notify callbacks
		mtm.notifyTransferCallbacks(transferID, progressCopy)
	} else {
		mtm.transferMu.Unlock()
	}
}

// addTransferStreamID associates a tensor with its streaming ID
func (mtm *ModelTransferManager) addTransferStreamID(transferID, tensorID, streamID string) {
	mtm.transferMu.Lock()
	defer mtm.transferMu.Unlock()
	
	if progress, exists := mtm.transfers[transferID]; exists {
		progress.TransferIDs[tensorID] = streamID
		mtm.transfers[transferID] = progress
	}
}

// notifyTransferCallbacks notifies callbacks about transfer progress changes
func (mtm *ModelTransferManager) notifyTransferCallbacks(transferID string, progress TransferProgress) {
	mtm.callbackMu.RLock()
	defer mtm.callbackMu.RUnlock()
	
	for _, callback := range mtm.transferCallbacks {
		go callback(transferID, progress)
	}
}

// RegisterTransferCallback adds a callback function for transfer progress updates
func (mtm *ModelTransferManager) RegisterTransferCallback(callback func(transferID string, progress TransferProgress)) {
	mtm.callbackMu.Lock()
	defer mtm.callbackMu.Unlock()
	
	mtm.transferCallbacks = append(mtm.transferCallbacks, callback)
}

// GetTransferProgress retrieves the current progress of a transfer
func (mtm *ModelTransferManager) GetTransferProgress(transferID string) (TransferProgress, error) {
	mtm.transferMu.RLock()
	defer mtm.transferMu.RUnlock()
	
	if progress, exists := mtm.transfers[transferID]; exists {
		return progress, nil
	}
	
	return TransferProgress{}, fmt.Errorf("transfer %s not found", transferID)
}

// CancelTransfer cancels an in-progress transfer
func (mtm *ModelTransferManager) CancelTransfer(transferID string) error {
	mtm.transferMu.Lock()
	progress, exists := mtm.transfers[transferID]
	if !exists {
		mtm.transferMu.Unlock()
		return fmt.Errorf("transfer %s not found", transferID)
	}
	
	// Only cancel if not already completed or failed
	if progress.State == TransferStateCompleted || 
	   progress.State == TransferStateFailed ||
	   progress.State == TransferStateCancelled {
		mtm.transferMu.Unlock()
		return fmt.Errorf("cannot cancel transfer %s in state %s", transferID, progress.State)
	}
	
	// Update state to cancelled
	progress.State = TransferStateCancelled
	mtm.transfers[transferID] = progress
	
	// Make a copy of transfer IDs for cancellation
	transferIDs := make(map[string]string)
	for k, v := range progress.TransferIDs {
		transferIDs[k] = v
	}
	mtm.transferMu.Unlock()
	
	// Cancel each individual tensor transfer
	// In a real implementation, would send cancellation messages to nodes
	
	fmt.Printf("Transfer %s cancelled\n", transferID)
	
	// Notify callbacks
	mtm.notifyTransferCallbacks(transferID, progress)
	
	return nil
}

// CleanupTransfers removes completed and failed transfers older than the given duration
func (mtm *ModelTransferManager) CleanupTransfers(age time.Duration) {
	mtm.transferMu.Lock()
	defer mtm.transferMu.Unlock()
	
	now := time.Now()
	for id, progress := range mtm.transfers {
		// Remove if completed/failed/cancelled and old enough
		if (progress.State == TransferStateCompleted || 
		    progress.State == TransferStateFailed ||
		    progress.State == TransferStateCancelled) && 
		   !progress.CompleteTime.IsZero() &&
		   now.Sub(progress.CompleteTime) > age {
			delete(mtm.transfers, id)
		}
	}
}

// Close releases all resources used by the transfer manager
func (mtm *ModelTransferManager) Close() error {
	// Close all protocol connections
	mtm.protocolManager.CloseAllProtocols()
	
	// Clean up transfers map
	mtm.transferMu.Lock()
	mtm.transfers = make(map[string]TransferProgress)
	mtm.transferMu.Unlock()
	
	return nil
}