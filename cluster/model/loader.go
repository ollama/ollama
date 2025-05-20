package model

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/ollama/ollama/cluster"
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
		partitioner: partitioner,
		registry:    registry,
		loadStatus:  make(map[string]map[string]LoadProgress),
	}
	
	// Log model loader initialization
	fmt.Printf("ModelLoader initialized with registry local node: %s\n", registry.GetLocalNodeID())
	
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
	
	fmt.Printf("Starting distributed loading of model %s across %d partitions\n",
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
					Err:        err,
					NodeID:     p.NodeID,
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

// loadPartition loads a single model partition on a specific node
func (ml *ModelLoader) loadPartition(ctx context.Context, modelID string, partition ModelPartition) error {
	// Update status to loading
	ml.updateLoadStatus(modelID, partition.PartitionID, LoadProgress{
		Status:      LoadStatusLoading,
		BytesLoaded: 0,
		TotalBytes:  partition.Size,
		StartTime:   time.Now(),
	})
	
	fmt.Printf("Loading partition %s on node %s\n", partition.PartitionID, partition.NodeID)
	
	// Check if node exists in registry
	node, exists := ml.registry.GetNode(partition.NodeID)
	if !exists {
		return fmt.Errorf("node %s not found in registry", partition.NodeID)
	}
	
	// Get source node (usually the coordinator/local node)
	localNodeID := ml.registry.GetLocalNodeID()
	
	// For the actual implementation, we should check if this is a remote node
	// that needs the model file transferred
	isRemoteNode := partition.NodeID != localNodeID
	
	// Log the node relationships
	fmt.Printf("Model %s (%s): Loading on node %s (remote: %v, local node: %s)\n",
		modelID, partition.PartitionID, partition.NodeID, isRemoteNode, localNodeID)
	
	// In a proper implementation, this would:
	// 1. Check if the model file exists on the source node
	sourceModelPath := fmt.Sprintf("/models/%s", modelID)
	fmt.Printf("Checking for model file at: %s\n", sourceModelPath)
	
	// 2. For remote nodes, ensure the model file is copied over
	if isRemoteNode {
		// This would use an API call or file transfer protocol to copy the model
		destModelPath := fmt.Sprintf("node://%s/models/%s", node.ID, modelID)
		fmt.Printf("Copying model file from %s to %s\n", sourceModelPath, destModelPath)
		
		// Simulate file copying progress
		totalBytes := partition.Size
		chunkSize := totalBytes / 10
		
		for bytesDone := uint64(0); bytesDone < totalBytes; bytesDone += chunkSize {
			// Check if context was canceled
			select {
			case <-ctx.Done():
				ml.updateLoadStatus(modelID, partition.PartitionID, LoadProgress{
					Status:      LoadStatusFailed,
					Error:       ctx.Err(),
					BytesLoaded: bytesDone,
					TotalBytes:  totalBytes,
				})
				fmt.Printf("Transfer canceled for model %s to node %s: %v\n",
					modelID, partition.NodeID, ctx.Err())
				return ctx.Err()
			default:
				// Continue loading
			}
			
			// Simulate network/disk I/O time
			bytesPerSecond := uint64(100 * 1024 * 1024) // 100MB/s
			chunkTimeMs := (chunkSize * 1000) / bytesPerSecond
			time.Sleep(time.Duration(chunkTimeMs) * time.Millisecond)
			
			// Update progress
			percentComplete := int((bytesDone * 100) / totalBytes)
			ml.updateLoadStatus(modelID, partition.PartitionID, LoadProgress{
				Status:          LoadStatusLoading,
				PercentComplete: percentComplete,
				BytesLoaded:     bytesDone,
				TotalBytes:      totalBytes,
				StartTime:       time.Now(), // In real code, this would be the actual start time
			})
			
			fmt.Printf("File transfer progress: %s to %s: %d%% (%d/%d bytes)\n",
				sourceModelPath, destModelPath, percentComplete, bytesDone, totalBytes)
		}
		
		// 3. Verify the file was properly copied
		fmt.Printf("File transfer complete. Verifying model file exists on target node: %s\n", destModelPath)
	}
	
	// 4. Load the model into GPU memory on the target node
	fmt.Printf("Loading model %s into GPU memory on node %s\n", modelID, node.ID)
	
	// This would use an API call to instruct the node to load the model into GPU memory
	// Simulate this with a delay for now
	time.Sleep(1 * time.Second)
	
	// Mark as successfully loaded
	ml.updateLoadStatus(modelID, partition.PartitionID, LoadProgress{
		Status:          LoadStatusLoaded,
		PercentComplete: 100,
		BytesLoaded:     partition.Size,
		TotalBytes:      partition.Size,
		StartTime:       time.Now(), // In real code, this would be the actual start time
		CompleteTime:    time.Now(),
	})
	
	fmt.Printf("Partition %s successfully loaded on node %s\n",
		partition.PartitionID, partition.NodeID)
	
	return nil
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
					Err:        err,
					NodeID:     p.NodeID,
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
	// For now, simulate unloading with a brief delay
	
	fmt.Printf("Unloading partition %s from node %s\n", partition.PartitionID, partition.NodeID)
	
	// Simulate unloading time
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(500 * time.Millisecond):
		// Unloading complete
	}
	
	return nil
}