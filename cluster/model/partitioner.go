package model

import (
	"errors"
	"fmt"
	"sync"

	"github.com/ollama/ollama/cluster"
)

// PartitioningStrategy defines how models are divided across nodes
type PartitioningStrategy string

const (
	// StrategyLayer divides models by layers
	StrategyLayer PartitioningStrategy = "layer"
	
	// StrategyTensor divides models by tensor dimensions
	StrategyTensor PartitioningStrategy = "tensor"
	
	// StrategyPipeline creates a pipeline of processing stages
	StrategyPipeline PartitioningStrategy = "pipeline"
	
	// StrategyHybrid combines multiple strategies
	StrategyHybrid PartitioningStrategy = "hybrid"
)

// PartitioningOptions configures how models are divided
type PartitioningOptions struct {
	// Strategy determines the partitioning approach
	Strategy PartitioningStrategy
	
	// MaxPartitionsPerNode limits partitions per node
	MaxPartitionsPerNode int
	
	// BalanceFactor controls how evenly to distribute partitions
	// 0 = equal size, 1 = equal count
	BalanceFactor float64
	
	// MinPartitionSize is the minimum size in bytes for a partition
	MinPartitionSize uint64
	
	// EnableOverlap allows some tensors to exist on multiple nodes
	EnableOverlap bool
	
	// ReplicationFactor is how many copies of each partition to maintain
	ReplicationFactor int
}

// DefaultPartitioningOptions provides sensible defaults
var DefaultPartitioningOptions = PartitioningOptions{
	Strategy:            StrategyLayer,
	MaxPartitionsPerNode: 4,
	BalanceFactor:       0.5,
	MinPartitionSize:    1024 * 1024 * 1, // 1MB - lowered to allow distribution of smaller models
	EnableOverlap:       true,
	ReplicationFactor:   1, // No replication by default
}

// ModelPartition represents a portion of a model assigned to a node
type ModelPartition struct {
	// PartitionID uniquely identifies this partition
	PartitionID string
	
	// NodeID where this partition is assigned
	NodeID string
	
	// ModelID identifies which model this partition belongs to
	ModelID string
	
	// StartLayer is the first layer in this partition
	StartLayer int
	
	// EndLayer is the last layer in this partition
	EndLayer int
	
	// Size is the memory footprint in bytes
	Size uint64
	
	// TensorIDs lists tensors in this partition
	TensorIDs []string
}

// ModelPartitioner manages splitting models across nodes
type ModelPartitioner struct {
	// options configures partitioning behavior
	options PartitioningOptions
	
	// registry provides access to available nodes
	registry *cluster.NodeRegistry
	
	// partitionMap tracks model → partitions
	// modelID → []ModelPartition
	partitionMap map[string][]ModelPartition
	
	// mu protects the partition map
	mu sync.RWMutex
}

// NewModelPartitioner creates a new model partitioner
func NewModelPartitioner(registry *cluster.NodeRegistry, options PartitioningOptions) *ModelPartitioner {
	return &ModelPartitioner{
		options:      options,
		registry:     registry,
		partitionMap: make(map[string][]ModelPartition),
	}
}

// PartitionModel divides a model across available nodes
func (mp *ModelPartitioner) PartitionModel(modelID string, modelSize uint64) ([]ModelPartition, error) {
	// Check if model is already partitioned
	mp.mu.RLock()
	if partitions, exists := mp.partitionMap[modelID]; exists {
		mp.mu.RUnlock()
		return partitions, nil
	}
	mp.mu.RUnlock()
	
	// Get available nodes
	nodes := mp.registry.GetAllNodes()
	activeNodes := make([]cluster.NodeInfo, 0)
	
	// Filter for online nodes only
	for _, node := range nodes {
		if node.Status == cluster.NodeStatusOnline {
			activeNodes = append(activeNodes, node)
		}
	}
	
	if len(activeNodes) == 0 {
		return nil, errors.New("no active nodes available for partitioning")
	}
	
	fmt.Printf("Partitioning model %s (%d bytes) across %d nodes using %s strategy\n",
		modelID, modelSize, len(activeNodes), mp.options.Strategy)
	
	var partitions []ModelPartition
	var err error
	
	// Apply the selected partitioning strategy
	switch mp.options.Strategy {
	case StrategyLayer:
		partitions, err = mp.partitionByLayer(modelID, modelSize, activeNodes)
	case StrategyTensor:
		partitions, err = mp.partitionByTensor(modelID, modelSize, activeNodes)
	case StrategyPipeline:
		partitions, err = mp.partitionByPipeline(modelID, modelSize, activeNodes)
	case StrategyHybrid:
		partitions, err = mp.partitionHybrid(modelID, modelSize, activeNodes)
	default:
		return nil, fmt.Errorf("unsupported partitioning strategy: %s", mp.options.Strategy)
	}
	
	if err != nil {
		return nil, err
	}
	
	// Store the partitioning
	mp.mu.Lock()
	mp.partitionMap[modelID] = partitions
	mp.mu.Unlock()
	
	return partitions, nil
}

// GetModelPartitions retrieves the partitioning for a model
func (mp *ModelPartitioner) GetModelPartitions(modelID string) ([]ModelPartition, bool) {
	mp.mu.RLock()
	defer mp.mu.RUnlock()
	
	partitions, exists := mp.partitionMap[modelID]
	return partitions, exists
}

// ListModels returns all partitioned models
func (mp *ModelPartitioner) ListModels() map[string][]ModelPartition {
	mp.mu.RLock()
	defer mp.mu.RUnlock()
	
	// Return a copy to avoid map modifications
	result := make(map[string][]ModelPartition)
	for modelID, partitions := range mp.partitionMap {
		partitionsCopy := make([]ModelPartition, len(partitions))
		copy(partitionsCopy, partitions)
		result[modelID] = partitionsCopy
	}
	
	return result
}
// partitionByLayer divides the model by layers across nodes
func (mp *ModelPartitioner) partitionByLayer(modelID string, modelSize uint64, nodes []cluster.NodeInfo) ([]ModelPartition, error) {
	// In transformer models, layers are a natural division point
	// For simplicity, we'll assume 32 layers in a typical LLM
	// In a real implementation, we would examine the model structure
	
	// This is a simplified implementation
	const assumedLayers = 32
	layersPerNode := assumedLayers / len(nodes)
	
	// Ensure at least one layer per node
	if layersPerNode == 0 {
		layersPerNode = 1
		fmt.Printf("Warning: More nodes than can be efficiently used. Using only %d nodes\n", assumedLayers)
	}
	
	// Size per layer (approximate)
	layerSize := modelSize / assumedLayers
	
	partitions := make([]ModelPartition, 0)
	
	// Distribute layers across nodes
	for i, node := range nodes {
		if i >= assumedLayers {
			break // Don't use more nodes than we have layers
		}
		
		startLayer := i * layersPerNode
		endLayer := (i + 1) * layersPerNode - 1
		
		// Adjust for the last partition
		if i == len(nodes)-1 || endLayer >= assumedLayers {
			endLayer = assumedLayers - 1
		}
		
		// Skip if out of layers
		if startLayer > endLayer {
			continue
		}
		
		// Calculate partition size
		partitionSize := layerSize * uint64(endLayer-startLayer+1)
		
		// Create partition with tensor IDs (simplified)
		tensorIDs := make([]string, 0)
		for layer := startLayer; layer <= endLayer; layer++ {
			// In a real implementation, we would list actual tensor IDs
			tensorIDs = append(tensorIDs, fmt.Sprintf("layer%d_weight", layer))
			tensorIDs = append(tensorIDs, fmt.Sprintf("layer%d_bias", layer))
		}
		
		partition := ModelPartition{
			PartitionID: fmt.Sprintf("%s-part%d", modelID, i),
			NodeID:      node.ID,
			ModelID:     modelID,
			StartLayer:  startLayer,
			EndLayer:    endLayer,
			Size:        partitionSize,
			TensorIDs:   tensorIDs,
		}
		
		partitions = append(partitions, partition)
	}
	
	fmt.Printf("Created %d partitions for model %s using layer strategy\n",
		len(partitions), modelID)
	
	return partitions, nil
}

// partitionByTensor divides the model by tensor dimensions
func (mp *ModelPartitioner) partitionByTensor(modelID string, modelSize uint64, nodes []cluster.NodeInfo) ([]ModelPartition, error) {
	// In this strategy, we would split large tensors across multiple nodes
	// For example, splitting attention matrices by head
	
	// This is a simplified placeholder implementation
	// In reality, this would require detailed knowledge of tensor dimensions
	
	partitionsCount := len(nodes)
	partitionSize := modelSize / uint64(partitionsCount)
	
	partitions := make([]ModelPartition, 0)
	
	for i, node := range nodes {
		// Create tensor IDs - in a real implementation these would be actual tensor IDs
		tensorIDs := make([]string, 0)
		tensorIDs = append(tensorIDs, fmt.Sprintf("embedding_shard%d", i))
		tensorIDs = append(tensorIDs, fmt.Sprintf("attention_shard%d", i))
		
		partition := ModelPartition{
			PartitionID: fmt.Sprintf("%s-tensor%d", modelID, i),
			NodeID:      node.ID,
			ModelID:     modelID,
			StartLayer:  0, // This field is less relevant for tensor partitioning
			EndLayer:    0, // This field is less relevant for tensor partitioning
			Size:        partitionSize,
			TensorIDs:   tensorIDs,
		}
		
		partitions = append(partitions, partition)
	}
	
	fmt.Printf("Created %d partitions for model %s using tensor strategy\n",
		len(partitions), modelID)
	
	return partitions, nil
}

// partitionByPipeline divides the model into pipeline stages
func (mp *ModelPartitioner) partitionByPipeline(modelID string, modelSize uint64, nodes []cluster.NodeInfo) ([]ModelPartition, error) {
	// In pipeline parallelism, each node handles a stage of processing
	// Stages are typically groups of layers that execute in sequence
	
	// For a typical LLM, pipeline stages might be:
	// 1. Embedding
	// 2. Early transformer blocks
	// 3. Middle transformer blocks
	// 4. Late transformer blocks
	// 5. Output layer
	
	// Determine how many stages we can use
	stageCount := min(len(nodes), 5) // Maximum 5 stages in this simplified implementation
	
	// Assuming layers are distributed evenly
	const assumedLayers = 32
	layersPerStage := assumedLayers / stageCount
	
	partitions := make([]ModelPartition, 0)
	
	for i := 0; i < stageCount; i++ {
		startLayer := i * layersPerStage
		endLayer := (i + 1) * layersPerStage - 1
		
		// Adjust for the last stage
		if i == stageCount-1 {
			endLayer = assumedLayers - 1
		}
		
		// Estimate size for this stage
		stageSize := (modelSize * uint64(endLayer-startLayer+1)) / assumedLayers
		
		// Create tensor IDs (simplified)
		tensorIDs := make([]string, 0)
		for layer := startLayer; layer <= endLayer; layer++ {
			tensorIDs = append(tensorIDs, fmt.Sprintf("layer%d_weight", layer))
			tensorIDs = append(tensorIDs, fmt.Sprintf("layer%d_bias", layer))
		}
		
		// For stage 0, add embedding
		if i == 0 {
			tensorIDs = append(tensorIDs, "token_embedding")
		}
		
		// For the last stage, add output layer
		if i == stageCount-1 {
			tensorIDs = append(tensorIDs, "output_layer")
		}
		
		partition := ModelPartition{
			PartitionID: fmt.Sprintf("%s-pipe%d", modelID, i),
			NodeID:      nodes[i].ID,
			ModelID:     modelID,
			StartLayer:  startLayer,
			EndLayer:    endLayer,
			Size:        stageSize,
			TensorIDs:   tensorIDs,
		}
		
		partitions = append(partitions, partition)
	}
	
	fmt.Printf("Created %d pipeline stages for model %s\n", stageCount, modelID)
	
	return partitions, nil
}

// partitionHybrid combines multiple partitioning strategies
func (mp *ModelPartitioner) partitionHybrid(modelID string, modelSize uint64, nodes []cluster.NodeInfo) ([]ModelPartition, error) {
	// Hybrid parallelism combines different strategies
	// Typically used for very large models on many nodes
	
	// If we have few nodes, default to layer-based partitioning
	if len(nodes) < 4 {
		return mp.partitionByLayer(modelID, modelSize, nodes)
	}
	
	// For more nodes, use a combination of strategies
	// Nodes are first divided into pipeline stages
	pipelineStages := min(len(nodes)/2, 4) // Use at most 4 pipeline stages
	
	nodesPerStage := len(nodes) / pipelineStages
	remainingNodes := len(nodes) % pipelineStages
	
	partitions := make([]ModelPartition, 0)
	nodeIndex := 0
	
	const assumedLayers = 32
	layersPerStage := assumedLayers / pipelineStages
	
	// Create pipeline stages
	for stage := 0; stage < pipelineStages; stage++ {
		startLayer := stage * layersPerStage
		endLayer := (stage + 1) * layersPerStage - 1
		
		// Adjust for last stage
		if stage == pipelineStages-1 {
			endLayer = assumedLayers - 1
		}
		
		// How many nodes for this stage (distributed fairly)
		stageNodes := nodesPerStage
		if stage < remainingNodes {
			stageNodes++
		}
		
		// Tensor partition within this stage
		layersInStage := endLayer - startLayer + 1
		stageModelSize := (modelSize * uint64(layersInStage)) / assumedLayers
		sizePerNode := stageModelSize / uint64(stageNodes)
		
		for i := 0; i < stageNodes; i++ {
			// For each node in this stage, create a partition
			node := nodes[nodeIndex]
			nodeIndex++
			
			// Create tensor IDs for this partition (simplified)
			tensorIDs := make([]string, 0)
			
			// In a real implementation, we would have a detailed mapping
			// between layers and tensors, and would distribute tensors across
			// nodes within the same stage
			
			// For this simplified version, we'll just assign some layer tensors
			layersPerNode := layersInStage / stageNodes
			nodeStartLayer := startLayer + (i * layersPerNode)
			nodeEndLayer := nodeStartLayer + layersPerNode - 1
			
			// Adjust for last node in stage
			if i == stageNodes-1 {
				nodeEndLayer = endLayer
			}
			
			for layer := nodeStartLayer; layer <= nodeEndLayer; layer++ {
				tensorIDs = append(tensorIDs, fmt.Sprintf("layer%d_weight", layer))
				tensorIDs = append(tensorIDs, fmt.Sprintf("layer%d_bias", layer))
			}
			
			// For first node in first stage, add embedding
			if stage == 0 && i == 0 {
				tensorIDs = append(tensorIDs, "token_embedding")
			}
			
			// For last node in last stage, add output layer
			if stage == pipelineStages-1 && i == stageNodes-1 {
				tensorIDs = append(tensorIDs, "output_layer")
			}
			
			partition := ModelPartition{
				PartitionID: fmt.Sprintf("%s-hybrid-p%d-n%d", modelID, stage, i),
				NodeID:      node.ID,
				ModelID:     modelID,
				StartLayer:  nodeStartLayer,
				EndLayer:    nodeEndLayer,
				Size:        sizePerNode,
				TensorIDs:   tensorIDs,
			}
			
			partitions = append(partitions, partition)
		}
	}
	
	fmt.Printf("Created %d partitions for model %s using hybrid strategy\n",
		len(partitions), modelID)
	
	return partitions, nil
}

// ReassignPartition moves a partition to a different node
func (mp *ModelPartitioner) ReassignPartition(modelID, partitionID, newNodeID string) error {
	mp.mu.Lock()
	defer mp.mu.Unlock()
	
	// Find the model
	partitions, exists := mp.partitionMap[modelID]
	if !exists {
		return fmt.Errorf("model %s not found", modelID)
	}
	
	// Find the partition
	found := false
	for i, partition := range partitions {
		if partition.PartitionID == partitionID {
			// Update the node ID
			partitions[i].NodeID = newNodeID
			found = true
			break
		}
	}
	
	if !found {
		return fmt.Errorf("partition %s not found for model %s", partitionID, modelID)
	}
	
	// Update the partition map
	mp.partitionMap[modelID] = partitions
	
	fmt.Printf("Reassigned partition %s of model %s to node %s\n",
		partitionID, modelID, newNodeID)
	
	return nil
}

// DeletePartitioning removes a model's partitioning information
func (mp *ModelPartitioner) DeletePartitioning(modelID string) error {
	mp.mu.Lock()
	defer mp.mu.Unlock()
	
	if _, exists := mp.partitionMap[modelID]; !exists {
		return fmt.Errorf("model %s not found", modelID)
	}
	
	delete(mp.partitionMap, modelID)
	fmt.Printf("Deleted partitioning for model %s\n", modelID)
	
	return nil
}

// UpdateOptions changes the partitioning options
func (mp *ModelPartitioner) UpdateOptions(options PartitioningOptions) {
	mp.mu.Lock()
	defer mp.mu.Unlock()
	
	mp.options = options
}

// GetOptions returns the current partitioning options
func (mp *ModelPartitioner) GetOptions() PartitioningOptions {
	return mp.options
}