package model

import (
	"errors"
	"testing"
)

// Mock model structure for testing
type mockModel struct {
	Layers        int
	Parameters    int64
	MemoryRequired int64
}

// Mock node for testing
type mockNode struct {
	ID        string
	Name      string
	MemoryMB  int64
	Available bool
}

func TestNewShardedModelPartitioner(t *testing.T) {
	partitioner := NewShardedModelPartitioner()
	
	if partitioner == nil {
		t.Fatal("Expected non-nil partitioner")
	}
	
	if partitioner.PartitioningStrategy != ShardedPartitioning {
		t.Errorf("Expected strategy %s, got %s", 
			ShardedPartitioning, partitioner.PartitioningStrategy)
	}
}

func TestNewPipelinedModelPartitioner(t *testing.T) {
	partitioner := NewPipelinedModelPartitioner()
	
	if partitioner == nil {
		t.Fatal("Expected non-nil partitioner")
	}
	
	if partitioner.PartitioningStrategy != PipelinedPartitioning {
		t.Errorf("Expected strategy %s, got %s", 
			PipelinedPartitioning, partitioner.PartitioningStrategy)
	}
}

func TestCalculatePartitions_Sharded(t *testing.T) {
	partitioner := NewShardedModelPartitioner()
	
	// Test model requiring 24GB total
	model := ModelInfo{
		Name:         "test-model-24GB",
		TotalSizeGB:  24,
		LayerCount:   32,
	}
	
	// Test nodes with 8GB each (need at least 3)
	nodes := []NodeResourceInfo{
		{
			NodeID:       "node-1",
			AvailableRAM: 8 * 1024, // MB
		},
		{
			NodeID:       "node-2",
			AvailableRAM: 8 * 1024, // MB
		},
		{
			NodeID:       "node-3",
			AvailableRAM: 8 * 1024, // MB
		},
		{
			NodeID:       "node-4",
			AvailableRAM: 8 * 1024, // MB
		},
	}
	
	partitions, err := partitioner.CalculatePartitions(model, nodes)
	
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	
	// Should create 3 partitions (3 x 8GB = 24GB)
	if len(partitions) != 3 {
		t.Errorf("Expected 3 partitions, got %d", len(partitions))
	}
	
	// Verify memory distribution is somewhat even across partitions
	totalAllocated := int64(0)
	for _, p := range partitions {
		totalAllocated += p.MemoryMB
		
		// Each partition should be approximately 8GB (8192 MB)
		// Allow for some variance in distribution
		if p.MemoryMB < 7000 || p.MemoryMB > 9000 {
			t.Errorf("Partition memory outside expected range: %d MB", p.MemoryMB)
		}
	}
	
	// Total should match model size (24GB)
	expectedTotalMB := int64(24 * 1024)
	if totalAllocated != expectedTotalMB {
		t.Errorf("Expected total memory %d MB, got %d MB", expectedTotalMB, totalAllocated)
	}
}

func TestCalculatePartitions_Insufficient_Memory(t *testing.T) {
	partitioner := NewShardedModelPartitioner()
	
	// Test model requiring 32GB
	model := ModelInfo{
		Name:         "test-model-32GB",
		TotalSizeGB:  32,
		LayerCount:   32,
	}
	
	// Nodes with only 24GB total (8GB each, 3 nodes)
	nodes := []NodeResourceInfo{
		{
			NodeID:       "node-1",
			AvailableRAM: 8 * 1024, // MB
		},
		{
			NodeID:       "node-2",
			AvailableRAM: 8 * 1024, // MB
		},
		{
			NodeID:       "node-3",
			AvailableRAM: 8 * 1024, // MB
		},
	}
	
	_, err := partitioner.CalculatePartitions(model, nodes)
	
	// Should return an error due to insufficient memory
	if err == nil {
		t.Error("Expected error for insufficient memory, but got nil")
	}
	
	if !errors.Is(err, ErrInsufficientResources) {
		t.Errorf("Expected InsufficientResources error, got %v", err)
	}
}

func TestCalculatePartitions_Pipeline(t *testing.T) {
	partitioner := NewPipelinedModelPartitioner()
	
	// Model with 32 layers
	model := ModelInfo{
		Name:         "test-model",
		LayerCount:   32,
		TotalSizeGB:  16,
	}
	
	// 4 nodes with equal memory
	nodes := []NodeResourceInfo{
		{
			NodeID:       "node-1",
			AvailableRAM: 4 * 1024, // MB
		},
		{
			NodeID:       "node-2",
			AvailableRAM: 4 * 1024, // MB
		},
		{
			NodeID:       "node-3",
			AvailableRAM: 4 * 1024, // MB
		},
		{
			NodeID:       "node-4",
			AvailableRAM: 4 * 1024, // MB
		},
	}
	
	partitions, err := partitioner.CalculatePartitions(model, nodes)
	
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	
	// Should create 4 partitions (one per node)
	if len(partitions) != 4 {
		t.Errorf("Expected 4 partitions, got %d", len(partitions))
	}
	
	// Pipeline partitioning should distribute layers across nodes
	// Each partition should have approximately 32/4 = 8 layers
	layerCounts := map[int]int{}
	for _, p := range partitions {
		layerCounts[len(p.LayerIndices)]++
		
		// Each partition should have around 8 layers
		// Allow for some variance in distribution
		if len(p.LayerIndices) < 6 || len(p.LayerIndices) > 10 {
			t.Errorf("Partition layer count outside expected range: %d layers", 
				len(p.LayerIndices))
		}
	}
	
	// Check that we have all 32 layers distributed
	totalLayers := 0
	for _, p := range partitions {
		totalLayers += len(p.LayerIndices)
	}
	
	if totalLayers != 32 {
		t.Errorf("Expected 32 total layers, got %d", totalLayers)
	}
}