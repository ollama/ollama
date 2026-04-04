package llm

import (
	"context"
	"fmt"
	"log/slog"
	"math"
	"os"
	"runtime"
	"strconv"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/gpu"
)

type LlamaServer struct {
	model      *Model
	opts       api.Options
	estimate   MemoryEstimate
	gpuLayers  int
	totalLayers int
}

type MemoryEstimate struct {
	ParametersSize uint64
	ContextSize    uint64
	TotalSize      uint64
	GPUSize        uint64
	CPUSize        uint64
}

// EstimateGPULayers calculates optimal GPU layer distribution
func (s *LlamaServer) EstimateGPULayers() error {
	if s.model == nil {
		return fmt.Errorf("model not loaded")
	}

	// Get GPU memory info
	memInfo, err := gpu.CheckVRAM()
	if err != nil {
		slog.Debug("No GPU available, using CPU", "error", err)
		s.gpuLayers = 0
		return nil
	}

	// Calculate memory requirements
	parameterSize := s.model.ModelSize()
	contextSize := s.calculateContextSize()
	totalModelSize := parameterSize + contextSize

	// Get total number of layers from model
	s.totalLayers = s.model.NumLayers()
	if s.totalLayers == 0 {
		slog.Debug("Unable to determine model layers, defaulting to CPU")
		s.gpuLayers = 0
		return nil
	}

	// Calculate layer size (approximate)
	layerSize := parameterSize / uint64(s.totalLayers)
	if layerSize == 0 {
		layerSize = 1024 * 1024 // 1MB minimum
	}

	// Add overhead for each layer (activations, gradients, etc.)
	layerOverhead := layerSize / 10 // 10% overhead per layer
	effectiveLayerSize := layerSize + layerOverhead

	// Calculate available GPU memory for model layers
	// Reserve memory for context and other GPU operations
	reservedForContext := contextSize + (contextSize / 4) // 25% extra for context overhead
	availableForLayers := uint64(0)
	if memInfo.FreeMemory > reservedForContext {
		availableForLayers = memInfo.FreeMemory - reservedForContext
	}

	// Estimate GPU layers with conservative approach
	s.gpuLayers = gpu.EstimateGPULayers(availableForLayers, effectiveLayerSize, s.totalLayers)

	// Ensure we don't exceed total layers
	if s.gpuLayers > s.totalLayers {
		s.gpuLayers = s.totalLayers
	}

	// Update memory estimates
	s.estimate = MemoryEstimate{
		ParametersSize: parameterSize,
		ContextSize:    contextSize,
		TotalSize:      totalModelSize,
		GPUSize:        uint64(s.gpuLayers) * effectiveLayerSize,
		CPUSize:        uint64(s.totalLayers-s.gpuLayers) * effectiveLayerSize,
	}

	slog.Info("GPU layer estimation",
		"total_layers", s.totalLayers,
		"gpu_layers", s.gpuLayers,
		"gpu_memory_available", formatBytes(memInfo.FreeMemory),
		"gpu_memory_used", formatBytes(s.estimate.GPUSize),
		"layer_size", formatBytes(effectiveLayerSize))

	return nil
}

// calculateContextSize estimates memory needed for context
func (s *LlamaServer) calculateContextSize() uint64 {
	contextLength := s.opts.NumCtx
	if contextLength == 0 {
		contextLength = 2048 // default context length
	}

	// Estimate context memory usage
	// This is a simplified calculation based on typical transformer memory usage
	embeddingSize := uint64(4096) // typical embedding dimension
	bytesPerParam := uint64(2)    // assuming fp16

	contextMemory := uint64(contextLength) * embeddingSize * bytesPerParam
	return contextMemory
}

func formatBytes(bytes uint64) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%d B", bytes)
	}
	div, exp := uint64(unit), 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %ciB", float64(bytes)/float64(div), "KMGTPE"[exp])
}

// GetGPULayers returns the number of layers allocated to GPU
func (s *LlamaServer) GetGPULayers() int {
	return s.gpuLayers
}

// GetMemoryEstimate returns the current memory estimate
func (s *LlamaServer) GetMemoryEstimate() MemoryEstimate {
	return s.estimate
}