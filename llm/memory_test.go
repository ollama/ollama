package llm

import (
	"bytes"
	"fmt"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/discover"
)

func TestEstimateGPULayers(t *testing.T) {
	t.Setenv("OLLAMA_DEBUG", "1")
	t.Setenv("OLLAMA_CACHE_TYPE_K", "")
	t.Setenv("OLLAMA_CACHE_TYPE_V", "")

	modelName := "dummy"
	f, err := os.CreateTemp(t.TempDir(), modelName)
	require.NoError(t, err)
	defer f.Close()
	inputLayerCount := 5

	tensors := []Tensor{
		{Name: "blk.0.attn.weight", Kind: uint32(0), Offset: uint64(0), Shape: []uint64{1, 1, 1, 1}, WriterTo: bytes.NewReader(make([]byte, 32))},
		{Name: "blk.1.attn.weight", Kind: uint32(0), Offset: uint64(0), Shape: []uint64{1, 1, 1, 1}, WriterTo: bytes.NewReader(make([]byte, 32))},
		{Name: "blk.2.attn.weight", Kind: uint32(0), Offset: uint64(0), Shape: []uint64{1, 1, 1, 1}, WriterTo: bytes.NewReader(make([]byte, 32))},
		{Name: "blk.3.attn.weight", Kind: uint32(0), Offset: uint64(0), Shape: []uint64{1, 1, 1, 1}, WriterTo: bytes.NewReader(make([]byte, 32))},
		{Name: "blk.4.attn.weight", Kind: uint32(0), Offset: uint64(0), Shape: []uint64{1, 1, 1, 1}, WriterTo: bytes.NewReader(make([]byte, 32))},
		{Name: "output.weight", Kind: uint32(0), Offset: uint64(0), Shape: []uint64{1, 1, 1, 1}, WriterTo: bytes.NewReader(make([]byte, 32))},
	}
	assert.Len(t, tensors, inputLayerCount+1)
	err = WriteGGUF(f, KV{
		"general.architecture":          "llama",
		"llama.context_length":          uint32(32),
		"llama.embedding_length":        uint32(4096),
		"llama.block_count":             uint32(inputLayerCount),
		"llama.attention.head_count":    uint32(32),
		"llama.attention.head_count_kv": uint32(32),
		"tokenizer.ggml.tokens":         []string{" "},
		"tokenizer.ggml.scores":         []float32{0},
		"tokenizer.ggml.token_type":     []int32{0},
	}, tensors)
	require.NoError(t, err)

	ggml, err := LoadModel(f.Name(), 0)
	if err != nil {
		t.Fatal(err)
	}

	// Simple CPU scenario
	gpus := []discover.GpuInfo{
		{
			Library: "cpu",
		},
	}
	projectors := []string{}
	opts := api.DefaultOptions()

	t.Run("cpu", func(t *testing.T) {
		estimate := EstimateGPULayers(gpus, ggml, projectors, opts)
		assert.Equal(t, 0, estimate.Layers)
		assert.Equal(t, uint64(0), estimate.Graph)
	})

	// derived from the dummy ggml file above
	graphPartialOffload := uint64(202377216)
	graphFullOffload := uint64(171968512)
	layerSize := uint64(33554436)
	projectorSize := uint64(0)
	memoryLayerOutput := uint64(4)

	// Dual CUDA scenario with asymmetry
	gpuMinimumMemory := uint64(2048)
	gpus = []discover.GpuInfo{
		{
			Library:       "cuda",
			MinimumMemory: gpuMinimumMemory,
		},
		{
			Library:       "cuda",
			MinimumMemory: gpuMinimumMemory,
		},
	}
	// Nested array: GPU0 layer space, GPU1 layer space, expected gpu0, expected gpu1
	for i, s := range []struct {
		layer0, layer1   uint64
		expect0, expect1 uint64
	}{
		{1, 1, 1, 1},
		{2, 1, 2, 1},
		{2, 2, 2, 2},
		{1, 2, 1, 2},
		{3, 3, 3, 3},
		{4, 4, 3, 3},
		{6, 6, 3, 3},
		{0, 3, 0, 3},
	} {
		t.Run(fmt.Sprintf("%v", s), func(t *testing.T) {
			gpus[0].FreeMemory = 0
			gpus[1].FreeMemory = 0
			gpus[0].FreeMemory += projectorSize
			if s.layer0 > 0 {
				gpus[0].FreeMemory += memoryLayerOutput
			} else {
				gpus[1].FreeMemory += memoryLayerOutput
			}
			gpus[0].FreeMemory += gpuMinimumMemory + layerSize + s.layer0*layerSize + 1
			gpus[1].FreeMemory += gpuMinimumMemory + layerSize + s.layer1*layerSize + 1
			gpus[0].FreeMemory += max(graphFullOffload, graphPartialOffload)
			gpus[1].FreeMemory += max(graphFullOffload, graphPartialOffload)
			estimate := EstimateGPULayers(gpus, ggml, projectors, opts)
			assert.Equal(t, int(s.expect0+s.expect1), estimate.Layers, "scenario %d: %v", i, s)
			assert.Equal(t, fmt.Sprintf("%d,%d", s.expect0, s.expect1), estimate.TensorSplit, "scenario %d: %v", i, s)
			var layerSums uint64
			for _, b := range estimate.GPUSizes {
				layerSums += b
			}
			if estimate.Layers < inputLayerCount+1 {
				assert.Less(t, estimate.VRAMSize, estimate.TotalSize, "scenario %d: %v %+v", i, s, estimate)
				assert.Equal(t, estimate.VRAMSize, layerSums, "scenario %d: %v %+v", i, s, estimate)
			} else {
				assert.Equal(t, estimate.VRAMSize, estimate.TotalSize, "scenario %d: %v %+v", i, s, estimate)
				assert.Equal(t, estimate.TotalSize, layerSums, "scenario %d: %v %+v", i, s, estimate)
			}
		})
	}
}

func TestEstimateKvCacheSize(t *testing.T) {
	tests := []struct {
		name               string
		cacheType          string
		numCtx             uint64
		blockCount         uint64
		embeddingHeadCount uint64
		headCountKV        uint64
		isEmbeddingModel   bool
		expected           uint64
	}{
		{
			name:               "f32 cache type",
			cacheType:          "f32",
			numCtx:             1024,
			blockCount:         32,
			embeddingHeadCount: 32,
			headCountKV:        32,
			isEmbeddingModel:   false,
			expected:           134217728, // 128 MB
		},
		{
			name:               "f16 cache type",
			cacheType:          "f16",
			numCtx:             1024,
			blockCount:         32,
			embeddingHeadCount: 32,
			headCountKV:        32,
			isEmbeddingModel:   false,
			expected:           67108864, // 64 MB
		},
		{
			name:               "q4_0 cache type",
			cacheType:          "q4_0",
			numCtx:             1024,
			blockCount:         32,
			embeddingHeadCount: 32,
			headCountKV:        32,
			isEmbeddingModel:   false,
			expected:           16777216, // 16 MB
		},
		{
			name:               "q8_0 cache type",
			cacheType:          "q8_0",
			numCtx:             1024,
			blockCount:         32,
			embeddingHeadCount: 32,
			headCountKV:        32,
			isEmbeddingModel:   false,
			expected:           33554432, // 32 MB
		},
		{
			name:               "unknown cache type",
			cacheType:          "unknown",
			numCtx:             1024,
			blockCount:         32,
			embeddingHeadCount: 32,
			headCountKV:        32,
			isEmbeddingModel:   false,
			expected:           67108864, // 64 MB (defaults to f16)
		},
		{
			name:               "empty cache type",
			cacheType:          "",
			numCtx:             1024,
			blockCount:         32,
			embeddingHeadCount: 32,
			headCountKV:        32,
			isEmbeddingModel:   false,
			expected:           67108864, // 64 MB (defaults to f16)
		},
		{
			name:               "rounding test",
			cacheType:          "f32",
			numCtx:             1000,
			blockCount:         32,
			embeddingHeadCount: 32,
			headCountKV:        32,
			isEmbeddingModel:   false,
			expected:           131072000, // Rounded up to nearest multiple of 64
		},
		{
			name:               "embedding model with q4_0 (should default to f16)",
			cacheType:          "q4_0",
			numCtx:             1024,
			blockCount:         32,
			embeddingHeadCount: 32,
			headCountKV:        32,
			isEmbeddingModel:   true,
			expected:           67108864, // 64 MB (defaults to f16)
		},
		{
			name:               "embedding model with f32",
			cacheType:          "f32",
			numCtx:             1024,
			blockCount:         32,
			embeddingHeadCount: 32,
			headCountKV:        32,
			isEmbeddingModel:   true,
			expected:           134217728, // 128 MB
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := estimateKvCacheSize(tt.cacheType, tt.numCtx, tt.blockCount, tt.embeddingHeadCount, tt.headCountKV, tt.isEmbeddingModel)
			assert.Equal(t, tt.expected, result, "Estimated KV cache size does not match expected value")
		})
	}
}
