package llm

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/gpu"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEstimateGPULayers(t *testing.T) {
	envconfig.Debug = true
	modelName := "dummy"
	f, err := os.CreateTemp(t.TempDir(), modelName)
	assert.Nil(t, err)
	defer f.Close()
	gguf := NewGGUFV3(binary.LittleEndian)
	inputLayerCount := 5
	tensors := []Tensor{
		{Name: "blk.0.attn.weight", Kind: uint32(0), Offset: uint64(0), Shape: []uint64{1, 1, 1, 1}, WriterTo: &bytes.Reader{}},
		{Name: "blk.1.attn.weight", Kind: uint32(0), Offset: uint64(0), Shape: []uint64{1, 1, 1, 1}, WriterTo: &bytes.Reader{}},
		{Name: "blk.2.attn.weight", Kind: uint32(0), Offset: uint64(0), Shape: []uint64{1, 1, 1, 1}, WriterTo: &bytes.Reader{}},
		{Name: "blk.3.attn.weight", Kind: uint32(0), Offset: uint64(0), Shape: []uint64{1, 1, 1, 1}, WriterTo: &bytes.Reader{}},
		{Name: "blk.4.attn.weight", Kind: uint32(0), Offset: uint64(0), Shape: []uint64{1, 1, 1, 1}, WriterTo: &bytes.Reader{}},
		{Name: "output.weight", Kind: uint32(0), Offset: uint64(0), Shape: []uint64{1, 1, 1, 1}, WriterTo: &bytes.Reader{}},
	}
	assert.Equal(t, inputLayerCount+1, len(tensors))
	err = gguf.Encode(f, KV{
		"general.architecture":          "llama",
		"general.name":                  "name",
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

	ggml, err := LoadModel(f.Name())
	require.NoError(t, err)

	// Simple CPU scenario
	gpus := []gpu.GpuInfo{
		{
			Library: "cpu",
		},
	}
	projectors := []string{}
	opts := api.DefaultOptions()
	estimate := EstimateGPULayers(gpus, ggml, projectors, opts)
	assert.Equal(t, 0, estimate.Layers)
	assert.Equal(t, uint64(0), estimate.Graph)

	// derived from the dummy ggml file above
	graphPartialOffload := uint64(202377216)
	graphFullOffload := uint64(171968512)
	layerSize := uint64(33554436)
	projectorSize := uint64(0)
	memoryLayerOutput := uint64(4)

	// Dual CUDA scenario with assymetry
	gpuMinimumMemory := uint64(2048)
	gpus = []gpu.GpuInfo{
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
	for i, s := range [][]uint64{
		{1, 1, 1, 1},
		{2, 1, 2, 1},
		{2, 2, 2, 2},
		{1, 2, 1, 2},
		{3, 3, 3, 3},
		{4, 4, 3, 3},
		{6, 6, 3, 3},
		{0, 3, 0, 3},
	} {
		gpus[0].FreeMemory = 0
		gpus[1].FreeMemory = 0
		gpus[0].FreeMemory += projectorSize + memoryLayerOutput
		gpus[0].FreeMemory += gpuMinimumMemory + layerSize + s[0]*layerSize + 1
		gpus[1].FreeMemory += gpuMinimumMemory + layerSize + s[1]*layerSize + 1
		gpus[0].FreeMemory += max(graphFullOffload, graphPartialOffload)
		gpus[1].FreeMemory += max(graphFullOffload, graphPartialOffload)
		estimate = EstimateGPULayers(gpus, ggml, projectors, opts)
		assert.Equal(t, int(s[2]+s[3]), estimate.Layers, "scenario %d: %v", i, s)
		assert.Equal(t, fmt.Sprintf("%d,%d", s[2], s[3]), estimate.TensorSplit, "scenario %d: %v", i, s)
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
	}

}
