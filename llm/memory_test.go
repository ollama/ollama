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
	t.Setenv("OLLAMA_KV_CACHE_TYPE", "") // Ensure default f16

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
	const (
		baseSize = 0x10000000 // Base size for standard model with 1024 context using f32
		baseCtx  = uint64(1024)
	)

	var (
		tokenizer = []string{" "}
		scores    = []float32{0}
		tokenType = []int32{0}
	)

	// Helper to create a test model with given architecture
	createTestModel := func(arch string, poolingType string) *GGML {
		f, err := os.CreateTemp(t.TempDir(), "dummy")
		require.NoError(t, err)
		defer f.Close()

		tensor := Tensor{
			Name:     "output.weight",
			Kind:     uint32(0),
			Offset:   uint64(0),
			Shape:    []uint64{1, 1, 1, 1},
			WriterTo: bytes.NewReader(make([]byte, 32)),
		}

		kv := KV{
			"general.architecture":            arch,
			arch + ".context_length":          uint32(32),
			arch + ".embedding_length":        uint32(4096),
			arch + ".block_count":             uint32(32),
			arch + ".attention.head_count":    uint32(32),
			arch + ".attention.head_count_kv": uint32(32),
			"tokenizer.ggml.tokens":           tokenizer,
			"tokenizer.ggml.scores":           scores,
			"tokenizer.ggml.token_type":       tokenType,
		}

		if poolingType != "" {
			kv[arch+".pooling_type"] = poolingType
		}

		require.NoError(t, WriteGGUF(f, kv, []Tensor{tensor}))

		ggml, err := LoadModel(f.Name(), 0)
		require.NoError(t, err)
		return ggml
	}

	tests := []struct {
		name        string
		ctx         uint64
		cacheType   string
		isEmbedding bool
		want        uint64
	}{
		{"f16 standard", baseCtx, "f16", false, baseSize * 2},     // f16 uses 2 bytes
		{"q8_0 standard", baseCtx, "q8_0", false, baseSize},       // q8_0 uses 1 byte
		{"q4_0 standard", baseCtx, "q4_0", false, baseSize / 2},   // q4_0 uses 0.5 bytes
		{"empty type", baseCtx, "", false, baseSize * 2},          // defaults to f16
		{"unknown type", baseCtx, "unknown", false, baseSize * 2}, // defaults to f16
		{"f16 embedding", baseCtx, "f16", true, baseSize * 2},     // embedding models still use same memory
		{"q8_0 embedding", baseCtx, "q8_0", true, baseSize},       // q8_0 still uses 1 byte
		{"q4_0 embedding", baseCtx, "q4_0", true, baseSize / 2},   // q4_0 still uses 0.5 bytes
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			arch := "llama"
			poolingType := ""
			if tt.isEmbedding {
				poolingType = "mean"
			}

			ggml := createTestModel(arch, poolingType)

			got := estimateKvCacheSize(tt.cacheType, tt.ctx, ggml)

			assert.Equal(t, tt.want, got,
				fmt.Sprintf("ctx=%d requested_cache=%s embedding=%v",
					tt.ctx, tt.cacheType, tt.isEmbedding))
		})
	}
}
