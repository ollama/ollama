package llm

import (
	"strconv"
	"testing"

	"github.com/ollama/ollama/discover"
)

// testGGML implements GGMLModel for testing
type testGGML struct {
	kv KV
}

func (g *testGGML) KV() KV {
	return g.kv
}

func TestKVCacheQuantization(t *testing.T) {
	tests := []struct {
		name         string
		kvData       map[string]any
		gpus         discover.GpuInfoList
		envKVCache   string
		envFlashAttn string
		numGPULayers int
		expected     string
	}{
		// Basic validation cases
		{
			name: "empty cache type",
			kvData: map[string]any{
				"general.architecture": "llama",
			},
			gpus: discover.GpuInfoList{
				{Library: "cuda", DriverMajor: 8},
			},
			envKVCache:   "",
			envFlashAttn: "false",
			numGPULayers: -1,
			expected:     "f16",
		},
		{
			name: "invalid cache type",
			kvData: map[string]any{
				"general.architecture": "llama",
			},
			gpus: discover.GpuInfoList{
				{Library: "cuda", DriverMajor: 8},
			},
			envKVCache:   "invalid",
			envFlashAttn: "true",
			numGPULayers: -1,
			expected:     "f16",
		},
		// Embedding model cases
		{
			name: "embedding model with q4_0",
			kvData: map[string]any{
				"general.architecture": "bert",
				"bert.pooling_type":    "mean",
			},
			gpus: discover.GpuInfoList{
				{Library: "cuda", DriverMajor: 8},
			},
			envKVCache:   "q4_0",
			envFlashAttn: "true",
			numGPULayers: -1,
			expected:     "f16",
		},
		{
			name: "embedding model with f32",
			kvData: map[string]any{
				"general.architecture": "bert",
				"bert.pooling_type":    "mean",
			},
			gpus: discover.GpuInfoList{
				{Library: "cuda", DriverMajor: 8},
			},
			envKVCache:   "f32",
			envFlashAttn: "true",
			numGPULayers: -1,
			expected:     "f32",
		},
		// Flash attention cases
		{
			name: "non-embedding model with q4_0 and flash attention",
			kvData: map[string]any{
				"general.architecture":         "llama",
				"llama.attention.key_length":   uint32(32),
				"llama.attention.value_length": uint32(32),
			},
			gpus: discover.GpuInfoList{
				{Library: "cuda", DriverMajor: 8},
			},
			envKVCache:   "q4_0",
			envFlashAttn: "true",
			numGPULayers: -1,
			expected:     "q4_0",
		},
		{
			name: "non-embedding model with q4_0 without flash attention support",
			kvData: map[string]any{
				"general.architecture": "llama",
			},
			gpus: discover.GpuInfoList{
				{Library: "cuda", DriverMajor: 6},
			},
			envKVCache:   "q4_0",
			envFlashAttn: "true",
			numGPULayers: -1,
			expected:     "f16",
		},
		{
			name: "flash attention disabled via env var",
			kvData: map[string]any{
				"general.architecture":         "llama",
				"llama.attention.key_length":   uint32(32),
				"llama.attention.value_length": uint32(32),
			},
			gpus: discover.GpuInfoList{
				{Library: "cuda", DriverMajor: 8},
			},
			envKVCache:   "q4_0",
			envFlashAttn: "false",
			numGPULayers: -1,
			expected:     "f16",
		},
		// Multi-GPU cases
		{
			name: "multiple GPUs with flash attention support",
			kvData: map[string]any{
				"general.architecture":         "llama",
				"llama.attention.key_length":   uint32(32),
				"llama.attention.value_length": uint32(32),
			},
			gpus: discover.GpuInfoList{
				{Library: "cuda", DriverMajor: 8},
				{Library: "cuda", DriverMajor: 8},
			},
			envKVCache:   "q4_0",
			envFlashAttn: "true",
			numGPULayers: -1,
			expected:     "q4_0",
		},
		// Partial offload cases
		{
			name: "partial offload with q4_0 cache type",
			kvData: map[string]any{
				"general.architecture":         "llama",
				"llama.attention.key_length":   uint32(32),
				"llama.attention.value_length": uint32(32),
				"llama.block_count":            uint32(32),
			},
			gpus: discover.GpuInfoList{
				{Library: "cuda", DriverMajor: 8},
			},
			envKVCache:   "q4_0",
			envFlashAttn: "true",
			numGPULayers: 16,
			expected:     "q4_0",
		},
		{
			name: "minimal partial offload",
			kvData: map[string]any{
				"general.architecture":         "llama",
				"llama.attention.key_length":   uint32(32),
				"llama.attention.value_length": uint32(32),
				"llama.block_count":            uint32(32),
			},
			gpus: discover.GpuInfoList{
				{Library: "cuda", DriverMajor: 8},
			},
			envKVCache:   "q4_0",
			envFlashAttn: "true",
			numGPULayers: 1, // Test minimum offload case
			expected:     "q4_0",
		},
		{
			name: "partial offload with insufficient VRAM",
			kvData: map[string]any{
				"general.architecture":         "llama",
				"llama.attention.key_length":   uint32(32),
				"llama.attention.value_length": uint32(32),
				"llama.block_count":            uint32(32),
			},
			gpus: discover.GpuInfoList{
				{
					Library:       "cuda",
					DriverMajor:   8,
					MinimumMemory: 1024 * 1024 * 1024,
				},
			},
			envKVCache:   "q4_0",
			envFlashAttn: "true",
			numGPULayers: 4,
			expected:     "q4_0",
		},
		// Different GPU types
		{
			name: "metal GPU",
			kvData: map[string]any{
				"general.architecture":         "llama",
				"llama.attention.key_length":   uint32(32),
				"llama.attention.value_length": uint32(32),
			},
			gpus: discover.GpuInfoList{
				{Library: "metal"},
			},
			envKVCache:   "q4_0",
			envFlashAttn: "true",
			numGPULayers: -1,
			expected:     "q4_0",
		},
		{
			name: "rocm GPU",
			kvData: map[string]any{
				"general.architecture":         "llama",
				"llama.attention.key_length":   uint32(32),
				"llama.attention.value_length": uint32(32),
			},
			gpus: discover.GpuInfoList{
				{Library: "rocm", DriverMajor: 5},
			},
			envKVCache:   "q4_0",
			envFlashAttn: "true",
			numGPULayers: -1,
			expected:     "q4_0",
		},
		{
			name: "zero GPU layers", // Updated name
			kvData: map[string]any{
				"general.architecture":         "llama",
				"llama.attention.key_length":   uint32(32),
				"llama.attention.value_length": uint32(32),
			},
			gpus: discover.GpuInfoList{
				{Library: "cuda", DriverMajor: 8},
			},
			envKVCache:   "q4_0",
			envFlashAttn: "true",
			numGPULayers: 0,
			expected:     "q4_0", // The implementation considers GPU capabilities even with zero layers
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Reset environment before each test
			t.Setenv("OLLAMA_KV_CACHE_TYPE", tt.envKVCache)
			t.Setenv("OLLAMA_FLASH_ATTENTION", tt.envFlashAttn)
			if tt.numGPULayers >= 0 {
				t.Setenv("OLLAMA_NUM_GPU_LAYERS", strconv.Itoa(tt.numGPULayers))
			}

			ggml := &testGGML{kv: tt.kvData}
			result := kVCacheQuantization(tt.envKVCache, ggml)

			// Check the result
			if result != tt.expected {
				t.Errorf("expected %s, got %s for test case: %s", tt.expected, result, tt.name)
				t.Logf("Test configuration: %+v", tt)
			}
		})
	}
}

func TestValidateFASupport(t *testing.T) {
	tests := []struct {
		name               string
		kvData             map[string]any
		gpus               discover.GpuInfoList
		flashAttnRequested bool
		want               bool
	}{
		{
			name: "supported model and hardware",
			kvData: map[string]any{
				"general.architecture":         "llama",
				"llama.attention.key_length":   uint32(32),
				"llama.attention.value_length": uint32(32),
			},
			gpus: discover.GpuInfoList{
				{Library: "cuda", DriverMajor: 8},
			},
			flashAttnRequested: true,
			want:               true,
		},
		{
			name: "embedding model",
			kvData: map[string]any{
				"general.architecture":        "bert",
				"bert.attention.key_length":   uint32(32),
				"bert.attention.value_length": uint32(32),
				"bert.pooling_type":           "mean",
			},
			gpus: discover.GpuInfoList{
				{Library: "cuda", DriverMajor: 8},
			},
			flashAttnRequested: true,
			want:               false,
		},
		{
			name: "unsupported hardware",
			kvData: map[string]any{
				"general.architecture":         "llama",
				"llama.attention.key_length":   uint32(32),
				"llama.attention.value_length": uint32(32),
			},
			gpus: discover.GpuInfoList{
				{Library: "cuda", DriverMajor: 6},
			},
			flashAttnRequested: true,
			want:               false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ggml := &testGGML{kv: tt.kvData}
			got := tt.gpus.FlashAttentionSupported() && modelSupportsFlashAttention(ggml) && tt.flashAttnRequested
			if tt.want != got {
				t.Errorf("expected %v, got %v", tt.want, got)
			}
		})
	}
}
