package llm

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/ml"
	"golang.org/x/sync/semaphore"
)

func TestLLMServerFitGPU(t *testing.T) {
	minMemory := 457 * format.MebiByte

	tests := []struct {
		name        string
		gpus        []ml.DeviceInfo
		layers      []int
		numGPU      int
		requireFull bool
		expected    ml.GPULayersList
		expectedErr error
	}{
		{
			name:        "No GPU",
			layers:      []int{50 * format.MebiByte, 50 * format.MebiByte, 50 * format.MebiByte},
			numGPU:      -1,
			expected:    ml.GPULayersList{},
			requireFull: true, // Should not try to evict even though we can't load any layers
		},
		{
			name:     "Full single GPU",
			gpus:     []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "gpu0"}, FreeMemory: uint64(256*format.MebiByte + minMemory)}},
			layers:   []int{50 * format.MebiByte, 50 * format.MebiByte, 50 * format.MebiByte},
			numGPU:   -1,
			expected: ml.GPULayersList{{DeviceID: ml.DeviceID{ID: "gpu0"}, Layers: []int{0, 1, 2}}},
		},
		{
			name:     "Partial single GPU",
			gpus:     []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "gpu0"}, FreeMemory: uint64(256*format.MebiByte + minMemory)}},
			layers:   []int{100 * format.MebiByte, 100 * format.MebiByte, 100 * format.MebiByte, 100 * format.MebiByte},
			numGPU:   -1,
			expected: ml.GPULayersList{{DeviceID: ml.DeviceID{ID: "gpu0"}, Layers: []int{1, 2}}},
		},
		{
			name:     "Single GPU with numGPU 1",
			gpus:     []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "gpu0"}, FreeMemory: uint64(256*format.MebiByte + minMemory)}},
			layers:   []int{50 * format.MebiByte, 50 * format.MebiByte, 50 * format.MebiByte},
			numGPU:   1,
			expected: ml.GPULayersList{{DeviceID: ml.DeviceID{ID: "gpu0"}, Layers: []int{1}}},
		},
		{
			name:     "Single GPU with numGPU 0",
			gpus:     []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "gpu0"}, FreeMemory: uint64(256*format.MebiByte + minMemory)}},
			layers:   []int{50 * format.MebiByte, 50 * format.MebiByte, 50 * format.MebiByte},
			numGPU:   0,
			expected: ml.GPULayersList{},
		},
		{
			name:     "Single GPU with numGPU 999",
			gpus:     []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "gpu0"}, FreeMemory: uint64(256*format.MebiByte + minMemory)}},
			layers:   []int{100 * format.MebiByte, 100 * format.MebiByte, 100 * format.MebiByte, 100 * format.MebiByte},
			numGPU:   999,
			expected: ml.GPULayersList{{DeviceID: ml.DeviceID{ID: "gpu0"}, Layers: []int{0, 1, 2, 3}}},
		},
		{
			name:     "Multi GPU fits on one",
			gpus:     []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "gpu0"}, FreeMemory: uint64(128*format.MebiByte + minMemory)}, {DeviceID: ml.DeviceID{ID: "gpu1"}, FreeMemory: uint64(256*format.MebiByte + minMemory)}},
			layers:   []int{50 * format.MebiByte, 50 * format.MebiByte, 50 * format.MebiByte},
			numGPU:   -1,
			expected: ml.GPULayersList{{DeviceID: ml.DeviceID{ID: "gpu1"}, Layers: []int{0, 1, 2}}},
		},
		{
			name:     "Multi GPU split",
			gpus:     []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "gpu0"}, FreeMemory: uint64(128*format.MebiByte + minMemory)}, {DeviceID: ml.DeviceID{ID: "gpu1"}, FreeMemory: uint64(256*format.MebiByte + minMemory)}},
			layers:   []int{256 * format.MebiByte, 50 * format.MebiByte, 50 * format.MebiByte},
			numGPU:   -1,
			expected: ml.GPULayersList{{DeviceID: ml.DeviceID{ID: "gpu1"}, Layers: []int{0}}, {DeviceID: ml.DeviceID{ID: "gpu0"}, Layers: []int{1, 2}}},
		},
		{
			name:     "Multi GPU partial",
			gpus:     []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "gpu0"}, FreeMemory: uint64(128*format.MebiByte + minMemory)}, {DeviceID: ml.DeviceID{ID: "gpu1"}, FreeMemory: uint64(256*format.MebiByte + minMemory)}},
			layers:   []int{256 * format.MebiByte, 256 * format.MebiByte, 50 * format.MebiByte},
			numGPU:   -1,
			expected: ml.GPULayersList{{DeviceID: ml.DeviceID{ID: "gpu1"}, Layers: []int{1}}},
		},
		{
			name:     "Multi GPU numGPU 1",
			gpus:     []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "gpu0"}, FreeMemory: uint64(128*format.MebiByte + minMemory)}, {DeviceID: ml.DeviceID{ID: "gpu1"}, FreeMemory: uint64(256*format.MebiByte + minMemory)}},
			layers:   []int{50 * format.MebiByte, 50 * format.MebiByte, 50 * format.MebiByte},
			numGPU:   1,
			expected: ml.GPULayersList{{DeviceID: ml.DeviceID{ID: "gpu1"}, Layers: []int{1}}},
		},
		{
			name:     "Multi GPU numGPU 2",
			gpus:     []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "gpu0"}, FreeMemory: uint64(128*format.MebiByte + minMemory)}, {DeviceID: ml.DeviceID{ID: "gpu1"}, FreeMemory: uint64(256*format.MebiByte + minMemory)}},
			layers:   []int{256 * format.MebiByte, 50 * format.MebiByte, 50 * format.MebiByte},
			numGPU:   2,
			expected: ml.GPULayersList{{DeviceID: ml.DeviceID{ID: "gpu1"}, Layers: []int{0}}, {DeviceID: ml.DeviceID{ID: "gpu0"}, Layers: []int{1}}},
		},
		{
			name:     "Multi GPU numGPU 999",
			gpus:     []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "gpu0"}, FreeMemory: uint64(128*format.MebiByte + minMemory)}, {DeviceID: ml.DeviceID{ID: "gpu1"}, FreeMemory: uint64(256*format.MebiByte + minMemory)}},
			layers:   []int{256 * format.MebiByte, 256 * format.MebiByte, 50 * format.MebiByte},
			numGPU:   999,
			expected: ml.GPULayersList{{DeviceID: ml.DeviceID{ID: "gpu1"}, Layers: []int{0, 1}}, {DeviceID: ml.DeviceID{ID: "gpu0"}, Layers: []int{2}}},
		},
		{
			name:     "Multi GPU different libraries",
			gpus:     []ml.DeviceInfo{{DeviceID: ml.DeviceID{Library: "CUDA", ID: "gpu0"}, FreeMemory: uint64(128*format.MebiByte + minMemory)}, {DeviceID: ml.DeviceID{Library: "ROCm", ID: "gpu1"}, FreeMemory: uint64(256*format.MebiByte + minMemory)}},
			layers:   []int{128 * format.MebiByte, 128 * format.MebiByte, 50 * format.MebiByte},
			numGPU:   -1,
			expected: ml.GPULayersList{{DeviceID: ml.DeviceID{ID: "gpu1", Library: "ROCm"}, Layers: []int{0, 1}}},
		},
		{
			name:        "requireFull",
			gpus:        []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "gpu0"}, FreeMemory: uint64(256*format.MebiByte + minMemory)}},
			layers:      []int{100 * format.MebiByte, 100 * format.MebiByte, 100 * format.MebiByte, 100 * format.MebiByte},
			numGPU:      -1,
			requireFull: true,
			expectedErr: ErrLoadRequiredFull,
		},
		{
			name:        "requireFull numGPU",
			gpus:        []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "gpu0"}, FreeMemory: uint64(256 * format.MebiByte)}},
			layers:      []int{100 * format.MebiByte, 100 * format.MebiByte, 100 * format.MebiByte, 100 * format.MebiByte},
			numGPU:      4,
			requireFull: true,
			expectedErr: ErrLoadRequiredFull,
		},
		{
			name:     "iGPU",
			gpus:     []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "gpu0"}, Integrated: true, FreeMemory: uint64(256*format.MebiByte + minMemory)}},
			layers:   []int{50 * format.MebiByte, 50 * format.MebiByte, 50 * format.MebiByte},
			numGPU:   -1,
			expected: ml.GPULayersList{{DeviceID: ml.DeviceID{ID: "gpu0"}, Layers: []int{0, 1, 2}}},
		},
		{
			name:     "iGPU + dGPU",
			gpus:     []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "gpu0"}, FreeMemory: uint64(128*format.MebiByte + minMemory)}, {DeviceID: ml.DeviceID{ID: "gpu1"}, Integrated: true, FreeMemory: uint64(256*format.MebiByte + minMemory)}},
			layers:   []int{50 * format.MebiByte, 50 * format.MebiByte, 50 * format.MebiByte},
			numGPU:   -1,
			expected: ml.GPULayersList{{DeviceID: ml.DeviceID{ID: "gpu1"}, Layers: []int{0}}, {DeviceID: ml.DeviceID{ID: "gpu0"}, Layers: []int{1, 2}}},
		},
		{
			name:     "iGPU + dGPU fits on one",
			gpus:     []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "gpu0"}, FreeMemory: uint64(128*format.MebiByte + minMemory)}, {DeviceID: ml.DeviceID{ID: "gpu1"}, Integrated: true, FreeMemory: uint64(256*format.MebiByte + minMemory)}},
			layers:   []int{50 * format.MebiByte, 50 * format.MebiByte},
			numGPU:   -1,
			expected: ml.GPULayersList{{DeviceID: ml.DeviceID{ID: "gpu0"}, Layers: []int{0, 1}}},
		},
		{
			name:     "iGPU + dGPU partial",
			gpus:     []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "gpu0"}, FreeMemory: uint64(128*format.MebiByte + minMemory)}, {DeviceID: ml.DeviceID{ID: "gpu1"}, Integrated: true, FreeMemory: uint64(256*format.MebiByte + minMemory)}},
			layers:   []int{100 * format.MebiByte, 100 * format.MebiByte, 100 * format.MebiByte, 100 * format.MebiByte},
			numGPU:   -1,
			expected: ml.GPULayersList{{DeviceID: ml.DeviceID{ID: "gpu1"}, Layers: []int{0, 1}}, {DeviceID: ml.DeviceID{ID: "gpu0"}, Layers: []int{2}}},
		},
		{
			name:     "iGPU + dGPU numGPU 1",
			gpus:     []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "gpu0"}, FreeMemory: uint64(128*format.MebiByte + minMemory)}, {DeviceID: ml.DeviceID{ID: "gpu1"}, Integrated: true, FreeMemory: uint64(256*format.MebiByte + minMemory)}},
			layers:   []int{100 * format.MebiByte, 100 * format.MebiByte, 100 * format.MebiByte, 100 * format.MebiByte},
			numGPU:   1,
			expected: ml.GPULayersList{{DeviceID: ml.DeviceID{ID: "gpu0"}, Layers: []int{2}}},
		},
		{
			name:     "iGPU + dGPU numGPU 999",
			gpus:     []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "gpu0"}, FreeMemory: uint64(128*format.MebiByte + minMemory)}, {DeviceID: ml.DeviceID{ID: "gpu1"}, Integrated: true, FreeMemory: uint64(256*format.MebiByte + minMemory)}},
			layers:   []int{100 * format.MebiByte, 100 * format.MebiByte, 100 * format.MebiByte, 100 * format.MebiByte},
			numGPU:   999,
			expected: ml.GPULayersList{{DeviceID: ml.DeviceID{ID: "gpu0"}, Layers: []int{0}}, {DeviceID: ml.DeviceID{ID: "gpu1"}, Layers: []int{1, 2, 3}}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var systemInfo ml.SystemInfo
			systemInfo.TotalMemory = format.GibiByte
			systemInfo.FreeMemory = 512 * format.MebiByte
			systemInfo.FreeSwap = 256 * format.MebiByte

			s := &ollamaServer{
				llmServer: llmServer{
					totalLayers: uint64(len(tt.layers)),
					options: api.Options{
						Runner: api.Runner{
							NumGPU: tt.numGPU,
						},
					},
				},
			}

			s.mem = &ml.BackendMemory{CPU: ml.DeviceMemory{
				Weights: make([]uint64, s.totalLayers),
				Cache:   make([]uint64, s.totalLayers),
			}, GPUs: make([]ml.DeviceMemory, len(tt.gpus))}

			for i := range tt.layers {
				s.mem.CPU.Weights[i] = uint64(tt.layers[i])
			}

			for i := range s.mem.GPUs {
				s.mem.GPUs[i].DeviceID = tt.gpus[i].DeviceID
				s.mem.GPUs[i].Weights = make([]uint64, s.totalLayers)
				s.mem.GPUs[i].Cache = make([]uint64, s.totalLayers)
			}

			gpuLayers, _, err := s.createLayout(systemInfo, tt.gpus, s.mem, tt.requireFull, 0)
			if err != tt.expectedErr {
				t.Fatalf("fitGPU returned error: %v", err)
			}
			if gpuLayers.Hash() != tt.expected.Hash() {
				t.Errorf("fitGPU assigned %v, want %v", gpuLayers, tt.expected)
			}
		})
	}
}

func TestLLMServerMoESplitRejectsMultipleGPUs(t *testing.T) {
	t.Setenv("OLLAMA_MOE_GPU_LAYERS", "-1")

	minMemory := uint64(457 * format.MebiByte)
	gpus := []ml.DeviceInfo{
		{DeviceID: ml.DeviceID{ID: "gpu-small"}, FreeMemory: minMemory + 320*format.MebiByte},
		{DeviceID: ml.DeviceID{ID: "gpu-large"}, FreeMemory: minMemory + 360*format.MebiByte},
	}

	s := &ollamaServer{
		llmServer: llmServer{
			totalLayers: 2,
			options: api.Options{
				Runner: api.Runner{NumGPU: -1},
			},
		},
	}

	s.mem = &ml.BackendMemory{CPU: ml.DeviceMemory{
		Weights:    []uint64{250 * format.MebiByte, 200 * format.MebiByte},
		MoEWeights: []uint64{50 * format.MebiByte, 50 * format.MebiByte},
		Cache:      make([]uint64, s.totalLayers),
	}, GPUs: make([]ml.DeviceMemory, len(gpus))}

	for i := range s.mem.GPUs {
		s.mem.GPUs[i].DeviceID = gpus[i].DeviceID
		s.mem.GPUs[i].Weights = make([]uint64, s.totalLayers)
		s.mem.GPUs[i].MoEWeights = make([]uint64, s.totalLayers)
		s.mem.GPUs[i].Cache = make([]uint64, s.totalLayers)
	}

	systemInfo := ml.SystemInfo{
		TotalMemory: 4 * format.GibiByte,
		FreeMemory:  2 * format.GibiByte,
		FreeSwap:    2 * format.GibiByte,
	}

	if _, _, err := s.createLayout(systemInfo, gpus, s.mem, false, 0); err == nil || !strings.Contains(err.Error(), "supports one GPU") {
		t.Fatalf("createLayout error = %v, want single-GPU guard", err)
	}
}

func TestLLMServerVerifyLayoutMoESplitCPUMemory(t *testing.T) {
	gpuID := ml.DeviceID{ID: "gpu0"}
	gpus := []ml.DeviceInfo{{DeviceID: gpuID}}
	layers := []uint64{1056 * format.MebiByte, 1056 * format.MebiByte}
	memory := &ml.BackendMemory{
		CPU: ml.DeviceMemory{
			Weights:    []uint64{1024 * format.MebiByte, 1024 * format.MebiByte},
			MoEWeights: []uint64{64 * format.MebiByte, 64 * format.MebiByte},
			Cache:      []uint64{32 * format.MebiByte, 32 * format.MebiByte},
		},
		GPUs: []ml.DeviceMemory{{
			DeviceID:   gpuID,
			Weights:    make([]uint64, len(layers)),
			MoEWeights: make([]uint64, len(layers)),
			Cache:      make([]uint64, len(layers)),
		}},
	}
	systemInfo := ml.SystemInfo{
		TotalMemory: 4 * format.GibiByte,
		FreeMemory:  200 * format.MebiByte,
	}
	s := &llmServer{}
	denseGPULayers := ml.GPULayersList{{DeviceID: gpuID, Layers: []int{0, 1}}}

	if err := s.verifyLayout(systemInfo, gpus, memory, false, nil, denseGPULayers, layers); err != nil {
		t.Fatalf("verifyLayout with MoE split returned error: %v", err)
	}

	if err := s.verifyLayout(systemInfo, gpus, memory, false, nil, nil, layers); err == nil {
		t.Fatal("verifyLayout without MoE split succeeded, want system memory error")
	}
}

func TestLLMServerMoESplitBudgetExcludesCacheFromMoELayers(t *testing.T) {
	t.Setenv("OLLAMA_MOE_GPU_LAYERS", "-1")
	t.Setenv("OLLAMA_GPU_OVERHEAD", "0")

	gpuID := ml.DeviceID{ID: "gpu0"}
	minMemory := uint64(457 * format.MebiByte)
	denseSize := uint64(100 * format.MebiByte)
	moeSize := uint64(50 * format.MebiByte)
	cacheSize := uint64(40 * format.MebiByte)
	denseCacheTotal := 2 * (denseSize + cacheSize)
	gpus := []ml.DeviceInfo{{DeviceID: gpuID, FreeMemory: minMemory + denseCacheTotal + moeSize}}

	s := &ollamaServer{
		llmServer: llmServer{
			totalLayers: 2,
			options: api.Options{
				Runner: api.Runner{NumGPU: -1},
			},
		},
	}
	s.mem = &ml.BackendMemory{CPU: ml.DeviceMemory{
		Weights:    []uint64{denseSize + moeSize, denseSize + moeSize},
		MoEWeights: []uint64{moeSize, moeSize},
		Cache:      []uint64{cacheSize, cacheSize},
	}, GPUs: []ml.DeviceMemory{{
		DeviceID:   gpuID,
		Weights:    make([]uint64, s.totalLayers),
		MoEWeights: make([]uint64, s.totalLayers),
		Cache:      make([]uint64, s.totalLayers),
	}}}

	gpuLayers, denseGPULayers, _, _ := s.buildLayout(gpus, s.mem, false, 0)
	if gpuLayers.Sum() != 1 {
		t.Fatalf("MoE GPU layers = %v, want one layer", gpuLayers)
	}
	if len(denseGPULayers) != 1 || denseGPULayers[0].DeviceID != gpuID || denseGPULayers.Sum() != 2 {
		t.Fatalf("dense GPU layers = %v, want all layers on %v", denseGPULayers, gpuID)
	}
}

func TestLLMServerMoEPrefetchReserveUsesLargestExpertTensor(t *testing.T) {
	t.Setenv("OLLAMA_MOE_PINNED", "1")
	t.Setenv("OLLAMA_MOE_PREFETCH", "1")

	memory := &ml.BackendMemory{
		CPU: ml.DeviceMemory{
			MoEMaxTensor: []uint64{20 * format.MebiByte, 4 * format.MebiByte},
		},
		GPUs: []ml.DeviceMemory{{
			MoEMaxTensor: []uint64{8 * format.MebiByte, 12 * format.MebiByte},
		}},
	}
	moeSize := []uint64{96 * format.MebiByte, 96 * format.MebiByte}

	if got, want := moePrefetchReserve(memory, moeSize, []int{0, 1}), uint64(40*format.MebiByte); got != want {
		t.Fatalf("moePrefetchReserve = %d, want %d", got, want)
	}
}

func TestLLMServerMoERunnerEnvOverrides(t *testing.T) {
	tests := []struct {
		name           string
		env            map[string]string
		wantPinned     bool
		wantPrefetch   bool
		wantNoOverride bool
	}{
		{
			name:           "no split leaves env unchanged",
			env:            map[string]string{"OLLAMA_MOE_PINNED": "1", "OLLAMA_MOE_PREFETCH": "1"},
			wantNoOverride: true,
		},
		{
			name:         "prefetch false does not enable internal flag",
			env:          map[string]string{"OLLAMA_MOE_CPU_LAYERS": "1", "OLLAMA_MOE_PINNED": "1", "OLLAMA_MOE_PREFETCH": "false"},
			wantPinned:   true,
			wantPrefetch: false,
		},
		{
			name:         "prefetch requires pinned memory",
			env:          map[string]string{"OLLAMA_MOE_CPU_LAYERS": "1", "OLLAMA_MOE_PREFETCH": "1"},
			wantPrefetch: false,
		},
		{
			name:         "pinned prefetch split enables internal flag",
			env:          map[string]string{"OLLAMA_MOE_GPU_LAYERS": "-1", "OLLAMA_MOE_PINNED": "1", "OLLAMA_MOE_PREFETCH": "1"},
			wantPinned:   true,
			wantPrefetch: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for key, value := range tt.env {
				t.Setenv(key, value)
			}

			env := runnerEnvOverrides(nil)
			if tt.wantNoOverride {
				if len(env) != 0 {
					t.Fatalf("runnerEnvOverrides = %v, want no overrides", env)
				}
				return
			}
			if _, ok := env["GGML_CUDA_REGISTER_HOST"]; ok != tt.wantPinned {
				t.Fatalf("GGML_CUDA_REGISTER_HOST present = %v, want %v in %v", ok, tt.wantPinned, env)
			}
			if _, ok := env["OLLAMA_MOE_PREFETCH_ENABLED"]; ok != tt.wantPrefetch {
				t.Fatalf("OLLAMA_MOE_PREFETCH_ENABLED present = %v, want %v in %v", ok, tt.wantPrefetch, env)
			}
		})
	}
}

func TestLLMServerMoECPULayersMatchesFirstExpertLayers(t *testing.T) {
	t.Setenv("OLLAMA_MOE_CPU_LAYERS", "2")
	t.Setenv("OLLAMA_GPU_OVERHEAD", "0")

	gpuID := ml.DeviceID{ID: "gpu0"}
	minMemory := uint64(457 * format.MebiByte)
	denseSize := uint64(50 * format.MebiByte)
	moeSize := uint64(10 * format.MebiByte)
	cacheSize := uint64(5 * format.MebiByte)
	gpus := []ml.DeviceInfo{{DeviceID: gpuID, FreeMemory: minMemory + 4*(denseSize+cacheSize) + 2*moeSize}}

	s := &ollamaServer{
		llmServer: llmServer{
			totalLayers: 4,
			options: api.Options{
				Runner: api.Runner{NumGPU: -1},
			},
		},
	}
	s.mem = &ml.BackendMemory{CPU: ml.DeviceMemory{
		Weights:    []uint64{denseSize + moeSize, denseSize, denseSize + moeSize, denseSize + moeSize},
		MoEWeights: []uint64{moeSize, 0, moeSize, moeSize},
		Cache:      []uint64{cacheSize, cacheSize, cacheSize, cacheSize},
	}, GPUs: []ml.DeviceMemory{{
		DeviceID:   gpuID,
		Weights:    make([]uint64, s.totalLayers),
		MoEWeights: make([]uint64, s.totalLayers),
		Cache:      make([]uint64, s.totalLayers),
	}}}

	gpuLayers, denseGPULayers, _, err := s.buildLayout(gpus, s.mem, false, 0)
	if err != nil {
		t.Fatalf("buildLayout returned error: %v", err)
	}
	expectedMoE := ml.GPULayersList{{DeviceID: gpuID, Layers: []int{2, 3}}}
	if gpuLayers.Hash() != expectedMoE.Hash() {
		t.Fatalf("MoE GPU layers = %v, want %v", gpuLayers, expectedMoE)
	}
	if len(denseGPULayers) != 1 || denseGPULayers[0].DeviceID != gpuID || denseGPULayers.Sum() != 4 {
		t.Fatalf("dense GPU layers = %v, want all model layers on %v", denseGPULayers, gpuID)
	}
}

func TestLLMServerMoEGPULayersMatchesLastExpertLayers(t *testing.T) {
	t.Setenv("OLLAMA_MOE_GPU_LAYERS", "2")
	t.Setenv("OLLAMA_GPU_OVERHEAD", "0")

	gpuID := ml.DeviceID{ID: "gpu0"}
	minMemory := uint64(457 * format.MebiByte)
	denseSize := uint64(50 * format.MebiByte)
	moeSize := uint64(10 * format.MebiByte)
	cacheSize := uint64(5 * format.MebiByte)
	gpus := []ml.DeviceInfo{{DeviceID: gpuID, FreeMemory: minMemory + 4*(denseSize+cacheSize) + 2*moeSize}}

	s := &ollamaServer{
		llmServer: llmServer{
			totalLayers: 4,
			options: api.Options{
				Runner: api.Runner{NumGPU: -1},
			},
		},
	}
	s.mem = &ml.BackendMemory{CPU: ml.DeviceMemory{
		Weights:    []uint64{denseSize + moeSize, denseSize, denseSize + moeSize, denseSize + moeSize},
		MoEWeights: []uint64{moeSize, 0, moeSize, moeSize},
		Cache:      []uint64{cacheSize, cacheSize, cacheSize, cacheSize},
	}, GPUs: []ml.DeviceMemory{{
		DeviceID:   gpuID,
		Weights:    make([]uint64, s.totalLayers),
		MoEWeights: make([]uint64, s.totalLayers),
		Cache:      make([]uint64, s.totalLayers),
	}}}

	gpuLayers, _, _, err := s.buildLayout(gpus, s.mem, false, 0)
	if err != nil {
		t.Fatalf("buildLayout returned error: %v", err)
	}
	expectedMoE := ml.GPULayersList{{DeviceID: gpuID, Layers: []int{2, 3}}}
	if gpuLayers.Hash() != expectedMoE.Hash() {
		t.Fatalf("MoE GPU layers = %v, want %v", gpuLayers, expectedMoE)
	}
}

func TestLLMServerMoEForcedSplitFailsWhenItCannotFit(t *testing.T) {
	t.Setenv("OLLAMA_MOE_CPU_LAYERS", "1")
	t.Setenv("OLLAMA_GPU_OVERHEAD", "0")

	gpuID := ml.DeviceID{ID: "gpu0"}
	minMemory := uint64(457 * format.MebiByte)
	denseSize := uint64(50 * format.MebiByte)
	moeSize := uint64(10 * format.MebiByte)
	cacheSize := uint64(5 * format.MebiByte)
	gpus := []ml.DeviceInfo{{DeviceID: gpuID, FreeMemory: minMemory + 3*(denseSize+cacheSize) + moeSize}}

	s := &ollamaServer{
		llmServer: llmServer{
			totalLayers: 3,
			options: api.Options{
				Runner: api.Runner{NumGPU: -1},
			},
		},
	}
	s.mem = &ml.BackendMemory{CPU: ml.DeviceMemory{
		Weights:    []uint64{denseSize + moeSize, denseSize + moeSize, denseSize + moeSize},
		MoEWeights: []uint64{moeSize, moeSize, moeSize},
		Cache:      []uint64{cacheSize, cacheSize, cacheSize},
	}, GPUs: []ml.DeviceMemory{{
		DeviceID:   gpuID,
		Weights:    make([]uint64, s.totalLayers),
		MoEWeights: make([]uint64, s.totalLayers),
		Cache:      make([]uint64, s.totalLayers),
	}}}

	if _, _, _, err := s.buildLayout(gpus, s.mem, false, 0); err == nil {
		t.Fatal("buildLayout succeeded, want forced MoE split allocation error")
	}
}

func TestLLMServerMoESplitRejectsConflictingOverrides(t *testing.T) {
	t.Setenv("OLLAMA_MOE_CPU_LAYERS", "1")
	t.Setenv("OLLAMA_MOE_GPU_LAYERS", "1")

	_, _, err := (&llmServer{totalLayers: 1}).createLayout(ml.SystemInfo{}, nil, &ml.BackendMemory{
		CPU: ml.DeviceMemory{
			Weights:    []uint64{1},
			MoEWeights: []uint64{1},
			Cache:      []uint64{0},
		},
	}, false, 0)
	if err == nil {
		t.Fatal("createLayout succeeded, want conflicting MoE override error")
	}
}

func TestLLMServerMoESplitRejectsInvalidNegativeGPULayers(t *testing.T) {
	t.Setenv("OLLAMA_MOE_GPU_LAYERS", "-2")

	_, _, err := (&llmServer{totalLayers: 1}).createLayout(ml.SystemInfo{}, nil, &ml.BackendMemory{
		CPU: ml.DeviceMemory{
			Weights:    []uint64{1},
			MoEWeights: []uint64{1},
			Cache:      []uint64{0},
		},
	}, false, 0)
	if err == nil {
		t.Fatal("createLayout succeeded, want invalid GPU override error")
	}
}

func TestLLMServerMoESplitRejectsOutOfRangeOverrides(t *testing.T) {
	tests := []struct {
		name string
		env  map[string]string
		want string
	}{
		{
			name: "gpu layers exceed expert layers",
			env:  map[string]string{"OLLAMA_MOE_GPU_LAYERS": "3"},
			want: "exceeds MoE expert layer count",
		},
		{
			name: "cpu layers exceed model layers",
			env:  map[string]string{"OLLAMA_MOE_CPU_LAYERS": "3"},
			want: "exceeds model layer count",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for key, value := range tt.env {
				t.Setenv(key, value)
			}

			_, _, err := (&llmServer{totalLayers: 2}).createLayout(ml.SystemInfo{}, nil, &ml.BackendMemory{
				CPU: ml.DeviceMemory{
					Weights:    []uint64{1, 1},
					MoEWeights: []uint64{1, 1},
					Cache:      []uint64{0, 0},
				},
			}, false, 0)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("createLayout error = %v, want %q", err, tt.want)
			}
		})
	}
}

func TestLLMServerMoEForcedSplitFailsWhenNoExpertsDetected(t *testing.T) {
	t.Setenv("OLLAMA_MOE_GPU_LAYERS", "-1")

	_, _, err := (&llmServer{totalLayers: 1}).createLayout(ml.SystemInfo{}, nil, &ml.BackendMemory{
		CPU: ml.DeviceMemory{
			Weights: []uint64{1},
			Cache:   []uint64{0},
		},
	}, false, 0)
	if err == nil || !strings.Contains(err.Error(), "no MoE expert tensors were detected") {
		t.Fatalf("createLayout error = %v, want missing MoE tensor error", err)
	}
}

func TestLLMServerMoEForcedSplitFailsWithoutDetectedExperts(t *testing.T) {
	t.Setenv("OLLAMA_MOE_CPU_LAYERS", "1")

	_, _, err := (&llmServer{totalLayers: 1}).createLayout(ml.SystemInfo{}, nil, &ml.BackendMemory{
		CPU: ml.DeviceMemory{
			Weights:    []uint64{1},
			MoEWeights: []uint64{0},
			Cache:      []uint64{0},
		},
	}, false, 0)
	if err == nil {
		t.Fatal("createLayout succeeded, want missing MoE expert tensor error")
	}
	if !strings.Contains(err.Error(), "no MoE expert tensors were detected") {
		t.Fatalf("createLayout error = %v, want missing MoE expert tensor error", err)
	}
}

func TestLLMServerMoEForcedSplitFailsWhenDenseCacheCannotFit(t *testing.T) {
	t.Setenv("OLLAMA_MOE_CPU_LAYERS", "1")
	t.Setenv("OLLAMA_GPU_OVERHEAD", "0")

	gpuID := ml.DeviceID{ID: "gpu0"}
	minMemory := uint64(457 * format.MebiByte)
	denseSize := uint64(100 * format.MebiByte)
	moeSize := uint64(10 * format.MebiByte)
	cacheSize := uint64(10 * format.MebiByte)
	gpus := []ml.DeviceInfo{{DeviceID: gpuID, FreeMemory: minMemory + denseSize + cacheSize}}

	s := &ollamaServer{
		llmServer: llmServer{
			totalLayers: 2,
			options: api.Options{
				Runner: api.Runner{NumGPU: -1},
			},
		},
	}
	s.mem = &ml.BackendMemory{CPU: ml.DeviceMemory{
		Weights:    []uint64{denseSize + moeSize, denseSize + moeSize},
		MoEWeights: []uint64{moeSize, moeSize},
		Cache:      []uint64{cacheSize, cacheSize},
	}, GPUs: []ml.DeviceMemory{{
		DeviceID:   gpuID,
		Weights:    make([]uint64, s.totalLayers),
		MoEWeights: make([]uint64, s.totalLayers),
		Cache:      make([]uint64, s.totalLayers),
	}}}

	if _, _, _, err := s.buildLayout(gpus, s.mem, false, 0); err == nil {
		t.Fatal("buildLayout succeeded, want dense/cache fit error")
	}
}

func TestLLMServerCompletionFormat(t *testing.T) {
	// This test was written to fix an already deployed issue. It is a bit
	// of a mess, and but it's good enough, until we can refactoring the
	// Completion method to be more testable.

	ctx, cancel := context.WithCancel(t.Context())
	s := &llmServer{
		sem: semaphore.NewWeighted(1), // required to prevent nil panic
	}

	checkInvalid := func(format string) {
		t.Helper()
		err := s.Completion(ctx, CompletionRequest{
			Options: new(api.Options),
			Format:  []byte(format),
		}, nil)

		want := fmt.Sprintf("invalid format: %q; expected \"json\" or a valid JSON Schema", format)
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Fatalf("err = %v; want %q", err, want)
		}
	}

	checkInvalid("X")   // invalid format
	checkInvalid(`"X"`) // invalid JSON Schema

	cancel() // prevent further processing if request makes it past the format check

	checkValid := func(err error) {
		t.Helper()
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Completion: err = %v; expected context.Canceled", err)
		}
	}

	valids := []string{
		// "missing"
		``,
		`""`,
		`null`,

		// JSON
		`"json"`,
		`{"type":"object"}`,
	}
	for _, valid := range valids {
		err := s.Completion(ctx, CompletionRequest{
			Options: new(api.Options),
			Format:  []byte(valid),
		}, nil)
		checkValid(err)
	}

	err := s.Completion(ctx, CompletionRequest{
		Options: new(api.Options),
		Format:  nil, // missing format
	}, nil)
	checkValid(err)
}
