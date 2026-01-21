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

			gpuLayers, err := s.createLayout(systemInfo, tt.gpus, s.mem, tt.requireFull, 0)
			if err != tt.expectedErr {
				t.Fatalf("fitGPU returned error: %v", err)
			}
			if gpuLayers.Hash() != tt.expected.Hash() {
				t.Errorf("fitGPU assigned %v, want %v", gpuLayers, tt.expected)
			}
		})
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
