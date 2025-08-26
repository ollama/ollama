package llm

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/ml"
	"golang.org/x/sync/semaphore"
)

func TestLLMServerFitGPU(t *testing.T) {
	type gpu struct {
		library string
		free    int
	}

	tests := []struct {
		name        string
		gpus        []gpu
		layers      []int
		numGPU      int
		requireFull bool
		expected    ml.GPULayersList
		expectedErr error
	}{
		{
			name:     "No GPU",
			layers:   []int{50 * format.MebiByte, 50 * format.MebiByte, 50 * format.MebiByte},
			numGPU:   -1,
			expected: ml.GPULayersList{},
		},
		{
			name:     "Full single GPU",
			gpus:     []gpu{{free: 256 * format.MebiByte}},
			layers:   []int{50 * format.MebiByte, 50 * format.MebiByte, 50 * format.MebiByte},
			numGPU:   -1,
			expected: ml.GPULayersList{{ID: "gpu0", Layers: []int{0, 1, 2}}},
		},
		{
			name:     "Partial single GPU",
			gpus:     []gpu{{free: 256 * format.MebiByte}},
			layers:   []int{100 * format.MebiByte, 100 * format.MebiByte, 100 * format.MebiByte, 100 * format.MebiByte},
			numGPU:   -1,
			expected: ml.GPULayersList{{ID: "gpu0", Layers: []int{1, 2}}},
		},
		{
			name:     "Single GPU with numGPU 1",
			gpus:     []gpu{{free: 256 * format.MebiByte}},
			layers:   []int{50 * format.MebiByte, 50 * format.MebiByte, 50 * format.MebiByte},
			numGPU:   1,
			expected: ml.GPULayersList{{ID: "gpu0", Layers: []int{1}}},
		},
		{
			name:     "Single GPU with numGPU 0",
			gpus:     []gpu{{free: 256 * format.MebiByte}},
			layers:   []int{50 * format.MebiByte, 50 * format.MebiByte, 50 * format.MebiByte},
			numGPU:   0,
			expected: ml.GPULayersList{},
		},
		{
			name:     "Single GPU with numGPU 999",
			gpus:     []gpu{{free: 256 * format.MebiByte}},
			layers:   []int{100 * format.MebiByte, 100 * format.MebiByte, 100 * format.MebiByte, 100 * format.MebiByte},
			numGPU:   999,
			expected: ml.GPULayersList{{ID: "gpu0", Layers: []int{0, 1, 2, 3}}},
		},
		{
			name:     "Multi GPU fits on one",
			gpus:     []gpu{{free: 128 * format.MebiByte}, {free: 256 * format.MebiByte}},
			layers:   []int{50 * format.MebiByte, 50 * format.MebiByte, 50 * format.MebiByte},
			numGPU:   -1,
			expected: ml.GPULayersList{{ID: "gpu1", Layers: []int{0, 1, 2}}},
		},
		{
			name:     "Multi GPU split",
			gpus:     []gpu{{free: 128 * format.MebiByte}, {free: 256 * format.MebiByte}},
			layers:   []int{256 * format.MebiByte, 50 * format.MebiByte, 50 * format.MebiByte},
			numGPU:   -1,
			expected: ml.GPULayersList{{ID: "gpu1", Layers: []int{0}}, {ID: "gpu0", Layers: []int{1, 2}}},
		},
		{
			name:     "Multi GPU partial",
			gpus:     []gpu{{free: 128 * format.MebiByte}, {free: 256 * format.MebiByte}},
			layers:   []int{256 * format.MebiByte, 256 * format.MebiByte, 50 * format.MebiByte},
			numGPU:   -1,
			expected: ml.GPULayersList{{ID: "gpu1", Layers: []int{1}}},
		},
		{
			name:     "Multi GPU numGPU 1",
			gpus:     []gpu{{free: 128 * format.MebiByte}, {free: 256 * format.MebiByte}},
			layers:   []int{50 * format.MebiByte, 50 * format.MebiByte, 50 * format.MebiByte},
			numGPU:   1,
			expected: ml.GPULayersList{{ID: "gpu1", Layers: []int{1}}},
		},
		{
			name:     "Multi GPU numGPU 2",
			gpus:     []gpu{{free: 128 * format.MebiByte}, {free: 256 * format.MebiByte}},
			layers:   []int{256 * format.MebiByte, 50 * format.MebiByte, 50 * format.MebiByte},
			numGPU:   2,
			expected: ml.GPULayersList{{ID: "gpu1", Layers: []int{0}}, {ID: "gpu0", Layers: []int{1}}},
		},
		{
			name:     "Multi GPU numGPU 999",
			gpus:     []gpu{{free: 128 * format.MebiByte}, {free: 256 * format.MebiByte}},
			layers:   []int{256 * format.MebiByte, 256 * format.MebiByte, 50 * format.MebiByte},
			numGPU:   999,
			expected: ml.GPULayersList{{ID: "gpu1", Layers: []int{0, 1}}, {ID: "gpu0", Layers: []int{2}}},
		},
		{
			name:     "Multi GPU different libraries",
			gpus:     []gpu{{library: "cuda", free: 128 * format.MebiByte}, {library: "rocm", free: 256 * format.MebiByte}},
			layers:   []int{128 * format.MebiByte, 128 * format.MebiByte, 50 * format.MebiByte},
			numGPU:   -1,
			expected: ml.GPULayersList{{ID: "gpu1", Layers: []int{0, 1}}},
		},
		{
			name:        "requireFull",
			gpus:        []gpu{{free: 256 * format.MebiByte}},
			layers:      []int{100 * format.MebiByte, 100 * format.MebiByte, 100 * format.MebiByte, 100 * format.MebiByte},
			numGPU:      -1,
			requireFull: true,
			expectedErr: ErrLoadRequiredFull,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var systemInfo discover.SystemInfo
			systemInfo.System.TotalMemory = format.GibiByte
			systemInfo.System.FreeMemory = 512 * format.MebiByte
			systemInfo.System.FreeSwap = 256 * format.MebiByte

			gpus := make(discover.GpuInfoList, len(tt.gpus))
			for i := range tt.gpus {
				gpus[i].ID = fmt.Sprintf("gpu%d", i)
				gpus[i].Library = tt.gpus[i].library
				gpus[i].FreeMemory = uint64(tt.gpus[i].free)
			}

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
				Weights: make([]ml.Memory, s.totalLayers),
				Cache:   make([]ml.Memory, s.totalLayers),
			}, GPUs: make([]ml.DeviceMemory, len(gpus))}

			for i := range tt.layers {
				s.mem.CPU.Weights[i].Size = uint64(tt.layers[i])
			}

			for i := range s.mem.GPUs {
				s.mem.GPUs[i].ID = fmt.Sprintf("gpu%d", i)
				s.mem.GPUs[i].Weights = make([]ml.Memory, s.totalLayers)
				s.mem.GPUs[i].Cache = make([]ml.Memory, s.totalLayers)
			}

			gpuLayers, err := s.createLayout(systemInfo, gpus, s.mem, tt.requireFull, 0)
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

func TestUnicodeBufferHandler(t *testing.T) {
	tests := []struct {
		name              string
		inputResponses    []CompletionResponse
		expectedResponses []CompletionResponse
		description       string
	}{
		{
			name: "complete_unicode",
			inputResponses: []CompletionResponse{
				{Content: "Hello", Done: false},
				{Content: " world", Done: false},
				{Content: "!", Done: true},
			},
			expectedResponses: []CompletionResponse{
				{Content: "Hello", Done: false},
				{Content: " world", Done: false},
				{Content: "!", Done: true},
			},
			description: "All responses with valid unicode should pass through unchanged",
		},
		{
			name: "incomplete_unicode_at_end_with_done",
			inputResponses: []CompletionResponse{
				{Content: "Hello", Done: false},
				{Content: string([]byte{0xF0, 0x9F}), Done: true}, // Incomplete emoji with Done=true
			},
			expectedResponses: []CompletionResponse{
				{Content: "Hello", Done: false},
				{Content: "", Done: true}, // Content is trimmed but response is still sent with Done=true
			},
			description: "When Done=true, incomplete Unicode at the end should be trimmed",
		},
		{
			name: "split_unicode_across_responses",
			inputResponses: []CompletionResponse{
				{Content: "Hello " + string([]byte{0xF0, 0x9F}), Done: false}, // First part of 😀
				{Content: string([]byte{0x98, 0x80}) + " world!", Done: true}, // Second part of 😀 and more text
			},
			expectedResponses: []CompletionResponse{
				{Content: "Hello ", Done: false},  // Incomplete Unicode trimmed
				{Content: "😀 world!", Done: true}, // Complete emoji in second response
			},
			description: "Unicode split across responses should be handled correctly",
		},
		{
			name: "incomplete_unicode_buffered",
			inputResponses: []CompletionResponse{
				{Content: "Test " + string([]byte{0xF0, 0x9F}), Done: false}, // Incomplete emoji
				{Content: string([]byte{0x98, 0x80}), Done: false},           // Complete the emoji
				{Content: " done", Done: true},
			},
			expectedResponses: []CompletionResponse{
				{Content: "Test ", Done: false}, // First part without incomplete unicode
				{Content: "😀", Done: false},     // Complete emoji
				{Content: " done", Done: true},
			},
			description: "Incomplete unicode should be buffered and combined with next response",
		},
		{
			name: "empty_response_with_done",
			inputResponses: []CompletionResponse{
				{Content: "Complete response", Done: false},
				{Content: "", Done: true}, // Empty response with Done=true
			},
			expectedResponses: []CompletionResponse{
				{Content: "Complete response", Done: false},
				{Content: "", Done: true}, // Should still be sent because Done=true
			},
			description: "Empty final response with Done=true should still be sent",
		},
		{
			name: "done_reason_preserved",
			inputResponses: []CompletionResponse{
				{Content: "Response", Done: false},
				{Content: " complete", Done: true, DoneReason: DoneReasonStop},
			},
			expectedResponses: []CompletionResponse{
				{Content: "Response", Done: false},
				{Content: " complete", Done: true, DoneReason: DoneReasonStop},
			},
			description: "DoneReason should be preserved in the final response",
		},
		{
			name: "only_incomplete_unicode_not_done",
			inputResponses: []CompletionResponse{
				{Content: string([]byte{0xF0, 0x9F}), Done: false}, // Only incomplete unicode
			},
			expectedResponses: []CompletionResponse{
				// No response expected - should be buffered
			},
			description: "Response with only incomplete unicode should be buffered if not done",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var actualResponses []CompletionResponse

			// Create a callback that collects responses
			callback := func(resp CompletionResponse) {
				actualResponses = append(actualResponses, resp)
			}

			// Create the unicode buffer handler
			handler := unicodeBufferHandler(callback)

			// Send all input responses through the handler
			for _, resp := range tt.inputResponses {
				handler(resp)
			}

			// Verify the number of responses
			if len(actualResponses) != len(tt.expectedResponses) {
				t.Fatalf("%s: got %d responses, want %d responses",
					tt.description, len(actualResponses), len(tt.expectedResponses))
			}

			// Verify each response matches the expected one
			for i, expected := range tt.expectedResponses {
				if i >= len(actualResponses) {
					t.Fatalf("%s: missing response at index %d", tt.description, i)
					continue
				}

				actual := actualResponses[i]

				// Verify content
				if actual.Content != expected.Content {
					t.Errorf("%s: response[%d].Content = %q, want %q",
						tt.description, i, actual.Content, expected.Content)
				}

				// Verify Done flag
				if actual.Done != expected.Done {
					t.Errorf("%s: response[%d].Done = %v, want %v",
						tt.description, i, actual.Done, expected.Done)
				}

				// Verify DoneReason if specified
				if actual.DoneReason != expected.DoneReason {
					t.Errorf("%s: response[%d].DoneReason = %v, want %v",
						tt.description, i, actual.DoneReason, expected.DoneReason)
				}
			}
		})
	}
}
