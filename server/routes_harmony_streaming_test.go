package server

// this test file is to test integration of harmony parser into routes.go (as
// opposed to harmonyparser_test.go, which tests the parser in isolation)

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llm"
)

func getTestTools() []api.Tool {
	return []api.Tool{
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "get_weather",
				Description: "Get the current weather in a given location",
				Parameters: struct {
					Type       string                      `json:"type"`
					Defs       any                         `json:"$defs,omitempty"`
					Items      any                         `json:"items,omitempty"`
					Required   []string                    `json:"required"`
					Properties map[string]api.ToolProperty `json:"properties"`
				}{
					Type:     "object",
					Required: []string{"location"},
					Properties: map[string]api.ToolProperty{
						"location": {
							Type:        api.PropertyType{"string"},
							Description: "The city and state, e.g. San Francisco, CA",
						},
					},
				},
			},
		},
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "calculate",
				Description: "Calculate a mathematical expression",
				Parameters: struct {
					Type       string                      `json:"type"`
					Defs       any                         `json:"$defs,omitempty"`
					Items      any                         `json:"items,omitempty"`
					Required   []string                    `json:"required"`
					Properties map[string]api.ToolProperty `json:"properties"`
				}{
					Type:     "object",
					Required: []string{"expression"},
					Properties: map[string]api.ToolProperty{
						"expression": {
							Type:        api.PropertyType{"string"},
							Description: "The mathematical expression to calculate",
						},
					},
				},
			},
		},
	}
}

func createHarmonyTestModel(t *testing.T) (string, string) {
	t.Helper()

	return createBinFile(t, ggml.KV{
		"general.architecture":          "gptoss",
		"llama.block_count":             uint32(1),
		"llama.context_length":          uint32(8192),
		"llama.embedding_length":        uint32(4096),
		"llama.attention.head_count":    uint32(32),
		"llama.attention.head_count_kv": uint32(8),
		"tokenizer.ggml.tokens":         []string{""},
		"tokenizer.ggml.scores":         []float32{0},
		"tokenizer.ggml.token_type":     []int32{0},
	}, []*ggml.Tensor{
		{Name: "token_embd.weight", Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))},
		{Name: "blk.0.attn_norm.weight", Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))},
		{Name: "blk.0.ffn_down.weight", Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))},
		{Name: "blk.0.ffn_gate.weight", Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))},
		{Name: "blk.0.ffn_up.weight", Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))},
		{Name: "blk.0.ffn_norm.weight", Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))},
		{Name: "blk.0.attn_k.weight", Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))},
		{Name: "blk.0.attn_output.weight", Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))},
		{Name: "blk.0.attn_q.weight", Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))},
		{Name: "blk.0.attn_v.weight", Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))},
		{Name: "output.weight", Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))},
	})
}

// TestChatHarmonyParserStreamingRealtime verifies that chunks are emitted as soon as they're available
func TestChatHarmonyParserStreamingRealtime(t *testing.T) {
	gin.SetMode(gin.TestMode)

	type step struct {
		input         llm.CompletionResponse
		wantContent   string
		wantThinking  string
		wantToolCalls []api.ToolCall
	}

	testCases := []struct {
		name  string
		steps []step
		only  bool
	}{
		{
			name: "content streams as it arrives",
			steps: []step{
				{
					input:       llm.CompletionResponse{Content: "Hello", Done: false},
					wantContent: "Hello",
				},
				{
					input:       llm.CompletionResponse{Content: ", world", Done: false},
					wantContent: ", world",
				},
				{
					input:       llm.CompletionResponse{Content: "!", Done: true, DoneReason: llm.DoneReasonStop},
					wantContent: "!",
				},
			},
		},
		{
			name: "thinking streams separately from content",
			steps: []step{
				{
					input:        llm.CompletionResponse{Thinking: "Thinking...", Done: false},
					wantThinking: "Thinking...",
				},
				{
					input:       llm.CompletionResponse{Content: "Answer", Done: false},
					wantContent: "Answer",
				},
				{
					input: llm.CompletionResponse{Done: true, DoneReason: llm.DoneReasonStop},
				},
			},
		},
		{
			name: "partial tags buffer until complete",
			steps: []step{
				{
					input:        llm.CompletionResponse{Thinking: "Deep ", Done: false},
					wantThinking: "Deep ",
				},
				{
					input:        llm.CompletionResponse{Thinking: "thought", Done: false},
					wantThinking: "thought",
				},
				{
					input:       llm.CompletionResponse{Content: "Done", Done: true, DoneReason: llm.DoneReasonStop},
					wantContent: "Done",
				},
			},
		},
		{
			name: "simple assistant after analysis",
			steps: []step{
				{
					input:        llm.CompletionResponse{Thinking: "Think", Content: "Answer", Done: true, DoneReason: llm.DoneReasonStop},
					wantContent:  "Answer",
					wantThinking: "Think",
				},
			},
		},
		{
			name: "tool call parsed and returned correctly",
			steps: []step{
				{
					input:       llm.CompletionResponse{Content: "The weather is sunny", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "get_weather", Arguments: api.ToolCallFunctionArguments{"location": "San Francisco"}}}}, Done: true, DoneReason: llm.DoneReasonStop},
					wantContent: "The weather is sunny",
					wantToolCalls: []api.ToolCall{
						{
							Function: api.ToolCallFunction{
								Name: "get_weather",
								Arguments: api.ToolCallFunctionArguments{
									"location": "San Francisco",
								},
							},
						},
					},
				},
			},
		},
		{
			name: "tool call with streaming JSON across chunks",
			steps: []step{
				{
					input: llm.CompletionResponse{Done: false},
				},
				{
					input: llm.CompletionResponse{ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "calculate", Arguments: api.ToolCallFunctionArguments{"expression": "2+2"}}}}, Done: true},
					wantToolCalls: []api.ToolCall{
						{
							Function: api.ToolCallFunction{
								Name: "calculate",
								Arguments: api.ToolCallFunctionArguments{
									"expression": "2+2",
								},
							},
						},
					},
				},
			},
		},
	}

	anyOnlies := false
	for _, tc := range testCases {
		if tc.only {
			anyOnlies = true
		}
	}

	for _, tc := range testCases {
		if anyOnlies && !tc.only {
			continue
		}

		t.Run(tc.name, func(t *testing.T) {
			var chunks []api.ChatResponse
			chunkIdx := 0

			mockResponses := make([]llm.CompletionResponse, len(tc.steps))
			for i, step := range tc.steps {
				mockResponses[i] = step.input
			}

			mock := mockRunner{
				CompletionFn: func(ctx context.Context, r llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
					for _, resp := range mockResponses {
						fn(resp)
						// Give the handler time to process each response
						time.Sleep(30 * time.Millisecond)
					}
					return nil
				},
			}

			s := Server{
				sched: &Scheduler{
					pendingReqCh:  make(chan *LlmRequest, 1),
					finishedReqCh: make(chan *LlmRequest, 1),
					expiredCh:     make(chan *runnerRef, 1),
					unloadedCh:    make(chan any, 1),
					loaded:        make(map[string]*runnerRef),
					newServerFn:   newMockServer(&mock),
					getGpuFn:      discover.GetGPUInfo,
					getCpuFn:      discover.GetCPUInfo,
					reschedDelay:  100 * time.Millisecond,
					loadFn: func(req *LlmRequest, _ *ggml.GGML, _ discover.GpuInfoList, _ bool) bool {
						req.successCh <- &runnerRef{
							llama: &mock,
						}
						return false
					},
				},
			}

			go s.sched.Run(t.Context())

			// Create a simple test model
			_, digest := createHarmonyTestModel(t)

			streamFalse := false
			w := createRequest(t, s.CreateHandler, api.CreateRequest{
				Model:    "harmony-test-streaming",
				Files:    map[string]string{"test.gguf": digest},
				Template: `<|start|><|end|>{{ with .Tools }}{{ end }}{{ .Prompt }}`,
				Stream:   &streamFalse,
			})

			if w.Code != 200 {
				t.Fatalf("failed to create model: %d", w.Code)
			}

			// Test chat endpoint with streaming
			streamTrue := true
			w = createRequest(t, s.ChatHandler, api.ChatRequest{
				Model:    "harmony-test-streaming",
				Messages: []api.Message{{Role: "user", Content: "Hello"}},
				Stream:   &streamTrue,
				Tools:    getTestTools(),
			})

			if w.Code != 200 {
				t.Fatalf("chat request failed: %d - %s", w.Code, w.Body.String())
			}

			// Parse all chunks
			decoder := json.NewDecoder(w.Body)
			for decoder.More() {
				var chunk api.ChatResponse
				if err := decoder.Decode(&chunk); err != nil {
					t.Fatalf("failed to decode chunk: %v", err)
				}
				if chunk.Message.Content != "" || chunk.Message.Thinking != "" || len(chunk.Message.ToolCalls) > 0 {
					chunks = append(chunks, chunk)
				}
			}

			// Log received chunks for debugging
			if t.Failed() || len(chunks) == 0 {
				t.Logf("Received %d chunks:", len(chunks))
				for i, chunk := range chunks {
					t.Logf("  Chunk %d: content=%q thinking=%q", i, chunk.Message.Content, chunk.Message.Thinking)
				}
			}

			// Verify chunks match expected steps
			for i, step := range tc.steps {
				// Skip steps that don't expect any output
				if step.wantContent == "" && step.wantThinking == "" && len(step.wantToolCalls) == 0 {
					continue
				}

				if chunkIdx >= len(chunks) {
					t.Errorf("step %d: expected chunk not received (wanted content=%q thinking=%q)",
						i, step.wantContent, step.wantThinking)
					continue
				}

				chunk := chunks[chunkIdx]
				if chunk.Message.Content != step.wantContent || chunk.Message.Thinking != step.wantThinking {
					t.Errorf("step %d: chunk mismatch: got (content=%q, thinking=%q), want (content=%q, thinking=%q)",
						i, chunk.Message.Content, chunk.Message.Thinking, step.wantContent, step.wantThinking)
				}

				// Check tool calls if expected
				if len(step.wantToolCalls) > 0 {
					if len(chunk.Message.ToolCalls) != len(step.wantToolCalls) {
						t.Errorf("step %d: tool calls count mismatch: got %d, want %d",
							i, len(chunk.Message.ToolCalls), len(step.wantToolCalls))
					} else {
						for j, wantCall := range step.wantToolCalls {
							if j >= len(chunk.Message.ToolCalls) {
								break
							}
							gotCall := chunk.Message.ToolCalls[j]
							if gotCall.Function.Name != wantCall.Function.Name {
								t.Errorf("step %d, tool call %d: name mismatch: got %q, want %q",
									i, j, gotCall.Function.Name, wantCall.Function.Name)
							}
							// Compare arguments as JSON strings for simplicity
							gotArgs, _ := json.Marshal(gotCall.Function.Arguments)
							wantArgs, _ := json.Marshal(wantCall.Function.Arguments)
							if string(gotArgs) != string(wantArgs) {
								t.Errorf("step %d, tool call %d: arguments mismatch: got %s, want %s",
									i, j, string(gotArgs), string(wantArgs))
							}
						}
					}
				}
				chunkIdx++
			}

			// Check if we have extra chunks
			if chunkIdx < len(chunks) {
				t.Errorf("received %d extra chunks", len(chunks)-chunkIdx)
				for i := chunkIdx; i < len(chunks); i++ {
					t.Logf("  extra chunk %d: content=%q thinking=%q",
						i-chunkIdx, chunks[i].Message.Content, chunks[i].Message.Thinking)
				}
			}
		})
	}
}

// TestChatHarmonyParserStreamingSimple is a simpler test that just verifies basic streaming
func TestChatHarmonyParserStreamingSimple(t *testing.T) {
	gin.SetMode(gin.TestMode)

	mockResponses := []llm.CompletionResponse{
		{Content: "First ", Done: false},
		{Content: "chunk ", Done: false},
		{Content: "here", Done: true, DoneReason: llm.DoneReasonStop},
	}

	mock := mockRunner{
		CompletionFn: func(ctx context.Context, r llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
			t.Logf("Mock received prompt: %q", r.Prompt)
			t.Logf("Mock sending %d responses", len(mockResponses))
			for i, resp := range mockResponses {
				t.Logf("Sending response %d: %q", i, resp.Content)
				fn(resp)
			}
			return nil
		},
	}

	s := Server{
		sched: &Scheduler{
			pendingReqCh:  make(chan *LlmRequest, 1),
			finishedReqCh: make(chan *LlmRequest, 1),
			expiredCh:     make(chan *runnerRef, 1),
			unloadedCh:    make(chan any, 1),
			loaded:        make(map[string]*runnerRef),
			newServerFn:   newMockServer(&mock),
			getGpuFn:      discover.GetGPUInfo,
			getCpuFn:      discover.GetCPUInfo,
			reschedDelay:  100 * time.Millisecond,
			loadFn: func(req *LlmRequest, _ *ggml.GGML, _ discover.GpuInfoList, _ bool) bool {
				req.successCh <- &runnerRef{
					llama: &mock,
				}
				return false
			},
		},
	}

	go s.sched.Run(t.Context())

	// Create model
	_, digest := createHarmonyTestModel(t)
	streamFalse := false
	w := createRequest(t, s.CreateHandler, api.CreateRequest{
		Model:    "gpt-oss",
		Files:    map[string]string{"test.gguf": digest},
		Template: `<|start|><|end|>{{ .Tools }}{{ .Prompt }}`,
		Stream:   &streamFalse,
	})

	if w.Code != 200 {
		t.Fatalf("failed to create model: %d", w.Code)
	}

	// Test streaming
	streamTrue := true
	w = createRequest(t, s.ChatHandler, api.ChatRequest{
		Model:    "gpt-oss",
		Messages: []api.Message{{Role: "user", Content: "Hello"}},
		Stream:   &streamTrue,
		Tools:    getTestTools(),
	})

	if w.Code != 200 {
		t.Fatalf("chat request failed: %d - %s", w.Code, w.Body.String())
	}

	// Parse chunks
	var chunks []api.ChatResponse
	decoder := json.NewDecoder(w.Body)
	for decoder.More() {
		var chunk api.ChatResponse
		if err := decoder.Decode(&chunk); err != nil {
			t.Fatalf("failed to decode chunk: %v", err)
		}
		chunks = append(chunks, chunk)
		t.Logf("Received chunk %d: content=%q thinking=%q done=%v",
			len(chunks), chunk.Message.Content, chunk.Message.Thinking, chunk.Done)
	}

	// Verify we got chunks
	if len(chunks) == 0 {
		t.Fatal("expected streaming chunks, got none")
	}

	// Verify content
	var content strings.Builder
	for _, chunk := range chunks {
		content.WriteString(chunk.Message.Content)
	}

	expectedContent := "First chunk here"
	if content.String() != expectedContent {
		t.Errorf("content mismatch: got %q, want %q", content.String(), expectedContent)
	}

	// Verify we got multiple chunks (streaming)
	contentChunks := 0
	for _, chunk := range chunks {
		if chunk.Message.Content != "" {
			contentChunks++
		}
	}

	if contentChunks < 2 {
		t.Errorf("expected at least 2 content chunks for streaming, got %d", contentChunks)
	}
}
