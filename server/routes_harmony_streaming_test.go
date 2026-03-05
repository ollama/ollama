package server

// this test file is to test integration of harmony parser into routes.go (as
// opposed to harmonyparser_test.go, which tests the parser in isolation)

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

func getTestTools() []api.Tool {
	return []api.Tool{
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "get_weather",
				Description: "Get the current weather in a given location",
				Parameters: api.ToolFunctionParameters{
					Type:     "object",
					Required: []string{"location"},
					Properties: testPropsMap(map[string]api.ToolProperty{
						"location": {
							Type:        api.PropertyType{"string"},
							Description: "The city and state, e.g. San Francisco, CA",
						},
					}),
				},
			},
		},
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "calculate",
				Description: "Calculate a mathematical expression",
				Parameters: api.ToolFunctionParameters{
					Type:     "object",
					Required: []string{"expression"},
					Properties: testPropsMap(map[string]api.ToolProperty{
						"expression": {
							Type:        api.PropertyType{"string"},
							Description: "The mathematical expression to calculate",
						},
					}),
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
					input:       llm.CompletionResponse{Content: "<|message|>Hello", Done: false},
					wantContent: "Hello",
				},
				{
					input:       llm.CompletionResponse{Content: ", world", Done: false},
					wantContent: ", world",
				},
				{
					input:       llm.CompletionResponse{Content: "!<|end|>", Done: true, DoneReason: llm.DoneReasonStop},
					wantContent: "!",
				},
			},
		},
		{
			name: "thinking streams separately from content",
			steps: []step{
				{
					input:        llm.CompletionResponse{Content: "<|channel|>analysis<|message|>Thinking...", Done: false},
					wantThinking: "Thinking...",
				},
				{
					input: llm.CompletionResponse{Content: "<|end|>", Done: false},
					// No output expected - just closes the analysis message and resets state to normal
				},
				{
					input:       llm.CompletionResponse{Content: "<|start|>assistant<|message|>Answer", Done: false},
					wantContent: "Answer", // After message end, state is reset to normal
				},
				{
					input: llm.CompletionResponse{Content: "<|end|>", Done: true, DoneReason: llm.DoneReasonStop},
					// No output expected - just closes the assistant message
				},
			},
		},
		{
			name: "partial tags buffer until complete",
			steps: []step{
				{
					input: llm.CompletionResponse{Content: "<|chan", Done: false},
					// No output - partial tag
				},
				{
					input: llm.CompletionResponse{Content: "nel|>analysis<|mess", Done: false},
					// No output - still building tags
				},
				{
					input:        llm.CompletionResponse{Content: "age|>Deep ", Done: false},
					wantThinking: "Deep ",
				},
				{
					input:        llm.CompletionResponse{Content: "thought<|end|>", Done: false},
					wantThinking: "thought",
				},
				{
					input:       llm.CompletionResponse{Content: "<|start|>assistant<|message|>Done<|end|>", Done: true, DoneReason: llm.DoneReasonStop},
					wantContent: "Done", // After message end, state is reset to normal
				},
			},
		},
		{
			name: "simple assistant after analysis",
			steps: []step{
				{
					input:        llm.CompletionResponse{Content: "<|channel|>analysis<|message|>Think<|end|><|start|>assistant<|message|>Answer<|end|>", Done: true, DoneReason: llm.DoneReasonStop},
					wantContent:  "Answer",
					wantThinking: "Think",
				},
			},
		},
		{
			name: "tool call parsed and returned correctly",
			steps: []step{
				{
					input:       llm.CompletionResponse{Content: "<|channel|>commentary to=functions.get_weather<|message|>{\"location\":\"San Francisco\"}<|end|><|start|>assistant<|message|>The weather is sunny<|end|>", Done: true, DoneReason: llm.DoneReasonStop},
					wantContent: "The weather is sunny",
					wantToolCalls: []api.ToolCall{
						{
							Function: api.ToolCallFunction{
								Name: "get_weather",
								Arguments: testArgs(map[string]any{
									"location": "San Francisco",
								}),
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
					input: llm.CompletionResponse{Content: "<|channel|>commentary to=functions.calculate<|message|>{\"expr", Done: false},
					// No output yet - incomplete JSON
				},
				{
					input: llm.CompletionResponse{Content: "ession\":\"2+", Done: false},
					// Still no output - incomplete JSON
				},
				{
					input: llm.CompletionResponse{Content: "2\"}", Done: true},
					wantToolCalls: []api.ToolCall{
						{
							Function: api.ToolCallFunction{
								Name: "calculate",
								Arguments: testArgs(map[string]any{
									"expression": "2+2",
								}),
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
					pendingReqCh:    make(chan *LlmRequest, 1),
					finishedReqCh:   make(chan *LlmRequest, 1),
					expiredCh:       make(chan *runnerRef, 1),
					unloadedCh:      make(chan any, 1),
					loaded:          make(map[string]*runnerRef),
					newServerFn:     newMockServer(&mock),
					getGpuFn:        getGpuFn,
					getSystemInfoFn: getSystemInfoFn,
					waitForRecovery: 100 * time.Millisecond,
					loadFn: func(req *LlmRequest, _ *ggml.GGML, _ ml.SystemInfo, _ []ml.DeviceInfo, _ bool) bool {
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
		{Content: "<|message|>First ", Done: false},
		{Content: "chunk ", Done: false},
		{Content: "here<|end|>", Done: true, DoneReason: llm.DoneReasonStop},
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
			pendingReqCh:    make(chan *LlmRequest, 1),
			finishedReqCh:   make(chan *LlmRequest, 1),
			expiredCh:       make(chan *runnerRef, 1),
			unloadedCh:      make(chan any, 1),
			loaded:          make(map[string]*runnerRef),
			newServerFn:     newMockServer(&mock),
			getGpuFn:        getGpuFn,
			getSystemInfoFn: getSystemInfoFn,
			waitForRecovery: 100 * time.Millisecond,
			loadFn: func(req *LlmRequest, _ *ggml.GGML, _ ml.SystemInfo, _ []ml.DeviceInfo, _ bool) bool {
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

func TestChatHarmonyParserStreaming(t *testing.T) {
	gin.SetMode(gin.TestMode)

	type expectedChunk struct {
		afterResponse int    // Which mock response this chunk should appear after
		content       string // Expected content in this chunk
		thinking      string // Expected thinking in this chunk
	}

	testCases := []struct {
		name           string
		mockResponses  []llm.CompletionResponse
		expectedChunks []expectedChunk
		wantContent    string
		wantThinking   string
	}{
		{
			name: "simple message without thinking",
			mockResponses: []llm.CompletionResponse{
				{Content: "<|start|>assistant<|message|>Hello, ", Done: false},
				{Content: "how can I help?", Done: false},
				{Content: "<|end|>", Done: true, DoneReason: llm.DoneReasonStop},
			},
			expectedChunks: []expectedChunk{
				{afterResponse: 1, content: "Hello, "},
				{afterResponse: 2, content: "how can I help?"},
			},
			wantContent: "Hello, how can I help?",
		},
		{
			name: "message with analysis channel for thinking",
			mockResponses: []llm.CompletionResponse{
				{Content: "<|channel|>analysis<|message|>", Done: false},
				{Content: "Let me think ", Done: false},
				{Content: "about this problem...", Done: false},
				{Content: "<|end|>", Done: false},
				{Content: "<|start|>assistant<|message|>", Done: false},
				{Content: "The answer ", Done: false},
				{Content: "is 42", Done: false},
				{Content: "<|end|>", Done: true, DoneReason: llm.DoneReasonStop},
			},
			expectedChunks: []expectedChunk{
				{afterResponse: 2, thinking: "Let me think "},
				{afterResponse: 3, thinking: "about this problem..."},
				{afterResponse: 6, content: "The answer "},
				{afterResponse: 7, content: "is 42"},
			},
			wantContent:  "The answer is 42",
			wantThinking: "Let me think about this problem...",
		},
		{
			name: "streaming with partial tags across boundaries",
			mockResponses: []llm.CompletionResponse{
				{Content: "<|chan", Done: false},
				{Content: "nel|>analy", Done: false},
				{Content: "sis<|mess", Done: false},
				{Content: "age|>Think", Done: false},
				{Content: "ing deeply...<|end|>", Done: false},
				{Content: "<|start|>assi", Done: false},
				{Content: "stant<|message|>Result ", Done: false},
				{Content: "computed<|e", Done: false},
				{Content: "nd|>", Done: true, DoneReason: llm.DoneReasonStop},
			},
			expectedChunks: []expectedChunk{
				{afterResponse: 4, thinking: "Think"},
				{afterResponse: 5, thinking: "ing deeply..."},
				{afterResponse: 7, content: "Result "},
				{afterResponse: 8, content: "computed"},
			},
			wantContent:  "Result computed",
			wantThinking: "Thinking deeply...",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Channel to synchronize mock responses with chunk verification
			responsesSent := make(chan int, len(tc.mockResponses))

			mock := mockRunner{
				CompletionFn: func(ctx context.Context, r llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
					// Send mock responses one at a time, notifying when each is sent
					for i, resp := range tc.mockResponses {
						fn(resp)
						responsesSent <- i + 1
					}
					close(responsesSent)
					return nil
				},
			}

			s := Server{
				sched: &Scheduler{
					pendingReqCh:    make(chan *LlmRequest, 1),
					finishedReqCh:   make(chan *LlmRequest, 1),
					expiredCh:       make(chan *runnerRef, 1),
					unloadedCh:      make(chan any, 1),
					loaded:          make(map[string]*runnerRef),
					newServerFn:     newMockServer(&mock),
					getGpuFn:        getGpuFn,
					getSystemInfoFn: getSystemInfoFn,
					waitForRecovery: 250 * time.Millisecond,
					loadFn: func(req *LlmRequest, _ *ggml.GGML, _ ml.SystemInfo, _ []ml.DeviceInfo, _ bool) bool {
						req.successCh <- &runnerRef{
							llama: &mock,
						}
						return false
					},
				},
			}

			go s.sched.Run(t.Context())

			// Create a minimal model
			_, digest := createHarmonyTestModel(t)

			// Create model with passthrough template
			stream := false
			w := createRequest(t, s.CreateHandler, api.CreateRequest{
				Model:    "harmony-test",
				Files:    map[string]string{"file.gguf": digest},
				Template: `<|start|><|end|>{{ with .Tools }}{{ end }}{{ .Prompt }}`,
				Stream:   &stream,
			})

			if w.Code != http.StatusOK {
				t.Fatalf("failed to create model: %d", w.Code)
			}

			// Test chat endpoint with streaming
			streamTrue := true
			w = createRequest(t, s.ChatHandler, api.ChatRequest{
				Model:    "harmony-test",
				Messages: []api.Message{{Role: "user", Content: "Hello"}},
				Stream:   &streamTrue,
				Tools:    getTestTools(),
			})

			if w.Code != http.StatusOK {
				t.Fatalf("chat request failed: %d - %s", w.Code, w.Body.String())
			}

			// Parse streaming response
			var chunks []api.ChatResponse
			var content, thinking strings.Builder

			decoder := json.NewDecoder(w.Body)
			for decoder.More() {
				var chunk api.ChatResponse
				if err := decoder.Decode(&chunk); err != nil {
					t.Fatalf("failed to decode chunk: %v", err)
				}
				chunks = append(chunks, chunk)

				// Accumulate content and thinking from each chunk
				content.WriteString(chunk.Message.Content)
				thinking.WriteString(chunk.Message.Thinking)

				// Debug output
				t.Logf("Chunk %d: content=%q thinking=%q done=%v", len(chunks), chunk.Message.Content, chunk.Message.Thinking, chunk.Done)
			}

			// Verify we got streaming chunks
			if len(chunks) == 0 {
				t.Fatal("expected streaming chunks, got none")
			}

			gotContent := content.String()
			gotThinking := thinking.String()

			if gotContent != tc.wantContent {
				t.Errorf("content mismatch: got %q, want %q", gotContent, tc.wantContent)
			}
			if gotThinking != tc.wantThinking {
				t.Errorf("thinking mismatch: got %q, want %q", gotThinking, tc.wantThinking)
			}

			// Verify last chunk has done=true
			lastChunk := chunks[len(chunks)-1]
			if !lastChunk.Done {
				t.Error("expected last chunk to have done=true")
			}
		})
	}
}
