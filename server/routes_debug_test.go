package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

func TestGenerateDebugRenderOnly(t *testing.T) {
	gin.SetMode(gin.TestMode)

	mock := mockRunner{
		CompletionResponse: llm.CompletionResponse{
			Done:               true,
			DoneReason:         llm.DoneReasonStop,
			PromptEvalCount:    1,
			PromptEvalDuration: 1,
			EvalCount:          1,
			EvalDuration:       1,
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
				// add small delay to simulate loading
				time.Sleep(time.Millisecond)
				req.successCh <- &runnerRef{
					llama: &mock,
				}
				return false
			},
		},
	}

	go s.sched.Run(t.Context())

	// Create a test model
	stream := false
	_, digest := createBinFile(t, ggml.KV{
		"general.architecture":          "llama",
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

	w := createRequest(t, s.CreateHandler, api.CreateRequest{
		Model:    "test-model",
		Files:    map[string]string{"file.gguf": digest},
		Template: "{{ .Prompt }}",
		Stream:   &stream,
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	tests := []struct {
		name            string
		request         api.GenerateRequest
		expectDebug     bool
		expectTemplate  string
		expectNumImages int
	}{
		{
			name: "debug render only enabled",
			request: api.GenerateRequest{
				Model:           "test-model",
				Prompt:          "Hello, world!",
				DebugRenderOnly: true,
			},
			expectDebug:    true,
			expectTemplate: "Hello, world!",
		},
		{
			name: "debug render only disabled",
			request: api.GenerateRequest{
				Model:           "test-model",
				Prompt:          "Hello, world!",
				DebugRenderOnly: false,
			},
			expectDebug: false,
		},
		{
			name: "debug render only with system prompt",
			request: api.GenerateRequest{
				Model:           "test-model",
				Prompt:          "User question",
				System:          "You are a helpful assistant",
				DebugRenderOnly: true,
			},
			expectDebug:    true,
			expectTemplate: "User question",
		},
		{
			name: "debug render only with template",
			request: api.GenerateRequest{
				Model:           "test-model",
				Prompt:          "Hello",
				Template:        "PROMPT: {{ .Prompt }}",
				DebugRenderOnly: true,
			},
			expectDebug:    true,
			expectTemplate: "PROMPT: Hello",
		},
		{
			name: "debug render only with images",
			request: api.GenerateRequest{
				Model:           "test-model",
				Prompt:          "Describe this image",
				Images:          []api.ImageData{[]byte("fake-image-data")},
				DebugRenderOnly: true,
			},
			expectDebug:     true,
			expectTemplate:  "[img-0]Describe this image",
			expectNumImages: 1,
		},
		{
			name: "debug render only with raw mode",
			request: api.GenerateRequest{
				Model:           "test-model",
				Prompt:          "Raw prompt text",
				Raw:             true,
				DebugRenderOnly: true,
			},
			expectDebug:    true,
			expectTemplate: "Raw prompt text",
		},
	}

	for _, tt := range tests {
		// Test both with and without streaming
		streamValues := []bool{false, true}
		for _, stream := range streamValues {
			streamSuffix := ""
			if stream {
				streamSuffix = " (streaming)"
			}
			t.Run(tt.name+streamSuffix, func(t *testing.T) {
				req := tt.request
				req.Stream = &stream
				w := createRequest(t, s.GenerateHandler, req)

				if tt.expectDebug {
					if w.Code != http.StatusOK {
						t.Errorf("expected status %d, got %d, body: %s", http.StatusOK, w.Code, w.Body.String())
					}

					var response api.GenerateResponse
					if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
						t.Fatalf("failed to unmarshal response: %v", err)
					}

					if response.Model != tt.request.Model {
						t.Errorf("expected model %s, got %s", tt.request.Model, response.Model)
					}

					if tt.expectTemplate != "" && response.DebugInfo.RenderedTemplate != tt.expectTemplate {
						t.Errorf("expected template %q, got %q", tt.expectTemplate, response.DebugInfo.RenderedTemplate)
					}

					if tt.expectNumImages > 0 && response.DebugInfo.ImageCount != tt.expectNumImages {
						t.Errorf("expected image count %d, got %d", tt.expectNumImages, response.DebugInfo.ImageCount)
					}
				} else {
					// When debug is disabled, it should attempt normal processing
					if w.Code != http.StatusOK {
						t.Errorf("expected status %d, got %d", http.StatusOK, w.Code)
					}
				}
			})
		}
	}
}

func TestChatDebugRenderOnly(t *testing.T) {
	gin.SetMode(gin.TestMode)

	mock := mockRunner{
		CompletionResponse: llm.CompletionResponse{
			Done:               true,
			DoneReason:         llm.DoneReasonStop,
			PromptEvalCount:    1,
			PromptEvalDuration: 1,
			EvalCount:          1,
			EvalDuration:       1,
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
				// add small delay to simulate loading
				time.Sleep(time.Millisecond)
				req.successCh <- &runnerRef{
					llama: &mock,
				}
				return false
			},
		},
	}

	go s.sched.Run(t.Context())

	// Create a test model
	stream := false
	_, digest := createBinFile(t, ggml.KV{
		"general.architecture":          "llama",
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

	w := createRequest(t, s.CreateHandler, api.CreateRequest{
		Model:    "test-model",
		Files:    map[string]string{"file.gguf": digest},
		Template: "{{ if .Tools }}{{ .Tools }}{{ end }}{{ range .Messages }}{{ .Role }}: {{ .Content }}\n{{ end }}",
		Stream:   &stream,
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	tests := []struct {
		name            string
		request         api.ChatRequest
		expectDebug     bool
		expectTemplate  string
		expectNumImages int
	}{
		{
			name: "chat debug render only enabled",
			request: api.ChatRequest{
				Model: "test-model",
				Messages: []api.Message{
					{Role: "system", Content: "You are a helpful assistant"},
					{Role: "user", Content: "Hello"},
				},
				DebugRenderOnly: true,
			},
			expectDebug:    true,
			expectTemplate: "system: You are a helpful assistant\nuser: Hello\n",
		},
		{
			name: "chat debug render only disabled",
			request: api.ChatRequest{
				Model: "test-model",
				Messages: []api.Message{
					{Role: "user", Content: "Hello"},
				},
				DebugRenderOnly: false,
			},
			expectDebug: false,
		},
		{
			name: "chat debug with assistant message",
			request: api.ChatRequest{
				Model: "test-model",
				Messages: []api.Message{
					{Role: "user", Content: "Hello"},
					{Role: "assistant", Content: "Hi there!"},
					{Role: "user", Content: "How are you?"},
				},
				DebugRenderOnly: true,
			},
			expectDebug:    true,
			expectTemplate: "user: Hello\nassistant: Hi there!\nuser: How are you?\n",
		},
		{
			name: "chat debug with images",
			request: api.ChatRequest{
				Model: "test-model",
				Messages: []api.Message{
					{
						Role:    "user",
						Content: "What's in this image?",
						Images:  []api.ImageData{[]byte("fake-image-data")},
					},
				},
				DebugRenderOnly: true,
			},
			expectDebug:     true,
			expectTemplate:  "user: [img-0]What's in this image?\n",
			expectNumImages: 1,
		},
		{
			name: "chat debug with tools",
			request: api.ChatRequest{
				Model: "test-model",
				Messages: []api.Message{
					{Role: "user", Content: "Get the weather"},
				},
				Tools: api.Tools{
					{
						Type: "function",
						Function: api.ToolFunction{
							Name:        "get_weather",
							Description: "Get weather information",
						},
					},
				},
				DebugRenderOnly: true,
			},
			expectDebug:    true,
			expectTemplate: "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get weather information\",\"parameters\":{\"type\":\"\",\"properties\":null}}}]user: Get the weather\n",
		},
	}

	for _, tt := range tests {
		// Test both with and without streaming
		streamValues := []bool{false, true}
		for _, stream := range streamValues {
			streamSuffix := ""
			if stream {
				streamSuffix = " (streaming)"
			}
			t.Run(tt.name+streamSuffix, func(t *testing.T) {
				req := tt.request
				req.Stream = &stream
				w := createRequest(t, s.ChatHandler, req)

				if tt.expectDebug {
					if w.Code != http.StatusOK {
						t.Errorf("expected status %d, got %d, body: %s", http.StatusOK, w.Code, w.Body.String())
					}

					var response api.ChatResponse
					if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
						t.Fatalf("failed to unmarshal response: %v", err)
					}

					if response.Model != tt.request.Model {
						t.Errorf("expected model %s, got %s", tt.request.Model, response.Model)
					}

					if tt.expectTemplate != "" && response.DebugInfo.RenderedTemplate != tt.expectTemplate {
						t.Errorf("expected template %q, got %q", tt.expectTemplate, response.DebugInfo.RenderedTemplate)
					}

					if tt.expectNumImages > 0 && response.DebugInfo.ImageCount != tt.expectNumImages {
						t.Errorf("expected image count %d, got %d", tt.expectNumImages, response.DebugInfo.ImageCount)
					}
				} else {
					// When debug is disabled, it should attempt normal processing
					if w.Code != http.StatusOK {
						t.Errorf("expected status %d, got %d", http.StatusOK, w.Code)
					}
				}
			})
		}
	}
}
