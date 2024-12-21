package server

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/llm"
)

type mockRunner struct {
	llm.LlamaServer

	// CompletionRequest is only valid until the next call to Completion
	llm.CompletionRequest
	llm.CompletionResponse
	CompletionFn func(context.Context, llm.CompletionRequest, func(llm.CompletionResponse)) error
}

func (m *mockRunner) Completion(ctx context.Context, r llm.CompletionRequest, fn func(r llm.CompletionResponse)) error {
	m.CompletionRequest = r
	if m.CompletionFn != nil {
		return m.CompletionFn(ctx, r, fn)
	}
	fn(m.CompletionResponse)
	return nil
}

func (mockRunner) Tokenize(_ context.Context, s string) (tokens []int, err error) {
	for range strings.Fields(s) {
		tokens = append(tokens, len(tokens))
	}

	return
}

func (mockRunner) Detokenize(_ context.Context, tokens []int) (string, error) {
	var strs []string
	for _, t := range tokens {
		strs = append(strs, fmt.Sprint(t))
	}
	return strings.Join(strs, " "), nil
}

func newMockServer(mock *mockRunner) func(discover.GpuInfoList, string, *llm.GGML, []string, []string, api.Options, int) (llm.LlamaServer, error) {
	return func(gpus discover.GpuInfoList, model string, ggml *llm.GGML, projectors, system []string, opts api.Options, numParallel int) (llm.LlamaServer, error) {
		return mock, nil
	}
}

func TestGenerateChat(t *testing.T) {
	gin.SetMode(gin.TestMode)

	mock := mockRunner{
		CompletionResponse: llm.CompletionResponse{
			Done:               true,
			DoneReason:         "stop",
			PromptEvalCount:    1,
			PromptEvalDuration: 1,
			EvalCount:          1,
			EvalDuration:       1,
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
			reschedDelay:  250 * time.Millisecond,
			loadFn: func(req *LlmRequest, ggml *llm.GGML, gpus discover.GpuInfoList, numParallel int) {
				// add small delay to simulate loading
				time.Sleep(time.Millisecond)
				req.successCh <- &runnerRef{
					llama: &mock,
				}
			},
		},
	}

	go s.sched.Run(context.TODO())

	w := createRequest(t, s.CreateHandler, api.CreateRequest{
		Model: "test",
		Modelfile: fmt.Sprintf(`FROM %s
		TEMPLATE """
{{- if .Tools }}
{{ .Tools }}
{{ end }}
{{- range .Messages }}
{{- .Role }}: {{ .Content }}
{{- range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{- end }}
{{ end }}"""
`, createBinFile(t, llm.KV{
			"general.architecture":          "llama",
			"llama.block_count":             uint32(1),
			"llama.context_length":          uint32(8192),
			"llama.embedding_length":        uint32(4096),
			"llama.attention.head_count":    uint32(32),
			"llama.attention.head_count_kv": uint32(8),
			"tokenizer.ggml.tokens":         []string{""},
			"tokenizer.ggml.scores":         []float32{0},
			"tokenizer.ggml.token_type":     []int32{0},
		}, []llm.Tensor{
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
		})),
		Stream: &stream,
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	t.Run("missing body", func(t *testing.T) {
		w := createRequest(t, s.ChatHandler, nil)
		if w.Code != http.StatusBadRequest {
			t.Errorf("expected status 400, got %d", w.Code)
		}

		if diff := cmp.Diff(w.Body.String(), `{"error":"model is required"}`); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}
	})

	t.Run("missing model", func(t *testing.T) {
		w := createRequest(t, s.ChatHandler, api.ChatRequest{})
		if w.Code != http.StatusBadRequest {
			t.Errorf("expected status 400, got %d", w.Code)
		}

		if diff := cmp.Diff(w.Body.String(), `{"error":"model is required"}`); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}
	})

	t.Run("missing capabilities chat", func(t *testing.T) {
		w := createRequest(t, s.CreateHandler, api.CreateRequest{
			Model: "bert",
			Modelfile: fmt.Sprintf("FROM %s", createBinFile(t, llm.KV{
				"general.architecture": "bert",
				"bert.pooling_type":    uint32(0),
			}, []llm.Tensor{})),
			Stream: &stream,
		})

		if w.Code != http.StatusOK {
			t.Fatalf("expected status 200, got %d", w.Code)
		}

		w = createRequest(t, s.ChatHandler, api.ChatRequest{
			Model: "bert",
		})

		if w.Code != http.StatusBadRequest {
			t.Errorf("expected status 400, got %d", w.Code)
		}

		if diff := cmp.Diff(w.Body.String(), `{"error":"\"bert\" does not support chat"}`); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}
	})

	t.Run("load model", func(t *testing.T) {
		w := createRequest(t, s.ChatHandler, api.ChatRequest{
			Model: "test",
		})

		if w.Code != http.StatusOK {
			t.Errorf("expected status 200, got %d", w.Code)
		}

		var actual api.ChatResponse
		if err := json.NewDecoder(w.Body).Decode(&actual); err != nil {
			t.Fatal(err)
		}

		if actual.Model != "test" {
			t.Errorf("expected model test, got %s", actual.Model)
		}

		if !actual.Done {
			t.Errorf("expected done true, got false")
		}

		if actual.DoneReason != "load" {
			t.Errorf("expected done reason load, got %s", actual.DoneReason)
		}
	})

	checkChatResponse := func(t *testing.T, body io.Reader, model, content string) {
		t.Helper()

		var actual api.ChatResponse
		if err := json.NewDecoder(body).Decode(&actual); err != nil {
			t.Fatal(err)
		}

		if actual.Model != model {
			t.Errorf("expected model test, got %s", actual.Model)
		}

		if !actual.Done {
			t.Errorf("expected done false, got true")
		}

		if actual.DoneReason != "stop" {
			t.Errorf("expected done reason stop, got %s", actual.DoneReason)
		}

		if diff := cmp.Diff(actual.Message, api.Message{
			Role:    "assistant",
			Content: content,
		}); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}

		if actual.PromptEvalCount == 0 {
			t.Errorf("expected prompt eval count > 0, got 0")
		}

		if actual.PromptEvalDuration == 0 {
			t.Errorf("expected prompt eval duration > 0, got 0")
		}

		if actual.EvalCount == 0 {
			t.Errorf("expected eval count > 0, got 0")
		}

		if actual.EvalDuration == 0 {
			t.Errorf("expected eval duration > 0, got 0")
		}

		if actual.LoadDuration == 0 {
			t.Errorf("expected load duration > 0, got 0")
		}

		if actual.TotalDuration == 0 {
			t.Errorf("expected total duration > 0, got 0")
		}
	}

	mock.CompletionResponse.Content = "Hi!"
	t.Run("messages", func(t *testing.T) {
		w := createRequest(t, s.ChatHandler, api.ChatRequest{
			Model: "test",
			Messages: []api.Message{
				{Role: "user", Content: "Hello!"},
			},
			Stream: &stream,
		})

		if w.Code != http.StatusOK {
			t.Errorf("expected status 200, got %d", w.Code)
		}

		if diff := cmp.Diff(mock.CompletionRequest.Prompt, "user: Hello!\n"); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}

		checkChatResponse(t, w.Body, "test", "Hi!")
	})

	w = createRequest(t, s.CreateHandler, api.CreateRequest{
		Model:     "test-system",
		Modelfile: "FROM test\nSYSTEM You are a helpful assistant.",
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	t.Run("messages with model system", func(t *testing.T) {
		w := createRequest(t, s.ChatHandler, api.ChatRequest{
			Model: "test-system",
			Messages: []api.Message{
				{Role: "user", Content: "Hello!"},
			},
			Stream: &stream,
		})

		if w.Code != http.StatusOK {
			t.Errorf("expected status 200, got %d", w.Code)
		}

		if diff := cmp.Diff(mock.CompletionRequest.Prompt, "system: You are a helpful assistant.\nuser: Hello!\n"); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}

		checkChatResponse(t, w.Body, "test-system", "Hi!")
	})

	mock.CompletionResponse.Content = "Abra kadabra!"
	t.Run("messages with system", func(t *testing.T) {
		w := createRequest(t, s.ChatHandler, api.ChatRequest{
			Model: "test-system",
			Messages: []api.Message{
				{Role: "system", Content: "You can perform magic tricks."},
				{Role: "user", Content: "Hello!"},
			},
			Stream: &stream,
		})

		if w.Code != http.StatusOK {
			t.Errorf("expected status 200, got %d", w.Code)
		}

		if diff := cmp.Diff(mock.CompletionRequest.Prompt, "system: You can perform magic tricks.\nuser: Hello!\n"); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}

		checkChatResponse(t, w.Body, "test-system", "Abra kadabra!")
	})

	t.Run("messages with interleaved system", func(t *testing.T) {
		w := createRequest(t, s.ChatHandler, api.ChatRequest{
			Model: "test-system",
			Messages: []api.Message{
				{Role: "user", Content: "Hello!"},
				{Role: "assistant", Content: "I can help you with that."},
				{Role: "system", Content: "You can perform magic tricks."},
				{Role: "user", Content: "Help me write tests."},
			},
			Stream: &stream,
		})

		if w.Code != http.StatusOK {
			t.Errorf("expected status 200, got %d", w.Code)
		}

		if diff := cmp.Diff(mock.CompletionRequest.Prompt, "system: You are a helpful assistant.\nuser: Hello!\nassistant: I can help you with that.\nsystem: You can perform magic tricks.\nuser: Help me write tests.\n"); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}

		checkChatResponse(t, w.Body, "test-system", "Abra kadabra!")
	})

	t.Run("messages with tools (non-streaming)", func(t *testing.T) {
		if w.Code != http.StatusOK {
			t.Fatalf("failed to create test-system model: %d", w.Code)
		}

		tools := []api.Tool{
			{
				Type: "function",
				Function: api.ToolFunction{
					Name:        "get_weather",
					Description: "Get the current weather",
					Parameters: struct {
						Type       string   `json:"type"`
						Required   []string `json:"required"`
						Properties map[string]struct {
							Type        string   `json:"type"`
							Description string   `json:"description"`
							Enum        []string `json:"enum,omitempty"`
						} `json:"properties"`
					}{
						Type:     "object",
						Required: []string{"location"},
						Properties: map[string]struct {
							Type        string   `json:"type"`
							Description string   `json:"description"`
							Enum        []string `json:"enum,omitempty"`
						}{
							"location": {
								Type:        "string",
								Description: "The city and state",
							},
							"unit": {
								Type: "string",
								Enum: []string{"celsius", "fahrenheit"},
							},
						},
					},
				},
			},
		}

		mock.CompletionResponse = llm.CompletionResponse{
			Content:            `{"name":"get_weather","arguments":{"location":"Seattle, WA","unit":"celsius"}}`,
			Done:               true,
			DoneReason:         "done",
			PromptEvalCount:    1,
			PromptEvalDuration: 1,
			EvalCount:          1,
			EvalDuration:       1,
		}

		streamRequest := true

		w := createRequest(t, s.ChatHandler, api.ChatRequest{
			Model: "test-system",
			Messages: []api.Message{
				{Role: "user", Content: "What's the weather in Seattle?"},
			},
			Tools:  tools,
			Stream: &streamRequest,
		})

		if w.Code != http.StatusOK {
			var errResp struct {
				Error string `json:"error"`
			}
			if err := json.NewDecoder(w.Body).Decode(&errResp); err != nil {
				t.Logf("Failed to decode error response: %v", err)
			} else {
				t.Logf("Error response: %s", errResp.Error)
			}
		}

		if w.Code != http.StatusOK {
			t.Errorf("expected status 200, got %d", w.Code)
		}

		var resp api.ChatResponse
		if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
			t.Fatal(err)
		}

		if resp.Message.ToolCalls == nil {
			t.Error("expected tool calls, got nil")
		}

		expectedToolCall := api.ToolCall{
			Function: api.ToolCallFunction{
				Name: "get_weather",
				Arguments: api.ToolCallFunctionArguments{
					"location": "Seattle, WA",
					"unit":     "celsius",
				},
			},
		}

		if diff := cmp.Diff(resp.Message.ToolCalls[0], expectedToolCall); diff != "" {
			t.Errorf("tool call mismatch (-got +want):\n%s", diff)
		}
	})

	t.Run("messages with tools (streaming)", func(t *testing.T) {
		tools := []api.Tool{
			{
				Type: "function",
				Function: api.ToolFunction{
					Name:        "get_weather",
					Description: "Get the current weather",
					Parameters: struct {
						Type       string   `json:"type"`
						Required   []string `json:"required"`
						Properties map[string]struct {
							Type        string   `json:"type"`
							Description string   `json:"description"`
							Enum        []string `json:"enum,omitempty"`
						} `json:"properties"`
					}{
						Type:     "object",
						Required: []string{"location"},
						Properties: map[string]struct {
							Type        string   `json:"type"`
							Description string   `json:"description"`
							Enum        []string `json:"enum,omitempty"`
						}{
							"location": {
								Type:        "string",
								Description: "The city and state",
							},
							"unit": {
								Type: "string",
								Enum: []string{"celsius", "fahrenheit"},
							},
						},
					},
				},
			},
		}

		// Simulate streaming response with multiple chunks
		var wg sync.WaitGroup
		wg.Add(1)

		mock.CompletionFn = func(ctx context.Context, r llm.CompletionRequest, fn func(r llm.CompletionResponse)) error {
			defer wg.Done()

			// Send chunks with small delays to simulate streaming
			responses := []llm.CompletionResponse{
				{
					Content:            `{"name":"get_`,
					Done:               false,
					PromptEvalCount:    1,
					PromptEvalDuration: 1,
				},
				{
					Content:            `weather","arguments":{"location":"Seattle`,
					Done:               false,
					PromptEvalCount:    2,
					PromptEvalDuration: 1,
				},
				{
					Content:            `, WA","unit":"celsius"}}`,
					Done:               true,
					DoneReason:         "tool_call",
					PromptEvalCount:    3,
					PromptEvalDuration: 1,
				},
			}

			for _, resp := range responses {
				select {
				case <-ctx.Done():
					return ctx.Err()
				default:
					fn(resp)
					time.Sleep(10 * time.Millisecond) // Small delay between chunks
				}
			}
			return nil
		}

		w := createRequest(t, s.ChatHandler, api.ChatRequest{
			Model: "test-system",
			Messages: []api.Message{
				{Role: "user", Content: "What's the weather in Seattle?"},
			},
			Tools:  tools,
			Stream: &stream,
		})

		wg.Wait()

		if w.Code != http.StatusOK {
			t.Errorf("expected status 200, got %d", w.Code)
		}

		// Read and validate the streamed responses
		decoder := json.NewDecoder(w.Body)
		var finalToolCall api.ToolCall

		for {
			var resp api.ChatResponse
			if err := decoder.Decode(&resp); err == io.EOF {
				break
			} else if err != nil {
				t.Fatal(err)
			}

			if resp.Done {
				if len(resp.Message.ToolCalls) != 1 {
					t.Errorf("expected 1 tool call in final response, got %d", len(resp.Message.ToolCalls))
				}
				finalToolCall = resp.Message.ToolCalls[0]
			}
		}

		expectedToolCall := api.ToolCall{
			Function: api.ToolCallFunction{
				Name: "get_weather",
				Arguments: api.ToolCallFunctionArguments{
					"location": "Seattle, WA",
					"unit":     "celsius",
				},
			},
		}

		if diff := cmp.Diff(finalToolCall, expectedToolCall); diff != "" {
			t.Errorf("final tool call mismatch (-got +want):\n%s", diff)
		}
	})
}

func TestGenerate(t *testing.T) {
	gin.SetMode(gin.TestMode)

	mock := mockRunner{
		CompletionResponse: llm.CompletionResponse{
			Done:               true,
			DoneReason:         "stop",
			PromptEvalCount:    1,
			PromptEvalDuration: 1,
			EvalCount:          1,
			EvalDuration:       1,
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
			reschedDelay:  250 * time.Millisecond,
			loadFn: func(req *LlmRequest, ggml *llm.GGML, gpus discover.GpuInfoList, numParallel int) {
				// add small delay to simulate loading
				time.Sleep(time.Millisecond)
				req.successCh <- &runnerRef{
					llama: &mock,
				}
			},
		},
	}

	go s.sched.Run(context.TODO())

	w := createRequest(t, s.CreateHandler, api.CreateRequest{
		Model: "test",
		Modelfile: fmt.Sprintf(`FROM %s
		TEMPLATE """
{{- if .System }}System: {{ .System }} {{ end }}
{{- if .Prompt }}User: {{ .Prompt }} {{ end }}
{{- if .Response }}Assistant: {{ .Response }} {{ end }}"""
`, createBinFile(t, llm.KV{
			"general.architecture":          "llama",
			"llama.block_count":             uint32(1),
			"llama.context_length":          uint32(8192),
			"llama.embedding_length":        uint32(4096),
			"llama.attention.head_count":    uint32(32),
			"llama.attention.head_count_kv": uint32(8),
			"tokenizer.ggml.tokens":         []string{""},
			"tokenizer.ggml.scores":         []float32{0},
			"tokenizer.ggml.token_type":     []int32{0},
		}, []llm.Tensor{
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
		})),
		Stream: &stream,
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	t.Run("missing body", func(t *testing.T) {
		w := createRequest(t, s.GenerateHandler, nil)
		if w.Code != http.StatusNotFound {
			t.Errorf("expected status 404, got %d", w.Code)
		}

		if diff := cmp.Diff(w.Body.String(), `{"error":"model '' not found"}`); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}
	})

	t.Run("missing model", func(t *testing.T) {
		w := createRequest(t, s.GenerateHandler, api.GenerateRequest{})
		if w.Code != http.StatusNotFound {
			t.Errorf("expected status 404, got %d", w.Code)
		}

		if diff := cmp.Diff(w.Body.String(), `{"error":"model '' not found"}`); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}
	})

	t.Run("missing capabilities generate", func(t *testing.T) {
		w := createRequest(t, s.CreateHandler, api.CreateRequest{
			Model: "bert",
			Modelfile: fmt.Sprintf("FROM %s", createBinFile(t, llm.KV{
				"general.architecture": "bert",
				"bert.pooling_type":    uint32(0),
			}, []llm.Tensor{})),
			Stream: &stream,
		})

		if w.Code != http.StatusOK {
			t.Fatalf("expected status 200, got %d", w.Code)
		}

		w = createRequest(t, s.GenerateHandler, api.GenerateRequest{
			Model: "bert",
		})

		if w.Code != http.StatusBadRequest {
			t.Errorf("expected status 400, got %d", w.Code)
		}

		if diff := cmp.Diff(w.Body.String(), `{"error":"\"bert\" does not support generate"}`); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}
	})

	t.Run("missing capabilities suffix", func(t *testing.T) {
		w := createRequest(t, s.GenerateHandler, api.GenerateRequest{
			Model:  "test",
			Prompt: "def add(",
			Suffix: "    return c",
		})

		if w.Code != http.StatusBadRequest {
			t.Errorf("expected status 400, got %d", w.Code)
		}

		if diff := cmp.Diff(w.Body.String(), `{"error":"registry.ollama.ai/library/test:latest does not support insert"}`); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}
	})

	t.Run("load model", func(t *testing.T) {
		w := createRequest(t, s.GenerateHandler, api.GenerateRequest{
			Model: "test",
		})

		if w.Code != http.StatusOK {
			t.Errorf("expected status 200, got %d", w.Code)
		}

		var actual api.GenerateResponse
		if err := json.NewDecoder(w.Body).Decode(&actual); err != nil {
			t.Fatal(err)
		}

		if actual.Model != "test" {
			t.Errorf("expected model test, got %s", actual.Model)
		}

		if !actual.Done {
			t.Errorf("expected done true, got false")
		}

		if actual.DoneReason != "load" {
			t.Errorf("expected done reason load, got %s", actual.DoneReason)
		}
	})

	checkGenerateResponse := func(t *testing.T, body io.Reader, model, content string) {
		t.Helper()

		var actual api.GenerateResponse
		if err := json.NewDecoder(body).Decode(&actual); err != nil {
			t.Fatal(err)
		}

		if actual.Model != model {
			t.Errorf("expected model test, got %s", actual.Model)
		}

		if !actual.Done {
			t.Errorf("expected done false, got true")
		}

		if actual.DoneReason != "stop" {
			t.Errorf("expected done reason stop, got %s", actual.DoneReason)
		}

		if actual.Response != content {
			t.Errorf("expected response %s, got %s", content, actual.Response)
		}

		if actual.Context == nil {
			t.Errorf("expected context not nil")
		}

		if actual.PromptEvalCount == 0 {
			t.Errorf("expected prompt eval count > 0, got 0")
		}

		if actual.PromptEvalDuration == 0 {
			t.Errorf("expected prompt eval duration > 0, got 0")
		}

		if actual.EvalCount == 0 {
			t.Errorf("expected eval count > 0, got 0")
		}

		if actual.EvalDuration == 0 {
			t.Errorf("expected eval duration > 0, got 0")
		}

		if actual.LoadDuration == 0 {
			t.Errorf("expected load duration > 0, got 0")
		}

		if actual.TotalDuration == 0 {
			t.Errorf("expected total duration > 0, got 0")
		}
	}

	mock.CompletionResponse.Content = "Hi!"
	t.Run("prompt", func(t *testing.T) {
		w := createRequest(t, s.GenerateHandler, api.GenerateRequest{
			Model:  "test",
			Prompt: "Hello!",
			Stream: &stream,
		})

		if w.Code != http.StatusOK {
			t.Errorf("expected status 200, got %d", w.Code)
		}

		if diff := cmp.Diff(mock.CompletionRequest.Prompt, "User: Hello! "); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}

		checkGenerateResponse(t, w.Body, "test", "Hi!")
	})

	w = createRequest(t, s.CreateHandler, api.CreateRequest{
		Model:     "test-system",
		Modelfile: "FROM test\nSYSTEM You are a helpful assistant.",
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	t.Run("prompt with model system", func(t *testing.T) {
		w := createRequest(t, s.GenerateHandler, api.GenerateRequest{
			Model:  "test-system",
			Prompt: "Hello!",
			Stream: &stream,
		})

		if w.Code != http.StatusOK {
			t.Errorf("expected status 200, got %d", w.Code)
		}

		if diff := cmp.Diff(mock.CompletionRequest.Prompt, "System: You are a helpful assistant. User: Hello! "); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}

		checkGenerateResponse(t, w.Body, "test-system", "Hi!")
	})

	mock.CompletionResponse.Content = "Abra kadabra!"
	t.Run("prompt with system", func(t *testing.T) {
		w := createRequest(t, s.GenerateHandler, api.GenerateRequest{
			Model:  "test-system",
			Prompt: "Hello!",
			System: "You can perform magic tricks.",
			Stream: &stream,
		})

		if w.Code != http.StatusOK {
			t.Errorf("expected status 200, got %d", w.Code)
		}

		if diff := cmp.Diff(mock.CompletionRequest.Prompt, "System: You can perform magic tricks. User: Hello! "); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}

		checkGenerateResponse(t, w.Body, "test-system", "Abra kadabra!")
	})

	t.Run("prompt with template", func(t *testing.T) {
		w := createRequest(t, s.GenerateHandler, api.GenerateRequest{
			Model:  "test-system",
			Prompt: "Help me write tests.",
			System: "You can perform magic tricks.",
			Template: `{{- if .System }}{{ .System }} {{ end }}
{{- if .Prompt }}### USER {{ .Prompt }} {{ end }}
{{- if .Response }}### ASSISTANT {{ .Response }} {{ end }}`,
			Stream: &stream,
		})

		if w.Code != http.StatusOK {
			t.Errorf("expected status 200, got %d", w.Code)
		}

		if diff := cmp.Diff(mock.CompletionRequest.Prompt, "You can perform magic tricks. ### USER Help me write tests. "); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}

		checkGenerateResponse(t, w.Body, "test-system", "Abra kadabra!")
	})

	w = createRequest(t, s.CreateHandler, api.CreateRequest{
		Model: "test-suffix",
		Modelfile: `FROM test
TEMPLATE """{{- if .Suffix }}<PRE> {{ .Prompt }} <SUF>{{ .Suffix }} <MID>
{{- else }}{{ .Prompt }}
{{- end }}"""`,
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	t.Run("prompt with suffix", func(t *testing.T) {
		w := createRequest(t, s.GenerateHandler, api.GenerateRequest{
			Model:  "test-suffix",
			Prompt: "def add(",
			Suffix: "    return c",
		})

		if w.Code != http.StatusOK {
			t.Errorf("expected status 200, got %d", w.Code)
		}

		if diff := cmp.Diff(mock.CompletionRequest.Prompt, "<PRE> def add( <SUF>    return c <MID>"); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}
	})

	t.Run("prompt without suffix", func(t *testing.T) {
		w := createRequest(t, s.GenerateHandler, api.GenerateRequest{
			Model:  "test-suffix",
			Prompt: "def add(",
		})

		if w.Code != http.StatusOK {
			t.Errorf("expected status 200, got %d", w.Code)
		}

		if diff := cmp.Diff(mock.CompletionRequest.Prompt, "def add("); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}
	})

	t.Run("raw", func(t *testing.T) {
		w := createRequest(t, s.GenerateHandler, api.GenerateRequest{
			Model:  "test-system",
			Prompt: "Help me write tests.",
			Raw:    true,
			Stream: &stream,
		})

		if w.Code != http.StatusOK {
			t.Errorf("expected status 200, got %d", w.Code)
		}

		if diff := cmp.Diff(mock.CompletionRequest.Prompt, "Help me write tests."); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}
	})
}
