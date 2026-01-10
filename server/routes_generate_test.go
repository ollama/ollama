package server

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

// testPropsMap creates a ToolPropertiesMap from a map (convenience function for tests)
func testPropsMap(m map[string]api.ToolProperty) *api.ToolPropertiesMap {
	props := api.NewToolPropertiesMap()
	for k, v := range m {
		props.Set(k, v)
	}
	return props
}

// testArgs creates ToolCallFunctionArguments from a map (convenience function for tests)
func testArgs(m map[string]any) api.ToolCallFunctionArguments {
	args := api.NewToolCallFunctionArguments()
	for k, v := range m {
		args.Set(k, v)
	}
	return args
}

// argsComparer provides cmp options for comparing ToolCallFunctionArguments by value
var argsComparer = cmp.Comparer(func(a, b api.ToolCallFunctionArguments) bool {
	return cmp.Equal(a.ToMap(), b.ToMap())
})

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

func newMockServer(mock *mockRunner) func(ml.SystemInfo, []ml.DeviceInfo, string, *ggml.GGML, []string, []string, api.Options, int) (llm.LlamaServer, error) {
	return func(_ ml.SystemInfo, _ []ml.DeviceInfo, _ string, _ *ggml.GGML, _, _ []string, _ api.Options, _ int) (llm.LlamaServer, error) {
		return mock, nil
	}
}

func TestGenerateChatRemote(t *testing.T) {
	gin.SetMode(gin.TestMode)

	rs := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("Expected POST request, got %s", r.Method)
		}
		if r.URL.Path != "/api/chat" {
			t.Errorf("Expected path '/api/chat', got %s", r.URL.Path)
		}

		w.WriteHeader(http.StatusOK)
		w.Header().Set("Content-Type", "application/json")
		resp := api.ChatResponse{
			Model:      "test",
			Done:       true,
			DoneReason: "load",
		}
		if err := json.NewEncoder(w).Encode(&resp); err != nil {
			t.Fatal(err)
		}
	}))
	defer rs.Close()

	p, err := url.Parse(rs.URL)
	if err != nil {
		t.Fatal(err)
	}

	t.Setenv("OLLAMA_REMOTES", p.Hostname())
	s := Server{}
	w := createRequest(t, s.CreateHandler, api.CreateRequest{
		Model:      "test-cloud",
		RemoteHost: rs.URL,
		From:       "test",
		Info: map[string]any{
			"capabilities": []string{"completion", "thinking"},
		},
		Stream: &stream,
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	t.Run("missing messages", func(t *testing.T) {
		w := createRequest(t, s.ChatHandler, api.ChatRequest{
			Model: "test-cloud",
		})
		if w.Code != http.StatusOK {
			t.Errorf("expected status 200, got %d", w.Code)
		}

		var actual api.ChatResponse
		if err := json.NewDecoder(w.Body).Decode(&actual); err != nil {
			t.Fatal(err)
		}

		if actual.Model != "test-cloud" {
			t.Errorf("expected model test-cloud, got %s", actual.Model)
		}

		if actual.RemoteModel != "test" {
			t.Errorf("expected remote model test, got %s", actual.RemoteModel)
		}

		if actual.RemoteHost != rs.URL {
			t.Errorf("expected remote host '%s', got %s", rs.URL, actual.RemoteHost)
		}

		if !actual.Done {
			t.Errorf("expected done true, got false")
		}

		if actual.DoneReason != "load" {
			t.Errorf("expected done reason load, got %s", actual.DoneReason)
		}
	})
}

func TestGenerateChat(t *testing.T) {
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
		Model: "test",
		Files: map[string]string{"file.gguf": digest},
		Template: `
{{- if .Tools }}
{{ .Tools }}
{{ end }}
{{- range .Messages }}
{{- .Role }}: {{ .Content }}
{{- range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{- end }}
{{ end }}`,
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

	t.Run("missing thinking capability", func(t *testing.T) {
		think := true
		w := createRequest(t, s.ChatHandler, api.ChatRequest{
			Model: "test",
			Messages: []api.Message{
				{Role: "user", Content: "Hello!"},
			},
			Think: &api.ThinkValue{Value: think},
		})

		if w.Code != http.StatusBadRequest {
			t.Errorf("expected status 400, got %d", w.Code)
		}

		if diff := cmp.Diff(w.Body.String(), `{"error":"\"test\" does not support thinking"}`); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}
	})

	t.Run("model can't think but think set false", func(t *testing.T) {
		think := false
		w := createRequest(t, s.ChatHandler, api.ChatRequest{
			Model: "test",
			Messages: []api.Message{
				{Role: "user", Content: "Hello!"},
			},
			Think: &api.ThinkValue{Value: think},
		})

		if w.Code != http.StatusOK {
			t.Errorf("expected status 200, got %d", w.Code)
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
		_, digest := createBinFile(t, ggml.KV{
			"general.architecture": "bert",
			"bert.pooling_type":    uint32(0),
		}, []*ggml.Tensor{})
		w := createRequest(t, s.CreateHandler, api.CreateRequest{
			Model:  "bert",
			Files:  map[string]string{"bert.gguf": digest},
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
		Model:  "test-system",
		From:   "test",
		System: "You are a helpful assistant.",
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
					Parameters: api.ToolFunctionParameters{
						Type:     "object",
						Required: []string{"location"},
						Properties: testPropsMap(map[string]api.ToolProperty{
							"location": {
								Type:        api.PropertyType{"string"},
								Description: "The city and state",
							},
							"unit": {
								Type: api.PropertyType{"string"},
								Enum: []any{"celsius", "fahrenheit"},
							},
						}),
					},
				},
			},
		}

		mock.CompletionResponse = llm.CompletionResponse{
			Content:            `{"name":"get_weather","arguments":{"location":"Seattle, WA","unit":"celsius"}}`,
			Done:               true,
			DoneReason:         llm.DoneReasonStop,
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

		gotToolCall := resp.Message.ToolCalls[0]
		if gotToolCall.ID == "" {
			t.Error("expected tool call ID to be populated")
		}
		if !strings.HasPrefix(gotToolCall.ID, "call_") {
			t.Errorf("expected tool call ID to have call_ prefix, got %q", gotToolCall.ID)
		}

		expectedToolCall := api.ToolCall{
			Function: api.ToolCallFunction{
				Name: "get_weather",
				Arguments: testArgs(map[string]any{
					"location": "Seattle, WA",
					"unit":     "celsius",
				}),
			},
		}

		expectedToolCall.ID = gotToolCall.ID
		if diff := cmp.Diff(gotToolCall, expectedToolCall, argsComparer); diff != "" {
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
					Parameters: api.ToolFunctionParameters{
						Type:     "object",
						Required: []string{"location"},
						Properties: testPropsMap(map[string]api.ToolProperty{
							"location": {
								Type:        api.PropertyType{"string"},
								Description: "The city and state",
							},
							"unit": {
								Type: api.PropertyType{"string"},
								Enum: []any{"celsius", "fahrenheit"},
							},
						}),
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
					DoneReason:         llm.DoneReasonStop,
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

			if len(resp.Message.ToolCalls) > 0 {
				for _, call := range resp.Message.ToolCalls {
					if call.ID == "" {
						t.Fatal("expected streaming tool call to have an ID")
					}
					if !strings.HasPrefix(call.ID, "call_") {
						t.Fatalf("expected streaming tool call ID to have call_ prefix, got %q", call.ID)
					}
				}
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
				Arguments: testArgs(map[string]any{
					"location": "Seattle, WA",
					"unit":     "celsius",
				}),
			},
		}

		if finalToolCall.ID == "" {
			t.Fatal("expected final tool call to have an ID")
		}
		if !strings.HasPrefix(finalToolCall.ID, "call_") {
			t.Fatalf("expected final tool call ID to have call_ prefix, got %q", finalToolCall.ID)
		}

		expectedToolCall.ID = finalToolCall.ID
		if diff := cmp.Diff(finalToolCall, expectedToolCall, argsComparer); diff != "" {
			t.Errorf("final tool call mismatch (-got +want):\n%s", diff)
		}
	})

	t.Run("messages with tools and logprobs (streaming)", func(t *testing.T) {
		tools := []api.Tool{
			{
				Type: "function",
				Function: api.ToolFunction{
					Name: "get_weather",
					Parameters: api.ToolFunctionParameters{
						Type: "object",
						Properties: testPropsMap(map[string]api.ToolProperty{
							"location": {Type: api.PropertyType{"string"}},
						}),
					},
				},
			},
		}

		var wg sync.WaitGroup
		wg.Add(1)

		mock.CompletionFn = func(ctx context.Context, r llm.CompletionRequest, fn func(r llm.CompletionResponse)) error {
			defer wg.Done()

			// Simulate a response where logprobs are sent while the tool call is being buffered
			responses := []llm.CompletionResponse{
				{
					Content:  `{ "name": "get_weather"`,
					Done:     false,
					Logprobs: []llm.Logprob{{}},
				},
				{
					Content:  `,"arguments":{"location":"Seattle, WA","unit":"celsius"}}`,
					Done:     false,
					Logprobs: []llm.Logprob{{}},
				},
				{
					Content:    ``,
					Done:       true,
					DoneReason: llm.DoneReasonStop,
					Logprobs:   nil,
				},
			}

			for _, resp := range responses {
				select {
				case <-ctx.Done():
					return ctx.Err()
				default:
					fn(resp)
					time.Sleep(10 * time.Millisecond)
				}
			}
			return nil
		}

		w := createRequest(t, s.ChatHandler, api.ChatRequest{
			Model: "test-system",
			Messages: []api.Message{
				{Role: "user", Content: "Weather?"},
			},
			Tools:  tools,
			Stream: &stream,
		})

		wg.Wait()

		if w.Code != http.StatusOK {
			t.Errorf("expected status 200, got %d", w.Code)
		}

		decoder := json.NewDecoder(w.Body)
		var totalLogprobs int

		for {
			var resp api.ChatResponse
			if err := decoder.Decode(&resp); err == io.EOF {
				break
			} else if err != nil {
				t.Fatal(err)
			}

			totalLogprobs += len(resp.Logprobs)
		}

		expectedLogprobs := 2
		if totalLogprobs != expectedLogprobs {
			t.Errorf("expected %d logprobs, got %d", expectedLogprobs, totalLogprobs)
		}
	})

	t.Run("status error non-streaming", func(t *testing.T) {
		mock.CompletionFn = func(ctx context.Context, r llm.CompletionRequest, fn func(r llm.CompletionResponse)) error {
			return api.StatusError{
				StatusCode:   http.StatusServiceUnavailable,
				Status:       "Service Unavailable",
				ErrorMessage: "model is overloaded",
			}
		}

		stream := false
		w := createRequest(t, s.ChatHandler, api.ChatRequest{
			Model: "test",
			Messages: []api.Message{
				{Role: "user", Content: "Hello!"},
			},
			Stream: &stream,
		})

		if w.Code != http.StatusServiceUnavailable {
			t.Errorf("expected status 503, got %d", w.Code)
		}

		if diff := cmp.Diff(w.Body.String(), `{"error":"model is overloaded"}`); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}
	})

	t.Run("status error streaming", func(t *testing.T) {
		mock.CompletionFn = func(ctx context.Context, r llm.CompletionRequest, fn func(r llm.CompletionResponse)) error {
			return api.StatusError{
				StatusCode:   http.StatusTooManyRequests,
				Status:       "Too Many Requests",
				ErrorMessage: "rate limit exceeded",
			}
		}

		w := createRequest(t, s.ChatHandler, api.ChatRequest{
			Model: "test",
			Messages: []api.Message{
				{Role: "user", Content: "Hello!"},
			},
		})

		if w.Code != http.StatusTooManyRequests {
			t.Errorf("expected status 429, got %d", w.Code)
		}

		if diff := cmp.Diff(w.Body.String(), `{"error":"rate limit exceeded"}`); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}
	})
}

func TestGenerate(t *testing.T) {
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
		Model: "test",
		Files: map[string]string{"file.gguf": digest},
		Template: `
{{- if .System }}System: {{ .System }} {{ end }}
{{- if .Prompt }}User: {{ .Prompt }} {{ end }}
{{- if .Response }}Assistant: {{ .Response }} {{ end }}
`,
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
		_, digest := createBinFile(t, ggml.KV{
			"general.architecture": "bert",
			"bert.pooling_type":    uint32(0),
		}, []*ggml.Tensor{})

		w := createRequest(t, s.CreateHandler, api.CreateRequest{
			Model:  "bert",
			Files:  map[string]string{"file.gguf": digest},
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
		Model:  "test-system",
		From:   "test",
		System: "You are a helpful assistant.",
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
		Template: `{{- if .Suffix }}<PRE> {{ .Prompt }} <SUF>{{ .Suffix }} <MID>
{{- else }}{{ .Prompt }}
{{- end }}`,
		From: "test",
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

	t.Run("status error non-streaming", func(t *testing.T) {
		mock.CompletionFn = func(ctx context.Context, r llm.CompletionRequest, fn func(r llm.CompletionResponse)) error {
			return api.StatusError{
				StatusCode:   http.StatusServiceUnavailable,
				Status:       "Service Unavailable",
				ErrorMessage: "model is overloaded",
			}
		}

		streamRequest := false
		w := createRequest(t, s.GenerateHandler, api.GenerateRequest{
			Model:  "test",
			Prompt: "Hello!",
			Stream: &streamRequest,
		})

		if w.Code != http.StatusServiceUnavailable {
			t.Errorf("expected status 503, got %d", w.Code)
		}

		if diff := cmp.Diff(w.Body.String(), `{"error":"model is overloaded"}`); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}
	})

	t.Run("status error streaming", func(t *testing.T) {
		mock.CompletionFn = func(ctx context.Context, r llm.CompletionRequest, fn func(r llm.CompletionResponse)) error {
			return api.StatusError{
				StatusCode:   http.StatusTooManyRequests,
				Status:       "Too Many Requests",
				ErrorMessage: "rate limit exceeded",
			}
		}

		w := createRequest(t, s.GenerateHandler, api.GenerateRequest{
			Model:  "test",
			Prompt: "Hello!",
			Stream: &stream,
		})

		if w.Code != http.StatusTooManyRequests {
			t.Errorf("expected status 429, got %d", w.Code)
		}

		if diff := cmp.Diff(w.Body.String(), `{"error":"rate limit exceeded"}`); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}
	})
}

func TestGenerateLogprobs(t *testing.T) {
	t.Run("invalid top_logprobs negative", func(t *testing.T) {
		gin.SetMode(gin.TestMode)
		s := Server{}
		w := createRequest(t, s.GenerateHandler, api.GenerateRequest{
			Model:       "test",
			Prompt:      "Hello",
			TopLogprobs: -1,
		})

		if w.Code != http.StatusBadRequest {
			t.Errorf("expected status 400, got %d", w.Code)
		}

		if diff := cmp.Diff(w.Body.String(), `{"error":"top_logprobs must be between 0 and 20"}`); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}
	})

	t.Run("invalid top_logprobs too high", func(t *testing.T) {
		gin.SetMode(gin.TestMode)
		s := Server{}
		w := createRequest(t, s.GenerateHandler, api.GenerateRequest{
			Model:       "test",
			Prompt:      "Hello",
			TopLogprobs: 21,
		})

		if w.Code != http.StatusBadRequest {
			t.Errorf("expected status 400, got %d", w.Code)
		}

		if diff := cmp.Diff(w.Body.String(), `{"error":"top_logprobs must be between 0 and 20"}`); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}
	})

	t.Run("returns logprob bytes when requested", func(t *testing.T) {
		gin.SetMode(gin.TestMode)

		mock := &mockRunner{}
		expectedPrimary := llm.TokenLogprob{
			Token:   "Hi",
			Logprob: -0.01,
		}
		expectedAlternatives := []llm.TokenLogprob{
			{
				Token:   "Hello",
				Logprob: -0.25,
			},
			{
				Token:   "Hey",
				Logprob: -0.5,
			},
		}

		mock.CompletionFn = func(ctx context.Context, r llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
			fn(llm.CompletionResponse{
				Content:            "Hi",
				Done:               true,
				DoneReason:         llm.DoneReasonStop,
				PromptEvalCount:    1,
				PromptEvalDuration: 1,
				EvalCount:          1,
				EvalDuration:       1,
				Logprobs: []llm.Logprob{
					{
						TokenLogprob: expectedPrimary,
						TopLogprobs:  expectedAlternatives,
					},
				},
			})
			return nil
		}

		s := &Server{
			sched: &Scheduler{
				pendingReqCh:    make(chan *LlmRequest, 1),
				finishedReqCh:   make(chan *LlmRequest, 1),
				expiredCh:       make(chan *runnerRef, 1),
				unloadedCh:      make(chan any, 1),
				loaded:          make(map[string]*runnerRef),
				newServerFn:     newMockServer(mock),
				getGpuFn:        getGpuFn,
				getSystemInfoFn: getSystemInfoFn,
				waitForRecovery: 250 * time.Millisecond,
				loadFn: func(req *LlmRequest, _ *ggml.GGML, _ ml.SystemInfo, _ []ml.DeviceInfo, _ bool) bool {
					req.successCh <- &runnerRef{llama: mock}
					return false
				},
			},
		}

		go s.sched.Run(t.Context())

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

		if w := createRequest(t, s.CreateHandler, api.CreateRequest{
			Model:    "test-logprob-bytes",
			Files:    map[string]string{"file.gguf": digest},
			Template: `{{ .Prompt }}`,
			Stream:   &stream,
		}); w.Code != http.StatusOK {
			t.Fatalf("expected status 200, got %d", w.Code)
		}

		stream := false
		w := createRequest(t, s.GenerateHandler, api.GenerateRequest{
			Model:       "test-logprob-bytes",
			Prompt:      "Hi",
			Stream:      &stream,
			Logprobs:    true,
			TopLogprobs: len(expectedAlternatives),
		})

		if w.Code != http.StatusOK {
			t.Fatalf("expected status 200, got %d", w.Code)
		}

		var resp api.GenerateResponse
		if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
			t.Fatalf("failed to decode response: %v", err)
		}

		if len(resp.Logprobs) != 1 {
			t.Fatalf("expected 1 logprob entry, got %d", len(resp.Logprobs))
		}

		expectedPrimaryBytes := stringToByteInts(expectedPrimary.Token)
		expectedAlternativesBytes := make([][]int, len(expectedAlternatives))
		for i, alternative := range expectedAlternatives {
			expectedAlternativesBytes[i] = stringToByteInts(alternative.Token)
		}
		if diff := cmp.Diff(expectedPrimaryBytes, resp.Logprobs[0].Bytes); diff != "" {
			t.Fatalf("primary token bytes mismatch (-want +got):\n%s", diff)
		}

		if len(resp.Logprobs[0].TopLogprobs) != len(expectedAlternatives) {
			t.Fatalf("expected %d top logprobs, got %d", len(expectedAlternatives), len(resp.Logprobs[0].TopLogprobs))
		}

		for i, top := range resp.Logprobs[0].TopLogprobs {
			if diff := cmp.Diff(expectedAlternativesBytes[i], top.Bytes); diff != "" {
				t.Fatalf("top logprob[%d] bytes mismatch (-want +got):\n%s", i, diff)
			}
		}
	})
}

func TestChatLogprobs(t *testing.T) {
	t.Run("invalid top_logprobs negative", func(t *testing.T) {
		gin.SetMode(gin.TestMode)
		s := Server{}
		w := createRequest(t, s.ChatHandler, api.ChatRequest{
			Model: "test",
			Messages: []api.Message{
				{Role: "user", Content: "Hello"},
			},
			TopLogprobs: -1,
		})

		if w.Code != http.StatusBadRequest {
			t.Errorf("expected status 400, got %d", w.Code)
		}

		if diff := cmp.Diff(w.Body.String(), `{"error":"top_logprobs must be between 0 and 20"}`); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}
	})

	t.Run("invalid top_logprobs too high", func(t *testing.T) {
		gin.SetMode(gin.TestMode)
		s := Server{}
		w := createRequest(t, s.ChatHandler, api.ChatRequest{
			Model: "test",
			Messages: []api.Message{
				{Role: "user", Content: "Hello"},
			},
			TopLogprobs: 21,
		})

		if w.Code != http.StatusBadRequest {
			t.Errorf("expected status 400, got %d", w.Code)
		}

		if diff := cmp.Diff(w.Body.String(), `{"error":"top_logprobs must be between 0 and 20"}`); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}
	})

	t.Run("returns logprob bytes when requested", func(t *testing.T) {
		gin.SetMode(gin.TestMode)

		mock := &mockRunner{}
		expectedPrimary := llm.TokenLogprob{
			Token:   "Hi",
			Logprob: -0.02,
		}
		expectedAlternatives := []llm.TokenLogprob{
			{
				Token:   "Hello",
				Logprob: -0.3,
			},
			{
				Token:   "Hey",
				Logprob: -0.45,
			},
		}
		expectedPrimaryBytes := stringToByteInts(expectedPrimary.Token)
		expectedAlternativesBytes := make([][]int, len(expectedAlternatives))
		for i, alternative := range expectedAlternatives {
			expectedAlternativesBytes[i] = stringToByteInts(alternative.Token)
		}

		mock.CompletionFn = func(ctx context.Context, r llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
			fn(llm.CompletionResponse{
				Content:            "Hi",
				Done:               true,
				DoneReason:         llm.DoneReasonStop,
				PromptEvalCount:    1,
				PromptEvalDuration: 1,
				EvalCount:          1,
				EvalDuration:       1,
				Logprobs: []llm.Logprob{
					{
						TokenLogprob: expectedPrimary,
						TopLogprobs:  expectedAlternatives,
					},
				},
			})
			return nil
		}

		s := &Server{
			sched: &Scheduler{
				pendingReqCh:    make(chan *LlmRequest, 1),
				finishedReqCh:   make(chan *LlmRequest, 1),
				expiredCh:       make(chan *runnerRef, 1),
				unloadedCh:      make(chan any, 1),
				loaded:          make(map[string]*runnerRef),
				newServerFn:     newMockServer(mock),
				getGpuFn:        getGpuFn,
				getSystemInfoFn: getSystemInfoFn,
				waitForRecovery: 250 * time.Millisecond,
				loadFn: func(req *LlmRequest, _ *ggml.GGML, _ ml.SystemInfo, _ []ml.DeviceInfo, _ bool) bool {
					req.successCh <- &runnerRef{llama: mock}
					return false
				},
			},
		}

		go s.sched.Run(t.Context())

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

		if w := createRequest(t, s.CreateHandler, api.CreateRequest{
			Model: "test-chat-logprob-bytes",
			Files: map[string]string{"file.gguf": digest},
			Template: `{{- range .Messages }}{{ .Role }}: {{ .Content }}
{{ end }}`,
			Stream: &stream,
		}); w.Code != http.StatusOK {
			t.Fatalf("expected status 200, got %d", w.Code)
		}

		stream := false
		w := createRequest(t, s.ChatHandler, api.ChatRequest{
			Model: "test-chat-logprob-bytes",
			Messages: []api.Message{
				{Role: "user", Content: "Say hi"},
			},
			Stream:      &stream,
			Logprobs:    true,
			TopLogprobs: len(expectedAlternatives),
		})

		if w.Code != http.StatusOK {
			t.Fatalf("expected status 200, got %d", w.Code)
		}

		var resp api.ChatResponse
		if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
			t.Fatalf("failed to decode response: %v", err)
		}

		if len(resp.Logprobs) != 1 {
			t.Fatalf("expected 1 logprob entry, got %d", len(resp.Logprobs))
		}

		if diff := cmp.Diff(expectedPrimaryBytes, resp.Logprobs[0].Bytes); diff != "" {
			t.Fatalf("primary token bytes mismatch (-want +got):\n%s", diff)
		}

		if len(resp.Logprobs[0].TopLogprobs) != len(expectedAlternatives) {
			t.Fatalf("expected %d top logprobs, got %d", len(expectedAlternatives), len(resp.Logprobs[0].TopLogprobs))
		}

		for i, top := range resp.Logprobs[0].TopLogprobs {
			if diff := cmp.Diff(expectedAlternativesBytes[i], top.Bytes); diff != "" {
				t.Fatalf("top logprob[%d] bytes mismatch (-want +got):\n%s", i, diff)
			}
		}
	})
}

func TestChatWithPromptEndingInThinkTag(t *testing.T) {
	gin.SetMode(gin.TestMode)

	// Helper to create a standard thinking test setup
	setupThinkingTest := func(t *testing.T) (*mockRunner, *Server) {
		mock := &mockRunner{
			CompletionResponse: llm.CompletionResponse{
				Done:               true,
				DoneReason:         llm.DoneReasonStop,
				PromptEvalCount:    1,
				PromptEvalDuration: 1,
				EvalCount:          1,
				EvalDuration:       1,
			},
		}

		s := &Server{
			sched: &Scheduler{
				pendingReqCh:    make(chan *LlmRequest, 1),
				finishedReqCh:   make(chan *LlmRequest, 1),
				expiredCh:       make(chan *runnerRef, 1),
				unloadedCh:      make(chan any, 1),
				loaded:          make(map[string]*runnerRef),
				newServerFn:     newMockServer(mock),
				getGpuFn:        getGpuFn,
				getSystemInfoFn: getSystemInfoFn,
				waitForRecovery: 250 * time.Millisecond,
				loadFn: func(req *LlmRequest, _ *ggml.GGML, _ ml.SystemInfo, _ []ml.DeviceInfo, _ bool) bool {
					time.Sleep(time.Millisecond)
					req.successCh <- &runnerRef{llama: mock}
					return false
				},
			},
		}

		go s.sched.Run(t.Context())

		// Create a model with thinking support
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

		// Create model with thinking template that adds <think> at the end
		w := createRequest(t, s.CreateHandler, api.CreateRequest{
			Model: "test-thinking",
			Files: map[string]string{"file.gguf": digest},
			Template: `{{- range .Messages }}
{{- if eq .Role "user" }}user: {{ .Content }}
{{ else if eq .Role "assistant" }}assistant: {{ if .Thinking }}<think>{{ .Thinking }}</think>{{ end }}{{ .Content }}
{{ end }}{{ end }}<think>`,
			Stream: &stream,
		})

		if w.Code != http.StatusOK {
			t.Fatalf("expected status 200, got %d", w.Code)
		}

		return mock, s
	}

	mock, s := setupThinkingTest(t)

	// Helper to test chat responses
	testChatRequest := func(t *testing.T, name string, userContent string, modelResponse string, expectedThinking string, expectedContent string, think bool) {
		t.Run(name, func(t *testing.T) {
			mock.CompletionResponse = llm.CompletionResponse{
				Content:            modelResponse,
				Done:               true,
				DoneReason:         llm.DoneReasonStop,
				PromptEvalCount:    1,
				PromptEvalDuration: 1,
				EvalCount:          1,
				EvalDuration:       1,
			}
			mock.CompletionFn = nil

			streamRequest := false
			req := api.ChatRequest{
				Model: "test-thinking",
				Messages: []api.Message{
					{Role: "user", Content: userContent},
				},
				Stream: &streamRequest,
			}
			if think {
				req.Think = &api.ThinkValue{Value: think}
			}

			w := createRequest(t, s.ChatHandler, req)
			if w.Code != http.StatusOK {
				t.Fatalf("expected status 200, got %d", w.Code)
			}

			var resp api.ChatResponse
			if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
				t.Fatal(err)
			}

			if resp.Message.Thinking != expectedThinking {
				t.Errorf("expected thinking %q, got %q", expectedThinking, resp.Message.Thinking)
			}

			if resp.Message.Content != expectedContent {
				t.Errorf("expected content %q, got %q", expectedContent, resp.Message.Content)
			}
		})
	}

	// Test cases - Note: Template adds <think> at the end, and leading whitespace after <think> is eaten by the parser
	testChatRequest(t, "basic thinking response",
		"Help me solve this problem",
		" Let me think about this step by step... </think> The answer is 42.",
		"Let me think about this step by step... ",
		"The answer is 42.",
		true)

	testChatRequest(t, "thinking with multiple sentences",
		"Explain quantum computing",
		" First, I need to understand the basics. Quantum bits can be in superposition. </think> Quantum computing uses quantum mechanics principles.",
		"First, I need to understand the basics. Quantum bits can be in superposition. ",
		"Quantum computing uses quantum mechanics principles.",
		true)

	testChatRequest(t, "no thinking content",
		"What is 2+2?",
		"</think> The answer is 4.",
		"",
		"The answer is 4.",
		true)

	// Test streaming response with template-added <think>
	t.Run("streaming with thinking", func(t *testing.T) {
		var wg sync.WaitGroup
		wg.Add(1)

		mock.CompletionFn = func(ctx context.Context, r llm.CompletionRequest, fn func(r llm.CompletionResponse)) error {
			defer wg.Done()

			// Verify the prompt ends with <think> due to template
			if !strings.HasSuffix(r.Prompt, "<think>") {
				t.Errorf("expected prompt to end with <think>, got: %q", r.Prompt)
			}

			// Simulate streaming chunks
			responses := []llm.CompletionResponse{
				{Content: " I need to consider", Done: false, PromptEvalCount: 1, PromptEvalDuration: 1},
				{Content: " multiple factors here...", Done: false, PromptEvalCount: 1, PromptEvalDuration: 1},
				{Content: " </think> Based on my analysis,", Done: false, PromptEvalCount: 1, PromptEvalDuration: 1},
				{Content: " the solution is straightforward.", Done: true, DoneReason: llm.DoneReasonStop, PromptEvalCount: 1, PromptEvalDuration: 1, EvalCount: 1, EvalDuration: 1},
			}

			for _, resp := range responses {
				select {
				case <-ctx.Done():
					return ctx.Err()
				default:
					fn(resp)
					time.Sleep(10 * time.Millisecond)
				}
			}
			return nil
		}

		think := true
		w := createRequest(t, s.ChatHandler, api.ChatRequest{
			Model:    "test-thinking",
			Messages: []api.Message{{Role: "user", Content: "Analyze this complex problem"}},
			Think:    &api.ThinkValue{Value: think},
			Stream:   &stream,
		})

		wg.Wait()

		if w.Code != http.StatusOK {
			t.Fatalf("expected status 200, got %d", w.Code)
		}

		// Parse streaming responses
		decoder := json.NewDecoder(w.Body)
		var allThinking, allContent strings.Builder

		for {
			var resp api.ChatResponse
			if err := decoder.Decode(&resp); err == io.EOF {
				break
			} else if err != nil {
				t.Fatal(err)
			}
			allThinking.WriteString(resp.Message.Thinking)
			allContent.WriteString(resp.Message.Content)
		}

		// Note: Leading whitespace after <think> is eaten by the parser
		if got := allThinking.String(); got != "I need to consider multiple factors here... " {
			t.Errorf("expected thinking %q, got %q", "I need to consider multiple factors here... ", got)
		}

		if got := allContent.String(); got != "Based on my analysis, the solution is straightforward." {
			t.Errorf("expected content %q, got %q", "Based on my analysis, the solution is straightforward.", got)
		}
	})

	t.Run("structured outputs restart non-stream", func(t *testing.T) {
		var (
			requestsMu sync.Mutex
			requests   []llm.CompletionRequest
			wg         sync.WaitGroup
		)

		wg.Add(2)

		format := json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}}}`)

		mock.CompletionFn = func(ctx context.Context, r llm.CompletionRequest, fn func(r llm.CompletionResponse)) error {
			defer wg.Done()

			requestsMu.Lock()
			requests = append(requests, r)
			callNum := len(requests)
			requestsMu.Unlock()

			switch callNum {
			case 1:
				fn(llm.CompletionResponse{
					Content:            " I am thinking through this problem. </think> {\"answer\":\"42\"}",
					Done:               false,
					PromptEvalCount:    1,
					PromptEvalDuration: 1,
				})

				select {
				case <-ctx.Done():
					return ctx.Err()
				case <-time.After(time.Second):
					t.Fatalf("timeout waiting for structured outputs cancellation")
					return nil
				}
			case 2:
				fn(llm.CompletionResponse{
					Content:            `{"answer":"42"}`,
					Done:               true,
					DoneReason:         llm.DoneReasonStop,
					PromptEvalCount:    1,
					PromptEvalDuration: 1,
					EvalCount:          1,
					EvalDuration:       1,
				})
				return nil
			default:
				t.Fatalf("unexpected number of completion calls: %d", callNum)
				return nil
			}
		}

		think := true
		streamRequest := false
		w := createRequest(t, s.ChatHandler, api.ChatRequest{
			Model:    "test-thinking",
			Messages: []api.Message{{Role: "user", Content: "Please respond in JSON."}},
			Think:    &api.ThinkValue{Value: think},
			Stream:   &streamRequest,
			Format:   format,
		})

		wg.Wait()
		mock.CompletionFn = nil

		if w.Code != http.StatusOK {
			t.Fatalf("expected status 200, got %d", w.Code)
		}

		if len(requests) != 2 {
			t.Fatalf("expected two completion calls, got %d", len(requests))
		}

		if requests[0].Format != nil {
			t.Errorf("expected first completion format to be nil, got %q", requests[0].Format)
		}

		if !bytes.Equal([]byte(format), []byte(requests[1].Format)) {
			t.Errorf("expected second completion format to match original format")
		}

		var resp api.ChatResponse
		if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
			t.Fatal(err)
		}

		if resp.Message.Thinking != "I am thinking through this problem. " {
			t.Errorf("expected thinking %q, got %q", "I am thinking through this problem. ", resp.Message.Thinking)
		}

		if resp.Message.Content != `{"answer":"42"}` {
			t.Errorf("expected content %q, got %q", `{"answer":"42"}`, resp.Message.Content)
		}

		if !resp.Done {
			t.Errorf("expected response to be done")
		}

		if resp.DoneReason != "stop" {
			t.Errorf("expected done reason stop, got %s", resp.DoneReason)
		}
	})

	t.Run("structured outputs restart streaming", func(t *testing.T) {
		var (
			requestsMu sync.Mutex
			requests   []llm.CompletionRequest
			wg         sync.WaitGroup
		)

		wg.Add(2)

		format := json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}}}`)

		mock.CompletionFn = func(ctx context.Context, r llm.CompletionRequest, fn func(r llm.CompletionResponse)) error {
			defer wg.Done()

			requestsMu.Lock()
			requests = append(requests, r)
			callNum := len(requests)
			requestsMu.Unlock()

			switch callNum {
			case 1:
				fn(llm.CompletionResponse{
					Content:            " I am thinking through this problem. </think> {\"answer\":\"42\"}",
					Done:               false,
					PromptEvalCount:    1,
					PromptEvalDuration: 1,
				})

				select {
				case <-ctx.Done():
					return ctx.Err()
				case <-time.After(time.Second):
					t.Fatalf("timeout waiting for structured outputs cancellation")
					return nil
				}
			case 2:
				fn(llm.CompletionResponse{
					Content:            `{"answer":"42"}`,
					Done:               true,
					DoneReason:         llm.DoneReasonStop,
					PromptEvalCount:    1,
					PromptEvalDuration: 1,
					EvalCount:          1,
					EvalDuration:       1,
				})
				return nil
			default:
				t.Fatalf("unexpected number of completion calls: %d", callNum)
				return nil
			}
		}

		think := true
		streamRequest := true
		w := createRequest(t, s.ChatHandler, api.ChatRequest{
			Model:    "test-thinking",
			Messages: []api.Message{{Role: "user", Content: "Please respond in JSON."}},
			Think:    &api.ThinkValue{Value: think},
			Stream:   &streamRequest,
			Format:   format,
		})

		wg.Wait()
		mock.CompletionFn = nil

		if w.Code != http.StatusOK {
			t.Fatalf("expected status 200, got %d", w.Code)
		}

		if len(requests) != 2 {
			t.Fatalf("expected two completion calls, got %d", len(requests))
		}

		if requests[0].Format != nil {
			t.Errorf("expected first completion format to be nil, got %q", requests[0].Format)
		}

		if !bytes.Equal([]byte(format), []byte(requests[1].Format)) {
			t.Errorf("expected second completion format to match original format")
		}

		decoder := json.NewDecoder(w.Body)
		var events []api.ChatResponse
		for {
			var event api.ChatResponse
			if err := decoder.Decode(&event); err == io.EOF {
				break
			} else if err != nil {
				t.Fatal(err)
			}
			events = append(events, event)
			if event.Done {
				break
			}
		}

		if len(events) < 2 {
			t.Fatalf("expected at least two streaming events, got %d", len(events))
		}

		first := events[0]
		if first.Message.Thinking != "I am thinking through this problem. " {
			t.Errorf("expected first event thinking %q, got %q", "I am thinking through this problem. ", first.Message.Thinking)
		}

		if first.Message.Content != "" {
			t.Errorf("expected first event content to be empty, got %q", first.Message.Content)
		}

		if first.Done {
			t.Error("expected first event to be non-terminal")
		}

		last := events[len(events)-1]
		if last.Message.Thinking != "" {
			t.Errorf("expected final event thinking to be empty, got %q", last.Message.Thinking)
		}

		if last.Message.Content != `{"answer":"42"}` {
			t.Errorf("expected final event content %q, got %q", `{"answer":"42"}`, last.Message.Content)
		}

		if !last.Done {
			t.Error("expected final event to be done")
		}

		if last.DoneReason != "stop" {
			t.Errorf("expected final done reason stop, got %s", last.DoneReason)
		}
	})
}
