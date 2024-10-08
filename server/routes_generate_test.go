package server

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
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
}

func (m *mockRunner) Completion(_ context.Context, r llm.CompletionRequest, fn func(r llm.CompletionResponse)) error {
	m.CompletionRequest = r
	fn(m.CompletionResponse)
	return nil
}

func (mockRunner) Tokenize(_ context.Context, s string) (tokens []int, err error) {
	for range strings.Fields(s) {
		tokens = append(tokens, len(tokens))
	}

	return
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

		if diff := cmp.Diff(mock.CompletionRequest.Prompt, "User: Hello! "); diff != "" {
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

		if diff := cmp.Diff(mock.CompletionRequest.Prompt, "System: You are a helpful assistant. User: Hello! "); diff != "" {
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

		if diff := cmp.Diff(mock.CompletionRequest.Prompt, "System: You can perform magic tricks. User: Hello! "); diff != "" {
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

		if diff := cmp.Diff(mock.CompletionRequest.Prompt, "System: You are a helpful assistant. User: Hello! Assistant: I can help you with that. System: You can perform magic tricks. User: Help me write tests. "); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}

		checkChatResponse(t, w.Body, "test-system", "Abra kadabra!")
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
		if w.Code != http.StatusBadRequest {
			t.Errorf("expected status 400, got %d", w.Code)
		}

		if diff := cmp.Diff(w.Body.String(), `{"error":"model is required"}`); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}
	})

	t.Run("missing model", func(t *testing.T) {
		w := createRequest(t, s.GenerateHandler, api.GenerateRequest{})
		if w.Code != http.StatusBadRequest {
			t.Errorf("expected status 400, got %d", w.Code)
		}

		if diff := cmp.Diff(w.Body.String(), `{"error":"model is required"}`); diff != "" {
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

		if diff := cmp.Diff(w.Body.String(), `{"error":"test does not support insert"}`); diff != "" {
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
