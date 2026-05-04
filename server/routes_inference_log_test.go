package server

import (
	"bytes"
	"log/slog"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

// captureLogOutput redirects the default slog logger to a buffer for the
// duration of the test and restores the original logger on cleanup.
func captureLogOutput(t *testing.T) *bytes.Buffer {
	t.Helper()
	var buf bytes.Buffer
	old := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug})))
	t.Cleanup(func() { slog.SetDefault(old) })
	return &buf
}

// findLogRecord returns the first log line whose msg field matches the given
// message, or the empty string if no such line is found.
func findLogRecord(output, msg string) string {
	needle := `msg="` + msg + `"`
	for _, line := range strings.Split(output, "\n") {
		if strings.Contains(line, needle) {
			return line
		}
	}
	return ""
}

func newInferenceLogTestServer(t *testing.T) (s *Server, createModel func(name string)) {
	t.Helper()
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_CONTEXT_LENGTH", "4096")

	mock := &mockRunner{
		CompletionResponse: llm.CompletionResponse{
			Done:               true,
			DoneReason:         llm.DoneReasonStop,
			PromptEvalCount:    42,
			PromptEvalDuration: 500 * time.Millisecond,
			EvalCount:          10,
			EvalDuration:       200 * time.Millisecond,
		},
	}

	srv := &Server{
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
			loadFn: func(req *LlmRequest, _ ml.SystemInfo, _ []ml.DeviceInfo, _ bool) bool {
				time.Sleep(time.Millisecond)
				req.successCh <- &runnerRef{llama: mock}
				return false
			},
		},
	}
	go srv.sched.Run(t.Context())

	// createModel builds fresh GGML fixtures each call so the io.Reader in each
	// tensor is not already-consumed from a previous invocation.
	createModel = func(name string) {
		t.Helper()
		kv := ggml.KV{
			"general.architecture":          "llama",
			"llama.block_count":             uint32(1),
			"llama.context_length":          uint32(8192),
			"llama.embedding_length":        uint32(4096),
			"llama.attention.head_count":    uint32(32),
			"llama.attention.head_count_kv": uint32(8),
			"tokenizer.ggml.tokens":         []string{""},
			"tokenizer.ggml.scores":         []float32{0},
			"tokenizer.ggml.token_type":     []int32{0},
		}
		tensors := []*ggml.Tensor{
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
		}
		_, digest := createBinFile(t, kv, tensors)
		w := createRequest(t, srv.CreateHandler, api.CreateRequest{
			Model:  name,
			Files:  map[string]string{"file.gguf": digest},
			Stream: &stream,
		})
		if w.Code != 200 {
			t.Fatalf("create model %q: status %d", name, w.Code)
		}
	}

	return srv, createModel
}

func TestGenerateHandlerLogsRequestComplete(t *testing.T) {
	s, createModel := newInferenceLogTestServer(t)
	createModel("logtest")

	logs := captureLogOutput(t)

	w := createRequest(t, s.GenerateHandler, api.GenerateRequest{
		Model:  "logtest",
		Prompt: "hello",
		Stream: &stream,
	})

	if w.Code != 200 {
		t.Fatalf("expected status 200, got %d: %s", w.Code, w.Body.String())
	}

	record := findLogRecord(logs.String(), "request complete")
	if record == "" {
		t.Fatalf("expected 'request complete' log record, got:\n%s", logs.String())
	}
	for _, want := range []string{
		"model=logtest",
		"prompt_eval_count=42",
		"prompt_eval_duration=500ms",
		"eval_count=10",
		"eval_duration=200ms",
	} {
		if !strings.Contains(record, want) {
			t.Errorf("expected %q in 'request complete' record, got:\n%s", want, record)
		}
	}
}

func TestChatHandlerLogsRequestComplete(t *testing.T) {
	s, createModel := newInferenceLogTestServer(t)
	createModel("logtest-chat")

	logs := captureLogOutput(t)

	w := createRequest(t, s.ChatHandler, api.ChatRequest{
		Model: "logtest-chat",
		Messages: []api.Message{
			{Role: "user", Content: "Hello!"},
		},
		Stream: &stream,
	})

	if w.Code != 200 {
		t.Fatalf("expected status 200, got %d: %s", w.Code, w.Body.String())
	}

	record := findLogRecord(logs.String(), "request complete")
	if record == "" {
		t.Fatalf("expected 'request complete' log record, got:\n%s", logs.String())
	}
	for _, want := range []string{
		"model=logtest-chat",
		"prompt_eval_count=42",
		"prompt_eval_duration=500ms",
		"eval_count=10",
		"eval_duration=200ms",
	} {
		if !strings.Contains(record, want) {
			t.Errorf("expected %q in 'request complete' record, got:\n%s", want, record)
		}
	}
}
