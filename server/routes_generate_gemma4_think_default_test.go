package server

import (
	"bytes"
	"context"
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

// TestGenerateGemma4ParserThinkingDefaultsOn pins the /api/generate handler's
// ordering invariant: when a model has CapabilityThinking, the handler must
// auto-default req.Think to true BEFORE initialising the builtin parser so
// thinking-aware parsers see thinkValue.Bool()==true and surface channel
// content through the Thinking field instead of silently dropping it.
//
// Before the fix, the parser was initialised with req.Think still nil, so
// Gemma4Parser.thinkingEnabled stayed false. Channel content was discarded
// and short-num_predict requests came back with an empty response and no
// thinking — see the empty-response repro on gemma4:31b that motivated the
// fix.
func TestGenerateGemma4ParserThinkingDefaultsOn(t *testing.T) {
	gin.SetMode(gin.TestMode)

	mock := &mockRunner{}
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
			loadFn: func(req *LlmRequest, _ ml.SystemInfo, _ []ml.DeviceInfo, _ bool) bool {
				time.Sleep(time.Millisecond)
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

	streamCreate := false
	if w := createRequest(t, s.CreateHandler, api.CreateRequest{
		Model:    "test-gemma4-think-default",
		Files:    map[string]string{"file.gguf": digest},
		Parser:   "gemma4",
		Template: `{{ .Prompt }}`,
		Stream:   &streamCreate,
	}); w.Code != http.StatusOK {
		t.Fatalf("create: expected status 200, got %d: %s", w.Code, w.Body.String())
	}

	// Mock the runner to emit the gemma4 channel-thinking format the real
	// model produces — opening <|channel>, a "thought\n" channel-name prefix
	// the parser strips, the thinking body, the <channel|> close tag, and
	// finally the user-visible content.
	mock.CompletionFn = func(ctx context.Context, _ llm.CompletionRequest, fn func(r llm.CompletionResponse)) error {
		chunks := []llm.CompletionResponse{
			{Content: "<|channel>thought\nReasoning about the question."},
			{
				Content:            "<channel|>Paris.",
				Done:               true,
				DoneReason:         llm.DoneReasonStop,
				PromptEvalCount:    1,
				PromptEvalDuration: 1,
				EvalCount:          1,
				EvalDuration:       1,
			},
		}
		for _, c := range chunks {
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
				fn(c)
			}
		}
		return nil
	}

	noStream := false

	// Case 1 — no Think param set; the handler must auto-default it to true
	// before Init so the parser surfaces thinking content.
	t.Run("default_think_surfaces_thinking", func(t *testing.T) {
		w := createRequest(t, s.GenerateHandler, api.GenerateRequest{
			Model:  "test-gemma4-think-default",
			Prompt: "The capital of France is",
			Stream: &noStream,
		})
		if w.Code != http.StatusOK {
			t.Fatalf("expected status 200, got %d: %s", w.Code, w.Body.String())
		}

		var resp api.GenerateResponse
		if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
			t.Fatalf("decode response: %v", err)
		}

		if resp.Thinking != "Reasoning about the question." {
			t.Errorf("Thinking = %q, want %q", resp.Thinking, "Reasoning about the question.")
		}
		if resp.Response != "Paris." {
			t.Errorf("Response = %q, want %q", resp.Response, "Paris.")
		}
	})

	// Case 2 — explicit think=false; the parser should silently drop channel
	// thinking content but still surface user-visible content after <channel|>.
	t.Run("explicit_think_false_hides_thinking", func(t *testing.T) {
		think := false
		w := createRequest(t, s.GenerateHandler, api.GenerateRequest{
			Model:  "test-gemma4-think-default",
			Prompt: "The capital of France is",
			Think:  &api.ThinkValue{Value: think},
			Stream: &noStream,
		})
		if w.Code != http.StatusOK {
			t.Fatalf("expected status 200, got %d: %s", w.Code, w.Body.String())
		}

		var resp api.GenerateResponse
		if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
			t.Fatalf("decode response: %v", err)
		}

		if resp.Thinking != "" {
			t.Errorf("Thinking = %q, want empty (think=false hides reasoning)", resp.Thinking)
		}
		if resp.Response != "Paris." {
			t.Errorf("Response = %q, want %q", resp.Response, "Paris.")
		}
	})
}
