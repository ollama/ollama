package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

// TestGenerateWithBuiltinRenderer tests that api/generate uses built-in renderers
// when in chat-like flow (messages present, no suffix, no template)
func TestGenerateWithBuiltinRenderer(t *testing.T) {
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
				time.Sleep(time.Millisecond)
				req.successCh <- &runnerRef{
					llama: &mock,
				}
				return false
			},
		},
	}

	go s.sched.Run(t.Context())

	// Create a model with a built-in renderer (qwen3-coder)
	_, digest := createBinFile(t, ggml.KV{
		"general.architecture":          "qwen3",
		"qwen3.block_count":             uint32(1),
		"qwen3.context_length":          uint32(8192),
		"qwen3.embedding_length":        uint32(4096),
		"qwen3.attention.head_count":    uint32(32),
		"qwen3.attention.head_count_kv": uint32(8),
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

	// Create a model with the qwen3-coder renderer
	w := createRequest(t, s.CreateHandler, api.CreateRequest{
		Model:    "test-renderer",
		Files:    map[string]string{"file.gguf": digest},
		Renderer: "qwen3-coder",
		Stream:   &stream,
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	mock.CompletionResponse.Content = "Hi!"

	t.Run("chat-like flow uses renderer", func(t *testing.T) {
		// Test that when using messages (chat-like flow), the built-in renderer is used
		w := createRequest(t, s.GenerateHandler, api.GenerateRequest{
			Model:  "test-renderer",
			Prompt: "Write a hello world function",
			Stream: &stream,
		})

		if w.Code != http.StatusOK {
			t.Errorf("expected status 200, got %d", w.Code)
		}

		// The qwen3-coder renderer produces output with <|im_start|> and <|im_end|> tags
		// When messages are built internally from prompt, it should use the renderer
		if !strings.Contains(mock.CompletionRequest.Prompt, "<|im_start|>") {
			t.Errorf("expected prompt to contain <|im_start|> from qwen3-coder renderer, got: %s", mock.CompletionRequest.Prompt)
		}

		if !strings.Contains(mock.CompletionRequest.Prompt, "<|im_end|>") {
			t.Errorf("expected prompt to contain <|im_end|> from qwen3-coder renderer, got: %s", mock.CompletionRequest.Prompt)
		}
	})

	t.Run("chat-like flow with system message uses renderer", func(t *testing.T) {
		// Test that system messages work with the renderer
		w := createRequest(t, s.GenerateHandler, api.GenerateRequest{
			Model:  "test-renderer",
			Prompt: "Write a hello world function",
			System: "You are a helpful coding assistant.",
			Stream: &stream,
		})

		if w.Code != http.StatusOK {
			t.Errorf("expected status 200, got %d", w.Code)
		}

		// Should contain the system message and use renderer format
		if !strings.Contains(mock.CompletionRequest.Prompt, "<|im_start|>system") {
			t.Errorf("expected prompt to contain system message with renderer format, got: %s", mock.CompletionRequest.Prompt)
		}

		if !strings.Contains(mock.CompletionRequest.Prompt, "You are a helpful coding assistant.") {
			t.Errorf("expected prompt to contain system message content, got: %s", mock.CompletionRequest.Prompt)
		}
	})

	t.Run("custom template bypasses renderer", func(t *testing.T) {
		// Test that providing a custom template uses the legacy flow
		w := createRequest(t, s.GenerateHandler, api.GenerateRequest{
			Model:    "test-renderer",
			Prompt:   "Write a hello world function",
			Template: "{{ .Prompt }}",
			Stream:   &stream,
		})

		if w.Code != http.StatusOK {
			t.Errorf("expected status 200, got %d", w.Code)
		}

		// Should NOT use the renderer format when custom template is provided
		if strings.Contains(mock.CompletionRequest.Prompt, "<|im_start|>") {
			t.Errorf("expected prompt to NOT use renderer when custom template provided, got: %s", mock.CompletionRequest.Prompt)
		}

		// Should just be the raw prompt from the template
		if diff := cmp.Diff(mock.CompletionRequest.Prompt, "Write a hello world function"); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}
	})

	// Create a model with suffix support for the next test
	w = createRequest(t, s.CreateHandler, api.CreateRequest{
		Model: "test-suffix-renderer",
		From:  "test-renderer",
		Template: `{{- if .Suffix }}<PRE> {{ .Prompt }} <SUF>{{ .Suffix }} <MID>
{{- else }}{{ .Prompt }}
{{- end }}`,
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	t.Run("suffix bypasses renderer", func(t *testing.T) {
		// Test that providing a suffix uses the legacy flow
		w := createRequest(t, s.GenerateHandler, api.GenerateRequest{
			Model:  "test-suffix-renderer",
			Prompt: "def add(",
			Suffix: "    return c",
		})

		if w.Code != http.StatusOK {
			t.Errorf("expected status 200, got %d", w.Code)
		}

		// Should NOT use the renderer format when suffix is provided
		if strings.Contains(mock.CompletionRequest.Prompt, "<|im_start|>") {
			t.Errorf("expected prompt to NOT use renderer when suffix provided, got: %s", mock.CompletionRequest.Prompt)
		}

		// Should use the suffix template format
		if diff := cmp.Diff(mock.CompletionRequest.Prompt, "<PRE> def add( <SUF>    return c <MID>"); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}
	})
}

// TestGenerateWithDebugRenderOnly tests that debug_render_only works with built-in renderers
func TestGenerateWithDebugRenderOnly(t *testing.T) {
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
				time.Sleep(time.Millisecond)
				req.successCh <- &runnerRef{
					llama: &mock,
				}
				return false
			},
		},
	}

	go s.sched.Run(t.Context())

	// Create a model with a built-in renderer
	_, digest := createBinFile(t, ggml.KV{
		"general.architecture":          "qwen3",
		"qwen3.block_count":             uint32(1),
		"qwen3.context_length":          uint32(8192),
		"qwen3.embedding_length":        uint32(4096),
		"qwen3.attention.head_count":    uint32(32),
		"qwen3.attention.head_count_kv": uint32(8),
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
		Model:    "test-debug-renderer",
		Files:    map[string]string{"file.gguf": digest},
		Renderer: "qwen3-coder",
		Stream:   &stream,
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	t.Run("debug_render_only with renderer", func(t *testing.T) {
		w := createRequest(t, s.GenerateHandler, api.GenerateRequest{
			Model:           "test-debug-renderer",
			Prompt:          "Write a hello world function",
			System:          "You are a coding assistant",
			DebugRenderOnly: true,
		})

		if w.Code != http.StatusOK {
			t.Errorf("expected status 200, got %d", w.Code)
		}

		var resp api.GenerateResponse
		if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
			t.Fatal(err)
		}

		if resp.DebugInfo == nil {
			t.Fatalf("expected debug info, got nil")
		}

		// Verify that the rendered template uses the built-in renderer
		if !strings.Contains(resp.DebugInfo.RenderedTemplate, "<|im_start|>") {
			t.Errorf("expected rendered template to use qwen3-coder renderer format, got: %s", resp.DebugInfo.RenderedTemplate)
		}

		if !strings.Contains(resp.DebugInfo.RenderedTemplate, "You are a coding assistant") {
			t.Errorf("expected rendered template to contain system message, got: %s", resp.DebugInfo.RenderedTemplate)
		}

		if !strings.Contains(resp.DebugInfo.RenderedTemplate, "Write a hello world function") {
			t.Errorf("expected rendered template to contain prompt, got: %s", resp.DebugInfo.RenderedTemplate)
		}
	})
}
