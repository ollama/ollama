package server

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/llm"
)

func TestTokenize(t *testing.T) {
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

	t.Run("missing body", func(t *testing.T) {
		w := httptest.NewRecorder()
		r := httptest.NewRequest(http.MethodPost, "/api/tokenize", nil)
		s.TokenizeHandler(w, r)

		if w.Code != http.StatusBadRequest {
			t.Errorf("expected status 400, got %d", w.Code)
		}

		if diff := cmp.Diff(w.Body.String(), "missing request body\n"); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}
	})

	t.Run("missing model", func(t *testing.T) {
		w := httptest.NewRecorder()
		r := httptest.NewRequest(http.MethodPost, "/api/tokenize", strings.NewReader("{}"))
		s.TokenizeHandler(w, r)

		if w.Code != http.StatusBadRequest {
			t.Errorf("expected status 400, got %d", w.Code)
		}

		if diff := cmp.Diff(w.Body.String(), "missing `text` for tokenization\n"); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}
	})
	t.Run("tokenize text", func(t *testing.T) {
		// First create the model
		w := createRequest(t, s.CreateHandler, api.CreateRequest{
			Model: "test",
			Modelfile: fmt.Sprintf(`FROM %s`, createBinFile(t, llm.KV{
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
				{Name: "output.weight", Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))},
			})),
		})
		if w.Code != http.StatusOK {
			t.Fatalf("failed to create model: %d", w.Code)
		}

		// Now test tokenization
		body, err := json.Marshal(api.TokenizeRequest{
			Model: "test",
			Text:  "Hello world how are you",
		})
		if err != nil {
			t.Fatalf("failed to marshal request: %v", err)
		}

		w = httptest.NewRecorder()
		r := httptest.NewRequest(http.MethodPost, "/api/tokenize", bytes.NewReader(body))
		r.Header.Set("Content-Type", "application/json")
		s.TokenizeHandler(w, r)

		if w.Code != http.StatusOK {
			t.Errorf("expected status 200, got %d: %s", w.Code, w.Body.String())
		}

		var resp api.TokenizeResponse
		if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
			t.Errorf("failed to decode response: %v", err)
		}

		// Our mock tokenizer creates sequential tokens based on word count
		expected := []int{0, 1, 2, 3, 4}
		if diff := cmp.Diff(resp.Tokens, expected); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}
	})

	t.Run("tokenize empty text", func(t *testing.T) {
		body, err := json.Marshal(api.TokenizeRequest{
			Model: "test",
			Text:  "",
		})
		if err != nil {
			t.Fatalf("failed to marshal request: %v", err)
		}

		w := httptest.NewRecorder()
		r := httptest.NewRequest(http.MethodPost, "/api/tokenize", bytes.NewReader(body))
		r.Header.Set("Content-Type", "application/json")
		s.TokenizeHandler(w, r)

		if w.Code != http.StatusBadRequest {
			t.Errorf("expected status 400, got %d", w.Code)
		}

		if diff := cmp.Diff(w.Body.String(), "missing `text` for tokenization\n"); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}
	})
}

func TestDetokenize(t *testing.T) {
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

	t.Run("detokenize tokens", func(t *testing.T) {
		// Create the model first
		w := createRequest(t, s.CreateHandler, api.CreateRequest{
			Model: "test",
			Modelfile: fmt.Sprintf(`FROM %s`, createBinFile(t, llm.KV{
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
				{Name: "output.weight", Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))},
			})),
			Stream: &stream,
		})
		if w.Code != http.StatusOK {
			t.Fatalf("failed to create model: %d - %s", w.Code, w.Body.String())
		}

		body, err := json.Marshal(api.DetokenizeRequest{
			Model:  "test",
			Tokens: []int{0, 1, 2, 3, 4},
		})
		if err != nil {
			t.Fatalf("failed to marshal request: %v", err)
		}

		w = httptest.NewRecorder()
		r := httptest.NewRequest(http.MethodPost, "/api/detokenize", bytes.NewReader(body))
		r.Header.Set("Content-Type", "application/json")
		s.DetokenizeHandler(w, r)

		if w.Code != http.StatusOK {
			t.Errorf("expected status 200, got %d: %s", w.Code, w.Body.String())
		}

		var resp api.DetokenizeResponse
		if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
			t.Errorf("failed to decode response: %v", err)
		}
	})

	t.Run("detokenize empty tokens", func(t *testing.T) {
		body, err := json.Marshal(api.DetokenizeRequest{
			Model: "test",
		})
		if err != nil {
			t.Fatalf("failed to marshal request: %v", err)
		}

		w := httptest.NewRecorder()
		r := httptest.NewRequest(http.MethodPost, "/api/detokenize", bytes.NewReader(body))
		r.Header.Set("Content-Type", "application/json")
		s.DetokenizeHandler(w, r)

		if w.Code != http.StatusBadRequest {
			t.Errorf("expected status 400, got %d", w.Code)
		}

		if diff := cmp.Diff(w.Body.String(), "missing tokens for detokenization\n"); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}
	})

	t.Run("detokenize missing model", func(t *testing.T) {
		body, err := json.Marshal(api.DetokenizeRequest{
			Tokens: []int{0, 1, 2},
		})
		if err != nil {
			t.Fatalf("failed to marshal request: %v", err)
		}

		w := httptest.NewRecorder()
		r := httptest.NewRequest(http.MethodPost, "/api/detokenize", bytes.NewReader(body))
		r.Header.Set("Content-Type", "application/json")
		s.DetokenizeHandler(w, r)

		if w.Code != http.StatusNotFound {
			t.Errorf("expected status 404, got %d", w.Code)
		}

		if diff := cmp.Diff(w.Body.String(), "model '' not found\n"); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}
	})
}
