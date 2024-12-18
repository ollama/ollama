package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/llama"
	"github.com/ollama/ollama/llm"
)

type mockModelLoader struct {
	LoadModelFn func(string, llama.ModelParams) (*loadedModel, error)
}

func (ml *mockModelLoader) LoadModel(name string, params llama.ModelParams) (*loadedModel, error) {
	if ml.LoadModelFn != nil {
		return ml.LoadModelFn(name, params)
	}

	return nil, nil
}

type mockModel struct {
	llama.Model
	TokenizeFn     func(text string, addBos bool, addEos bool) ([]int, error)
	TokenToPieceFn func(token int) string
}

func (mockModel) Tokenize(text string, addBos bool, addEos bool) ([]int, error) {
	return []int{1, 2, 3}, nil
}

func (mockModel) TokenToPiece(token int) string {
	return fmt.Sprint(token)
}

func TestTokenizeHandler(t *testing.T) {
	gin.SetMode(gin.TestMode)

	mockModel := mockModel{}

	s := Server{
		sched: &Scheduler{
			pendingReqCh:  make(chan *LlmRequest, 1),
			finishedReqCh: make(chan *LlmRequest, 1),
			expiredCh:     make(chan *runnerRef, 1),
			unloadedCh:    make(chan any, 1),
			loaded:        make(map[string]*runnerRef),
			newServerFn:   newMockServer(&mockRunner{}),
			getGpuFn:      discover.GetGPUInfo,
			getCpuFn:      discover.GetCPUInfo,
			reschedDelay:  250 * time.Millisecond,
			loadFn: func(req *LlmRequest, ggml *llm.GGML, gpus discover.GpuInfoList, numParallel int) {
				time.Sleep(time.Millisecond)
				req.successCh <- &runnerRef{
					llama: &mockRunner{},
				}
			},
		},
		ml: mockLoader,
	}

	t.Run("method not allowed", func(t *testing.T) {
		w := createRequest(t, gin.WrapF(s.TokenizeHandler), nil)
		if w.Code != http.StatusMethodNotAllowed {
			t.Errorf("expected status %d, got %d", http.StatusMethodNotAllowed, w.Code)
		}
	})

	t.Run("missing body", func(t *testing.T) {
		w := createRequest(t, gin.WrapF(s.TokenizeHandler), nil)
		if w.Code != http.StatusBadRequest {
			t.Errorf("expected status %d, got %d", http.StatusBadRequest, w.Code)
		}
	})

	t.Run("missing text", func(t *testing.T) {
		w := createRequest(t, gin.WrapF(s.TokenizeHandler), api.TokenizeRequest{
			Model: "test",
		})
		if w.Code != http.StatusBadRequest {
			t.Errorf("expected status %d, got %d", http.StatusBadRequest, w.Code)
		}
	})

	t.Run("missing model", func(t *testing.T) {
		w := createRequest(t, gin.WrapF(s.TokenizeHandler), api.TokenizeRequest{
			Text: "test text",
		})
		if w.Code != http.StatusBadRequest {
			t.Errorf("expected status %d, got %d", http.StatusBadRequest, w.Code)
		}
	})

	t.Run("model not found", func(t *testing.T) {
		w := createRequest(t, gin.WrapF(s.TokenizeHandler), api.TokenizeRequest{
			Model: "nonexistent",
			Text:  "test text",
		})
		if w.Code != http.StatusInternalServerError {
			t.Errorf("expected status %d, got %d", http.StatusInternalServerError, w.Code)
		}
	})

	t.Run("successful tokenization", func(t *testing.T) {
		w := createRequest(t, gin.WrapF(s.TokenizeHandler), api.TokenizeRequest{
			Model: "test",
			Text:  "test text",
		})

		if w.Code != http.StatusOK {
			t.Errorf("expected status %d, got %d", http.StatusOK, w.Code)
		}

		var resp api.TokenizeResponse
		if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
			t.Fatal(err)
		}

		expectedTokens := []int{0, 1}
		if len(resp.Tokens) != len(expectedTokens) {
			t.Errorf("expected %d tokens, got %d", len(expectedTokens), len(resp.Tokens))
		}
		for i, token := range resp.Tokens {
			if token != expectedTokens[i] {
				t.Errorf("expected token %d at position %d, got %d", expectedTokens[i], i, token)
			}
		}
	})
}
