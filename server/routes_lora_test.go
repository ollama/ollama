package server

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

// MockLoraRunner extends mockLlm to support LoRA operations
type MockLoraRunner struct {
	*mockLlm
	adapters []api.LoraAdapter
}

func (m *MockLoraRunner) GetLoraAdapters(ctx context.Context) (api.LoraAdapterList, error) {
	return api.LoraAdapterList(m.adapters), nil
}

func (m *MockLoraRunner) SetLoraAdapterScales(ctx context.Context, adapters []api.LoraScaleRequest) (api.LoraAdapterList, error) {
	for _, req := range adapters {
		for i, existing := range m.adapters {
			if existing.ID == req.ID {
				m.adapters[i].Scale = req.Scale
			}
		}
	}
	return api.LoraAdapterList(m.adapters), nil
}

func TestLoraAdaptersHandler(t *testing.T) {
	gin.SetMode(gin.TestMode)

	// Setup Server with mocked Scheduler
	ctx := context.Background()
	sched := InitScheduler(ctx)

	// Create a mock runner with pre-loaded adapters
	mockRunner := &MockLoraRunner{
		mockLlm: &mockLlm{},
		adapters: []api.LoraAdapter{
			{ID: 0, Path: "/path/to/adapter1.gguf", Scale: 1.0},
			{ID: 1, Path: "/path/to/adapter2.gguf", Scale: 0.0},
		},
	}

	// Manually inject the runner into the scheduler's loaded map
	// We use the same model name key as we will request
	modelName := "test-model"
	modelPath := "/tmp/test-model"

	sched.loadedMu.Lock()
	sched.loaded[modelPath] = &runnerRef{
		llama:    mockRunner,
		model:    &Model{ModelPath: modelPath, Name: modelName, ShortName: modelName},
		refCount: 1, // Ensure it's not evicted immediately if we were using real scheduling loop
	}
	sched.loadedMu.Unlock()

	// Intercept newServerFn just in case, though we manually injected result
	sched.newServerFn = func(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, model string, f *ggml.GGML, adapters []string, projectors []string, opts api.Options, numParallel int) (llm.LlamaServer, error) {
		return mockRunner, nil
	}

	s := &Server{sched: sched}

	// We need to mount the routes.
	// Since GenerateRoutes takes a Registry which we don't want to mock fully if unnecessary,
	// we can just construct a router and register the handlers directly for this unit test.
	// This avoids dependency on Registry and other middleware.
	router := gin.New()
	router.GET("/api/lora-adapters", s.GetLoraAdaptersHandler)
	router.POST("/api/lora-adapters", s.SetLoraAdaptersHandler)

	t.Run("Get Adapters", func(t *testing.T) {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/api/lora-adapters?model=test-model", nil)
		router.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("expected status 200, got %d", w.Code)
		}

		var resp api.LoraAdaptersResponse
		if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
			t.Fatal(err)
		}

		if len(resp.Adapters) != 2 {
			t.Errorf("expected 2 adapters, got %d", len(resp.Adapters))
		}
		if resp.Adapters[0].Path != "/path/to/adapter1.gguf" || resp.Adapters[0].Scale != 1.0 {
			t.Errorf("unexpected adapter 0 data: %v", resp.Adapters[0])
		}
	})

	t.Run("Set Adapters", func(t *testing.T) {
		payload := api.SetLoraAdaptersRequest{
			Model: "test-model",
			Adapters: []api.LoraScaleRequest{
				{ID: 0, Scale: 0.5},
				{ID: 1, Scale: 0.8},
			},
		}
		body, _ := json.Marshal(payload)

		w := httptest.NewRecorder()
		req, _ := http.NewRequest("POST", "/api/lora-adapters", bytes.NewReader(body))
		router.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("expected status 200, got %d", w.Code)
		}

		// Verify scales were updated in the mock
		if mockRunner.adapters[0].Scale != 0.5 {
			t.Errorf("expected adapter 0 scale 0.5, got %f", mockRunner.adapters[0].Scale)
		}
		if mockRunner.adapters[1].Scale != 0.8 {
			t.Errorf("expected adapter 1 scale 0.8, got %f", mockRunner.adapters[1].Scale)
		}
	})
}
