package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
)

func TestUsageTrackerRecord(t *testing.T) {
	tracker := NewUsageTracker()

	tracker.Record("model-a", 10, 20)
	tracker.Record("model-a", 5, 15)
	tracker.Record("model-b", 100, 200)

	stats := tracker.Stats()

	if len(stats.Usage) != 2 {
		t.Fatalf("expected 2 models, got %d", len(stats.Usage))
	}

	lookup := make(map[string]api.ModelUsageData)
	for _, m := range stats.Usage {
		lookup[m.Model] = m
	}

	a := lookup["model-a"]
	if a.Requests != 2 {
		t.Errorf("model-a requests: expected 2, got %d", a.Requests)
	}
	if a.PromptTokens != 15 {
		t.Errorf("model-a prompt tokens: expected 15, got %d", a.PromptTokens)
	}
	if a.CompletionTokens != 35 {
		t.Errorf("model-a completion tokens: expected 35, got %d", a.CompletionTokens)
	}

	b := lookup["model-b"]
	if b.Requests != 1 {
		t.Errorf("model-b requests: expected 1, got %d", b.Requests)
	}
	if b.PromptTokens != 100 {
		t.Errorf("model-b prompt tokens: expected 100, got %d", b.PromptTokens)
	}
	if b.CompletionTokens != 200 {
		t.Errorf("model-b completion tokens: expected 200, got %d", b.CompletionTokens)
	}
}

func TestUsageTrackerConcurrent(t *testing.T) {
	tracker := NewUsageTracker()

	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			tracker.Record("model-a", 1, 2)
		}()
	}
	wg.Wait()

	stats := tracker.Stats()
	if len(stats.Usage) != 1 {
		t.Fatalf("expected 1 model, got %d", len(stats.Usage))
	}

	m := stats.Usage[0]
	if m.Requests != 100 {
		t.Errorf("requests: expected 100, got %d", m.Requests)
	}
	if m.PromptTokens != 100 {
		t.Errorf("prompt tokens: expected 100, got %d", m.PromptTokens)
	}
	if m.CompletionTokens != 200 {
		t.Errorf("completion tokens: expected 200, got %d", m.CompletionTokens)
	}
}

func TestUsageTrackerStart(t *testing.T) {
	tracker := NewUsageTracker()

	stats := tracker.Stats()
	if stats.Start.IsZero() {
		t.Error("expected non-zero start time")
	}
}

func TestUsageHandler(t *testing.T) {
	gin.SetMode(gin.TestMode)

	s := &Server{
		usage: NewUsageTracker(),
	}

	s.usage.Record("llama3", 50, 100)
	s.usage.Record("llama3", 25, 50)

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodGet, "/api/usage", nil)

	s.UsageHandler(c)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	var resp api.UsageResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to unmarshal response: %v", err)
	}

	if len(resp.Usage) != 1 {
		t.Fatalf("expected 1 model, got %d", len(resp.Usage))
	}

	m := resp.Usage[0]
	if m.Model != "llama3" {
		t.Errorf("expected model llama3, got %s", m.Model)
	}
	if m.Requests != 2 {
		t.Errorf("expected 2 requests, got %d", m.Requests)
	}
	if m.PromptTokens != 75 {
		t.Errorf("expected 75 prompt tokens, got %d", m.PromptTokens)
	}
	if m.CompletionTokens != 150 {
		t.Errorf("expected 150 completion tokens, got %d", m.CompletionTokens)
	}
}
