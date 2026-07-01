package server

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestMetricsHandler(t *testing.T) {
	// Scheduler with 2 queued requests (buffered channel) and 1 loaded model.
	sched := &Scheduler{
		pendingReqCh: make(chan *LlmRequest, 512),
		loaded:       map[string]*runnerRef{"model-a": {}},
	}
	sched.pendingReqCh <- &LlmRequest{}
	sched.pendingReqCh <- &LlmRequest{}

	s := &Server{sched: sched}
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.GET("/metrics", s.MetricsHandler)

	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/metrics", nil))

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", w.Code, http.StatusOK)
	}

	if ct := w.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/plain; version=0.0.4") {
		t.Errorf("Content-Type = %q, want prefix %q", ct, "text/plain; version=0.0.4")
	}

	body := w.Body.String()
	for _, want := range []string{
		"# HELP ollama_requests_queued Number of requests waiting for a model runner.",
		"# TYPE ollama_requests_queued gauge",
		"ollama_requests_queued 2",
		"ollama_queue_capacity 512",
		"ollama_models_loaded 1",
	} {
		if !strings.Contains(body, want) {
			t.Errorf("metrics output missing line %q\n--- got ---\n%s", want, body)
		}
	}
}

func TestMetricsHandlerEmpty(t *testing.T) {
	// No queued requests and no loaded models should report zeros, not error.
	s := &Server{sched: &Scheduler{
		pendingReqCh: make(chan *LlmRequest, 512),
		loaded:       map[string]*runnerRef{},
	}}
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.GET("/metrics", s.MetricsHandler)

	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/metrics", nil))

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", w.Code, http.StatusOK)
	}
	body := w.Body.String()
	for _, want := range []string{"ollama_requests_queued 0", "ollama_models_loaded 0"} {
		if !strings.Contains(body, want) {
			t.Errorf("metrics output missing line %q\n--- got ---\n%s", want, body)
		}
	}
}
