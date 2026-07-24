package server

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

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

func TestMetricsHandlerOmitsEmptyMetricFamilies(t *testing.T) {
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

	// Empty metric maps should only emit build-info and base gauges.
	for _, unwanted := range []string{
		"http_requests_total",
		"ollama_total_duration_seconds",
		"ollama_load_duration_seconds",
		"ollama_prompt_eval_total",
		"ollama_prompt_eval_duration_seconds",
		"ollama_eval_total",
		"ollama_eval_duration_seconds",
		"ollama_peak_memory_bytes",
	} {
		if strings.Contains(body, "# HELP "+unwanted) {
			t.Fatalf("metrics output should omit helper for %s when no values are recorded", unwanted)
		}
	}

	if !strings.Contains(body, `# HELP ollama_build_info Ollama build information.`) {
		t.Fatalf("metrics output missing build info metric")
	}
}

func TestMetricsHandlerSortedHTTPMetrics(t *testing.T) {
	sched := &Scheduler{
		pendingReqCh: make(chan *LlmRequest, 4),
		loaded:       map[string]*runnerRef{"model-a": {}},
	}

	sched.recordHTTPRequests("chat", http.StatusServiceUnavailable, http.StatusText(http.StatusServiceUnavailable))
	sched.recordHTTPRequests("chat", http.StatusOK, http.StatusText(http.StatusOK))
	sched.recordHTTPRequests("embed", http.StatusBadRequest, http.StatusText(http.StatusBadRequest))
	sched.recordHTTPRequests("all", http.StatusOK, http.StatusText(http.StatusOK))

	s := &Server{sched: sched}
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.GET("/metrics", s.MetricsHandler)

	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/metrics", nil))

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", w.Code, http.StatusOK)
	}

	body := w.Body.String()
	want := []string{
		`http_requests_total{action="all",status="Bad Request",status_code="400"} 1.000000`,
		`http_requests_total{action="all",status="OK",status_code="200"} 3.000000`,
		`http_requests_total{action="all",status="Service Unavailable",status_code="503"} 1.000000`,
		`http_requests_total{action="chat",status="OK",status_code="200"} 1.000000`,
		`http_requests_total{action="chat",status="Service Unavailable",status_code="503"} 1.000000`,
		`http_requests_total{action="embed",status="Bad Request",status_code="400"} 1.000000`,
	}

	prev := -1
	for _, wantLine := range want {
		pos := strings.Index(body, wantLine)
		if pos == -1 {
			t.Fatalf("metrics output missing line %q\n--- got ---\n%s", wantLine, body)
		}
		if pos < prev {
			t.Fatalf("metric sample ordering is incorrect; %q appeared after previous sample", wantLine)
		}
		prev = pos
	}
}

func TestMetricsHandlerReasonLabels(t *testing.T) {
	sched := &Scheduler{
		pendingReqCh: make(chan *LlmRequest, 4),
		loaded:       map[string]*runnerRef{"model-a": {}},
	}

	sched.recordPromptAndEvalMetricsWithMemory("with-reason", "timeout", true, 2*time.Second, time.Second, 500*time.Millisecond, 500*time.Millisecond, 200, 150, 4096)
	sched.recordPromptAndEvalMetricsWithMemory("no-reason", "ignore", false, time.Second, 250*time.Millisecond, 100*time.Millisecond, 150*time.Millisecond, 100, 50, 2048)

	s := &Server{sched: sched}
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.GET("/metrics", s.MetricsHandler)

	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/metrics", nil))

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", w.Code, http.StatusOK)
	}

	body := w.Body.String()

	if !strings.Contains(body, `ollama_prompt_eval_duration_seconds{model="with-reason",reason="timeout"} 0.500000`) {
		t.Fatalf("metrics output missing with-reason sample")
	}
	if !strings.Contains(body, `ollama_prompt_eval_duration_seconds{model="no-reason"} 0.100000`) {
		t.Fatalf("metrics output missing no-reason sample")
	}
	if strings.Contains(body, `ollama_prompt_eval_duration_seconds{model="no-reason",reason="ignore"}`) {
		t.Fatalf("prompt eval should not include reason when includeReason=false")
	}

	if !strings.Contains(body, `ollama_peak_memory_bytes{model="no-reason"} 2048`) {
		t.Fatalf("metrics output missing peak memory sample")
	}

	if !strings.Contains(body, `ollama_peak_memory_bytes{model="with-reason",reason="timeout"} 4096`) {
		t.Fatalf("metrics output missing reasoned peak memory sample")
	}

	if strings.Contains(body, `ollama_peak_memory_bytes 4096`) {
		t.Fatalf("metrics output unexpectedly used unlabeled peak memory sample")
	}
}

func TestMetricsHandlerExpandedMetrics(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	sched := InitScheduler(ctx)
	sched.pendingReqCh = make(chan *LlmRequest, 4)
	sched.loaded = map[string]*runnerRef{"model-a": {}}

	sched.recordHTTPRequests("chat", http.StatusOK, http.StatusText(http.StatusOK))
	sched.recordHTTPRequests("embed", http.StatusBadRequest, http.StatusText(http.StatusBadRequest))
	sched.recordPromptAndEvalMetricsWithMemory("llama3", "stop", true, 1500*time.Millisecond, 500*time.Millisecond, 300*time.Millisecond, 450*time.Millisecond, 131, 62, 120)
	sched.recordPromptAndEvalMetrics("phi4", "", false, 0, 0, 0, 0, 42, 0)

	s := &Server{sched: sched}
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.GET("/metrics", s.MetricsHandler)

	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/metrics", nil))

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", w.Code, http.StatusOK)
	}

	body := w.Body.String()
	for _, want := range []string{
		`# HELP http_requests_total The total number of requests on the endpoints.`,
		`http_requests_total{action="all",status="OK",status_code="200"} 1.000000`,
		`http_requests_total{action="chat",status="OK",status_code="200"} 1.000000`,
		`http_requests_total{action="embed",status="Bad Request",status_code="400"} 1.000000`,
		`http_requests_total{action="all",status="Bad Request",status_code="400"} 1.000000`,
		`# HELP ollama_build_info Ollama build information.`,
		`ollama_build_info{version="0.0.0"} `,
		`# HELP ollama_peak_memory_bytes The peak memory used during computation in bytes.`,
		`ollama_peak_memory_bytes{model="llama3",reason="stop"} 120`,
		"# HELP ollama_total_duration_seconds The request total duration in seconds.",
		`ollama_total_duration_seconds{model="llama3",reason="stop"} 1.500000`,
		`ollama_load_duration_seconds{model="llama3",reason="stop"} 0.500000`,
		`ollama_prompt_eval_total{model="llama3",reason="stop"} 131`,
		`ollama_prompt_eval_total{model="phi4"} 42`,
		`ollama_prompt_eval_duration_seconds{model="llama3",reason="stop"} 0.300000`,
		`ollama_eval_total{model="llama3",reason="stop"} 62`,
		`ollama_eval_duration_seconds{model="llama3",reason="stop"} 0.450000`,
	} {
		if !strings.Contains(body, want) {
			t.Errorf("metrics output missing line %q\n--- got ---\n%s", want, body)
		}
	}
}
