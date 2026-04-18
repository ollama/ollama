package server

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestObservabilityMiddlewareAddsTraceIDAndMetrics(t *testing.T) {
	gin.SetMode(gin.TestMode)
	obs := newObservabilityCollector()

	r := gin.New()
	r.Use(obs.middleware())
	r.POST("/api/chat", func(c *gin.Context) {
		c.Header("X-Cache-Hit", "true")
		c.JSON(http.StatusOK, gin.H{
			"prompt_eval_count": 12,
			"eval_count":        34,
		})
	})
	r.GET("/metrics", obs.metricsHandler)

	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"model":"x"}`))
	req.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", w.Code, http.StatusOK)
	}
	if got := w.Header().Get(traceIDHeader); got == "" {
		t.Fatalf("expected %s header to be set", traceIDHeader)
	}

	mw := httptest.NewRecorder()
	mreq := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	r.ServeHTTP(mw, mreq)

	if mw.Code != http.StatusOK {
		t.Fatalf("metrics status = %d, want %d", mw.Code, http.StatusOK)
	}
	body := mw.Body.String()
	if !strings.Contains(body, "ollama_http_requests_total") {
		t.Fatalf("metrics output missing request counter")
	}
	if !strings.Contains(body, `ollama_tokens_total{route="/api/chat",kind="prompt"} 12`) {
		t.Fatalf("metrics output missing prompt token count: %s", body)
	}
	if !strings.Contains(body, `ollama_tokens_total{route="/api/chat",kind="completion"} 34`) {
		t.Fatalf("metrics output missing completion token count: %s", body)
	}
	if !strings.Contains(body, `ollama_signal_hits_total{route="/api/chat",signal="cache_hit"} 1`) {
		t.Fatalf("metrics output missing cache hit signal: %s", body)
	}
}

func TestExtractTokenCountsFromNDJSON(t *testing.T) {
	payload := "{\"done\":false}\n{\"done\":true,\"prompt_eval_count\":5,\"eval_count\":7}\n"
	prompt, completion := extractTokenCounts([]byte(payload))

	if prompt != 5 || completion != 7 {
		t.Fatalf("prompt/completion = %d/%d, want 5/7", prompt, completion)
	}
}
