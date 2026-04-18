package server

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gin-gonic/gin"
)

const traceIDHeader = "X-Trace-Id"

var latencyBucketsMs = []int64{50, 100, 250, 500, 1000, 2500, 5000, 10000}

type observabilityCollector struct {
	requestsTotal      sync.Map // key route|status -> *atomic.Uint64
	failuresTotal      sync.Map // key route|reason -> *atomic.Uint64
	latencyBucketTotal sync.Map // key route|le -> *atomic.Uint64
	tokensTotal        sync.Map // key route|kind(prompt|completion|total) -> *atomic.Uint64
	cacheSignalsTotal  sync.Map // key route|signal(cache_hit|retrieval_hit) -> *atomic.Uint64
	inflight           atomic.Int64
}

func newObservabilityCollector() *observabilityCollector {
	return &observabilityCollector{}
}

func (o *observabilityCollector) middleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		traceID := c.GetHeader(traceIDHeader)
		if traceID == "" {
			traceID = newTraceID()
		}
		c.Writer.Header().Set(traceIDHeader, traceID)

		ow := &observabilityWriter{ResponseWriter: c.Writer}
		c.Writer = ow

		start := time.Now()
		o.inflight.Add(1)
		defer o.inflight.Add(-1)

		c.Next()

		route := c.FullPath()
		if route == "" {
			route = c.Request.URL.Path
		}
		status := c.Writer.Status()
		latencyMs := time.Since(start).Milliseconds()

		incCounter(&o.requestsTotal, route+"|"+strconv.Itoa(status), 1)
		observeLatency(&o.latencyBucketTotal, route, latencyMs)

		if status >= http.StatusBadRequest {
			reason := "client_error"
			if status >= http.StatusInternalServerError {
				reason = "server_error"
			}
			incCounter(&o.failuresTotal, route+"|"+reason, 1)
		}

		if isTruthy(c.Writer.Header().Get("X-Cache-Hit")) {
			incCounter(&o.cacheSignalsTotal, route+"|cache_hit", 1)
		}
		if isTruthy(c.Writer.Header().Get("X-Retrieval-Hit")) {
			incCounter(&o.cacheSignalsTotal, route+"|retrieval_hit", 1)
		}

		prompt, completion := extractTokenCounts(ow.tailBytes())
		// Extract model name for per-model token metrics (#380).
		model := modelFromGinContext(c)
		if prompt > 0 {
			incCounter(&o.tokensTotal, route+"|prompt", uint64(prompt))
			if model != "" {
				incCounter(&o.tokensTotal, route+"|model="+model+"|prompt", uint64(prompt))
			}
		}
		if completion > 0 {
			incCounter(&o.tokensTotal, route+"|completion", uint64(completion))
			if model != "" {
				incCounter(&o.tokensTotal, route+"|model="+model+"|completion", uint64(completion))
			}
		}
		if prompt+completion > 0 {
			incCounter(&o.tokensTotal, route+"|total", uint64(prompt+completion))
			if model != "" {
				incCounter(&o.tokensTotal, route+"|model="+model+"|total", uint64(prompt+completion))
			}
		}
	}
}

func (o *observabilityCollector) metricsHandler(c *gin.Context) {
	lines := make([]string, 0, 512)
	lines = append(lines,
		"# HELP ollama_http_requests_total Total HTTP requests by route and status.",
		"# TYPE ollama_http_requests_total counter",
	)
	appendCounterMap(&lines, "ollama_http_requests_total", []string{"route", "status"}, &o.requestsTotal)

	lines = append(lines,
		"# HELP ollama_http_failures_total Total failed HTTP requests by route and reason.",
		"# TYPE ollama_http_failures_total counter",
	)
	appendCounterMap(&lines, "ollama_http_failures_total", []string{"route", "reason"}, &o.failuresTotal)

	lines = append(lines,
		"# HELP ollama_http_request_duration_ms_bucket HTTP request latency bucketed in milliseconds.",
		"# TYPE ollama_http_request_duration_ms_bucket counter",
	)
	appendCounterMap(&lines, "ollama_http_request_duration_ms_bucket", []string{"route", "le"}, &o.latencyBucketTotal)

	lines = append(lines,
		"# HELP ollama_tokens_total Token accounting from response payload metrics.",
		"# TYPE ollama_tokens_total counter",
	)
	appendCounterMap(&lines, "ollama_tokens_total", []string{"route", "kind"}, &o.tokensTotal)

	lines = append(lines,
		"# HELP ollama_signal_hits_total Cache and retrieval quality signals from response headers.",
		"# TYPE ollama_signal_hits_total counter",
	)
	appendCounterMap(&lines, "ollama_signal_hits_total", []string{"route", "signal"}, &o.cacheSignalsTotal)

	lines = append(lines,
		"# HELP ollama_http_inflight_requests Current in-flight HTTP requests.",
		"# TYPE ollama_http_inflight_requests gauge",
		fmt.Sprintf("ollama_http_inflight_requests %d", o.inflight.Load()),
	)

	c.Data(http.StatusOK, "text/plain; version=0.0.4", []byte(strings.Join(lines, "\n")+"\n"))
}

type observabilityWriter struct {
	gin.ResponseWriter
	buf strings.Builder
}

func (w *observabilityWriter) Write(data []byte) (int, error) {
	// Keep bounded memory while still allowing token extraction from final payload.
	if w.buf.Len() < 1<<20 {
		remaining := (1 << 20) - w.buf.Len()
		if len(data) > remaining {
			data = data[:remaining]
		}
		w.buf.Write(data)
	}
	return w.ResponseWriter.Write(data)
}

func newTraceID() string {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		return fmt.Sprintf("fallback-%d", time.Now().UnixNano())
	}
	return hex.EncodeToString(b)
}

func isTruthy(v string) bool {
	v = strings.TrimSpace(strings.ToLower(v))
	return v == "1" || v == "true" || v == "yes" || v == "hit"
}

// modelFromGinContext extracts the model name for per-model metrics (#380).
// It checks the X-Ollama-Model header first, then falls back to the cached
// request body stored by body-buffering middleware.
func modelFromGinContext(c *gin.Context) string {
	if m := c.GetHeader("X-Ollama-Model"); m != "" {
		return m
	}
	if raw, ok := c.Get("requestBody"); ok {
		if b, ok2 := raw.([]byte); ok2 {
			var req struct {
				Model string `json:"model"`
			}
			if err := json.Unmarshal(b, &req); err == nil {
				return req.Model
			}
		}
	}
	return ""
}

func observeLatency(m *sync.Map, route string, latencyMs int64) {
	for _, le := range latencyBucketsMs {
		if latencyMs <= le {
			incCounter(m, route+"|"+strconv.FormatInt(le, 10), 1)
		}
	}
	incCounter(m, route+"|+Inf", 1)
}

func incCounter(m *sync.Map, key string, delta uint64) {
	v, _ := m.LoadOrStore(key, &atomic.Uint64{})
	v.(*atomic.Uint64).Add(delta)
}

func appendCounterMap(lines *[]string, metric string, labelNames []string, m *sync.Map) {
	entries := make([]string, 0, 128)
	m.Range(func(k, v any) bool {
		key, ok := k.(string)
		if !ok {
			return true
		}
		parts := strings.Split(key, "|")
		if len(parts) != len(labelNames) {
			return true
		}
		labels := make([]string, 0, len(parts))
		for i, p := range parts {
			labels = append(labels, fmt.Sprintf("%s=%q", labelNames[i], p))
		}
		count := v.(*atomic.Uint64).Load()
		entries = append(entries, fmt.Sprintf("%s{%s} %d", metric, strings.Join(labels, ","), count))
		return true
	})
	sort.Strings(entries)
	*lines = append(*lines, entries...)
}

func extractTokenCounts(payload []byte) (prompt int64, completion int64) {
	if len(payload) == 0 {
		return 0, 0
	}

	var data any
	if err := json.Unmarshal(payload, &data); err != nil {
		// Streaming responses may be NDJSON; try last line.
		lines := strings.Split(string(payload), "\n")
		for i := len(lines) - 1; i >= 0; i-- {
			line := strings.TrimSpace(lines[i])
			if line == "" {
				continue
			}
			if err := json.Unmarshal([]byte(line), &data); err == nil {
				break
			}
		}
	}

	if data == nil {
		return 0, 0
	}

	prompt = int64(findJSONNumber(data, "prompt_eval_count"))
	completion = int64(findJSONNumber(data, "eval_count"))

	if prompt == 0 {
		prompt = int64(findJSONNumber(data, "prompt_tokens"))
	}
	if completion == 0 {
		completion = int64(findJSONNumber(data, "completion_tokens"))
	}

	return maxInt64(prompt, 0), maxInt64(completion, 0)
}

func findJSONNumber(v any, key string) float64 {
	switch t := v.(type) {
	case map[string]any:
		if val, ok := t[key]; ok {
			switch n := val.(type) {
			case float64:
				return n
			case float32:
				return float64(n)
			case int:
				return float64(n)
			case int64:
				return float64(n)
			case json.Number:
				if f, err := n.Float64(); err == nil {
					return f
				}
			}
		}
		for _, child := range t {
			if found := findJSONNumber(child, key); found != 0 {
				return found
			}
		}
	case []any:
		for _, child := range t {
			if found := findJSONNumber(child, key); found != 0 {
				return found
			}
		}
	}
	return 0
}

func maxInt64(v, minV int64) int64 {
	return int64(math.Max(float64(v), float64(minV)))
}
