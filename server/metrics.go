package server

import (
	"cmp"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/version"
)

// metrics accumulates what the inference handlers already measure for every
// request, so that a scrape can report it without the server keeping any
// history of its own. It is nil unless OLLAMA_METRICS is set, and every
// method tolerates that, which keeps the call sites free of conditionals.
//
// The exposition is written by hand. The Prometheus text format is a few
// lines of text, and writing it here keeps the feature free of dependencies.
type metrics struct {
	mu      sync.Mutex
	byModel map[string]*modelMetrics
}

// modelMetrics are the counters for one model. Everything here only ever
// grows, which is what a Prometheus counter is, and the scrape reports the
// running totals.
type modelMetrics struct {
	requests           uint64
	promptTokens       uint64
	promptCachedTokens uint64
	generatedTokens    uint64

	// Durations are summed as nanoseconds and divided once, at the scrape.
	// Adding fractions of a second in float64 drifts, and these counters are
	// meant to run for the life of the server.
	promptEvalNanos uint64
	evalNanos       uint64
	loadNanos       uint64
}

func newMetrics() *metrics {
	return &metrics{byModel: make(map[string]*modelMetrics)}
}

// initMetrics turns the endpoint on when OLLAMA_METRICS is set. It is off by
// default: a server that nobody scrapes should not carry a route it never
// serves.
func (s *Server) initMetrics() {
	if !envconfig.Metrics() {
		return
	}

	s.metrics = newMetrics()
	slog.Info("metrics enabled, Prometheus exposition served at /metrics")
}

// record folds one finished request in. The durations come from the runner
// and the counts from the response that is about to go back to the client,
// so nothing here is measured twice or measured differently from what the
// caller is told.
func (m *metrics) record(name string, r api.Metrics) {
	if m == nil {
		return
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	c, ok := m.byModel[name]
	if !ok {
		c = &modelMetrics{}
		m.byModel[name] = c
	}

	c.requests++
	c.promptTokens += uint64(max(r.PromptEvalCount, 0))
	c.generatedTokens += uint64(max(r.EvalCount, 0))
	if r.PromptEvalCachedCount != nil {
		c.promptCachedTokens += uint64(max(*r.PromptEvalCachedCount, 0))
	}
	c.promptEvalNanos += uint64(max(r.PromptEvalDuration, 0))
	c.evalNanos += uint64(max(r.EvalDuration, 0))
	c.loadNanos += uint64(max(r.LoadDuration, 0))
}

// snapshot copies the counters so the exposition is written without holding
// the lock across a slow client's socket.
func (m *metrics) snapshot() map[string]modelMetrics {
	m.mu.Lock()
	defer m.mu.Unlock()

	out := make(map[string]modelMetrics, len(m.byModel))
	for name, c := range m.byModel {
		out[name] = *c
	}
	return out
}

// runningModel is what the scrape reports about a model held in memory. It is
// read from the scheduler at scrape time, the same source /api/ps uses.
type runningModel struct {
	name      string
	size      int64
	sizeVRAM  int64
	expiresAt time.Time
}

// MetricsHandler serves the Prometheus text exposition format.
func (s *Server) MetricsHandler(c *gin.Context) {
	c.Header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	c.Status(http.StatusOK)
	s.metrics.write(c.Writer, s.runningModelsForMetrics())
}

// metricsModelName is the name a model is reported under. It is the one
// /api/ps shows, so a reader correlating the two is not left wondering
// whether they are looking at the same model.
func metricsModelName(m *Model) string {
	return model.ParseName(m.ShortName).DisplayShortest()
}

func (s *Server) runningModelsForMetrics() []runningModel {
	var loaded []runningModel
	for _, v := range s.sched.loadedModels() {
		loaded = append(loaded, runningModel{
			name:      metricsModelName(v.model),
			size:      v.size,
			sizeVRAM:  v.sizeVRAM,
			expiresAt: v.expiresAt,
		})
	}
	return loaded
}

func (m *metrics) write(w io.Writer, loaded []runningModel) {
	io.WriteString(w, "# HELP ollama_build_info The version of the running server.\n")
	io.WriteString(w, "# TYPE ollama_build_info gauge\n")
	fmt.Fprintf(w, "ollama_build_info{version=\"%s\"} 1\n", escapeLabel(version.Version))

	counters := m.snapshot()
	models := make([]string, 0, len(counters))
	for name := range counters {
		models = append(models, name)
	}
	slices.Sort(models)

	counterMetrics := []struct {
		name string
		help string
		of   func(modelMetrics) any
	}{
		{"ollama_requests_total", "Inference requests that ran to completion.", func(c modelMetrics) any { return c.requests }},
		{"ollama_prompt_tokens_total", "Prompt tokens evaluated.", func(c modelMetrics) any { return c.promptTokens }},
		{"ollama_prompt_cached_tokens_total", "Prompt tokens served from the cache rather than evaluated.", func(c modelMetrics) any { return c.promptCachedTokens }},
		{"ollama_generated_tokens_total", "Tokens generated.", func(c modelMetrics) any { return c.generatedTokens }},
		{"ollama_prompt_eval_duration_seconds_total", "Time spent evaluating prompts.", func(c modelMetrics) any { return seconds(c.promptEvalNanos) }},
		{"ollama_eval_duration_seconds_total", "Time spent generating tokens.", func(c modelMetrics) any { return seconds(c.evalNanos) }},
		{"ollama_load_duration_seconds_total", "Time spent loading models for a request.", func(c modelMetrics) any { return seconds(c.loadNanos) }},
	}

	for _, cm := range counterMetrics {
		fmt.Fprintf(w, "# HELP %s %s\n", cm.name, cm.help)
		fmt.Fprintf(w, "# TYPE %s counter\n", cm.name)
		for _, name := range models {
			writeSample(w, cm.name, name, cm.of(counters[name]))
		}
	}

	slices.SortStableFunc(loaded, func(a, b runningModel) int { return cmp.Compare(a.name, b.name) })

	gaugeMetrics := []struct {
		name string
		help string
		of   func(runningModel) any
	}{
		{"ollama_loaded_model_size_bytes", "Size of a loaded model, across GPU and system memory.", func(l runningModel) any { return l.size }},
		{"ollama_loaded_model_vram_bytes", "Part of a loaded model held in GPU memory. Below the size means the rest is in system memory.", func(l runningModel) any { return l.sizeVRAM }},
		{"ollama_loaded_model_expiry_timestamp_seconds", "When the keep alive unloads a model, in unix seconds.", func(l runningModel) any {
			return float64(l.expiresAt.Unix())
		}},
	}

	for _, gm := range gaugeMetrics {
		fmt.Fprintf(w, "# HELP %s %s\n", gm.name, gm.help)
		fmt.Fprintf(w, "# TYPE %s gauge\n", gm.name)
		for _, l := range loaded {
			writeSample(w, gm.name, l.name, gm.of(l))
		}
	}
}

func seconds(nanos uint64) float64 {
	return float64(nanos) / float64(time.Second)
}

// writeSample writes one line of the exposition. The label value is escaped
// here rather than by %q, which would escape the escapes.
func writeSample(w io.Writer, name, label string, value any) {
	switch v := value.(type) {
	case float64:
		fmt.Fprintf(w, "%s{model=\"%s\"} %g\n", name, escapeLabel(label), v)
	default:
		fmt.Fprintf(w, "%s{model=\"%s\"} %v\n", name, escapeLabel(label), v)
	}
}

// escapeLabel escapes the three characters the text format reserves inside a
// label value. A model can be named anything, and one quote in a name would
// otherwise make the whole exposition unparseable.
func escapeLabel(s string) string {
	return strings.NewReplacer(`\`, `\\`, `"`, `\"`, "\n", `\n`).Replace(s)
}
