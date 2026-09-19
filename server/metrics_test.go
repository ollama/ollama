package server

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func cached(n int) *int { return &n }

// scrape renders the exposition the way MetricsHandler does.
func scrape(t *testing.T, m *metrics, loaded ...runningModel) string {
	t.Helper()
	var sb strings.Builder
	m.write(&sb, loaded)
	return sb.String()
}

func mustContain(t *testing.T, body string, lines ...string) {
	t.Helper()
	for _, want := range lines {
		if !strings.Contains(body, want+"\n") {
			t.Errorf("exposition is missing %q\n--- got ---\n%s", want, body)
		}
	}
}

// The counters are sums over requests, so two requests for one model are one
// line carrying their total.
func TestMetricsCountEveryRequest(t *testing.T) {
	m := newMetrics()

	m.record("qwen3:8b", api.Metrics{
		PromptEvalCount:       10,
		PromptEvalCachedCount: cached(4),
		PromptEvalDuration:    100 * time.Millisecond,
		EvalCount:             20,
		EvalDuration:          2 * time.Second,
		LoadDuration:          500 * time.Millisecond,
	})
	m.record("qwen3:8b", api.Metrics{
		PromptEvalCount:    5,
		PromptEvalDuration: 50 * time.Millisecond,
		EvalCount:          30,
		EvalDuration:       3 * time.Second,
	})

	body := scrape(t, m)
	mustContain(t, body,
		`ollama_requests_total{model="qwen3:8b"} 2`,
		`ollama_prompt_tokens_total{model="qwen3:8b"} 15`,
		`ollama_prompt_cached_tokens_total{model="qwen3:8b"} 4`,
		`ollama_generated_tokens_total{model="qwen3:8b"} 50`,
		`ollama_prompt_eval_duration_seconds_total{model="qwen3:8b"} 0.15`,
		`ollama_eval_duration_seconds_total{model="qwen3:8b"} 5`,
		`ollama_load_duration_seconds_total{model="qwen3:8b"} 0.5`,
	)
}

// Every metric carries its own model, so a server serving two of them can be
// read apart.
func TestMetricsKeepModelsApart(t *testing.T) {
	m := newMetrics()
	m.record("qwen3:8b", api.Metrics{EvalCount: 1})
	m.record("llama3:70b", api.Metrics{EvalCount: 2})

	body := scrape(t, m)
	mustContain(t, body,
		`ollama_generated_tokens_total{model="qwen3:8b"} 1`,
		`ollama_generated_tokens_total{model="llama3:70b"} 2`,
	)
}

// A model can be named anything. One quote in a name would otherwise make
// the whole exposition unparseable, and a scraper drops the lot.
func TestMetricsEscapeLabelValues(t *testing.T) {
	m := newMetrics()
	m.record(`we"ird\model`, api.Metrics{EvalCount: 1})

	body := scrape(t, m)
	mustContain(t, body, `ollama_generated_tokens_total{model="we\"ird\\model"} 1`)
}

// The loaded model gauges say how much of a model sits in GPU memory. A
// value below its size is the reason a model runs slower than it should,
// and nothing else on the endpoint reports it.
func TestMetricsReportWhatIsLoaded(t *testing.T) {
	expires := time.Unix(1_700_000_300, 0)
	m := newMetrics()

	body := scrape(t, m, runningModel{
		name:      "qwen3:8b",
		size:      8_000_000_000,
		sizeVRAM:  6_000_000_000,
		expiresAt: expires,
	})

	mustContain(t, body,
		`ollama_loaded_model_size_bytes{model="qwen3:8b"} 8000000000`,
		`ollama_loaded_model_vram_bytes{model="qwen3:8b"} 6000000000`,
		`ollama_loaded_model_expiry_timestamp_seconds{model="qwen3:8b"} 1.7000003e+09`,
	)
}

// Every metric needs its HELP and TYPE, because a scrape without them is
// accepted but arrives with nothing to say what it means.
func TestMetricsDescribeThemselves(t *testing.T) {
	m := newMetrics()
	m.record("qwen3:8b", api.Metrics{EvalCount: 1})

	body := scrape(t, m)
	for _, name := range []string{
		"ollama_build_info",
		"ollama_requests_total",
		"ollama_prompt_tokens_total",
		"ollama_prompt_cached_tokens_total",
		"ollama_generated_tokens_total",
		"ollama_prompt_eval_duration_seconds_total",
		"ollama_eval_duration_seconds_total",
		"ollama_load_duration_seconds_total",
		"ollama_loaded_model_size_bytes",
		"ollama_loaded_model_vram_bytes",
		"ollama_loaded_model_expiry_timestamp_seconds",
	} {
		if !strings.Contains(body, "# HELP "+name+" ") {
			t.Errorf("%s has no HELP line", name)
		}
		if !strings.Contains(body, "# TYPE "+name+" ") {
			t.Errorf("%s has no TYPE line", name)
		}
	}
}

// Recording into a server that has metrics turned off does nothing, which is
// what lets the handlers call it unconditionally.
func TestMetricsAreSilentWhenDisabled(t *testing.T) {
	var m *metrics
	m.record("qwen3:8b", api.Metrics{EvalCount: 1})
}

// The endpoint is off by default: a server nobody scrapes should not carry a
// route it never serves. OLLAMA_METRICS is what turns it on.
func TestMetricsRouteFollowsTheEnvironment(t *testing.T) {
	for _, tc := range []struct {
		value string
		want  int
	}{
		{"", http.StatusNotFound},
		{"1", http.StatusOK},
	} {
		t.Run("OLLAMA_METRICS="+tc.value, func(t *testing.T) {
			t.Setenv("OLLAMA_METRICS", tc.value)

			s := &Server{sched: &Scheduler{}}
			s.initMetrics()
			router, err := s.GenerateRoutes()
			if err != nil {
				t.Fatalf("generate routes: %v", err)
			}

			w := httptest.NewRecorder()
			router.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/metrics", nil))
			if w.Code != tc.want {
				t.Fatalf("GET /metrics = %d, want %d", w.Code, tc.want)
			}
			if tc.want == http.StatusOK && !strings.Contains(w.Body.String(), "ollama_build_info") {
				t.Errorf("the exposition is empty:\n%s", w.Body.String())
			}
		})
	}
}
