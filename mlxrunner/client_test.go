package mlxrunner

import (
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/llm"
)

func TestRequestGrammar(t *testing.T) {
	schema := `{"type":"object","properties":{"answer":{"type":"string"}}}`
	tag := `{"type":"structural_tag","format":{"type":"json_schema","json_schema":` + schema + `}}`
	for _, tt := range []struct {
		name string
		req  llm.CompletionRequest
		want string
	}{
		{name: "unset"},
		{name: "null", req: llm.CompletionRequest{Format: json.RawMessage(`null`)}},
		{name: "empty", req: llm.CompletionRequest{Format: json.RawMessage(`""`)}},
		{
			name: "json",
			req:  llm.CompletionRequest{Format: json.RawMessage(`"json"`)},
			want: `{"type":"structural_tag","format":{"type":"json_schema","json_schema":{"type":"object"}}}`,
		},
		{name: "schema", req: llm.CompletionRequest{Format: json.RawMessage(schema)}, want: tag},
		{
			name: "schema after thinking",
			req:  llm.CompletionRequest{Format: json.RawMessage(schema), ThinkingClose: []string{"</think>"}},
			want: `{"type":"structural_tag","format":{"type":"sequence","elements":[{"type":"any_text","excludes":["</think>"]},` +
				`{"type":"optional","content":{"type":"sequence","elements":[{"type":"const_string","value":"</think>"},` +
				`{"type":"json_schema","json_schema":` + schema + `}]}}]}}`,
		},
		{
			name: "schema after thinking with two closings",
			req:  llm.CompletionRequest{Format: json.RawMessage(schema), ThinkingClose: []string{"<|final|>", "<|final|>json"}},
			want: `{"type":"structural_tag","format":{"type":"sequence","elements":[{"type":"any_text","excludes":["<|final|>","<|final|>json"]},` +
				`{"type":"optional","content":{"type":"sequence","elements":[{"type":"or","elements":[{"type":"const_string","value":"<|final|>"},{"type":"const_string","value":"<|final|>json"}]},` +
				`{"type":"json_schema","json_schema":` + schema + `}]}}]}}`,
		},
		{name: "thinking without a format", req: llm.CompletionRequest{ThinkingClose: []string{"</think>"}}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := string(requestGrammar(tt.req)); got != tt.want {
				t.Fatalf("requestGrammar = %s, want %s", got, tt.want)
			}
		})
	}
}

// newTestClient points a Client at a stub runner serving /v1/status.
func newTestClient(t *testing.T, handler http.HandlerFunc) *Client {
	t.Helper()

	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)

	_, portString, err := net.SplitHostPort(srv.Listener.Addr().String())
	if err != nil {
		t.Fatalf("parse server port: %v", err)
	}
	port, err := strconv.Atoi(portString)
	if err != nil {
		t.Fatalf("parse server port: %v", err)
	}

	return &Client{
		port:      port,
		modelName: "test-model",
		done:      make(chan struct{}),
		client:    srv.Client(),
	}
}

// writeStatus responds the way the runner does: 200 throughout the load, with
// the load state carried in the body.
func writeStatus(w http.ResponseWriter, status statusResponse) {
	json.NewEncoder(w).Encode(status) //nolint:errcheck
}

func TestWaitUntilRunningAdoptsReadyStatus(t *testing.T) {
	// `ollama ps` reports the context length recorded here. Leaving it unset
	// makes the first ps after a load show 0 until some later call refreshes it.
	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		writeStatus(w, statusResponse{Status: llm.ServerStatusReady, Progress: 1, ContextLength: 262144, Memory: 55 << 30})
	})

	if err := c.WaitUntilRunning(t.Context()); err != nil {
		t.Fatalf("WaitUntilRunning error: %v", err)
	}
	if got := c.ContextLength(); got != 262144 {
		t.Errorf("ContextLength() = %d, want 262144", got)
	}
	if total, _ := c.MemorySize(); total != 55<<30 {
		t.Errorf("MemorySize() total = %d, want %d", total, uint64(55)<<30)
	}
}

func TestWaitUntilRunningTimesOutWhenLoadStalls(t *testing.T) {
	t.Setenv("OLLAMA_LOAD_TIMEOUT", "500ms")

	var polls atomic.Int32
	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		if polls.Add(1) == 1 {
			http.Error(w, "not listening yet", http.StatusServiceUnavailable)
			return
		}
		// Progress never advances - a wedged load.
		writeStatus(w, statusResponse{Status: llm.ServerStatusLoadingModel, Progress: 0.25})
	})

	err := c.WaitUntilRunning(t.Context())
	if err == nil {
		t.Fatal("expected a stall timeout")
	}
	if !strings.Contains(err.Error(), "timed out waiting for mlx runner to start") {
		t.Fatalf("expected a load timeout, got %q", err)
	}
	if !strings.Contains(err.Error(), "progress 0.25") {
		t.Fatalf("expected the error to report last progress, got %q", err)
	}
	if strings.Contains(err.Error(), "health check failed") {
		t.Fatalf("timeout reported a stale transient poll error: %q", err)
	}
}

func TestWaitUntilRunningOutlivesTimeoutWhileProgressing(t *testing.T) {
	// The load takes several times the stall timeout but keeps advancing, so
	// it must not be aborted. This is the case a flat deadline gets wrong.
	// Keep the window wide against the 100ms poll interval, or a loaded
	// machine reads as a regression.
	t.Setenv("OLLAMA_LOAD_TIMEOUT", "2s")

	start := time.Now()
	const loadDuration = 3 * time.Second

	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		elapsed := float32(time.Since(start)) / float32(loadDuration)
		if elapsed >= 1 {
			writeStatus(w, statusResponse{Status: llm.ServerStatusReady, Progress: 1})
			return
		}
		writeStatus(w, statusResponse{Status: llm.ServerStatusLoadingModel, Progress: elapsed})
	})

	if err := c.WaitUntilRunning(t.Context()); err != nil {
		t.Fatalf("WaitUntilRunning aborted a progressing load: %v", err)
	}
	if elapsed := time.Since(start); elapsed < loadDuration {
		t.Fatalf("returned after %v, before the load finished at %v", elapsed, loadDuration)
	}
}

func TestPingKeepsLoadingRunnerHealthy(t *testing.T) {
	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		writeStatus(w, statusResponse{Status: llm.ServerStatusLoadingModel, Progress: 0.5, ContextLength: 4096, Memory: 123})
	})
	c.memory.Store(999)

	if err := c.Ping(t.Context()); err != nil {
		t.Fatalf("Ping error while loading: %v", err)
	}
	if got := c.memory.Load(); got != 999 {
		t.Errorf("memory = %d, want the estimate 999 left untouched", got)
	}
	if got := c.contextLength.Load(); got != 0 {
		t.Errorf("contextLength = %d, want 0 while loading", got)
	}
}
