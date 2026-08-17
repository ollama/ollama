package mlxrunner

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/llm"
)

// newTestClient points a Client at a stub runner serving /v1/status.
func newTestClient(t *testing.T, handler http.HandlerFunc) *Client {
	t.Helper()

	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)

	var port int
	parts := strings.Split(srv.URL, ":")
	if _, err := fmt.Sscanf(parts[len(parts)-1], "%d", &port); err != nil {
		t.Fatalf("parsing stub runner port: %v", err)
	}

	return &Client{
		port:      port,
		modelName: "test-model",
		done:      make(chan struct{}),
		client:    http.DefaultClient,
		loadStart: time.Now(),
	}
}

// writeStatus responds the way the runner does: 200 throughout the load, with
// the load state carried in the body.
func writeStatus(w http.ResponseWriter, status statusResponse) {
	json.NewEncoder(w).Encode(status) //nolint:errcheck
}

func TestWaitUntilRunningReturnsWhenReady(t *testing.T) {
	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		writeStatus(w, statusResponse{Status: llm.ServerStatusReady, Progress: 1, ContextLength: 8192, Memory: 1024})
	})

	if err := c.WaitUntilRunning(t.Context()); err != nil {
		t.Fatalf("WaitUntilRunning error: %v", err)
	}
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

	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
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
}

func TestWaitUntilRunningOutlivesTimeoutWhileProgressing(t *testing.T) {
	// The load takes several times the stall timeout but keeps advancing, so
	// it must not be aborted. This is the case a flat deadline gets wrong.
	// Keep the window wide against the 250ms poll interval, or a loaded
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

func TestWaitUntilRunningWaitsForPostLoadWork(t *testing.T) {
	// Weights finish materializing before the runner is ready to serve -
	// wired memory, caches, and compile still run. Progress sits at 1.0
	// throughout, which must not be read as a stall.
	t.Setenv("OLLAMA_LOAD_TIMEOUT", "5s")

	var polls atomic.Int32
	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		if polls.Add(1) < 5 {
			writeStatus(w, statusResponse{Status: llm.ServerStatusLoadingModel, Progress: 1})
			return
		}
		writeStatus(w, statusResponse{Status: llm.ServerStatusReady, Progress: 1})
	})

	if err := c.WaitUntilRunning(t.Context()); err != nil {
		t.Fatalf("WaitUntilRunning error: %v", err)
	}
	if polls.Load() < 5 {
		t.Fatalf("returned after %d polls, before the runner reported ready", polls.Load())
	}
}

func TestWaitUntilRunningIgnoresProgressGoingBackwards(t *testing.T) {
	t.Setenv("OLLAMA_LOAD_TIMEOUT", "500ms")

	var polls atomic.Int32
	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		progress := float32(0.5)
		if polls.Add(1) > 1 {
			progress = 0.1
		}
		writeStatus(w, statusResponse{Status: llm.ServerStatusLoadingModel, Progress: progress})
	})

	err := c.WaitUntilRunning(t.Context())
	if err == nil {
		t.Fatal("expected a stall timeout, a regressing progress value reset the timer")
	}
	if !strings.Contains(err.Error(), "progress 0.50") {
		t.Fatalf("expected progress to stay at its high-water mark, got %q", err)
	}
}

func TestWaitUntilRunningTimeoutReportsWhyPollsFailed(t *testing.T) {
	// With nothing on stderr, the timeout must name why the polls failed.
	t.Setenv("OLLAMA_LOAD_TIMEOUT", "500ms")

	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "nope", http.StatusInternalServerError)
	})
	c.status = llm.NewStatusWriter(nil)

	err := c.WaitUntilRunning(t.Context())
	if err == nil {
		t.Fatal("expected a timeout")
	}
	if !strings.Contains(err.Error(), "health check failed: 500") {
		t.Fatalf("expected the error to name the failing poll, got %q", err)
	}
}

func TestWaitUntilRunningToleratesUnreachableRunnerWhileStarting(t *testing.T) {
	t.Setenv("OLLAMA_LOAD_TIMEOUT", "2s")

	var polls atomic.Int32
	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		// Simulate the runner not answering yet, then coming up.
		if polls.Add(1) < 3 {
			http.Error(w, "nope", http.StatusInternalServerError)
			return
		}
		writeStatus(w, statusResponse{Status: llm.ServerStatusReady, Progress: 1})
	})

	if err := c.WaitUntilRunning(t.Context()); err != nil {
		t.Fatalf("WaitUntilRunning gave up on a runner that was still starting: %v", err)
	}
}

func TestWaitUntilRunningReportsSubprocessExit(t *testing.T) {
	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		writeStatus(w, statusResponse{Status: llm.ServerStatusLoadingModel})
	})
	c.status = llm.NewStatusWriter(nil)
	c.status.SetLastError("mlx: out of memory")
	close(c.done)

	err := c.WaitUntilRunning(t.Context())
	if err == nil {
		t.Fatal("expected an error after the runner exited")
	}
	if !strings.Contains(err.Error(), "mlx: out of memory") {
		t.Fatalf("expected the runner's last error, got %q", err)
	}
}

func TestPingRejectsLoadingRunner(t *testing.T) {
	// The scheduler reads MemorySize before the load finishes. Ping must fail
	// then, so the manifest-derived estimate is not replaced by a partial one.
	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		writeStatus(w, statusResponse{Status: llm.ServerStatusLoadingModel, Progress: 0.5, ContextLength: 4096, Memory: 123})
	})
	c.memory.Store(999)

	if err := c.Ping(t.Context()); err == nil {
		t.Fatal("expected Ping to fail while the model is loading")
	}
	if got := c.memory.Load(); got != 999 {
		t.Errorf("memory = %d, want the estimate 999 left untouched", got)
	}
	if got := c.contextLength.Load(); got != 0 {
		t.Errorf("contextLength = %d, want 0 while loading", got)
	}
}

func TestPingUpdatesFromReadyRunner(t *testing.T) {
	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		writeStatus(w, statusResponse{Status: llm.ServerStatusReady, Progress: 1, ContextLength: 4096, Memory: 123})
	})
	c.memory.Store(999)

	if err := c.Ping(t.Context()); err != nil {
		t.Fatalf("Ping error: %v", err)
	}
	if got := c.memory.Load(); got != 123 {
		t.Errorf("memory = %d, want 123", got)
	}
	if got := c.contextLength.Load(); got != 4096 {
		t.Errorf("contextLength = %d, want 4096", got)
	}
}

func TestGetServerStatusDecodesLoadingBody(t *testing.T) {
	// A 200 with a loading body is the normal case during a load, not a
	// signal that the model is ready.
	c := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		writeStatus(w, statusResponse{Status: llm.ServerStatusLoadingModel, Progress: 0.75})
	})

	status, err := c.getServerStatus(t.Context())
	if err != nil {
		t.Fatalf("getServerStatus error: %v", err)
	}
	if status.Status != llm.ServerStatusLoadingModel {
		t.Errorf("Status = %v, want %v", status.Status, llm.ServerStatusLoadingModel)
	}
	if status.Progress != 0.75 {
		t.Errorf("Progress = %v, want 0.75", status.Progress)
	}
}
