package server

import (
	"io"
	"net"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"
)

var registerDefaultMuxHandler sync.Once

func TestServeDoesNotUseDefaultServeMux(t *testing.T) {
	t.Setenv("OLLAMA_NOPRUNE", "1")
	t.Setenv("OLLAMA_NO_CLOUD", "1")
	setTestHome(t, t.TempDir())

	// A handler registered on http.DefaultServeMux, the way net/http/pprof
	// registers /debug/pprof/, must not be reachable on the API port.
	const path = "/debug/default-serve-mux-test/"
	registerDefaultMuxHandler.Do(func() {
		http.HandleFunc(path, func(w http.ResponseWriter, r *http.Request) {
			io.WriteString(w, "default mux")
		})
	})

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	errCh := make(chan error, 1)
	go func() { errCh <- Serve(ln) }()
	t.Cleanup(func() {
		ln.Close()
		<-errCh
	})

	client := &http.Client{Timeout: 30 * time.Second}
	base := "http://" + ln.Addr().String()

	resp, err := client.Get(base + "/api/version")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /api/version: got status %d, want %d", resp.StatusCode, http.StatusOK)
	}

	resp, err = client.Get(base + path)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusNotFound || strings.Contains(string(body), "default mux") {
		t.Fatalf("GET %s: got status %d body %q, want 404 from the API routes", path, resp.StatusCode, body)
	}
}
