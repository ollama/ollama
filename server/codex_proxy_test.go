package server

import (
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/internal/proxy"
)

func TestCodexProxyHealthRoute(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	handler, err := (&Server{}).GenerateRoutes()
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodGet, "http://127.0.0.1/api/codex/_health", nil)
	req.RemoteAddr = "127.0.0.1:1234"
	recorder := httptest.NewRecorder()

	handler.ServeHTTP(recorder, req)
	if recorder.Code != http.StatusOK || recorder.Body.String() != `{"ok":true}` {
		t.Fatalf("health response = %d %q", recorder.Code, recorder.Body.String())
	}
}

func TestCodexProxyUpstreamDisconnect(t *testing.T) {
	for _, http2 := range []bool{false, true} {
		name := "HTTP1"
		if http2 {
			name = "HTTP2"
		}
		t.Run(name, func(t *testing.T) {
			const partial = "data: {\"type\":\"response.created\"}\n\n"
			disconnect := make(chan struct{})
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = io.WriteString(w, partial)
				w.(http.Flusher).Flush()
				select {
				case <-disconnect:
				case <-r.Context().Done():
				}
				panic(http.ErrAbortHandler)
			}))
			defer upstream.Close()
			defer close(disconnect)
			home := t.TempDir()
			t.Setenv("HOME", home)
			t.Setenv("OLLAMA_HOST", upstream.URL)
			catalogDir := filepath.Join(home, ".codex")
			if err := os.MkdirAll(catalogDir, 0o700); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(catalogDir, proxy.CodexDesktopRoutingCatalogFilename), []byte(`{"models":[{"slug":"glm-5.3-flash:cloud"}]}`), 0o600); err != nil {
				t.Fatal(err)
			}
			handler, err := (&Server{}).GenerateRoutes()
			if err != nil {
				t.Fatal(err)
			}
			server := httptest.NewUnstartedServer(handler)
			server.EnableHTTP2 = http2
			server.StartTLS()
			defer server.Close()
			client := server.Client()
			client.Timeout = 5 * time.Second
			resp, err := client.Post(server.URL+"/api/codex/v1/responses", "application/json", strings.NewReader(`{"model":"glm-5.3-flash:cloud","stream":true,"input":[{"role":"user","content":"List files"}]}`))
			if err != nil {
				t.Fatal(err)
			}
			defer resp.Body.Close()
			wantProto := 1
			if http2 {
				wantProto = 2
			}
			if resp.StatusCode != http.StatusOK || resp.ProtoMajor != wantProto {
				t.Fatalf("response = %s %s", resp.Proto, resp.Status)
			}
			prefix := make([]byte, len(partial))
			if _, err := io.ReadFull(resp.Body, prefix); err != nil || string(prefix) != partial {
				t.Fatalf("partial response = %q, error = %v", prefix, err)
			}
			disconnect <- struct{}{}
			rest, err := io.ReadAll(resp.Body)
			if err == nil {
				t.Errorf("truncated response ended successfully: %q", rest)
			} else if !http2 && !errors.Is(err, io.ErrUnexpectedEOF) {
				t.Errorf("HTTP/1 read error = %v, want unexpected EOF", err)
			} else if http2 && !strings.Contains(err.Error(), "INTERNAL_ERROR") {
				t.Errorf("HTTP/2 read error = %v, want stream reset", err)
			}
			logData, err := os.ReadFile(filepath.Join(home, ".ollama", "logs", codexDesktopLogFilename))
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(string(logData), "status=200 ") || !strings.Contains(string(logData), "result=stream_error") {
				t.Fatalf("activity log did not record failed stream: %s", logData)
			}
		})
	}
}

func TestServerRecoversOrdinaryPanic(t *testing.T) {
	handler, err := (&Server{}).GenerateRoutes()
	if err != nil {
		t.Fatal(err)
	}
	handler.(*gin.Engine).GET("/test-panic", func(*gin.Context) { panic("test panic") })
	recorder := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/test-panic", nil)
	req.RemoteAddr = "127.0.0.1:1234"
	handler.ServeHTTP(recorder, req)
	if recorder.Code != http.StatusInternalServerError {
		t.Fatalf("panic status = %d, want 500", recorder.Code)
	}
}

func TestCodexProxyWebSocketUpgradeRequestsHTTPFallback(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	handler, err := (&Server{}).GenerateRoutes()
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodGet, "http://127.0.0.1/api/codex/v1/responses", nil)
	req.RemoteAddr = "127.0.0.1:1234"
	req.Header.Set("Connection", "Upgrade")
	req.Header.Set("Upgrade", "websocket")
	recorder := httptest.NewRecorder()

	handler.ServeHTTP(recorder, req)
	if recorder.Code != http.StatusUpgradeRequired {
		t.Fatalf("WebSocket fallback response = %d %q, want 426", recorder.Code, recorder.Body.String())
	}
}

func TestCodexProxyRemainsLocalOnExposedOllamaListener(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	handler, err := (&Server{addr: &net.TCPAddr{IP: net.IPv4zero, Port: 11434}}).GenerateRoutes()
	if err != nil {
		t.Fatal(err)
	}

	for _, tt := range []struct {
		path string
		want int
	}{
		{path: "/", want: http.StatusOK},
		{path: "/api/codex/_health", want: http.StatusForbidden},
	} {
		req := httptest.NewRequest(http.MethodGet, "http://192.0.2.1:11434"+tt.path, nil)
		req.RemoteAddr = "192.0.2.10:1234"
		recorder := httptest.NewRecorder()

		handler.ServeHTTP(recorder, req)
		if recorder.Code != tt.want {
			t.Fatalf("%s status = %d, want %d", tt.path, recorder.Code, tt.want)
		}
	}
}
