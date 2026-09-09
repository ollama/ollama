package server

import (
	"encoding/json"
	"errors"
	"io"
	"log/slog"
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

func TestCodexProxyStreamTermination(t *testing.T) {
	for _, cloudHop := range []bool{false, true} {
		route := "direct"
		if cloudHop {
			route = "cloud-hop"
		}
		for _, http2 := range []bool{false, true} {
			protocol := "HTTP1"
			if http2 {
				protocol = "HTTP2"
			}
			for _, abort := range []bool{true, false} {
				outcome := "complete"
				if abort {
					outcome = "disconnect"
				}
				t.Run(route+"/"+protocol+"/"+outcome, func(t *testing.T) {
					const partial = "data: {\"type\":\"response.created\"}\n\n"
					const completed = "data: {\"type\":\"response.completed\"}\n\ndata: [DONE]\n\n"
					release := make(chan struct{})
					upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						w.Header().Set("Content-Type", "text/event-stream")
						_, _ = io.WriteString(w, partial)
						w.(http.Flusher).Flush()
						select {
						case <-release:
						case <-r.Context().Done():
							return
						}
						if abort {
							panic(http.ErrAbortHandler)
						}
						_, _ = io.WriteString(w, completed)
					}))
					defer upstream.Close()
					defer close(release)

					home := t.TempDir()
					t.Setenv("HOME", home)
					t.Setenv("OLLAMA_NO_CLOUD", "false")
					t.Setenv("OLLAMA_HOST", upstream.URL)
					catalogDir := filepath.Join(home, ".codex")
					if err := os.MkdirAll(catalogDir, 0o700); err != nil {
						t.Fatal(err)
					}
					if err := os.WriteFile(filepath.Join(catalogDir, proxy.CodexDesktopRoutingCatalogFilename), []byte(`{"models":[{"slug":"stream-probe:cloud"}]}`), 0o600); err != nil {
						t.Fatal(err)
					}

					serverLog, err := os.Create(filepath.Join(home, "server.log"))
					if err != nil {
						t.Fatal(err)
					}
					defer serverLog.Close()
					oldLogger, oldWriter, oldErrorWriter := slog.Default(), gin.DefaultWriter, gin.DefaultErrorWriter
					slog.SetDefault(slog.New(slog.NewTextHandler(serverLog, nil)))
					gin.DefaultWriter, gin.DefaultErrorWriter = serverLog, serverLog
					defer func() {
						slog.SetDefault(oldLogger)
						gin.DefaultWriter, gin.DefaultErrorWriter = oldWriter, oldErrorWriter
					}()

					// The extra HTTP listener exercises the same two-hop route used by
					// ollama serve: /api/codex/v1/responses -> /v1/responses -> cloud.
					var inner *httptest.Server
					if cloudHop {
						inner = httptest.NewUnstartedServer(nil)
						defer inner.Close()
						t.Setenv("OLLAMA_HOST", "http://"+inner.Listener.Addr().String())
						original := cloudProxyBaseURL
						cloudProxyBaseURL = upstream.URL
						defer func() { cloudProxyBaseURL = original }()
					}
					handler, err := (&Server{}).GenerateRoutes()
					if err != nil {
						t.Fatal(err)
					}
					if inner != nil {
						inner.Config.Handler = handler
						inner.Start()
					}
					done := make(chan struct{})
					server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						if r.URL.Path == "/api/codex/v1/responses" {
							defer close(done)
						}
						handler.ServeHTTP(w, r)
					}))
					server.EnableHTTP2 = http2
					server.StartTLS()
					defer server.Close()
					client := server.Client()
					client.Timeout = 5 * time.Second
					resp, err := client.Post(server.URL+"/api/codex/v1/responses", "application/json", strings.NewReader(`{"model":"stream-probe:cloud","stream":true,"input":[{"role":"user","content":"Synthetic stream probe"}]}`))
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
					// Disconnect only after the real client has received partial output.
					select {
					case release <- struct{}{}:
					case <-time.After(5 * time.Second):
						t.Fatal("upstream did not accept release")
					}
					rest, readErr := io.ReadAll(resp.Body)
					select {
					case <-done:
					case <-time.After(5 * time.Second):
						t.Fatal("proxy handler did not finish")
					}
					t.Logf("client: protocol=%s status=%d partial=%q remaining=%q read_error=%v", resp.Proto, resp.StatusCode, prefix, rest, readErr)
					if abort {
						if !http2 && !errors.Is(readErr, io.ErrUnexpectedEOF) {
							t.Errorf("HTTP/1 read error = %v, want unexpected EOF", readErr)
						} else if http2 && (readErr == nil || !strings.Contains(readErr.Error(), "INTERNAL_ERROR")) {
							t.Errorf("HTTP/2 read error = %v, want stream reset", readErr)
						}
						if len(rest) != 0 {
							t.Errorf("unexpected output after abort: %q", rest)
						}
					} else if readErr != nil || string(rest) != completed {
						t.Errorf("complete stream: remaining=%q error=%v", rest, readErr)
					}

					wantResult, wantErrors := "ok", 0
					if abort {
						wantResult, wantErrors = "stream_error", 1
					}
					logData, err := os.ReadFile(filepath.Join(home, ".ollama", "logs", codexDesktopLogFilename))
					if err != nil {
						t.Fatal(err)
					}
					t.Logf("activity: %s", logData)
					if !strings.Contains(string(logData), "status=200 ") || !strings.Contains(string(logData), "result="+wantResult) {
						t.Errorf("activity log did not record %s: %s", wantResult, logData)
					}
					status, err := client.Get(server.URL + "/api/codex/_status")
					if err != nil {
						t.Fatal(err)
					}
					defer status.Body.Close()
					var metrics struct {
						UpstreamErrors int `json:"upstream_errors"`
					}
					if err := json.NewDecoder(status.Body).Decode(&metrics); err != nil {
						t.Fatal(err)
					}
					if metrics.UpstreamErrors != wantErrors {
						t.Errorf("upstream_errors=%d, want %d", metrics.UpstreamErrors, wantErrors)
					}
					t.Logf("status: upstream_errors=%d", metrics.UpstreamErrors)
					logs, err := os.ReadFile(serverLog.Name())
					if err != nil {
						t.Fatal(err)
					}
					for _, line := range strings.Split(string(logs), "\n") {
						if strings.Contains(line, "level=WARN") || strings.Contains(line, "level=ERROR") || strings.Contains(line, "[Recovery]") {
							t.Logf("server: %s", line)
						}
					}
					if strings.Contains(string(logs), "panic recovered") || strings.Contains(string(logs), "override status code 200 with 500") {
						t.Errorf("middleware treated a stream abort as an application panic:\n%s", logs)
					}
					if abort && !strings.Contains(string(logs), "Codex proxy response stream aborted") {
						t.Error("server log lost the stream failure")
					}
				})
			}
		}
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
