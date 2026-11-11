package server

import (
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestClaudeProxySharedRouteAndBrowserBoundary(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	t.Setenv("OLLAMA_ORIGINS", "*")
	handler, err := (&Server{addr: &net.TCPAddr{IP: net.IPv4zero, Port: 11434}}).GenerateRoutes()
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		method, path, peer, origin, host string
		status                           int
	}{
		{http.MethodGet, "/api/claude/_ollama/health", "127.0.0.1:1234", "", "", http.StatusNoContent},
		{http.MethodGet, "/api/claude/_ollama/health", "[::1]:1234", "", "[::1]:11434", http.StatusNoContent},
		{http.MethodGet, "/api/claude/_ollama/health", "192.0.2.1:1234", "", "", http.StatusForbidden},
		{http.MethodGet, "/api/claude/_ollama/health", "127.0.0.1:1234", "", "evil.example:11434", http.StatusForbidden},
		{http.MethodOptions, "/api/claude/v1/messages", "127.0.0.1:1234", "http://localhost:3000", "", http.StatusForbidden},
		{http.MethodGet, "/api/claude/v1/models", "127.0.0.1:1234", "", "", http.StatusOK},
		{http.MethodGet, "/v1/models", "192.0.2.1:1234", "", "", http.StatusOK},
		{http.MethodGet, "/", "192.0.2.1:1234", "", "", http.StatusOK},
	} {
		req := httptest.NewRequest(test.method, "http://127.0.0.1:11434"+test.path, nil)
		req.RemoteAddr = test.peer
		req.Header.Set("Origin", test.origin)
		req.Header.Set("X-Forwarded-For", "127.0.0.1")
		if test.host != "" {
			req.Host = test.host
		}
		recorder := httptest.NewRecorder()
		handler.ServeHTTP(recorder, req)
		if recorder.Code != test.status {
			t.Errorf("%s %s peer=%s origin=%s: %d %s", test.method, test.path, test.peer, test.origin, recorder.Code, recorder.Body.String())
		}
	}
}
