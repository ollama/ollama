package server

import (
	"net/http"
	"net/http/httptest"
	"testing"
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
