package server

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/internal/proxy"
)

func TestClaudeProxySharedRouteAndBrowserBoundary(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	t.Setenv("OLLAMA_ORIGINS", "*")
	t.Setenv("OLLAMA_HOST", "http://ollama.local")
	handler, err := (&Server{addr: &net.TCPAddr{IP: net.IPv4zero, Port: 80}}).GenerateRoutes()
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		method, path, peer, origin, host string
		status                           int
	}{
		{http.MethodGet, "/api/claude/_ollama/health", "127.0.0.1:1234", "", "", http.StatusNoContent},
		{http.MethodGet, "/api/claude/_ollama/health", "127.0.0.1:1234", "", "localhost", http.StatusNoContent},
		{http.MethodGet, "/api/claude/_ollama/health", "[::1]:1234", "", "[::1]", http.StatusNoContent},
		{http.MethodGet, "/api/claude/_ollama/health", "127.0.0.1:1234", "", "ollama.local", http.StatusNoContent},
		{http.MethodGet, "/api/claude/_ollama/health", "10.0.0.7:1234", "", "ollama.local", http.StatusNoContent},
		{http.MethodGet, "/api/claude/_ollama/health", "10.0.0.8:1234", "", "ollama.local", http.StatusForbidden},
		{http.MethodGet, "/api/claude/_ollama/health", "10.0.0.7:1234", "", "10.0.0.7", http.StatusNoContent},
		{http.MethodGet, "/api/claude/_ollama/health", "10.0.0.8:1234", "", "10.0.0.7", http.StatusForbidden},
		{http.MethodGet, "/api/claude/_ollama/health", "[::1]:1234", "", "[::1]:11434", http.StatusNoContent},
		{http.MethodGet, "/api/claude/_ollama/health", "192.0.2.1:1234", "", "", http.StatusForbidden},
		{http.MethodGet, "/api/claude/_ollama/health", "127.0.0.1:1234", "", "evil.example:11434", http.StatusForbidden},
		{http.MethodOptions, "/api/claude/v1/messages", "127.0.0.1:1234", "http://localhost:3000", "", http.StatusForbidden},
		{http.MethodGet, "/api/claude/v1/models", "127.0.0.1:1234", "", "", http.StatusOK},
		{http.MethodGet, "/v1/models", "192.0.2.1:1234", "", "", http.StatusOK},
		{http.MethodGet, "/", "192.0.2.1:1234", "", "", http.StatusOK},
	} {
		req := httptest.NewRequest(test.method, "http://127.0.0.1"+test.path, nil)
		req.RemoteAddr = test.peer
		req = req.WithContext(context.WithValue(req.Context(), http.LocalAddrContextKey, &net.TCPAddr{IP: net.ParseIP("10.0.0.7"), Port: 80}))
		req.Header.Set("Origin", test.origin)
		req.Header.Set("X-Forwarded-For", "127.0.0.1")
		if test.host != "" {
			req.Host = test.host
		}
		recorder := httptest.NewRecorder()
		handler.ServeHTTP(recorder, req)
		if recorder.Code != test.status {
			t.Errorf("%s %s peer=%s host=%s origin=%s: %d %s", test.method, test.path, test.peer, req.Host, test.origin, recorder.Code, recorder.Body.String())
		}
	}
	t.Run("wildcard interface", func(t *testing.T) {
		t.Setenv("OLLAMA_HOST", "http://0.0.0.0:80")
		handler, err := (&Server{addr: &net.TCPAddr{IP: net.IPv4zero, Port: 80}}).GenerateRoutes()
		if err != nil {
			t.Fatal(err)
		}
		req := httptest.NewRequest(http.MethodGet, "http://10.0.0.7/api/claude/_ollama/health", nil)
		req.RemoteAddr = "10.0.0.7:1234"
		req = req.WithContext(context.WithValue(req.Context(), http.LocalAddrContextKey, &net.TCPAddr{IP: net.ParseIP("10.0.0.7"), Port: 80}))
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, req)
		if response.Code != http.StatusNoContent {
			t.Fatalf("wildcard interface: %d %s", response.Code, response.Body.String())
		}
	})
	t.Run("no home", func(t *testing.T) {
		t.Setenv("HOME", "")
		t.Setenv("USERPROFILE", "")
		handler, err := (&Server{addr: &net.TCPAddr{IP: net.IPv4zero, Port: 80}}).GenerateRoutes()
		if err != nil {
			t.Fatal(err)
		}
		for _, path := range []string{"/", "/api/claude/v1/models"} {
			req := httptest.NewRequest(http.MethodGet, "http://127.0.0.1"+path, nil)
			req.RemoteAddr = "127.0.0.1:1234"
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, req)
			if response.Code != http.StatusOK {
				t.Errorf("%s without a home: %d %s", path, response.Code, response.Body.String())
			}
		}
	})
}

func TestClaudeProxyCatalogChecksCurrentAccess(t *testing.T) {
	environmentProxy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("local access check used the environment proxy")
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer environmentProxy.Close()
	proxyURL, err := url.Parse(environmentProxy.URL)
	if err != nil {
		t.Fatal(err)
	}
	originalTransport := http.DefaultTransport
	transport := originalTransport.(*http.Transport).Clone()
	// Force proxy selection even for the loopback fixture to exercise the
	// default transport policy used for configured hostnames and interfaces.
	transport.Proxy = http.ProxyURL(proxyURL)
	http.DefaultTransport = transport
	t.Cleanup(func() {
		http.DefaultTransport = originalTransport
		transport.CloseIdleConnections()
	})
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	path := filepath.Join(home, ".ollama", proxy.ClaudeDesktopRoutingCatalogFilename)
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(`{"models":[{"route":"claude-sonnet-5","model":"test:cloud","required_plan":"pro"}]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	type accessState struct {
		plan          string
		accountStatus int
		cloudOff      bool
		want          int
	}
	var mu sync.Mutex
	var current accessState
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		switch r.URL.Path {
		case "/api/status":
			_ = json.NewEncoder(w).Encode(api.StatusResponse{Cloud: api.CloudStatus{Disabled: current.cloudOff}})
		case "/api/me":
			w.WriteHeader(current.accountStatus)
			_ = json.NewEncoder(w).Encode(api.UserResponse{Name: "test-account", Plan: current.plan})
		case "/api/tags":
			_ = json.NewEncoder(w).Encode(api.ListResponse{})
		case "/v1/messages":
			if current.want != http.StatusOK {
				t.Error("unavailable model reached inference")
			}
		default:
			if r.URL.Path != "/" {
				t.Errorf("unexpected discovery request: %s", r.URL.Path)
			}
		}
	}))
	defer upstream.Close()
	t.Setenv("OLLAMA_HOST", upstream.URL)
	gateway, err := newClaudeDesktopProxy()
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []accessState{
		{"pro", http.StatusOK, false, http.StatusOK},
		{"free", http.StatusOK, false, http.StatusForbidden},
		{"pro", http.StatusUnauthorized, false, http.StatusUnauthorized},
		{"pro", http.StatusOK, true, http.StatusForbidden},
		{"pro", http.StatusServiceUnavailable, false, http.StatusServiceUnavailable},
		{"pro", http.StatusOK, false, http.StatusOK},
	} {
		mu.Lock()
		current = test
		mu.Unlock()
		for _, route := range []string{"/v1/models", "/v1/messages"} {
			method, want := http.MethodGet, http.StatusOK
			if route == "/v1/messages" {
				method, want = http.MethodPost, test.want
			}
			req := httptest.NewRequest(method, upstream.URL+proxy.ClaudeDesktopPathPrefix+route, strings.NewReader(`{"model":"claude-sonnet-5","messages":[]}`))
			req.RemoteAddr = "127.0.0.1:1234"
			response := httptest.NewRecorder()
			gateway.ServeHTTP(response, req)
			if response.Code != want || route == "/v1/models" && strings.Contains(response.Body.String(), "claude-sonnet-5") != (test.want == http.StatusOK) {
				t.Errorf("state=%+v %s: %d %s", test, route, response.Code, response.Body.String())
			}
		}
	}
}
