package proxy

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

func sharedGatewayRequest(gateway *ClaudeDesktop, method, path, body string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(method, "http://localhost:11434"+ClaudeDesktopPathPrefix+path, strings.NewReader(body))
	req.RemoteAddr = "127.0.0.1:1234"
	req.Header.Set("Authorization", "Bearer private-placeholder")
	req.Header.Set("Cookie", "private=test")
	response := httptest.NewRecorder()
	gateway.ServeHTTP(response, req)
	return response
}

func TestClaudeSharedListenerKeepsConcurrentRequestMappings(t *testing.T) {
	for _, path := range []string{"/_ollama/status", "/v1/messages"} {
		for _, oldFirst := range []bool{true, false} {
			order := []int{0, 1}
			if !oldFirst {
				order = []int{1, 0}
			}
			t.Run(fmt.Sprintf("%s/old_first_%t", path, oldFirst), func(t *testing.T) {
				upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					if r.URL.Path == "/v1/messages" {
						_, _ = io.WriteString(w, requestModel(t, r))
					}
				}))
				defer upstream.Close()
				names := []string{"old-local", "new-local"}
				started := []chan struct{}{make(chan struct{}), make(chan struct{})}
				gates := []chan struct{}{make(chan struct{}), make(chan struct{})}
				release := []func(){sync.OnceFunc(func() { close(gates[0]) }), sync.OnceFunc(func() { close(gates[1]) })}
				t.Cleanup(release[0])
				t.Cleanup(release[1])
				var calls atomic.Int32
				gateway := startTestGateway(t, upstream.URL, ClaudeDesktopConfig{
					PathPrefix: ClaudeDesktopPathPrefix,
					RefreshModels: func(context.Context, []ClaudeDesktopModel) ([]ClaudeDesktopModel, error) {
						i := int(calls.Add(1)) - 1
						close(started[i])
						<-gates[i]
						return MapClaudeDesktopModels(nil, map[string]string{"claude-sonnet-5": names[i]}), nil
					},
				})
				method := http.MethodGet
				if path == "/v1/messages" {
					method = http.MethodPost
				}
				done := []chan *httptest.ResponseRecorder{make(chan *httptest.ResponseRecorder, 1), make(chan *httptest.ResponseRecorder, 1)}
				for i := range 2 {
					go func() {
						done[i] <- sharedGatewayRequest(gateway, method, path, `{"model":"claude-sonnet-5","messages":[]}`)
					}()
					<-started[i]
				}
				for _, i := range order {
					release[i]()
					response := <-done[i]
					if response.Code != http.StatusOK {
						t.Fatalf("%s response: %d %s", names[i], response.Code, response.Body.String())
					}
					got := response.Body.String()
					if path == "/_ollama/status" {
						var status claudeDesktopRuntimeStatus
						if err := json.Unmarshal(response.Body.Bytes(), &status); err != nil || len(status.Models) != 1 {
							t.Fatalf("status = %s, %v", response.Body.String(), err)
						}
						got = status.Models[0].Model.OllamaModel
					}
					if got != names[i] {
						t.Fatalf("request used mapping %q, want %q", got, names[i])
					}
				}
			})
		}
	}
}

func TestClaudeSharedListenerReloadsMappings(t *testing.T) {
	var mu sync.Mutex
	var models []ClaudeDesktopModel
	var configErr error
	var routed []string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/messages" {
			if r.Header.Get("Authorization") != "" || r.Header.Get("Cookie") != "" || r.Header.Get("X-Api-Key") != "ollama" {
				t.Error("unexpected forwarded credentials")
			}
			mu.Lock()
			routed = append(routed, requestModel(t, r)+" "+r.URL.RequestURI())
			mu.Unlock()
		}
	}))
	defer upstream.Close()
	gateway := startTestGateway(t, upstream.URL, ClaudeDesktopConfig{
		PathPrefix: ClaudeDesktopPathPrefix,
		RefreshModels: func(context.Context, []ClaudeDesktopModel) ([]ClaudeDesktopModel, error) {
			mu.Lock()
			defer mu.Unlock()
			return models, configErr
		},
		ListLocalModels: func(context.Context) ([]string, error) { return []string{"local-one", "local-two"}, nil },
	})
	if err := gateway.Start(); err == nil {
		t.Fatal("shared handler started a second listener")
	}
	readStatus := func() claudeDesktopRuntimeStatus {
		t.Helper()
		response := sharedGatewayRequest(gateway, http.MethodGet, "/_ollama/status", "")
		var status claudeDesktopRuntimeStatus
		if err := json.Unmarshal(response.Body.Bytes(), &status); err != nil || response.Code != http.StatusOK {
			t.Fatalf("status = %d, %v", response.Code, err)
		}
		return status
	}
	if status := readStatus(); len(status.Models) != 0 {
		t.Fatalf("empty config status = %+v", status)
	}
	for _, target := range []string{"local-one", "local-two"} {
		mu.Lock()
		models = MapClaudeDesktopModels(nil, map[string]string{"claude-sonnet-5": target})
		mu.Unlock()
		if status := readStatus(); len(status.Models) != 1 || status.Models[0].Model.OllamaModel != target {
			t.Fatalf("saved mapping status = %+v", status)
		}
		response := sharedGatewayRequest(gateway, http.MethodGet, "/v1/models", "")
		var catalog struct{ Data []gatewayModel }
		if err := json.Unmarshal(response.Body.Bytes(), &catalog); err != nil || response.Code != http.StatusOK || len(catalog.Data) != 1 || catalog.Data[0].ID != "claude-sonnet-5" || catalog.Data[0].DisplayName != target {
			t.Fatalf("catalog = %s, %v", response.Body.String(), err)
		}
		if response := sharedGatewayRequest(gateway, http.MethodPost, "/v1/messages?beta=true", `{"model":"claude-sonnet-5","messages":[]}`); response.Code != http.StatusOK {
			t.Fatalf("messages = %d", response.Code)
		}
	}
	mu.Lock()
	got := strings.Join(routed, ",")
	mu.Unlock()
	if got != "local-one /v1/messages?beta=true,local-two /v1/messages?beta=true" || readStatus().Counts.Routed != 2 {
		t.Fatalf("upstream requests = %s", got)
	}
	mu.Lock()
	configErr = errors.New("malformed saved selection")
	mu.Unlock()
	if response := sharedGatewayRequest(gateway, http.MethodGet, "/v1/models", ""); response.Code != http.StatusServiceUnavailable {
		t.Fatalf("invalid config = %d", response.Code)
	}
}
