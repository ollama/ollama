package proxy

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
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
	req.Header.Set("X-Api-Key", "private-placeholder")
	req.Header.Set("Proxy-Authorization", "private-placeholder")
	response := httptest.NewRecorder()
	gateway.ServeHTTP(response, req)
	return response
}

func writeClaudeRoutingCatalog(t *testing.T, path, data string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(data), 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestClaudeSharedListenerKeepsConcurrentRequestMappings(t *testing.T) {
	for _, oldFirst := range []bool{true, false} {
		t.Run(fmt.Sprintf("old_first_%t", oldFirst), func(t *testing.T) {
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/v1/messages" {
					_, _ = io.WriteString(w, requestModel(t, r))
				}
			}))
			defer upstream.Close()
			catalogPath := filepath.Join(t.TempDir(), "routing.json")
			names := []string{"old-local", "new-local"}
			started := []chan struct{}{make(chan struct{}), make(chan struct{})}
			gates := []chan struct{}{make(chan struct{}), make(chan struct{})}
			release := []func(){sync.OnceFunc(func() { close(gates[0]) }), sync.OnceFunc(func() { close(gates[1]) })}
			t.Cleanup(release[0])
			t.Cleanup(release[1])
			var calls atomic.Int32
			gateway := startTestGateway(t, upstream.URL, ClaudeDesktopConfig{
				PathPrefix:         ClaudeDesktopPathPrefix,
				RoutingCatalogPath: catalogPath,
				ListLocalModels: func(context.Context) ([]string, error) {
					i := int(calls.Add(1)) - 1
					close(started[i])
					<-gates[i]
					return names, nil
				},
			})
			done := []chan *httptest.ResponseRecorder{make(chan *httptest.ResponseRecorder, 1), make(chan *httptest.ResponseRecorder, 1)}
			for i := range 2 {
				writeClaudeRoutingCatalog(t, catalogPath, fmt.Sprintf(`{"models":[{"route":"claude-sonnet-5","model":%q}]}`, names[i]))
				go func() {
					done[i] <- sharedGatewayRequest(gateway, http.MethodPost, "/v1/messages", `{"model":"claude-sonnet-5","messages":[]}`)
				}()
				<-started[i]
			}
			order := []int{0, 1}
			if !oldFirst {
				order = []int{1, 0}
			}
			for _, i := range order {
				release[i]()
				response := <-done[i]
				if response.Code != http.StatusOK || response.Body.String() != names[i] {
					t.Fatalf("request used mapping %q (%d), want %q", response.Body.String(), response.Code, names[i])
				}
			}
		})
	}
}

func TestClaudeSharedListenerReloadsMappings(t *testing.T) {
	var mu sync.Mutex
	var routed []string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/messages" {
			if r.Header.Get("Authorization") != "" || r.Header.Get("Cookie") != "" || r.Header.Get("X-Api-Key") != "" || r.Header.Get("Proxy-Authorization") != "" {
				t.Error("unexpected forwarded credentials")
			}
			model := requestModel(t, r)
			mu.Lock()
			routed = append(routed, model+" "+r.URL.RequestURI())
			mu.Unlock()
			_, _ = io.WriteString(w, model)
		}
	}))
	defer upstream.Close()
	catalogPath := filepath.Join(t.TempDir(), "routing.json")
	gateway := startTestGateway(t, upstream.URL, ClaudeDesktopConfig{
		PathPrefix:         ClaudeDesktopPathPrefix,
		RoutingCatalogPath: catalogPath,
		ListLocalModels: func(context.Context) ([]string, error) {
			return []string{"local-one", "local-two", "claude-sonnet-5"}, nil
		},
		ResolveAccessState: func(context.Context) (ClaudeDesktopAccessState, error) {
			return ClaudeDesktopAccessState{Cloud: ClaudeDesktopCloudOn, Account: ClaudeDesktopAccountSignedIn, Plan: "free"}, nil
		},
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
		t.Fatalf("missing catalog status = %+v", status)
	}
	for _, target := range []string{"local-one", "local-two", "local-one:local", "registry.ollama.ai/library/local-one:latest:local"} {
		writeClaudeRoutingCatalog(t, catalogPath, fmt.Sprintf(`{"models":[{"route":"claude-opus-5","model":%q,"max_tokens":4096},{"route":"claude-sonnet-5","model":%q,"max_tokens":8192}]}`, target, target))
		if status := readStatus(); len(status.Models) != 2 || status.Models[0].Model.OllamaModel != target {
			t.Fatalf("saved mapping status = %+v", status)
		}
		response := sharedGatewayRequest(gateway, http.MethodGet, "/v1/models", "")
		var catalog struct{ Data []gatewayModel }
		if err := json.Unmarshal(response.Body.Bytes(), &catalog); err != nil || response.Code != http.StatusOK || len(catalog.Data) != 2 || catalog.Data[0].ID != "claude-opus-5" || catalog.Data[1].ID != "claude-sonnet-5" || catalog.Data[1].DisplayName != target || catalog.Data[0].MaxTokens != 4096 || catalog.Data[1].MaxTokens != 8192 {
			t.Fatalf("catalog = %s, %v", response.Body.String(), err)
		}
		if response := sharedGatewayRequest(gateway, http.MethodPost, "/v1/messages?beta=true", `{"model":"claude-sonnet-5","messages":[]}`); response.Code != http.StatusOK {
			t.Fatalf("messages = %d", response.Code)
		}
	}
	mu.Lock()
	got := strings.Join(routed, ",")
	mu.Unlock()
	if got != "local-one /v1/messages?beta=true,local-two /v1/messages?beta=true,local-one:local /v1/messages?beta=true,registry.ollama.ai/library/local-one:latest:local /v1/messages?beta=true" || readStatus().Counts.Routed != 4 {
		t.Fatalf("upstream requests = %s", got)
	}
	writeClaudeRoutingCatalog(t, catalogPath, `{"models":[{"route":"claude-opus-5","model":"claude-sonnet-5"},{"route":"claude-sonnet-5","model":"local-one"}]}`)
	for _, model := range []string{"claude-sonnet-5", "local-one"} {
		response := sharedGatewayRequest(gateway, http.MethodPost, "/v1/messages", fmt.Sprintf(`{"model":%q,"messages":[]}`, model))
		if response.Code != http.StatusOK || response.Body.String() != "local-one" {
			t.Fatalf("route or target alias %q = %d %s", model, response.Code, response.Body.String())
		}
	}
	writeClaudeRoutingCatalog(t, catalogPath, `{"models":[{"route":"claude-sonnet-5","model":"test:cloud","required_plan":"free"}]}`)
	if response := sharedGatewayRequest(gateway, http.MethodPost, "/v1/messages", `{"model":"claude-sonnet-5","messages":[]}`); response.Code != http.StatusOK {
		t.Fatalf("verified free model = %d %s", response.Code, response.Body.String())
	}
	for _, invalid := range []string{
		`{`, `null`,
		`{"models":[{"route":"unknown","model":"local-one"}]}`,
		`{"models":[{"route":"claude-sonnet-5","model":"local-one"},{"route":"claude-sonnet-5","model":"local-two"}]}`,
		`{"models":[{"route":"claude-sonnet-5","model":"test:cloud"}]}`,
		`{"models":[{"route":"claude-sonnet-5","model":"test","required_plan":"pro"}]}`,
		`{"models":[{"route":"claude-sonnet-5","model":":cloud","required_plan":"free"}]}`,
	} {
		writeClaudeRoutingCatalog(t, catalogPath, invalid)
		if response := sharedGatewayRequest(gateway, http.MethodGet, "/v1/models", ""); response.Code != http.StatusServiceUnavailable {
			t.Fatalf("invalid catalog %s: %d", invalid, response.Code)
		}
		if data, err := os.ReadFile(catalogPath); err != nil || string(data) != invalid {
			t.Fatalf("reader changed catalog: %s, %v", data, err)
		}
	}
}
