package proxy

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

func TestClaudeSharedListenerKeepsConcurrentRequestMappings(t *testing.T) {
	for _, path := range []string{"/_ollama/status", "/v1/messages"} {
		for _, oldFirst := range []bool{true, false} {
			order := "new completes first"
			if oldFirst {
				order = "old completes first"
			}
			t.Run(path+"/"+order, func(t *testing.T) {
				upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					if r.URL.Path == "/v1/messages" {
						_, _ = io.WriteString(w, requestModel(t, r))
					}
				}))
				defer upstream.Close()
				oldStarted, newStarted := make(chan struct{}), make(chan struct{})
				allowOld, allowNew := make(chan struct{}), make(chan struct{})
				releaseOld := sync.OnceFunc(func() { close(allowOld) })
				releaseNew := sync.OnceFunc(func() { close(allowNew) })
				defer releaseOld()
				defer releaseNew()
				var calls atomic.Int32
				gateway, err := NewClaudeDesktop(ClaudeDesktopConfig{
					PathPrefix: ClaudeDesktopPathPrefix, OllamaURL: upstream.URL,
					RefreshModels: func(context.Context, []ClaudeDesktopModel) ([]ClaudeDesktopModel, error) {
						if calls.Add(1) == 1 {
							close(oldStarted)
							<-allowOld
							return MapClaudeDesktopModels(nil, map[string]string{"claude-sonnet-5": "old-local"}), nil
						}
						close(newStarted)
						<-allowNew
						return MapClaudeDesktopModels(nil, map[string]string{"claude-sonnet-5": "new-local"}), nil
					},
				})
				if err != nil {
					t.Fatal(err)
				}
				defer gateway.Close(t.Context())
				request := func(result chan<- *httptest.ResponseRecorder) {
					method := http.MethodGet
					if path == "/v1/messages" {
						method = http.MethodPost
					}
					req := httptest.NewRequest(method, "http://localhost:11434"+ClaudeDesktopPathPrefix+path, strings.NewReader(`{"model":"claude-sonnet-5","messages":[]}`))
					req.RemoteAddr = "127.0.0.1:1234"
					recorder := httptest.NewRecorder()
					gateway.ServeHTTP(recorder, req)
					result <- recorder
				}
				oldDone, newDone := make(chan *httptest.ResponseRecorder, 1), make(chan *httptest.ResponseRecorder, 1)
				go request(oldDone)
				<-oldStarted
				// The next request reads the changed mapping while the old one waits.
				go request(newDone)
				<-newStarted
				var oldResponse, newResponse *httptest.ResponseRecorder
				if oldFirst {
					releaseOld()
					oldResponse = <-oldDone
					releaseNew()
					newResponse = <-newDone
				} else {
					releaseNew()
					newResponse = <-newDone
					releaseOld()
					oldResponse = <-oldDone
				}
				for want, response := range map[string]*httptest.ResponseRecorder{"old-local": oldResponse, "new-local": newResponse} {
					if response.Code != http.StatusOK {
						t.Fatalf("%s response: %d %s", want, response.Code, response.Body.String())
					}
					got := response.Body.String()
					if path == "/_ollama/status" {
						var status claudeDesktopRuntimeStatus
						if err := json.Unmarshal(response.Body.Bytes(), &status); err != nil || len(status.Models) != 1 {
							t.Fatalf("status = %s, %v", response.Body.String(), err)
						}
						got = status.Models[0].Model.OllamaModel
					}
					if got != want {
						t.Fatalf("request used mapping %q, want %q", got, want)
					}
				}
			})
		}
	}
}

func TestClaudeSharedListenerReloadsMappingsAndPreservesNativeAPI(t *testing.T) {
	var mu sync.Mutex
	var models []ClaudeDesktopModel
	var configErr error
	var routed []string
	var gateway *ClaudeDesktop
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, ClaudeDesktopPathPrefix+"/") {
			gateway.ServeHTTP(w, r)
			return
		}
		switch r.URL.Path {
		case "/":
			w.WriteHeader(http.StatusOK)
		case "/v1/models":
			_, _ = io.WriteString(w, `{"native":true}`)
		case "/v1/messages":
			if r.Header.Get("Authorization") != "" || r.Header.Get("Cookie") != "" || r.Header.Get("X-Api-Key") != "ollama" {
				t.Error("unexpected forwarded credentials")
			}
			model := requestModel(t, r)
			mu.Lock()
			routed = append(routed, model+" "+r.URL.RequestURI())
			mu.Unlock()
			_, _ = io.WriteString(w, `{"ok":true}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	var err error
	gateway, err = NewClaudeDesktop(ClaudeDesktopConfig{
		PathPrefix: ClaudeDesktopPathPrefix, OllamaURL: server.URL,
		RefreshModels: func(context.Context, []ClaudeDesktopModel) ([]ClaudeDesktopModel, error) {
			mu.Lock()
			defer mu.Unlock()
			return models, configErr
		},
		ListLocalModels: func(context.Context) ([]string, error) { return []string{"local-one", "local-two"}, nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	defer gateway.Close(context.Background())
	if err := gateway.Start(); err == nil {
		t.Fatal("shared handler started a second listener")
	}
	baseURL := server.URL + ClaudeDesktopPathPrefix
	readStatus := func() claudeDesktopRuntimeStatus {
		t.Helper()
		response, err := server.Client().Get(baseURL + "/_ollama/status")
		if err != nil {
			t.Fatal(err)
		}
		defer response.Body.Close()
		var status claudeDesktopRuntimeStatus
		if err := json.NewDecoder(response.Body).Decode(&status); err != nil || response.StatusCode != http.StatusOK {
			t.Fatalf("status = %d, %v", response.StatusCode, err)
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
		catalog, err := FetchClaudeDesktopCatalog(context.Background(), nil, baseURL, "test")
		if err != nil || len(catalog) != 1 || catalog[0].ID != "claude-sonnet-5" || catalog[0].DisplayName != target {
			t.Fatalf("catalog = %+v, %v", catalog, err)
		}
		req, _ := http.NewRequest(http.MethodPost, baseURL+"/v1/messages?beta=true", strings.NewReader(`{"model":"claude-sonnet-5","messages":[]}`))
		req.Header.Set("Authorization", "Bearer private-placeholder")
		req.Header.Set("Cookie", "private=test")
		response, err := server.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		response.Body.Close()
		if response.StatusCode != http.StatusOK {
			t.Fatalf("messages = %s", response.Status)
		}
	}
	mu.Lock()
	got := strings.Join(routed, ",")
	mu.Unlock()
	if got != "local-one /v1/messages?beta=true,local-two /v1/messages?beta=true" {
		t.Fatalf("upstream requests = %s", got)
	}
	if readStatus().Counts.Routed != 2 {
		t.Fatal("server request count was not shared")
	}
	response, err := server.Client().Get(server.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(response.Body)
	response.Body.Close()
	if string(body) != `{"native":true}` {
		t.Fatalf("native API changed: %s", body)
	}
	mu.Lock()
	configErr = errors.New("malformed saved selection")
	mu.Unlock()
	response, err = server.Client().Get(baseURL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	response.Body.Close()
	if response.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("invalid config = %s", response.Status)
	}
}

func TestClaudeSharedListenerRejectsRemotePeersAndBrowserRequests(t *testing.T) {
	gateway, err := NewClaudeDesktop(ClaudeDesktopConfig{PathPrefix: ClaudeDesktopPathPrefix, OllamaURL: "http://127.0.0.1:1"})
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		peer, host, origin string
		status             int
	}{
		{"127.0.0.1:40000", "localhost:11434", "", http.StatusNoContent},
		{"[::1]:40000", "[::1]:11434", "", http.StatusNoContent},
		{"192.0.2.10:40000", "localhost:11434", "", http.StatusForbidden},
		{"127.0.0.1:40000", "evil.example:11434", "", http.StatusForbidden},
		{"127.0.0.1:40000", "localhost:11434", "http://localhost:3000", http.StatusForbidden},
	} {
		r := httptest.NewRequest(http.MethodGet, "http://"+test.host+ClaudeDesktopPathPrefix+healthPath, nil)
		r.RemoteAddr, r.Host = test.peer, test.host
		r.Header.Set("Origin", test.origin)
		r.Header.Set("X-Forwarded-For", "127.0.0.1")
		w := httptest.NewRecorder()
		gateway.ServeHTTP(w, r)
		if w.Code != test.status {
			t.Errorf("peer=%s host=%s origin=%s: %d", test.peer, test.host, test.origin, w.Code)
		}
	}
}
