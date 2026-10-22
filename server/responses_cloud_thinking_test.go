package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

func TestCloudResponsesWebSearchThinking(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	t.Setenv("OLLAMA_NO_CLOUD", "")
	originalSignRequest := cloudProxySignRequest
	cloudProxySignRequest = func(context.Context, *http.Request) error { return nil }
	t.Cleanup(func() { cloudProxySignRequest = originalSignRequest })

	boolean := &model.Thinking{Values: []any{false, true}, Default: false}
	named := &model.Thinking{Values: []any{false, "low", "medium", "xhigh"}, Default: "medium"}
	always := &model.Thinking{Values: []any{true}, Default: true}
	for _, tc := range []struct {
		name     string
		thinking *model.Thinking
		fields   string
		want     any
		family   string
		showCode int
		wantCode int
	}{
		{name: "boolean effort", thinking: boolean, fields: `,"reasoning":{"effort":"high"}`, want: true},
		{name: "omitted", thinking: boolean},
		{name: "explicit off", thinking: boolean, fields: `,"reasoning":{"effort":"none"}`, want: false},
		{name: "native off overrides effort", thinking: named, fields: `,"think":false,"reasoning":{"effort":"high"}`, want: false},
		{name: "native on overrides off", thinking: named, fields: `,"think":true,"reasoning":{"effort":"none"}`, want: true},
		{name: "supported low", thinking: named, fields: `,"reasoning":{"effort":"low"}`, want: "low"},
		{name: "supported xhigh", thinking: named, fields: `,"reasoning":{"effort":"xhigh"}`, want: "xhigh"},
		{name: "native named override", thinking: named, fields: `,"think":"xhigh","reasoning":{"effort":"low"}`, want: "xhigh"},
		{name: "unsupported name is rejected", thinking: named, fields: `,"reasoning":{"effort":"max"}`, wantCode: http.StatusBadRequest},
		{name: "unknown name is rejected", thinking: named, fields: `,"reasoning":{"effort":"future"}`, wantCode: http.StatusBadRequest},
		{name: "always thinking omission", thinking: always},
		{name: "always thinking explicit off", thinking: always, fields: `,"reasoning":{"effort":"none"}`, wantCode: http.StatusBadRequest},
		{name: "always thinking native off", thinking: always, fields: `,"think":false`, wantCode: http.StatusBadRequest},
		{name: "nonthinking effort", thinking: &model.Thinking{Values: []any{false}, Default: false}, fields: `,"reasoning":{"effort":"high"}`, wantCode: http.StatusBadRequest},
		{name: "Harmony unsupported effort", thinking: &model.Thinking{Values: []any{"low", "medium", "high"}, Default: "medium"}, family: "gptoss", fields: `,"reasoning":{"effort":"xhigh"}`, wantCode: http.StatusBadRequest},
		{name: "unknown metadata rejects conversion", fields: `,"reasoning":{"effort":"xhigh"}`, wantCode: http.StatusServiceUnavailable},
		{name: "unavailable metadata rejects conversion", showCode: http.StatusServiceUnavailable, fields: `,"reasoning":{"effort":"xhigh"}`, wantCode: http.StatusServiceUnavailable},
		{name: "metadata needs sign in", showCode: http.StatusUnauthorized, fields: `,"reasoning":{"effort":"high"}`, wantCode: http.StatusUnauthorized},
		{name: "missing cloud model", showCode: http.StatusNotFound, fields: `,"reasoning":{"effort":"high"}`, wantCode: http.StatusNotFound},
		{name: "metadata rate limit", showCode: http.StatusTooManyRequests, fields: `,"reasoning":{"effort":"high"}`, wantCode: http.StatusTooManyRequests},
		{name: "invalid metadata rejects conversion", thinking: &model.Thinking{Values: []any{"high"}, Default: "missing"}, fields: `,"reasoning":{"effort":"future"}`, wantCode: http.StatusServiceUnavailable},
		{name: "unknown metadata rejects unknown effort", fields: `,"reasoning":{"effort":"future"}`, wantCode: http.StatusServiceUnavailable},
		{name: "invalid native type", thinking: named, fields: `,"think":75`, wantCode: http.StatusBadRequest},
	} {
		for _, streaming := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream=%t", tc.name, streaming), func(t *testing.T) {
				captured := make(chan *api.ThinkValue, 2)
				var showCalls atomic.Int32
				upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					switch r.URL.Path {
					case "/api/show":
						showCalls.Add(1)
						if tc.showCode != 0 {
							w.WriteHeader(tc.showCode)
							return
						}
						json.NewEncoder(w).Encode(api.ShowResponse{Thinking: tc.thinking, Details: api.ModelDetails{Family: tc.family}})
					case "/api/chat":
						var req api.ChatRequest
						if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
							t.Error(err)
							w.WriteHeader(http.StatusBadRequest)
							return
						}
						captured <- req.Think
						w.Header().Set("Content-Type", "application/x-ndjson")
						if len(captured) == 1 {
							fmt.Fprintln(w, `{"message":{"role":"assistant","tool_calls":[{"id":"call_1","function":{"name":"web_search","arguments":{"query":"test"}}}]},"done":true}`)
						} else {
							fmt.Fprintln(w, `{"message":{"role":"assistant","content":"answer"},"done":true}`)
						}
					case "/api/web_search":
						fmt.Fprintln(w, `{"results":[{"title":"Test","url":"https://example.com","content":"result"}]}`)
					default:
						t.Errorf("unexpected upstream path %q", r.URL.Path)
						w.WriteHeader(http.StatusNotFound)
					}
				}))
				defer upstream.Close()
				withCloudProxyBaseURL(t, upstream.URL)
				s := &Server{modelCaches: &modelCaches{show: newModelShowCache()}}
				router, err := s.GenerateRoutes()
				if err != nil {
					t.Fatal(err)
				}
				local := httptest.NewServer(router)
				defer local.Close()
				t.Setenv("OLLAMA_HOST", local.URL)
				body := fmt.Sprintf(`{"model":"thinking-test:cloud","input":"hello","tools":[{"type":"web_search"}],"stream":%t%s}`, streaming, tc.fields)
				w := httptest.NewRecorder()
				router.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body)))
				if tc.wantCode != 0 {
					if w.Code != tc.wantCode || len(captured) != 0 {
						t.Fatalf("status=%d model requests=%d, want %d with no inference: %s", w.Code, len(captured), tc.wantCode, w.Body.String())
					}
					if tc.wantCode == http.StatusServiceUnavailable && (!strings.Contains(w.Body.String(), "thinking_metadata_unavailable") || w.Header().Get("Retry-After") == "" || !strings.HasPrefix(w.Header().Get("Content-Type"), "application/json")) {
						t.Fatalf("metadata failure needs a retryable JSON error: headers=%v body=%s", w.Header(), w.Body.String())
					}
					if tc.wantCode == http.StatusBadRequest && tc.thinking.Valid() && tc.name != "invalid native type" && !strings.Contains(w.Body.String(), "supported values") {
						t.Fatalf("unsupported thinking needs supported choices: %s", w.Body.String())
					}
					return
				}
				if tc.fields == "" && showCalls.Load() != 0 {
					t.Fatal("omitted thinking must not need cloud metadata")
				}
				if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), "answer") {
					t.Fatalf("status %d: %s", w.Code, w.Body.String())
				}
				if len(captured) != 2 {
					t.Fatalf("got %d model requests, want initial request and search follow-up", len(captured))
				}
				for i := range 2 {
					think := <-captured
					var got any
					if think != nil {
						got = think.Value
					}
					if got != tc.want {
						t.Errorf("model request %d: think=%#v, want %#v", i, got, tc.want)
					}
				}
			})
		}
	}
}

func TestCloudThinkingLookupWaitAndRecovery(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	t.Setenv("OLLAMA_NO_CLOUD", "")
	originalSignRequest := cloudProxySignRequest
	cloudProxySignRequest = func(context.Context, *http.Request) error { return nil }
	t.Cleanup(func() { cloudProxySignRequest = originalSignRequest })

	release := make(chan struct{})
	var releaseOnce sync.Once
	unblock := func() { releaseOnce.Do(func() { close(release) }) }
	defer unblock()
	var showCalls, modelCalls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/show":
			showCalls.Add(1)
			select {
			case <-release:
			case <-r.Context().Done():
				return
			}
			json.NewEncoder(w).Encode(api.ShowResponse{Thinking: &model.Thinking{Values: []any{false, "xhigh"}, Default: "xhigh"}})
		case "/api/chat":
			modelCalls.Add(1)
			var req api.ChatRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Think == nil || req.Think.Value != "xhigh" {
				t.Errorf("incorrect recovered request: think=%v error=%v", req.Think, err)
			}
			fmt.Fprintln(w, `{"message":{"role":"assistant","content":"answer"},"done":true}`)
		default:
			t.Errorf("unexpected upstream path %q", r.URL.Path)
		}
	}))
	defer upstream.Close()
	// Unblock before closing the test server, including on an assertion failure.
	defer unblock()
	withCloudProxyBaseURL(t, upstream.URL)
	cache := newModelShowCache()
	s := &Server{modelCaches: &modelCaches{show: cache}}
	router, err := s.GenerateRoutes()
	if err != nil {
		t.Fatal(err)
	}
	request := func() *httptest.ResponseRecorder {
		w := httptest.NewRecorder()
		router.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"model":"thinking-test:cloud","input":"hello","tools":[{"type":"web_search"}],"reasoning":{"effort":"xhigh"}}`)))
		return w
	}
	responses := make(chan *httptest.ResponseRecorder, 2)
	for range 2 {
		go func() { responses <- request() }()
	}
	deadline := time.NewTimer(1500 * time.Millisecond)
	defer deadline.Stop()
	for range 2 {
		select {
		case w := <-responses:
			if w.Code != http.StatusServiceUnavailable || !strings.Contains(w.Body.String(), "thinking_metadata_unavailable") {
				t.Fatalf("slow metadata must produce a retryable error: status=%d body=%s", w.Code, w.Body.String())
			}
		case <-deadline.C:
			t.Fatal("inference request waited for the full metadata fetch")
		}
	}
	if showCalls.Load() != 1 || modelCalls.Load() != 0 {
		t.Fatalf("cold requests: metadata=%d inference=%d, want one shared metadata fetch and no inference", showCalls.Load(), modelCalls.Load())
	}
	unblock()
	for deadline := time.Now().Add(time.Second); ; {
		if _, ok := cache.getCloud(modelShowCloudKeyForModel("thinking-test", false)); ok {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("metadata fetch did not complete after the caller stopped waiting")
		}
		time.Sleep(time.Millisecond)
	}
	w := request()
	if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), "answer") || modelCalls.Load() != 1 || showCalls.Load() != 1 {
		t.Fatalf("retry did not recover: status=%d inference=%d body=%s", w.Code, modelCalls.Load(), w.Body.String())
	}
}

func TestCloudThinkingPassthrough(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	t.Setenv("OLLAMA_NO_CLOUD", "")
	originalSignRequest := cloudProxySignRequest
	cloudProxySignRequest = func(context.Context, *http.Request) error { return nil }
	t.Cleanup(func() { cloudProxySignRequest = originalSignRequest })
	captured := make(chan map[string]any, 1)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/responses" && r.URL.Path != "/v1/chat/completions" {
			t.Errorf("passthrough unexpectedly called %s", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
			return
		}
		var req map[string]any
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Error(err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		captured <- req
		fmt.Fprintln(w, `{"ok":true}`)
	}))
	defer upstream.Close()
	withCloudProxyBaseURL(t, upstream.URL)
	s := &Server{}
	router, err := s.GenerateRoutes()
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{"/v1/responses", "/v1/chat/completions"} {
		for _, fields := range []string{"", `,"reasoning":{"effort":"xhigh"},"reasoning_effort":"minimal"`, `,"think":false,"reasoning":{"effort":"high"}`, `,"reasoning":{"effort":"future"}`} {
			t.Run(path+fields, func(t *testing.T) {
				body := `{"model":"thinking-test:cloud","input":"hello","messages":[{"role":"user","content":"hello"}]` + fields + `}`
				w := httptest.NewRecorder()
				router.ServeHTTP(w, httptest.NewRequest(http.MethodPost, path, strings.NewReader(body)))
				if w.Code != http.StatusOK {
					t.Fatalf("status %d: %s", w.Code, w.Body.String())
				}
				var want map[string]any
				if err := json.Unmarshal([]byte(body), &want); err != nil {
					t.Fatal(err)
				}
				want["model"] = "thinking-test"
				select {
				case got := <-captured:
					if !reflect.DeepEqual(got, want) {
						t.Fatalf("passthrough changed thinking fields: got %v, want %v", got, want)
					}
				default:
					t.Fatal("cloud request not forwarded")
				}
			})
		}
	}
}
