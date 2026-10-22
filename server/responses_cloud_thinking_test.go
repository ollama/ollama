package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

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
		{name: "unsupported name reaches model", thinking: named, fields: `,"reasoning":{"effort":"max"}`, want: "max"},
		{name: "unknown name reaches model", thinking: named, fields: `,"reasoning":{"effort":"future"}`, want: "future"},
		{name: "always thinking omission", thinking: always},
		{name: "always thinking explicit off", thinking: always, fields: `,"reasoning":{"effort":"none"}`, want: false},
		{name: "nonthinking effort", thinking: &model.Thinking{Values: []any{false}, Default: false}, fields: `,"reasoning":{"effort":"high"}`, want: "high"},
		{name: "Harmony retains aliases", thinking: &model.Thinking{Values: []any{"low", "medium", "high"}, Default: "medium"}, family: "gptoss", fields: `,"reasoning":{"effort":"xhigh"}`, want: "max"},
		{name: "unknown metadata retains aliases", fields: `,"reasoning":{"effort":"xhigh"}`, want: "max"},
		{name: "unavailable metadata retains aliases", showCode: http.StatusServiceUnavailable, fields: `,"reasoning":{"effort":"xhigh"}`, want: "max"},
		{name: "invalid metadata retains validation", thinking: &model.Thinking{Values: []any{"high"}, Default: "missing"}, fields: `,"reasoning":{"effort":"future"}`, wantCode: http.StatusBadRequest},
		{name: "unknown metadata retains validation", fields: `,"reasoning":{"effort":"future"}`, wantCode: http.StatusBadRequest},
		{name: "invalid native type", thinking: named, fields: `,"think":75`, wantCode: http.StatusBadRequest},
	} {
		for _, streaming := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream=%t", tc.name, streaming), func(t *testing.T) {
				captured := make(chan *api.ThinkValue, 2)
				upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					switch r.URL.Path {
					case "/api/show":
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
					return
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
