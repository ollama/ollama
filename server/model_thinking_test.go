package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/middleware"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/types/model"
)

func TestThinkingInputErrors(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	t.Setenv("OLLAMA_NO_CLOUD", "")
	s := &Server{modelCaches: &modelCaches{show: newModelShowCache()}}
	createMinimalGGUFModel(t, s, "thinking-base", nil, "{{ .Prompt }}", nil)
	createMinimalGGUFModel(t, s, "thinking-harmony", nil, "<|start|>{{ .Prompt }}<|end|>", map[string]any{"model_family": "gptoss", "capabilities": []any{"completion", "thinking"}})
	w := createRequest(t, s.CreateHandler, api.CreateRequest{Model: "thinking-qwen", From: "thinking-base", Renderer: "qwen3.8", Parser: "qwen3.5", Stream: &stream})
	if w.Code != http.StatusOK {
		t.Fatal(w.Body.String())
	}

	showCalls := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/show" {
			t.Errorf("invalid thinking request reached %s", r.URL.Path)
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		showCalls++
		json.NewEncoder(w).Encode(api.ShowResponse{Thinking: &model.Thinking{Values: []any{false, "high", "max"}, Default: "high"}})
	}))
	defer upstream.Close()
	withCloudProxyBaseURL(t, upstream.URL)

	for _, tc := range []struct {
		name, values string
	}{
		{"thinking-qwen", `[false,"low","medium","xhigh"]`},
		{"thinking-qwen:local", `[false,"low","medium","xhigh"]`},
		{" THINKING-QWEN ", `[false,"low","medium","xhigh"]`},
		{"thinking-cloud:cloud", `[false,"high","max"]`},
		{"thinking-harmony", `["low","medium","high"]`},
		{"thinking-base", ""},
		{"missing", ""},
	} {
		for _, endpoint := range []struct {
			name    string
			handler gin.HandlerFunc
		}{{"chat", s.ChatHandler}, {"generate", s.GenerateHandler}} {
			for _, value := range []string{"75", "0.5", "{}", "[]"} {
				for _, body := range []string{
					fmt.Sprintf(`{"model":%q,"think":%s}`, tc.name, value),
					fmt.Sprintf(`{"think":%s,"model":%q}`, value, tc.name),
				} {
					t.Run(tc.name+"/"+endpoint.name+"/"+body, func(t *testing.T) {
						w := httptest.NewRecorder()
						c, _ := gin.CreateTestContext(w)
						c.Request = httptest.NewRequest("POST", "/api/"+endpoint.name, strings.NewReader(body))
						endpoint.handler(c)
						var response struct {
							Error string `json:"error"`
						}
						if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
							t.Fatal(err)
						}
						want := "think must be a boolean or string"
						if tc.values != "" {
							want += "; supported values: " + tc.values
						}
						if w.Code != 400 || response.Error != want {
							t.Fatalf("status=%d error=%q, want 400 %q", w.Code, response.Error, want)
						}
					})
				}
			}
		}
	}
	if showCalls != 1 {
		t.Fatalf("cloud show calls = %d, want one cold fetch then cached reads", showCalls)
	}
}

func TestModelThinking(t *testing.T) {
	t.Setenv("OLLAMA_GO_TEMPLATE", "1")
	known := "sha256:ae370d884f108d16e7cc8fd5259ebc5773a0afa6e078b11f4ed7e39a27e0dfc4"
	for _, tt := range []struct {
		name string
		m    Model
		want *model.Thinking
	}{
		{"local gemma default on", Model{Config: model.ConfigV2{Renderer: "gemma4", Parser: "gemma4"}}, &model.Thinking{Values: []any{false, true}, Default: true}},
		{"local qwen38 default medium", Model{Config: model.ConfigV2{Renderer: "qwen3.8", Parser: "qwen3.5"}}, &model.Thinking{Values: []any{false, "low", "medium", "xhigh"}, Default: "medium"}},
		{"renderer without thinking capability", Model{Config: model.ConfigV2{Renderer: "gemma4"}}, &model.Thinking{Values: []any{false}, Default: false}},
		{"unknown renderer", Model{Config: model.ConfigV2{Renderer: "unknown"}}, nil},
		{"nonthinking", Model{Config: model.ConfigV2{Renderer: "qwen3-coder"}}, &model.Thinking{Values: []any{false}, Default: false}},
		{"known template", Model{HasGoTemplate: true, templateDigest: known}, &model.Thinking{Values: []any{false, true}, Default: true}},
		{"custom template", Model{HasGoTemplate: true, templateDigest: "custom"}, nil},
		{"inactive template", Model{templateDigest: known}, nil},
		{"remote", Model{Config: model.ConfigV2{RemoteHost: "https://ollama.com", Renderer: "qwen3.8", Parser: "qwen3.5"}}, nil},
		{"unknown", Model{}, nil},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.m.Thinking()
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("got %#v, want %#v", got, tt.want)
			}
		})
	}
	t.Run("backfill stays off the Jinja path", func(t *testing.T) {
		t.Setenv("OLLAMA_GO_TEMPLATE", "0")
		m := &Model{HasGoTemplate: true, templateDigest: known}
		if m.Thinking() != nil {
			t.Fatal("inactive Go template must not supply backfill")
		}
	})
	tmpl, err := template.Parse("<|start|>{{ .Prompt }}<|end|>")
	if err != nil {
		t.Fatal(err)
	}
	for _, renderer := range []string{"", "harmony"} {
		harmony := &Model{Template: tmpl, Config: model.ConfigV2{ModelFamily: "gptoss", Renderer: renderer}}
		want := &model.Thinking{Values: []any{"low", "medium", "high"}, Default: "medium"}
		if got := harmony.Thinking(); !reflect.DeepEqual(got, want) {
			t.Fatalf("Harmony discovery = %#v, want %#v", got, want)
		}
		if harmony.genericThinking() != nil {
			t.Fatal("Harmony must retain legacy inference behavior")
		}
	}
}

func TestThinkingShowFollowsRendererChanges(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	var s Server
	createMinimalGGUFModel(t, &s, "thinking-base", nil, "{{ .Prompt }}", nil)
	for _, tt := range []struct {
		name, from, renderer, parser string
		want                         *model.Thinking
	}{
		{"thinking-generic", "thinking-base", "qwen3.8", "qwen3.5", &model.Thinking{Values: []any{false, "low", "medium", "xhigh"}, Default: "medium"}},
		{"thinking-inherited", "thinking-generic", "", "", &model.Thinking{Values: []any{false, "low", "medium", "xhigh"}, Default: "medium"}},
		{"thinking-changed", "thinking-generic", "qwen3-coder", "qwen3-coder", &model.Thinking{Values: []any{false}, Default: false}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			w := createRequest(t, s.CreateHandler, api.CreateRequest{Model: tt.name, From: tt.from, Renderer: tt.renderer, Parser: tt.parser, Stream: &stream})
			if w.Code != http.StatusOK {
				t.Fatalf("create: %d %s", w.Code, w.Body.String())
			}
			info, err := GetModelInfo(api.ShowRequest{Model: tt.name})
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(info.Thinking, tt.want) {
				t.Fatalf("show descriptor %#v, want %#v", info.Thinking, tt.want)
			}
			clone := cloneShowResponse(info)
			clone.Thinking.Values[0] = "changed"
			if info.Thinking.Values[0] == "changed" {
				t.Fatal("show cache clone shares thinking")
			}
		})
	}
	info, err := GetModelInfo(api.ShowRequest{Model: "thinking-base"})
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(info)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), `"thinking":`) {
		t.Fatal("unknown metadata must be omitted")
	}
}

func TestThinkingResolvedBeforeRenderAndParse(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	t.Setenv("OLLAMA_CONTEXT_LENGTH", "4096")
	mock := mockRunner{CompletionResponse: llm.CompletionResponse{Content: "reason</think>answer", Done: true, DoneReason: llm.DoneReasonStop}}
	s := newServerWithMockRunner(t, &mock)
	createMinimalGGUFModel(t, s, "thinking-base", nil, "{{ .Prompt }}", nil)
	w := createRequest(t, s.CreateHandler, api.CreateRequest{Model: "thinking-qwen", From: "thinking-base", Renderer: "qwen3.8", Parser: "qwen3.5", Stream: &stream})
	if w.Code != http.StatusOK {
		t.Fatalf("create: %s", w.Body.String())
	}
	for _, requested := range []any{nil, true, false, "low", "medium", "xhigh", "high", "max", "minimal", "future"} {
		for _, endpoint := range []string{"chat", "generate"} {
			label, _ := json.Marshal(requested)
			t.Run(endpoint+"/"+string(label), func(t *testing.T) {
				var think *api.ThinkValue
				if requested != nil {
					think = &api.ThinkValue{Value: requested}
				}
				var content, reasoning string
				if endpoint == "chat" {
					w = createRequest(t, s.ChatHandler, api.ChatRequest{Model: "thinking-qwen", Messages: []api.Message{{Role: "user", Content: "hello"}}, Think: think, Stream: &stream})
					var resp api.ChatResponse
					if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
						t.Fatal(err)
					}
					content, reasoning = resp.Message.Content, resp.Message.Thinking
				} else {
					w = createRequest(t, s.GenerateHandler, api.GenerateRequest{Model: "thinking-qwen", Prompt: "hello", Think: think, Stream: &stream})
					var resp api.GenerateResponse
					if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
						t.Fatal(err)
					}
					content, reasoning = resp.Response, resp.Thinking
				}
				if w.Code != http.StatusOK {
					t.Fatalf("status %d: %s", w.Code, w.Body.String())
				}
				prompt := mock.CompletionRequest.Prompt
				if strings.Contains(prompt, "Reasoning effort is set to xhigh") != (requested == "xhigh") {
					t.Fatalf("xhigh mismatch for %#v: %s", requested, prompt)
				}
				if strings.Contains(prompt, "Reasoning effort is set to low") != (requested == "low") {
					t.Fatalf("low mismatch for %#v: %s", requested, prompt)
				}
				if requested != false && (content != "answer" || reasoning != "reason") {
					t.Fatalf("parser mismatch: content=%q thinking=%q", content, reasoning)
				}
				if requested == false && reasoning != "" {
					t.Fatalf("off produced thinking %q", reasoning)
				}
			})
		}
	}
}

func TestGenerateExtractsThinkingWithNonThinkingBuiltinParser(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	t.Setenv("OLLAMA_CONTEXT_LENGTH", "4096")
	mock := mockRunner{CompletionResponse: llm.CompletionResponse{
		Content:    "reason</think>answer",
		Done:       true,
		DoneReason: llm.DoneReasonStop,
	}}
	s := newServerWithMockRunner(t, &mock)
	createMinimalGGUFModel(t, s, "thinking-base", nil, "{{ .Prompt }}", nil)

	w := createRequest(t, s.CreateHandler, api.CreateRequest{
		Model:    "qwen3-parser",
		From:     "thinking-base",
		Template: `{{ range .Messages }}{{ if .Thinking }}<think>{{ .Thinking }}</think>{{ end }}{{ .Content }}{{ end }}{{ if .Think }}<think>{{ end }}`,
		Parser:   "qwen3",
		Info:     map[string]any{"capabilities": []any{"completion", "thinking"}},
		Stream:   &stream,
	})
	if w.Code != http.StatusOK {
		t.Fatalf("create: %s", w.Body.String())
	}

	nonStreaming := false
	w = createRequest(t, s.GenerateHandler, api.GenerateRequest{
		Model:  "qwen3-parser",
		Prompt: "hello",
		Think:  &api.ThinkValue{Value: true},
		Stream: &nonStreaming,
	})
	if w.Code != http.StatusOK {
		t.Fatalf("generate: %d %s", w.Code, w.Body.String())
	}

	var resp api.GenerateResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Response != "answer" || resp.Thinking != "reason" {
		t.Fatalf("response=%q thinking=%q, want response=%q thinking=%q", resp.Response, resp.Thinking, "answer", "reason")
	}

	streaming := true
	w = createRequest(t, s.GenerateHandler, api.GenerateRequest{
		Model:  "qwen3-parser",
		Prompt: "hello",
		Think:  &api.ThinkValue{Value: true},
		Stream: &streaming,
	})
	if w.Code != http.StatusOK {
		t.Fatalf("streaming generate: %d %s", w.Code, w.Body.String())
	}
	var streamedResponse, streamedThinking string
	for _, line := range strings.Split(strings.TrimSpace(w.Body.String()), "\n") {
		var chunk api.GenerateResponse
		if err := json.Unmarshal([]byte(line), &chunk); err != nil {
			t.Fatal(err)
		}
		streamedResponse += chunk.Response
		streamedThinking += chunk.Thinking
	}
	if streamedResponse != "answer" || streamedThinking != "reason" {
		t.Fatalf("streamed response=%q thinking=%q, want response=%q thinking=%q", streamedResponse, streamedThinking, "answer", "reason")
	}
}

func TestThinkingNonthinkingFallback(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	t.Setenv("OLLAMA_CONTEXT_LENGTH", "4096")
	mock := mockRunner{CompletionResponse: llm.CompletionResponse{Content: "answer", Done: true, DoneReason: llm.DoneReasonStop}}
	s := newServerWithMockRunner(t, &mock)
	createMinimalGGUFModel(t, s, "thinking-base", nil, "{{ .Prompt }}", nil)
	for _, config := range []struct{ name, renderer, parser string }{
		{"thinking-coder", "qwen3-coder", "qwen3-coder"},
		{"thinking-no-parser", "gemma4", ""},
	} {
		w := createRequest(t, s.CreateHandler, api.CreateRequest{Model: config.name, From: "thinking-base", Renderer: config.renderer, Parser: config.parser, Stream: &stream})
		if w.Code != http.StatusOK {
			t.Fatal(w.Body.String())
		}
		show, err := GetModelInfo(api.ShowRequest{Model: config.name})
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(show.Thinking, &model.Thinking{Values: []any{false}, Default: false}) {
			t.Fatalf("%s advertised rejected controls: %+v", config.name, show.Thinking)
		}
		for _, value := range []any{nil, false, true, "", "high", "future"} {
			for _, endpoint := range []string{"chat", "generate"} {
				t.Run(fmt.Sprintf("%s/%s/%v", config.name, endpoint, value), func(t *testing.T) {
					var think *api.ThinkValue
					if value != nil {
						think = &api.ThinkValue{Value: value}
					}
					if endpoint == "chat" {
						w = createRequest(t, s.ChatHandler, api.ChatRequest{Model: config.name, Messages: []api.Message{{Role: "user", Content: "hello"}}, Think: think, Stream: &stream})
					} else {
						w = createRequest(t, s.GenerateHandler, api.GenerateRequest{Model: config.name, Prompt: "hello", Think: think, Stream: &stream})
					}
					want := http.StatusOK
					if value == true {
						want = http.StatusBadRequest
					}
					if w.Code != want {
						t.Fatalf("status=%d, want %d: %s", w.Code, want, w.Body.String())
					}
				})
			}
		}
	}
}

func TestThinkingLookupModelReferences(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	s := &Server{}
	createMinimalGGUFModel(t, s, "thinking-base", nil, "{{ .Prompt }}", nil)
	createMinimalGGUFModel(t, s, "thinking-harmony", nil, "<|start|>{{ .Prompt }}<|end|>", map[string]any{"model_family": "gptoss", "capabilities": []any{"completion", "thinking"}})
	w := createRequest(t, s.CreateHandler, api.CreateRequest{Model: "thinking-qwen", From: "thinking-base", Renderer: "qwen3.8", Parser: "qwen3.5", Stream: &stream})
	if w.Code != http.StatusOK {
		t.Fatal(w.Body.String())
	}
	for _, protocol := range []struct {
		name, fields string
		middleware   gin.HandlerFunc
	}{
		{"chat", `"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"xhigh"`, middleware.ChatMiddleware(lookupThinking)},
		{"responses", `"input":"hi","reasoning":{"effort":"xhigh"}`, middleware.ResponsesMiddleware(lookupThinking)},
		{"anthropic", `"messages":[{"role":"user","content":"hi"}],"max_tokens":64,"output_config":{"effort":"xhigh"}`, middleware.AnthropicMessagesMiddleware(lookupThinking)},
	} {
		for _, name := range []string{"thinking-qwen", "thinking-qwen:local", " THINKING-QWEN ", "thinking-harmony"} {
			t.Run(protocol.name+"/"+name, func(t *testing.T) {
				var req api.ChatRequest
				router := gin.New()
				router.POST("/", protocol.middleware, func(c *gin.Context) {
					if err := json.NewDecoder(c.Request.Body).Decode(&req); err != nil {
						t.Error(err)
					}
					c.Status(http.StatusOK)
				})
				r := httptest.NewRequest("POST", "/", strings.NewReader(fmt.Sprintf(`{"model":%q,%s}`, name, protocol.fields)))
				r.Header.Set("Content-Type", "application/json")
				w := httptest.NewRecorder()
				router.ServeHTTP(w, r)
				want := "xhigh"
				if name == "thinking-harmony" {
					want = "max"
					if protocol.name == "anthropic" {
						want = "high"
					}
				}
				if w.Code != http.StatusOK || req.Think == nil || req.Think.Value != want {
					t.Fatalf("status=%d think=%v: %s", w.Code, req.Think, w.Body.String())
				}
			})
		}
	}
	for _, name := range []string{"thinking-qwen:cloud", "missing", ""} {
		if got := lookupThinking(name); got != nil {
			t.Errorf("%q lookup=%+v, want no local metadata", name, got)
		}
	}
}

func TestThinkingHarmonyDiscoveryPreservesInference(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	t.Setenv("OLLAMA_CONTEXT_LENGTH", "4096")
	mock := mockRunner{CompletionResponse: llm.CompletionResponse{Done: true, DoneReason: llm.DoneReasonStop}}
	s := newServerWithMockRunner(t, &mock)
	createMinimalGGUFModel(t, s, "thinking-harmony", nil, "<|start|><|end|>Reasoning: {{ .ThinkLevel }} {{ .Prompt }}", map[string]any{"model_family": "gptoss", "capabilities": []any{"completion", "thinking"}})
	w := createRequest(t, s.ShowHandler, api.ShowRequest{Model: "thinking-harmony"})
	var show api.ShowResponse
	if err := json.Unmarshal(w.Body.Bytes(), &show); err != nil {
		t.Fatal(err)
	}
	want := &model.Thinking{Values: []any{"low", "medium", "high"}, Default: "medium"}
	if w.Code != http.StatusOK || !reflect.DeepEqual(show.Thinking, want) {
		t.Fatalf("show status=%d thinking=%#v, want %#v", w.Code, show.Thinking, want)
	}
	for _, tt := range []struct {
		name       string
		think      *api.ThinkValue
		want       string
		badRequest bool
	}{
		{"omitted", nil, "medium", false},
		{"low", &api.ThinkValue{Value: "low"}, "low", false},
		{"medium", &api.ThinkValue{Value: "medium"}, "medium", false},
		{"high", &api.ThinkValue{Value: "high"}, "high", false},
		{"max", &api.ThinkValue{Value: "max"}, "high", false},
		{"true", &api.ThinkValue{Value: true}, "medium", false},
		{"false", &api.ThinkValue{Value: false}, "", false},
		{"xhigh", &api.ThinkValue{Value: "xhigh"}, "", true},
		{"future", &api.ThinkValue{Value: "future"}, "", true},
	} {
		for _, endpoint := range []string{"chat", "generate"} {
			t.Run(endpoint+"/"+tt.name, func(t *testing.T) {
				think := tt.think
				if think != nil {
					think = &api.ThinkValue{Value: think.Value}
				}
				var w *httptest.ResponseRecorder
				if endpoint == "chat" {
					w = createRequest(t, s.ChatHandler, api.ChatRequest{Model: "thinking-harmony", Messages: []api.Message{{Role: "user", Content: "hello"}}, Think: think, Stream: &stream})
				} else {
					w = createRequest(t, s.GenerateHandler, api.GenerateRequest{Model: "thinking-harmony", Prompt: "hello", Think: think, Stream: &stream})
				}
				if tt.badRequest {
					if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "invalid think value") {
						t.Fatalf("expected legacy validation error: %d %s", w.Code, w.Body.String())
					}
					return
				}
				if w.Code != http.StatusOK {
					t.Fatalf("status %d: %s", w.Code, w.Body.String())
				}
				if !strings.Contains(mock.CompletionRequest.Prompt, "Reasoning: "+tt.want+" ") {
					t.Fatalf("expected reasoning %s: %s", tt.want, mock.CompletionRequest.Prompt)
				}
			})
		}
	}
}
