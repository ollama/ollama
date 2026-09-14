package server

import (
	"encoding/json"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/types/model"
)

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
		{"renderer without thinking capability keeps its own default", Model{Config: model.ConfigV2{Renderer: "gemma4"}}, &model.Thinking{Values: []any{false, true}, Default: false}},
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
	harmony := &Model{Template: tmpl, Config: model.ConfigV2{ModelFamily: "gptoss"}}
	if harmony.Thinking() != nil || harmony.genericThinking() != nil {
		t.Fatal("Harmony must retain legacy behavior")
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

func TestThinkingHarmonyKeepsLegacyDefaultAndMax(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	t.Setenv("OLLAMA_CONTEXT_LENGTH", "4096")
	mock := mockRunner{CompletionResponse: llm.CompletionResponse{Done: true, DoneReason: llm.DoneReasonStop}}
	s := newServerWithMockRunner(t, &mock)
	createMinimalGGUFModel(t, s, "thinking-harmony", nil, "<|start|><|end|>Reasoning: {{ .ThinkLevel }} {{ .Prompt }}", map[string]any{"model_family": "gptoss", "capabilities": []any{"completion", "thinking"}})
	for _, tt := range []struct {
		name  string
		think *api.ThinkValue
		want  string
	}{
		{"omitted", nil, "medium"}, {"max", &api.ThinkValue{Value: "max"}, "high"},
	} {
		for _, endpoint := range []string{"chat", "generate"} {
			t.Run(endpoint+"/"+tt.name, func(t *testing.T) {
				think := tt.think
				if think != nil {
					think = &api.ThinkValue{Value: think.Value}
				}
				if endpoint == "chat" {
					w := createRequest(t, s.ChatHandler, api.ChatRequest{Model: "thinking-harmony", Messages: []api.Message{{Role: "user", Content: "hello"}}, Think: think, Stream: &stream})
					if w.Code != http.StatusOK {
						t.Fatalf("status %d: %s", w.Code, w.Body.String())
					}
				} else {
					w := createRequest(t, s.GenerateHandler, api.GenerateRequest{Model: "thinking-harmony", Prompt: "hello", Think: think, Stream: &stream})
					if w.Code != http.StatusOK {
						t.Fatalf("status %d: %s", w.Code, w.Body.String())
					}
				}
				if !strings.Contains(mock.CompletionRequest.Prompt, "Reasoning: "+tt.want) {
					t.Fatalf("expected reasoning %s: %s", tt.want, mock.CompletionRequest.Prompt)
				}
			})
		}
	}
}
