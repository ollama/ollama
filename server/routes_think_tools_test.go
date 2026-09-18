package server

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

// TestChatHandlerDefaultThinkWithTools is the end-to-end regression for #10976:
// a thinking-capable model used with tools and no explicit thinking preference
// must default thinking OFF, so the model can emit tool tokens instead of
// burying its decision inside the reasoning block. The mock emits a fixed
// "<think>deciding</think>hello"; whether ChatHandler engages the thinking
// parser is observable on the response — Message.Thinking is populated when
// thinking is on, and the raw <think> text stays in Message.Content when off.
func TestChatHandlerDefaultThinkWithTools(t *testing.T) {
	t.Setenv("OLLAMA_GO_TEMPLATE", "1")
	gin.SetMode(gin.TestMode)

	mock := mockRunner{
		CompletionResponse: llm.CompletionResponse{
			Content:            "<think>deciding</think>hello",
			Done:               true,
			DoneReason:         llm.DoneReasonStop,
			PromptEvalCount:    1,
			PromptEvalDuration: 1,
			EvalCount:          1,
			EvalDuration:       1,
		},
	}
	s := newServerWithMockRunner(t, &mock)
	// Template is both thinking-capable (<think> tags) and tool-capable ({{ .Tools }}).
	createMinimalGGUFModel(t, s, "thinky", nil,
		`{{- if .Tools }}{{ .Tools }}{{ end }}{{ range .Messages }}{{ if .Thinking }}<think>{{ .Thinking }}</think>{{ end }}{{ .Content }}{{ end }}`,
		nil)

	weather := []api.Tool{{Type: "function", Function: api.ToolFunction{Name: "get_weather"}}}

	chat := func(t *testing.T, tools []api.Tool, think *api.ThinkValue) api.ChatResponse {
		t.Helper()
		w := createRequest(t, s.ChatHandler, api.ChatRequest{
			Model:    "thinky",
			Messages: []api.Message{{Role: "user", Content: "what is the weather?"}},
			Tools:    tools,
			Think:    think,
		})
		if w.Code != http.StatusOK {
			t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
		}
		var resp api.ChatResponse
		if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
			t.Fatal(err)
		}
		return resp
	}

	// First request loads the runner (DoneReason "load"); ignore it so the cases
	// below exercise real completions against a warm runner.
	_ = chat(t, nil, nil)

	cases := []struct {
		name      string
		tools     []api.Tool
		think     *api.ThinkValue
		wantThink bool
	}{
		{"tools + no preference defaults thinking off", weather, nil, false},
		{"no tools + no preference keeps thinking on", nil, nil, true},
		{"explicit think true is honored even with tools", weather, &api.ThinkValue{Value: true}, true},
		{"explicit think false is honored", nil, &api.ThinkValue{Value: false}, false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			resp := chat(t, tc.tools, tc.think)
			if tc.wantThink {
				if resp.Message.Thinking != "deciding" {
					t.Errorf("thinking should be engaged: got Thinking=%q Content=%q", resp.Message.Thinking, resp.Message.Content)
				}
			} else {
				if resp.Message.Thinking != "" {
					t.Errorf("thinking should be off: got Thinking=%q", resp.Message.Thinking)
				}
				if !strings.Contains(resp.Message.Content, "<think>") {
					t.Errorf("thinking off should leave raw <think> in content: got Content=%q", resp.Message.Content)
				}
			}
		})
	}

	// A multi-turn conversation carrying a prior assistant tool call with empty
	// content, with tools and no thinking preference, still defaults thinking off.
	t.Run("multi-turn tool loop defaults thinking off", func(t *testing.T) {
		w := createRequest(t, s.ChatHandler, api.ChatRequest{
			Model: "thinky",
			Messages: []api.Message{
				{Role: "user", Content: "weather in Paris?"},
				{Role: "assistant", Content: "", ToolCalls: []api.ToolCall{
					{Function: api.ToolCallFunction{Name: "get_weather", Arguments: api.NewToolCallFunctionArguments()}},
				}},
				{Role: "tool", Content: "18C"},
				{Role: "user", Content: "and tomorrow?"},
			},
			Tools: weather,
		})
		if w.Code != http.StatusOK {
			t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
		}
		var resp api.ChatResponse
		if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
			t.Fatal(err)
		}
		if resp.Message.Thinking != "" {
			t.Errorf("multi-turn with tools should default thinking off: got Thinking=%q", resp.Message.Thinking)
		}
	})
}
