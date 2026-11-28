package renderers

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/api"
)

type olalaCase struct {
	name     string
	messages []api.Message
	tools    []api.Tool
	think    *api.ThinkValue
}

// Golden files in testdata/olala were rendered from the publisher template
// testdata/olala_chat_template.jinja (Olala-v1.0-mini-7A1B-chat-template.jinja)
// with jinja2 (trim_blocks, lstrip_blocks, tojson = json.dumps(ensure_ascii=False)).
func TestOlalaRendererMatchesPublisherTemplate(t *testing.T) {
	weather := api.Tool{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "get_weather",
			Description: "Get the weather & more",
			Parameters: api.ToolFunctionParameters{
				Type:     "object",
				Required: []string{"city"},
				Properties: testPropsMap(map[string]api.ToolProperty{
					"city": {Type: api.PropertyType{"string"}, Description: "City name"},
				}),
			},
		},
	}

	clock := api.Tool{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "get_time",
			Description: "Current time",
			Parameters: api.ToolFunctionParameters{
				Type:     "object",
				Required: []string{"tz"},
				Properties: testPropsMap(map[string]api.ToolProperty{
					"tz": {Type: api.PropertyType{"string"}, Description: "IANA time zone"},
				}),
			},
		},
	}

	cases := []olalaCase{
		{
			name:     "simple",
			messages: []api.Message{{Role: "user", Content: "Hi"}},
		},
		{
			name:     "disabled",
			messages: []api.Message{{Role: "user", Content: "Hi"}},
			think:    &api.ThinkValue{Value: false},
		},
		{
			name: "low_system",
			messages: []api.Message{
				{Role: "system", Content: "  Réponds en français.  "},
				{Role: "user", Content: "Bonjour"},
			},
			think: &api.ThinkValue{Value: "low"},
		},
		{
			name: "blank_system",
			messages: []api.Message{
				{Role: "system", Content: "   "},
				{Role: "user", Content: "Hi"},
			},
			think: &api.ThinkValue{Value: true},
		},
		{
			name: "multi_turn",
			messages: []api.Message{
				{Role: "user", Content: "A?"},
				{Role: "assistant", Content: "A.", Thinking: "old thought"},
				{Role: "user", Content: "B?"},
			},
			think: &api.ThinkValue{Value: "high"},
		},
		{
			name: "tool_loop",
			messages: []api.Message{
				{Role: "system", Content: "Be brief."},
				{Role: "user", Content: "Quel temps à Lyon et Paris <b>?</b>"},
				{
					Role:     "assistant",
					Thinking: "Need weather.",
					ToolCalls: []api.ToolCall{
						{ID: "call_0", Function: api.ToolCallFunction{Name: "get_weather", Arguments: testArgsOrdered([]orderedArg{{"city", "Lyon"}})}},
						{ID: "call_1", Function: api.ToolCallFunction{Name: "get_weather", Arguments: testArgsOrdered([]orderedArg{{"city", "Paris"}, {"unit", "c"}})}},
					},
				},
				{Role: "tool", ToolCallID: "call_0", ToolName: "get_weather", Content: `{"temp": 12}`},
				{Role: "tool", ToolCallID: "call_1", ToolName: "get_weather", Content: `{"temp": 9}`},
			},
			tools: []api.Tool{weather},
			think: &api.ThinkValue{Value: "medium"},
		},
		{
			// tool calls without IDs get call_<index>; results without IDs carry only the name
			name: "tool_no_id",
			messages: []api.Message{
				{Role: "user", Content: "Heure à Tokyo ?"},
				{
					Role:     "assistant",
					Content:  "Je regarde.",
					Thinking: "Use the clock tool.",
					ToolCalls: []api.ToolCall{
						{Function: api.ToolCallFunction{Name: "get_time", Arguments: testArgsOrdered([]orderedArg{{"tz", "Asia/Tokyo"}})}},
						{Function: api.ToolCallFunction{Name: "get_time", Arguments: testArgsOrdered([]orderedArg{{"tz", "UTC"}})}},
					},
				},
				{Role: "tool", ToolName: "get_time", Content: "10:42"},
				{Role: "tool", ToolName: "get_time", Content: "01:42"},
			},
			tools: []api.Tool{clock},
			think: &api.ThinkValue{Value: "low"},
		},
		{
			name: "unicode",
			messages: []api.Message{
				{Role: "system", Content: "Tu es concis. \"Guillemets\" & <balises>."},
				{Role: "user", Content: "Écris « bonjour » en 日本語 😀"},
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			want, err := os.ReadFile(filepath.Join("testdata", "olala", tc.name+".txt"))
			if err != nil {
				t.Fatal(err)
			}
			got, err := RenderWithRenderer("olala", tc.messages, tc.tools, tc.think)
			if err != nil {
				t.Fatal(err)
			}
			if got != string(want) {
				t.Fatalf("olala renderer differs from publisher template:\ngot:  %q\nwant: %q", got, string(want))
			}
		})
	}
}

func TestOlalaRendererPrefill(t *testing.T) {
	got, err := RenderWithRenderer("olala", []api.Message{
		{Role: "user", Content: "Hi"},
		{Role: "assistant", Thinking: "Greeting.", Content: "Hel"},
	}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	want := "<|im_start|>assistant<|channel_start|>analysis<|content|>Greeting.<|channel_end|><|channel_start|>final<|content|>Hel"
	if len(got) < len(want) || got[len(got)-len(want):] != want {
		t.Fatalf("prefill not left open, tail: %q", got[max(0, len(got)-200):])
	}
}

func TestOlalaThinkingDescriptor(t *testing.T) {
	th := ThinkingForRenderer("olala")
	if th == nil || th.Default != "medium" || !th.Supports(false) || !th.Supports("high") {
		t.Fatalf("unexpected thinking descriptor: %+v", th)
	}
}
