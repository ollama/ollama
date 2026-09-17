package renderers

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestGlimmerRenderImages(t *testing.T) {
	tests := []struct {
		name        string
		renderTags  bool
		messages    []api.Message
		wantContent string
	}{
		{
			name: "reference patch token",
			messages: []api.Message{{
				Role:    "user",
				Content: "Describe.",
				Images:  []api.ImageData{{1}},
			}},
			wantContent: "<|start|>user<|message|><|patch|>Describe.<|eot|>",
		},
		{
			name:       "runner tags across turns",
			renderTags: true,
			messages: []api.Message{
				{Role: "user", Content: "First.", Images: []api.ImageData{{1}}},
				{Role: "assistant", Content: "Done."},
				{Role: "user", Content: "Compare.", Images: []api.ImageData{{2}, {3}}},
			},
			wantContent: "<|start|>user<|message|>[img-0] First.<|eot|>" +
				"<|start|>assistant to=user<|message|>Done.<|eot|>" +
				"<|start|>user<|message|>[img-1][img-2] Compare.<|eot|>",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := (&GlimmerRenderer{useImgTags: tt.renderTags}).Render(tt.messages, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(got, tt.wantContent) {
				t.Fatalf("rendered prompt missing image content:\ngot:  %q\nwant: %q", got, tt.wantContent)
			}
		})
	}
}

func TestGlimmerRenderATEMValues(t *testing.T) {
	arguments := api.NewToolCallFunctionArguments()
	arguments.Set("text", "keep  spaces")
	arguments.Set("count", 3)
	arguments.Set("enabled", true)
	arguments.Set("fallback", nil)
	arguments.Set("items", []any{"one", "two"})
	arguments.Set("config", map[string]any{"mode": "fast"})

	got, err := (&GlimmerRenderer{}).Render([]api.Message{
		{Role: "user", Content: "Run it."},
		{Role: "assistant", ToolCalls: []api.ToolCall{{
			Function: api.ToolCallFunction{Name: "tools.run", Arguments: arguments},
		}}},
	}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	want := `<atem:function_calls>
<atem:invoke name="tools.run">
<atem:parameter name="text">keep  spaces</atem:parameter>
<atem:parameter name="count">3</atem:parameter>
<atem:parameter name="enabled">true</atem:parameter>
<atem:parameter name="fallback">null</atem:parameter>
<atem:parameter name="items">["one", "two"]</atem:parameter>
<atem:parameter name="config">{"mode": "fast"}</atem:parameter>
</atem:invoke>
</atem:function_calls>`
	if !strings.Contains(got, want) {
		t.Fatalf("rendered prompt missing ATEM call:\ngot:  %q\nwant: %q", got, want)
	}
}

func TestGlimmerRenderReasoningStrength(t *testing.T) {
	tests := []struct {
		name  string
		think *api.ThinkValue
		want  string
	}{
		{name: "default", want: "Reasoning strength: high."},
		{name: "enabled", think: &api.ThinkValue{Value: true}, want: "Reasoning strength: high."},
		{name: "disabled", think: &api.ThinkValue{Value: false}, want: "Reasoning strength: none."},
		{name: "level", think: &api.ThinkValue{Value: "medium"}, want: "Reasoning strength: medium."},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := (&GlimmerRenderer{}).Render([]api.Message{{Role: "user", Content: "Hello"}}, nil, tt.think)
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(got, tt.want) {
				t.Fatalf("rendered prompt missing %q:\n%s", tt.want, got)
			}
		})
	}
}

func TestGlimmerRenderToolResultName(t *testing.T) {
	arguments := api.NewToolCallFunctionArguments()
	got, err := (&GlimmerRenderer{}).Render([]api.Message{
		{Role: "assistant", ToolCalls: []api.ToolCall{{
			ID:       "call_1",
			Function: api.ToolCallFunction{Name: "get_weather", Arguments: arguments},
		}}},
		{Role: "tool", ToolCallID: "call_1", Content: `{"temp":65}`},
		{Role: "tool", ToolCallID: "missing", Content: "not found"},
	}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	for _, want := range []string{
		`<|start|>tool get_weather<|message|><tool_output name="get_weather">` + "\n" + `{"temp":65}` + "\n</tool_output><|eot|>",
		`<|start|>tool missing<|message|><tool_output name="missing">` + "\nnot found\n</tool_output><|eot|>",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("rendered prompt missing tool result %q:\n%s", want, got)
		}
	}
}

func TestGlimmerRendererRegistered(t *testing.T) {
	if rendererForName("glimmer") == nil {
		t.Fatal("glimmer renderer is not registered")
	}
	if got := LeadingBOSForRenderer("glimmer"); got != glimmerBOS {
		t.Fatalf("LeadingBOSForRenderer(glimmer) = %q, want %q", got, glimmerBOS)
	}
}
