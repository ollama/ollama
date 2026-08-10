package renderers

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func apertus15WeatherTool(t *testing.T) api.Tool {
	t.Helper()
	var tool api.Tool
	if err := json.Unmarshal([]byte(`{
		"type": "function",
		"function": {
			"name": "get_weather",
			"description": "Get current weather.",
			"parameters": {
				"type": "object",
				"properties": {
					"city": {"type": "string", "description": "City name."},
					"unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
				},
				"required": ["city"]
			}
		}
	}`), &tool); err != nil {
		t.Fatal(err)
	}
	return tool
}

func apertus15ToolCall(name string, values map[string]any) api.ToolCall {
	arguments := api.NewToolCallFunctionArguments()
	for key, value := range values {
		arguments.Set(key, value)
	}
	return api.ToolCall{Function: api.ToolCallFunction{Name: name, Arguments: arguments}}
}

func TestApertus15RendererDefaultSystem(t *testing.T) {
	got, err := (&Apertus15Renderer{}).Render([]api.Message{
		{Role: "user", Content: "Hello"},
	}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	want := apertus15BOS +
		apertus15SystemStart + apertus15DefaultSystem + apertus15SystemEnd +
		apertus15DeveloperStart + "Deliberation: disabled\nTool Capabilities: disabled" + apertus15DeveloperEnd +
		apertus15UserStart + "Hello" + apertus15UserEnd + apertus15AssistantStart
	if got != want {
		t.Fatalf("rendered prompt mismatch\nwant: %q\n got: %q", want, got)
	}
	if leading := (&Apertus15Renderer{}).LeadingBOS(); leading != apertus15BOS {
		t.Fatalf("LeadingBOS() = %q, want %q", leading, apertus15BOS)
	}
}

func TestApertus15RendererThinkingHistory(t *testing.T) {
	got, err := (&Apertus15Renderer{}).Render([]api.Message{
		{Role: "system", Content: "Be concise."},
		{Role: "user", Content: "Question"},
		{Role: "assistant", Thinking: "Reasoning.", Content: "Answer."},
		{Role: "user", Content: "Next"},
	}, nil, &api.ThinkValue{Value: true})
	if err != nil {
		t.Fatal(err)
	}

	for _, want := range []string{
		apertus15SystemStart + "Be concise." + apertus15SystemEnd,
		apertus15DeveloperStart + "Deliberation: enabled\nTool Capabilities: disabled" + apertus15DeveloperEnd,
		apertus15AssistantStart + apertus15InnerStart + "Reasoning." + apertus15InnerEnd + "Answer." + apertus15AssistantEnd,
		apertus15UserStart + "Next" + apertus15UserEnd + apertus15AssistantStart,
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("rendered prompt missing %q:\n%s", want, got)
		}
	}
}

func TestApertus15RendererKeepsHistoricalThinkingWhenDisabled(t *testing.T) {
	got, err := (&Apertus15Renderer{}).Render([]api.Message{
		{Role: "user", Content: "Question"},
		{Role: "assistant", Thinking: "Stored reasoning.", Content: "Stored answer."},
	}, nil, &api.ThinkValue{Value: false})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(got, apertus15InnerStart+"Stored reasoning."+apertus15InnerEnd+"Stored answer.") {
		t.Fatalf("rendered prompt dropped historical thinking:\n%s", got)
	}
	if !strings.HasSuffix(got, "Stored answer.") {
		t.Fatalf("assistant prefill should remain open:\n%s", got)
	}
}

func TestApertus15RendererToolsAndOutputs(t *testing.T) {
	tool := apertus15WeatherTool(t)
	call := apertus15ToolCall("get_weather", map[string]any{"city": "Zurich"})
	got, err := (&Apertus15Renderer{}).Render([]api.Message{
		{Role: "user", Content: "Weather?"},
		{Role: "assistant", Thinking: "I should check.", ToolCalls: []api.ToolCall{call}},
		{Role: "tool", Content: `{"temperature":22}`},
		{Role: "tool", Content: `{"condition":"clear"}`},
	}, []api.Tool{tool}, &api.ThinkValue{Value: true})
	if err != nil {
		t.Fatal(err)
	}

	wantTools := "Tool Capabilities:\n" +
		"// Get current weather.\n" +
		"type get_weather = (_: {\n" +
		"// City name.\n" +
		"city: string,\n" +
		"unit?: \"celsius\" | \"fahrenheit\"\n" +
		"}) => any;"
	if !strings.Contains(got, wantTools) {
		t.Fatalf("rendered prompt missing TypeScript tools\nwant fragment: %q\nrendered: %s", wantTools, got)
	}
	wantTurn := apertus15AssistantStart + apertus15InnerStart + "I should check." +
		apertus15ToolsStart + `[{"get_weather": {"city": "Zurich"}}]` + apertus15ToolsEnd +
		apertus15ToolOutputStart + `{"temperature":22}, {"condition":"clear"}` + apertus15ToolOutputEnd +
		apertus15AssistantEnd + apertus15AssistantStart
	if !strings.Contains(got, wantTurn) {
		t.Fatalf("rendered prompt missing tool turn\nwant fragment: %q\nrendered: %s", wantTurn, got)
	}
}

func TestApertus15RendererAnyOfMatchesTemplateFallback(t *testing.T) {
	var tool api.Tool
	if err := json.Unmarshal([]byte(`{
		"type": "function",
		"function": {
			"name": "accept_value",
			"description": "Accept a value.",
			"parameters": {
				"type": "object",
				"properties": {
					"value": {"anyOf": [{"type": "string"}, {"type": "number"}]}
				},
				"required": ["value"]
			}
		}
	}`), &tool); err != nil {
		t.Fatal(err)
	}

	got, err := (&Apertus15Renderer{}).Render([]api.Message{{Role: "user", Content: "Use it."}}, []api.Tool{tool}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(got, "type accept_value = (_: {\nvalue: any\n}) => any;") {
		t.Fatalf("anyOf did not use the source template fallback:\n%s", got)
	}
}

func TestApertus15RendererUsesJinjaEnumStrings(t *testing.T) {
	var tool api.Tool
	if err := json.Unmarshal([]byte(`{
		"type": "function",
		"function": {
			"name": "choose",
			"description": "Choose a value.",
			"parameters": {
				"type": "object",
				"properties": {
					"value": {"type": "string", "enum": [true, false, null, 1]}
				},
				"required": ["value"]
			}
		}
	}`), &tool); err != nil {
		t.Fatal(err)
	}

	got, err := (&Apertus15Renderer{}).Render([]api.Message{{Role: "user", Content: "Choose."}}, []api.Tool{tool}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(got, `value: "True" | "False" | "None" | "1"`) {
		t.Fatalf("enum values do not match Jinja string conversion:\n%s", got)
	}
}

func TestApertus15RendererDoesNotEscapeHTMLToolArguments(t *testing.T) {
	call := apertus15ToolCall("echo", map[string]any{"value": "<tag>&value</tag>"})
	got, err := (&Apertus15Renderer{}).Render([]api.Message{
		{Role: "user", Content: "Echo"},
		{Role: "assistant", ToolCalls: []api.ToolCall{call}},
	}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(got, `[{"echo": {"value": "<tag>&value</tag>"}}]`) {
		t.Fatalf("tool arguments do not match Jinja escaping:\n%s", got)
	}
}

func TestApertus15RendererPreservesToolNameAndTopLevelArgumentOrder(t *testing.T) {
	arguments := api.NewToolCallFunctionArguments()
	arguments.Set("zeta", "<tag>&value")
	arguments.Set("alpha", 2)
	call := api.ToolCall{Function: api.ToolCallFunction{
		Name:      "echo<tag>",
		Arguments: arguments,
	}}

	got, err := (&Apertus15Renderer{}).Render([]api.Message{
		{Role: "user", Content: "Echo"},
		{Role: "assistant", ToolCalls: []api.ToolCall{call}},
	}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(got, `[{"echo<tag>": {"zeta": "<tag>&value", "alpha": 2}}]`) {
		t.Fatalf("tool call does not match Jinja spelling and order:\n%s", got)
	}
}

func TestApertus15RendererPreservesAdjacentAssistantBlockOrder(t *testing.T) {
	call := apertus15ToolCall("lookup", map[string]any{"id": "42"})
	got, err := (&Apertus15Renderer{}).Render([]api.Message{
		{Role: "user", Content: "Look up"},
		{Role: "assistant", ToolCalls: []api.ToolCall{call}},
		{Role: "assistant", Thinking: "Reconsider."},
		{Role: "assistant", Content: "Done."},
		{Role: "user", Content: "Next"},
	}, nil, &api.ThinkValue{Value: true})
	if err != nil {
		t.Fatal(err)
	}

	want := apertus15AssistantStart +
		apertus15ToolsStart + `[{"lookup": {"id": "42"}}]` + apertus15ToolsEnd +
		apertus15InnerStart + "Reconsider." + apertus15InnerEnd + "Done." +
		apertus15AssistantEnd + apertus15UserStart + "Next" + apertus15UserEnd
	if !strings.Contains(got, want) {
		t.Fatalf("adjacent assistant messages lost block order\nwant fragment: %q\nrendered: %s", want, got)
	}
}

func TestApertus15RendererPreservesRepeatedToolOutputBlocks(t *testing.T) {
	got, err := (&Apertus15Renderer{}).Render([]api.Message{
		{Role: "user", Content: "Observe"},
		{Role: "assistant"},
		{Role: "tool", Content: "one"},
		{Role: "assistant"},
		{Role: "tool", Content: "two"},
		{Role: "user", Content: "Next"},
	}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	want := apertus15AssistantStart +
		apertus15ToolOutputStart + "one" + apertus15ToolOutputEnd +
		apertus15ToolOutputStart + "two" + apertus15ToolOutputEnd +
		apertus15AssistantEnd
	if !strings.Contains(got, want) {
		t.Fatalf("repeated tool output boundaries were merged\nwant fragment: %q\nrendered: %s", want, got)
	}
}

func TestApertus15RendererThinkingWithDisplayAnswers(t *testing.T) {
	call := apertus15ToolCall("display_answers", map[string]any{"answer": "Done"})
	got, err := (&Apertus15Renderer{}).Render([]api.Message{
		{Role: "user", Content: "Solve"},
		{Role: "assistant", Thinking: "Finished.", ToolCalls: []api.ToolCall{call}},
	}, nil, &api.ThinkValue{Value: true})
	if err != nil {
		t.Fatal(err)
	}
	// The wrapped function-call representation does not trigger the template's
	// direct-call-only display_answers transition.
	want := apertus15InnerStart + "Finished." +
		apertus15ToolsStart + `[{"display_answers": {"answer": "Done"}}]` + apertus15ToolsEnd
	if !strings.Contains(got, want) {
		t.Fatalf("rendered prompt missing display_answers transition\nwant fragment: %q\nrendered: %s", want, got)
	}
}

func TestApertus15RendererImages(t *testing.T) {
	messages := []api.Message{
		{Role: "user", Content: "First", Images: []api.ImageData{api.ImageData("a")}},
		{Role: "assistant", Content: "Seen"},
		{Role: "user", Content: "Second", Images: []api.ImageData{api.ImageData("b"), api.ImageData("c")}},
	}

	canonical, err := (&Apertus15Renderer{}).Render(messages, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(canonical, apertus15UserStart+apertus15Image+"First"+apertus15UserEnd) ||
		!strings.Contains(canonical, apertus15UserStart+apertus15Image+apertus15Image+"Second"+apertus15UserEnd) {
		t.Fatalf("canonical rendering has incorrect image tokens:\n%s", canonical)
	}

	native, err := (&Apertus15Renderer{useImgTags: true}).Render(messages, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"[img-0] First", "[img-1][img-2] Second"} {
		if !strings.Contains(native, want) {
			t.Fatalf("native rendering missing %q:\n%s", want, native)
		}
	}
}

func TestApertus15RendererRejectsInvalidSequences(t *testing.T) {
	tests := []struct {
		name     string
		messages []api.Message
	}{
		{name: "non-initial system", messages: []api.Message{{Role: "user", Content: "Hi"}, {Role: "system", Content: "Late"}}},
		{name: "tool outside assistant", messages: []api.Message{{Role: "tool", Content: "result"}}},
		{name: "unknown role", messages: []api.Message{{Role: "developer", Content: "no"}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := (&Apertus15Renderer{}).Render(tt.messages, nil, nil); err == nil {
				t.Fatal("expected rendering error")
			}
		})
	}
}
