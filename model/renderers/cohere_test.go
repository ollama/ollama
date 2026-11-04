package renderers

import (
	"testing"

	"github.com/ollama/ollama/api"
)

// Ground truth in these tests comes from rendering North-Mini-Code-1.0's
// chat_template.jinja with HF jinja semantics (add_generation_prompt=true),
// minus the leading "<BOS>" from {{ bos_token }} (the tokenizer adds BOS as a
// token at encode time).

const cohereSystemTurnNoTools = "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><|START_TEXT|>" +
	"# Available Tools\n```json\n[\n\n\n\n]\n```" +
	"<|END_TEXT|><|END_OF_TURN_TOKEN|>"

func TestCohereRenderUserOnly(t *testing.T) {
	r := &CohereRenderer{}
	got, err := r.Render([]api.Message{{Role: "user", Content: "USERMSG"}}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	want := cohereSystemTurnNoTools +
		"<|START_OF_TURN_TOKEN|><|USER_TOKEN|><|START_TEXT|>USERMSG<|END_TEXT|><|END_OF_TURN_TOKEN|>" +
		"<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_THINKING|>"
	if got != want {
		t.Errorf("render mismatch:\ngot:  %q\nwant: %q", got, want)
	}
}

func TestCohereRenderSystemHistoryAndThinking(t *testing.T) {
	r := &CohereRenderer{}
	got, err := r.Render([]api.Message{
		{Role: "system", Content: "DEVPREAMBLE"},
		{Role: "user", Content: "Q1"},
		{Role: "assistant", Content: "A1", Thinking: "THINK1"},
		{Role: "user", Content: "Q2"},
	}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	want := "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><|START_TEXT|>" +
		"DEVPREAMBLE\n\n\n\n# Available Tools\n```json\n[\n\n\n\n]\n```" +
		"<|END_TEXT|><|END_OF_TURN_TOKEN|>" +
		"<|START_OF_TURN_TOKEN|><|USER_TOKEN|><|START_TEXT|>Q1<|END_TEXT|><|END_OF_TURN_TOKEN|>" +
		"<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_THINKING|>THINK1<|END_THINKING|><|START_TEXT|>A1<|END_TEXT|><|END_OF_TURN_TOKEN|>" +
		"<|START_OF_TURN_TOKEN|><|USER_TOKEN|><|START_TEXT|>Q2<|END_TEXT|><|END_OF_TURN_TOKEN|>" +
		"<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_THINKING|>"
	if got != want {
		t.Errorf("render mismatch:\ngot:  %q\nwant: %q", got, want)
	}
}

func TestCohereRenderReasoningOff(t *testing.T) {
	r := &CohereRenderer{}
	think := &api.ThinkValue{Value: false}
	got, err := r.Render([]api.Message{{Role: "user", Content: "Q"}}, nil, think)
	if err != nil {
		t.Fatal(err)
	}
	wantSuffix := "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_THINKING|><|END_THINKING|>"
	if got[len(got)-len(wantSuffix):] != wantSuffix {
		t.Errorf("reasoning-off generation prompt mismatch, got tail %q", got[len(got)-len(wantSuffix):])
	}
}

func TestCohereRenderToolFlow(t *testing.T) {
	r := &CohereRenderer{}
	tools := []api.Tool{{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "get_weather",
			Description: "Get weather",
			Parameters: api.ToolFunctionParameters{
				Type:     "object",
				Required: []string{"city"},
				Properties: testPropsOrdered([]orderedProp{
					{Key: "city", Value: api.ToolProperty{Type: api.PropertyType{"string"}}},
				}),
			},
		},
	}}
	args := api.ToolCallFunctionArguments{}
	args.Set("city", "Paris")
	got, err := r.Render([]api.Message{
		{Role: "user", Content: "weather in Paris?"},
		{Role: "assistant", Thinking: "I should call the tool", ToolCalls: []api.ToolCall{
			{ID: "call_x", Function: api.ToolCallFunction{Name: "get_weather", Arguments: args}},
		}},
		{Role: "tool", ToolCallID: "call_x", Content: "15C sunny"},
		{Role: "user", Content: "thanks"},
	}, tools, nil)
	if err != nil {
		t.Fatal(err)
	}

	// Tools section (from jinja): one entry, surrounded by the template's
	// whitespace. Key order inside "parameters" follows Go's struct order
	// (jinja preserves whatever order the client sent; both are valid JSON
	// schema).
	wantTools := "# Available Tools\n```json\n[\n\n    {\"name\": \"get_weather\", \"description\": \"Get weather\", \"parameters\": {\"type\": \"object\", \"required\": [\"city\"], \"properties\": {\"city\": {\"type\": \"string\"}}}, \"responses\": null}\n\n\n]\n```"
	if !contains(t, got, wantTools, "tools section") {
		return
	}

	// Assistant tool call turn.
	wantAction := "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>\n            \n                <|START_THINKING|>I should call the tool<|END_THINKING|><|START_ACTION|>[\n\n    {\"tool_call_id\": \"0\", \"tool_name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\"}}\n\n]<|END_ACTION|><|END_OF_TURN_TOKEN|>"
	if !contains(t, got, wantAction, "action turn") {
		return
	}

	// Tool result turn.
	wantResult := "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><|START_TOOL_RESULT|>[\n    {\n        \"tool_call_id\": \"0\",\n        \"results\": {\n\n            \n            \"0\": {\"content\": \"15C sunny\"}\n\n        },\n        \"is_error\": null\n    }\n\n]<|END_TOOL_RESULT|><|END_OF_TURN_TOKEN|>"
	contains(t, got, wantResult, "tool result turn")
}

func TestCohereRenderMultiToolCallsAndResults(t *testing.T) {
	r := &CohereRenderer{}
	emptyArgs := api.ToolCallFunctionArguments{}
	xArgs := api.ToolCallFunctionArguments{}
	xArgs.Set("x", 1)
	got, err := r.Render([]api.Message{
		{Role: "user", Content: "go"},
		{Role: "assistant", ToolCalls: []api.ToolCall{
			{ID: "a", Function: api.ToolCallFunction{Name: "t1", Arguments: emptyArgs}},
			{ID: "b", Function: api.ToolCallFunction{Name: "t2", Arguments: xArgs}},
		}},
		{Role: "tool", ToolCallID: "a", Content: "r1"},
		{Role: "tool", ToolCallID: "b", Content: "r2"},
	}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	wantAction := "<|START_ACTION|>[\n\n    {\"tool_call_id\": \"0\", \"tool_name\": \"t1\", \"parameters\": {}},\n\n    {\"tool_call_id\": \"1\", \"tool_name\": \"t2\", \"parameters\": {\"x\": 1}}\n\n]<|END_ACTION|>"
	if !contains(t, got, wantAction, "multi action") {
		return
	}

	// Consecutive tool messages merge into one result block with sequential
	// regenerated ids.
	wantResults := "<|START_TOOL_RESULT|>[\n    {\n        \"tool_call_id\": \"0\",\n        \"results\": {\n\n            \n            \"0\": {\"content\": \"r1\"}\n\n        },\n        \"is_error\": null\n    },\n    {\n        \"tool_call_id\": \"1\",\n        \"results\": {\n\n            \n            \"0\": {\"content\": \"r2\"}\n\n        },\n        \"is_error\": null\n    }\n\n]<|END_TOOL_RESULT|>"
	contains(t, got, wantResults, "merged tool results")
}

func TestCohereRenderAssistantPrefill(t *testing.T) {
	r := &CohereRenderer{}
	got, err := r.Render([]api.Message{
		{Role: "user", Content: "Q"},
		{Role: "assistant", Content: "partial"},
	}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	wantSuffix := "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_TEXT|>partial"
	if got[len(got)-len(wantSuffix):] != wantSuffix {
		t.Errorf("prefill should leave the text open, got tail %q", got[max(0, len(got)-90):])
	}
}

func TestCohereRendererRegistered(t *testing.T) {
	if rendererForName("cohere") == nil {
		t.Fatal("cohere renderer not registered")
	}
	if got := LeadingBOSForRenderer("cohere"); got != "<BOS_TOKEN>" {
		t.Errorf("LeadingBOS = %q, want <BOS_TOKEN>", got)
	}
}

func contains(t *testing.T, haystack, needle, what string) bool {
	t.Helper()
	if idx := indexOf(haystack, needle); idx == -1 {
		t.Errorf("missing %s:\nwant substring: %q\nin: %q", what, needle, haystack)
		return false
	}
	return true
}

func indexOf(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}
