package renderers

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/api"
)

const (
	lagunaV2Template = "testdata/laguna_v2_chat_template.jinja2"
	lagunaV8Template = "testdata/laguna_v8_chat_template.jinja2"
)

// lagunaToolJSON is the get_weather tool as serialized into <available_tools>,
// matching lagunaWeatherTool().
const lagunaToolJSON = `{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "required": ["location"], "properties": {"location": {"type": "string", "description": "City"}}}}}`
const lagunaMathToolJSON = `{"type": "function", "function": {"name": "add", "description": "Add numbers", "parameters": {"type": "object", "required": ["a", "b"], "properties": {"a": {"type": "number", "description": "First number"}, "b": {"type": "number", "description": "Second number"}}}}}`

// TestLagunaRendererReferenceFlowCoverage checks the renderer against byte-for-byte
// expected output from the Laguna v2 chat template. VERIFY_JINJA2=1 also verifies
// these expected values against the checked-in Jinja fixture.
func TestLagunaRendererReferenceFlowCoverage(t *testing.T) {
	weather := lagunaWeatherTool()
	think := func(v bool) *api.ThinkValue { return &api.ThinkValue{Value: v} }
	verifyJinja2 := lagunaVerifyJinja2(t)

	// system header is always emitted; with no system message the default is used
	defaultHeader := "〈|EOS|〉<system>\n\n" + lagunaDefaultSystem + "\n</system>\n"

	tests := []struct {
		name     string
		messages []api.Message
		tools    []api.Tool
		think    *api.ThinkValue
		want     string
	}{
		{
			name: "empty_messages",
			want: defaultHeader + "<assistant>\n</think>",
		},
		{
			name:     "user_only_default",
			messages: []api.Message{{Role: "user", Content: "Hello"}},
			want:     defaultHeader + "<user>\nHello\n</user>\n<assistant>\n</think>",
		},
		{
			name:     "user_only_think",
			messages: []api.Message{{Role: "user", Content: "Hello"}},
			think:    think(true),
			want:     defaultHeader + "<user>\nHello\n</user>\n<assistant>\n<think>",
		},
		{
			name:     "user_only_nothink",
			messages: []api.Message{{Role: "user", Content: "Hello"}},
			think:    think(false),
			want:     defaultHeader + "<user>\nHello\n</user>\n<assistant>\n</think>",
		},
		{
			name: "first_system_is_header",
			messages: []api.Message{
				{Role: "system", Content: "Stay concise.\n\n"},
				{Role: "user", Content: "Hi"},
			},
			want: "〈|EOS|〉<system>\n\nStay concise.\n</system>\n" +
				"<user>\nHi\n</user>\n<assistant>\n</think>",
		},
		{
			name: "empty_first_system_opts_out_of_header",
			messages: []api.Message{
				{Role: "system", Content: ""},
				{Role: "user", Content: "Hi"},
			},
			want: "〈|EOS|〉<user>\nHi\n</user>\n<assistant>\n</think>",
		},
		{
			name: "additional_system",
			messages: []api.Message{
				{Role: "system", Content: "Primary."},
				{Role: "user", Content: "Hi"},
				{Role: "system", Content: "Secondary."},
			},
			want: "〈|EOS|〉<system>\n\nPrimary.\n</system>\n" +
				"<user>\nHi\n</user>\n" +
				"<system>\nSecondary.\n</system>\n" +
				"<assistant>\n</think>",
		},
		{
			name: "empty_first_system_with_tools",
			messages: []api.Message{
				{Role: "system", Content: ""},
				{Role: "user", Content: "Weather?"},
			},
			tools: weather,
			want: "〈|EOS|〉<system>\n\n\n### Tools\n\n" +
				"You may call functions to assist with the user query.\n" +
				"All available function signatures are listed below:\n" +
				"<available_tools>\n" + lagunaToolJSON + "\n</available_tools>\n\n" +
				"For each function call, return an unescaped XML-like object with function name and arguments within '<tool_call>' and '</tool_call>' tags, like here:\n" +
				"<tool_call>function-name\n<arg_key>argument-key</arg_key>\n<arg_value>value-of-argument-key</arg_value>\n</tool_call>" +
				"\n</system>\n" +
				"<user>\nWeather?\n</user>\n<assistant>\n</think>",
		},
		{
			name: "tools_in_header",
			messages: []api.Message{
				{Role: "system", Content: "Stay concise."},
				{Role: "user", Content: "Weather?"},
			},
			tools: weather,
			think: think(true),
			want: "〈|EOS|〉<system>\n\nStay concise.\n\n### Tools\n\n" +
				"You may call functions to assist with the user query.\n" +
				"All available function signatures are listed below:\n" +
				"<available_tools>\n" + lagunaToolJSON + "\n</available_tools>\n\n" +
				"Wrap your thinking in '<think>', '</think>' tags, followed by a function call. For each function call, return an unescaped XML-like object with function name and arguments within '<tool_call>' and '</tool_call>' tags, like here:\n" +
				"<think> your thoughts here </think>\n" +
				"<tool_call>function-name\n<arg_key>argument-key</arg_key>\n<arg_value>value-of-argument-key</arg_value>\n</tool_call>" +
				"\n</system>\n" +
				"<user>\nWeather?\n</user>\n<assistant>\n<think>",
		},
		{
			name:     "tools_default",
			messages: []api.Message{{Role: "user", Content: "Weather?"}},
			tools:    weather,
			want: "〈|EOS|〉<system>\n\n" + lagunaDefaultSystem + "\n\n### Tools\n\n" +
				"You may call functions to assist with the user query.\n" +
				"All available function signatures are listed below:\n" +
				"<available_tools>\n" + lagunaToolJSON + "\n</available_tools>\n\n" +
				"For each function call, return an unescaped XML-like object with function name and arguments within '<tool_call>' and '</tool_call>' tags, like here:\n" +
				"<tool_call>function-name\n<arg_key>argument-key</arg_key>\n<arg_value>value-of-argument-key</arg_value>\n</tool_call>" +
				"\n</system>\n" +
				"<user>\nWeather?\n</user>\n<assistant>\n</think>",
		},
		{
			name:     "multiple_tools_in_header",
			messages: []api.Message{{Role: "user", Content: "Add then report weather"}},
			tools:    append(weather, lagunaMathTool()...),
			want: "〈|EOS|〉<system>\n\n" + lagunaDefaultSystem + "\n\n### Tools\n\n" +
				"You may call functions to assist with the user query.\n" +
				"All available function signatures are listed below:\n" +
				"<available_tools>\n" + lagunaToolJSON + "\n" + lagunaMathToolJSON + "\n</available_tools>\n\n" +
				"For each function call, return an unescaped XML-like object with function name and arguments within '<tool_call>' and '</tool_call>' tags, like here:\n" +
				"<tool_call>function-name\n<arg_key>argument-key</arg_key>\n<arg_value>value-of-argument-key</arg_value>\n</tool_call>" +
				"\n</system>\n" +
				"<user>\nAdd then report weather\n</user>\n<assistant>\n</think>",
		},
		{
			name: "assistant_history",
			messages: []api.Message{
				{Role: "user", Content: "Add these."},
				{
					Role:     "assistant",
					Content:  "\nCalling the tool.\n",
					Thinking: "Need addition.",
					ToolCalls: []api.ToolCall{{
						Function: api.ToolCallFunction{
							Name: "add",
							Arguments: testArgsOrdered([]orderedArg{
								{Key: "a", Value: 2},
								{Key: "b", Value: 3},
							}),
						},
					}},
				},
				{Role: "tool", Content: "5"},
				{Role: "user", Content: "Thanks"},
			},
			think: think(true),
			want: defaultHeader +
				"<user>\nAdd these.\n</user>\n" +
				"<assistant>\n" +
				"<think>\nNeed addition.\n</think>\n" +
				"Calling the tool.\n" +
				"<tool_call>add\n" +
				"<arg_key>a</arg_key>\n<arg_value>2</arg_value>\n" +
				"<arg_key>b</arg_key>\n<arg_value>3</arg_value>\n" +
				"</tool_call>\n" +
				"</assistant>\n" +
				"<tool_response>\n5\n</tool_response>\n" +
				"<user>\nThanks\n</user>\n<assistant>\n<think>",
		},
		{
			name: "assistant_extracts_thinking_from_content",
			messages: []api.Message{
				{Role: "user", Content: "Explain"},
				{Role: "assistant", Content: "<think>\nPlan\n</think>\nAnswer\n\n"},
				{Role: "user", Content: "Next"},
			},
			think: think(true),
			want: defaultHeader +
				"<user>\nExplain\n</user>\n" +
				"<assistant>\n<think>\nPlan\n</think>\nAnswer\n</assistant>\n" +
				"<user>\nNext\n</user>\n<assistant>\n<think>",
		},
		{
			name: "assistant_thinking_metadata_overrides_content_tags",
			messages: []api.Message{
				{Role: "user", Content: "Explain"},
				{Role: "assistant", Thinking: "Use metadata.", Content: "<think>Ignore this</think>\nAnswer"},
				{Role: "user", Content: "Next"},
			},
			want: defaultHeader +
				"<user>\nExplain\n</user>\n" +
				"<assistant>\n<think>\nUse metadata.\n</think>\nAnswer\n</assistant>\n" +
				"<user>\nNext\n</user>\n<assistant>\n</think>",
		},
		{
			name: "assistant_whitespace_content_only",
			messages: []api.Message{
				{Role: "user", Content: "Continue"},
				{Role: "assistant", Content: " \n\t "},
				{Role: "user", Content: "Next"},
			},
			want: defaultHeader +
				"<user>\nContinue\n</user>\n" +
				"<assistant>\n</think>\n</assistant>\n" +
				"<user>\nNext\n</user>\n<assistant>\n</think>",
		},
		{
			name: "assistant_multiple_tool_calls_mixed_args",
			messages: []api.Message{
				{Role: "user", Content: "Do calls"},
				{
					Role: "assistant",
					ToolCalls: []api.ToolCall{
						{Function: api.ToolCallFunction{
							Name: "echo",
							Arguments: testArgsOrdered([]orderedArg{
								{Key: "text", Value: "hello"},
								{Key: "count", Value: 2},
							}),
						}},
						{Function: api.ToolCallFunction{
							Name: "configure",
							Arguments: testArgsOrdered([]orderedArg{
								{Key: "flag", Value: true},
								{Key: "options", Value: map[string]any{"mode": "fast"}},
							}),
						}},
					},
				},
				{Role: "user", Content: "Done?"},
			},
			want: defaultHeader +
				"<user>\nDo calls\n</user>\n" +
				"<assistant>\n</think>\n" +
				"<tool_call>echo\n" +
				"<arg_key>text</arg_key>\n<arg_value>hello</arg_value>\n" +
				"<arg_key>count</arg_key>\n<arg_value>2</arg_value>\n" +
				"</tool_call>\n" +
				"<tool_call>configure\n" +
				"<arg_key>flag</arg_key>\n<arg_value>true</arg_value>\n" +
				"<arg_key>options</arg_key>\n<arg_value>{\"mode\": \"fast\"}</arg_value>\n" +
				"</tool_call>\n" +
				"</assistant>\n" +
				"<user>\nDone?\n</user>\n<assistant>\n</think>",
		},
	}

	renderer := &LagunaRenderer{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := renderer.Render(tt.messages, tt.tools, tt.think)
			if err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Fatalf("renderer output mismatch vs template (-want +got):\n%s", diff)
			}
			if verifyJinja2 {
				jinja := renderLagunaJinja2Template(t, lagunaV2Template, tt.messages, tt.tools, tt.think)
				if diff := cmp.Diff(jinja, tt.want); diff != "" {
					t.Fatalf("hardcoded expected mismatch vs Jinja2 template (-jinja +want):\n%s", diff)
				}
				if diff := cmp.Diff(jinja, got); diff != "" {
					t.Fatalf("renderer output mismatch vs Jinja2 template (-jinja +got):\n%s", diff)
				}
			}
		})
	}
}

func TestLagunaRendererAssistantPrefill(t *testing.T) {
	got, err := (&LagunaRenderer{}).Render([]api.Message{
		{Role: "user", Content: "Complete this"},
		{Role: "assistant", Content: "Partial"},
	}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	want := "〈|EOS|〉<system>\n\n" + lagunaDefaultSystem + "\n</system>\n" +
		"<user>\nComplete this\n</user>\n<assistant>\n</think>\nPartial\n"
	if diff := cmp.Diff(want, got); diff != "" {
		t.Fatalf("renderer prefill mismatch (-want +got):\n%s", diff)
	}
}

func TestLagunaRendererKnownJinja2Differences(t *testing.T) {
	if !lagunaVerifyJinja2(t) {
		t.Skip("set VERIFY_JINJA2=1 to run Jinja2 difference checks")
	}

	messages := []api.Message{
		{Role: "user", Content: "Complete this"},
		{Role: "assistant", Content: "Partial"},
	}
	got, err := (&LagunaRenderer{}).Render(messages, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	jinja := renderLagunaJinja2Template(t, lagunaV2Template, messages, nil, nil)
	if got == jinja {
		t.Fatal("v2 assistant prefill no longer differs from Jinja2 output")
	}

	wantJinja := "〈|EOS|〉<system>\n\n" + lagunaDefaultSystem + "\n</system>\n" +
		"<user>\nComplete this\n</user>\n<assistant>\n</think>\nPartial\n</assistant>\n<assistant>\n</think>"
	if diff := cmp.Diff(wantJinja, jinja); diff != "" {
		t.Fatalf("v2 assistant prefill Jinja2 reference mismatch (-want +jinja):\n%s", diff)
	}
}

func TestLagunaV8RendererReferenceFlowCoverage(t *testing.T) {
	weather := lagunaWeatherTool()
	think := func(v bool) *api.ThinkValue { return &api.ThinkValue{Value: v} }
	verifyJinja2 := lagunaVerifyJinja2(t)

	defaultHeader := "〈|EOS|〉<system>" + lagunaDefaultSystem + "</system>\n"

	tests := []struct {
		name     string
		messages []api.Message
		tools    []api.Tool
		think    *api.ThinkValue
		want     string
	}{
		{
			name: "empty_messages",
			want: defaultHeader + "<assistant></think>",
		},
		{
			name:     "user_only_default",
			messages: []api.Message{{Role: "user", Content: "Hello"}},
			want:     defaultHeader + "<user>Hello</user>\n<assistant></think>",
		},
		{
			name:     "user_only_think",
			messages: []api.Message{{Role: "user", Content: "Hello"}},
			think:    think(true),
			want:     defaultHeader + "<user>Hello</user>\n<assistant><think>",
		},
		{
			name:     "user_only_nothink",
			messages: []api.Message{{Role: "user", Content: "Hello"}},
			think:    think(false),
			want:     defaultHeader + "<user>Hello</user>\n<assistant></think>",
		},
		{
			name: "first_system_is_header",
			messages: []api.Message{
				{Role: "system", Content: "Stay concise.\n\n"},
				{Role: "user", Content: "Hi"},
			},
			want: "〈|EOS|〉<system>Stay concise.</system>\n" +
				"<user>Hi</user>\n<assistant></think>",
		},
		{
			name: "empty_first_system_opts_out_of_header",
			messages: []api.Message{
				{Role: "system", Content: ""},
				{Role: "user", Content: "Hi"},
			},
			want: "〈|EOS|〉<user>Hi</user>\n<assistant></think>",
		},
		{
			name: "empty_first_system_with_tools",
			messages: []api.Message{
				{Role: "system", Content: ""},
				{Role: "user", Content: "Weather?"},
			},
			tools: weather,
			want: "〈|EOS|〉<system>" +
				"### Tools\n\n" +
				"You may call functions to assist with the user query.\n" +
				"All available function signatures are listed below:\n" +
				"<available_tools>\n" + lagunaToolJSON + "\n</available_tools>" +
				"</system>\n" +
				"<user>Weather?</user>\n<assistant></think>",
		},
		{
			name: "empty_first_system_thinking_enabled",
			messages: []api.Message{
				{Role: "system", Content: ""},
				{Role: "user", Content: "Hi"},
			},
			think: think(true),
			want:  "〈|EOS|〉<system></system>\n<user>Hi</user>\n<assistant><think>",
		},
		{
			name: "additional_system",
			messages: []api.Message{
				{Role: "system", Content: "Primary."},
				{Role: "user", Content: "Hi"},
				{Role: "system", Content: "Secondary."},
			},
			want: "〈|EOS|〉<system>Primary.</system>\n" +
				"<user>Hi</user>\n" +
				"<system>Secondary.</system>\n" +
				"<assistant></think>",
		},
		{
			name: "tools_in_header",
			messages: []api.Message{
				{Role: "system", Content: "Stay concise."},
				{Role: "user", Content: "Weather?"},
			},
			tools: weather,
			think: think(true),
			want: "〈|EOS|〉<system>Stay concise.\n\n" +
				"### Tools\n\n" +
				"You may call functions to assist with the user query.\n" +
				"All available function signatures are listed below:\n" +
				"<available_tools>\n" + lagunaToolJSON + "\n</available_tools>" +
				"</system>\n" +
				"<user>Weather?</user>\n<assistant><think>",
		},
		{
			name:     "tools_default",
			messages: []api.Message{{Role: "user", Content: "Weather?"}},
			tools:    weather,
			want: "〈|EOS|〉<system>" + lagunaDefaultSystem + "\n\n" +
				"### Tools\n\n" +
				"You may call functions to assist with the user query.\n" +
				"All available function signatures are listed below:\n" +
				"<available_tools>\n" + lagunaToolJSON + "\n</available_tools>" +
				"</system>\n" +
				"<user>Weather?</user>\n<assistant></think>",
		},
		{
			name:     "multiple_tools_in_header",
			messages: []api.Message{{Role: "user", Content: "Add then report weather"}},
			tools:    append(weather, lagunaMathTool()...),
			want: "〈|EOS|〉<system>" + lagunaDefaultSystem + "\n\n" +
				"### Tools\n\n" +
				"You may call functions to assist with the user query.\n" +
				"All available function signatures are listed below:\n" +
				"<available_tools>\n" + lagunaToolJSON + "\n" + lagunaMathToolJSON + "\n</available_tools>" +
				"</system>\n" +
				"<user>Add then report weather</user>\n<assistant></think>",
		},
		{
			name: "assistant_history",
			messages: []api.Message{
				{Role: "user", Content: "Add these."},
				{
					Role:     "assistant",
					Content:  "\nCalling the tool.\n",
					Thinking: "Need addition.",
					ToolCalls: []api.ToolCall{{
						Function: api.ToolCallFunction{
							Name: "add",
							Arguments: testArgsOrdered([]orderedArg{
								{Key: "a", Value: 2},
								{Key: "b", Value: 3},
							}),
						},
					}},
				},
				{Role: "tool", Content: "5"},
				{Role: "user", Content: "Thanks"},
			},
			think: think(true),
			want: defaultHeader +
				"<user>Add these.</user>\n" +
				"<assistant>" +
				"<think>Need addition.</think>" +
				"\nCalling the tool.\n" +
				"<tool_call>add" +
				"<arg_key>a</arg_key><arg_value>2</arg_value>" +
				"<arg_key>b</arg_key><arg_value>3</arg_value>" +
				"</tool_call>" +
				"</assistant>\n" +
				"<tool_response>5</tool_response>\n" +
				"<user>Thanks</user>\n<assistant><think>",
		},
		{
			name: "assistant_reasoning_ignored_when_thinking_disabled",
			messages: []api.Message{
				{Role: "user", Content: "Explain"},
				{Role: "assistant", Thinking: "Hidden plan.", Content: "Answer"},
				{Role: "user", Content: "Next"},
			},
			want: defaultHeader +
				"<user>Explain</user>\n" +
				"<assistant></think>Answer</assistant>\n" +
				"<user>Next</user>\n<assistant></think>",
		},
		{
			name: "assistant_empty_reasoning_when_thinking_enabled",
			messages: []api.Message{
				{Role: "user", Content: "Explain"},
				{Role: "assistant", Content: "Answer"},
				{Role: "user", Content: "Next"},
			},
			think: think(true),
			want: defaultHeader +
				"<user>Explain</user>\n" +
				"<assistant><think></think>Answer</assistant>\n" +
				"<user>Next</user>\n<assistant><think>",
		},
		{
			name: "assistant_preserves_content_whitespace",
			messages: []api.Message{
				{Role: "user", Content: "Explain"},
				{Role: "assistant", Content: "\nAnswer\n"},
				{Role: "user", Content: "Next"},
			},
			want: defaultHeader +
				"<user>Explain</user>\n" +
				"<assistant></think>\nAnswer\n</assistant>\n" +
				"<user>Next</user>\n<assistant></think>",
		},
		{
			name: "assistant_multiple_tool_calls_mixed_args",
			messages: []api.Message{
				{Role: "user", Content: "Do calls"},
				{
					Role: "assistant",
					ToolCalls: []api.ToolCall{
						{Function: api.ToolCallFunction{
							Name: "echo",
							Arguments: testArgsOrdered([]orderedArg{
								{Key: "text", Value: "hello"},
								{Key: "count", Value: 2},
							}),
						}},
						{Function: api.ToolCallFunction{
							Name: "configure",
							Arguments: testArgsOrdered([]orderedArg{
								{Key: "flag", Value: true},
								{Key: "options", Value: map[string]any{"mode": "fast"}},
							}),
						}},
					},
				},
				{Role: "user", Content: "Done?"},
			},
			want: defaultHeader +
				"<user>Do calls</user>\n" +
				"<assistant></think>" +
				"<tool_call>echo" +
				"<arg_key>text</arg_key><arg_value>hello</arg_value>" +
				"<arg_key>count</arg_key><arg_value>2</arg_value>" +
				"</tool_call>" +
				"<tool_call>configure" +
				"<arg_key>flag</arg_key><arg_value>true</arg_value>" +
				"<arg_key>options</arg_key><arg_value>{\"mode\": \"fast\"}</arg_value>" +
				"</tool_call>" +
				"</assistant>\n" +
				"<user>Done?</user>\n<assistant></think>",
		},
		{
			name: "final_assistant_closes_then_generation_prompt",
			messages: []api.Message{
				{Role: "user", Content: "Complete this"},
				{Role: "assistant", Content: "Partial"},
			},
			want: defaultHeader +
				"<user>Complete this</user>\n" +
				"<assistant></think>Partial</assistant>\n" +
				"<assistant></think>",
		},
	}

	renderer := &LagunaV8Renderer{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := renderer.Render(tt.messages, tt.tools, tt.think)
			if err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Fatalf("renderer output mismatch vs template (-want +got):\n%s", diff)
			}
			if verifyJinja2 {
				jinja := renderLagunaJinja2Template(t, lagunaV8Template, tt.messages, tt.tools, tt.think)
				if diff := cmp.Diff(jinja, tt.want); diff != "" {
					t.Fatalf("hardcoded expected mismatch vs Jinja2 template (-jinja +want):\n%s", diff)
				}
				if diff := cmp.Diff(jinja, got); diff != "" {
					t.Fatalf("renderer output mismatch vs Jinja2 template (-jinja +got):\n%s", diff)
				}
			}
		})
	}
}

func TestLagunaRendererMatchesJinja2ExpandedParity(t *testing.T) {
	if os.Getenv("VERIFY_JINJA2") == "" {
		t.Skip("set VERIFY_JINJA2=1 to run expanded Jinja2 parity checks")
	}
	lagunaVerifyJinja2(t)

	tests := []struct {
		name     string
		messages []api.Message
		tools    []api.Tool
		think    *api.ThinkValue
	}{
		{
			name:     "user_only",
			messages: []api.Message{{Role: "user", Content: "Hello"}},
		},
		{
			name: "system_user",
			messages: []api.Message{
				{Role: "system", Content: "Stay concise.\n"},
				{Role: "user", Content: "Hello"},
			},
		},
		{
			name: "additional_system_and_tool_response",
			messages: []api.Message{
				{Role: "system", Content: "Primary."},
				{Role: "user", Content: "Weather?"},
				{Role: "assistant", Content: "Calling."},
				{Role: "tool", Content: "Sunny"},
				{Role: "system", Content: "Secondary."},
			},
		},
		{
			name:     "thinking_enabled",
			messages: []api.Message{{Role: "user", Content: "Think briefly."}},
			think:    &api.ThinkValue{Value: true},
		},
		{
			name:     "thinking_disabled",
			messages: []api.Message{{Role: "user", Content: "Answer directly."}},
			think:    &api.ThinkValue{Value: false},
		},
		{
			name: "tools_and_assistant_history",
			messages: []api.Message{
				{Role: "system", Content: "Stay concise."},
				{Role: "user", Content: "Weather?"},
				{Role: "assistant", Content: "Calling.", Thinking: "Need weather.", ToolCalls: []api.ToolCall{{
					Function: api.ToolCallFunction{
						Name:      "get_weather",
						Arguments: testArgsOrdered([]orderedArg{{Key: "location", Value: "Paris"}}),
					},
				}}},
				{Role: "tool", Content: "Sunny"},
				{Role: "user", Content: "Thanks"},
			},
			tools: lagunaWeatherTool(),
			think: &api.ThinkValue{Value: true},
		},
	}

	variants := []struct {
		name     string
		renderer Renderer
		template string
	}{
		{name: "v2", renderer: &LagunaRenderer{}, template: lagunaV2Template},
		{name: "v8", renderer: &LagunaV8Renderer{}, template: lagunaV8Template},
	}

	for _, variant := range variants {
		t.Run(variant.name, func(t *testing.T) {
			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					got, err := variant.renderer.Render(tt.messages, tt.tools, tt.think)
					if err != nil {
						t.Fatal(err)
					}
					want := renderLagunaJinja2Template(t, variant.template, tt.messages, tt.tools, tt.think)
					if diff := cmp.Diff(want, got); diff != "" {
						t.Fatalf("renderer output mismatch vs Jinja2 template (-jinja +got):\n%s", diff)
					}
				})
			}
		})
	}
}

func lagunaVerifyJinja2(t *testing.T) bool {
	t.Helper()
	if os.Getenv("VERIFY_JINJA2") == "" {
		return false
	}
	python := lagunaJinjaPython(t)
	cmd := exec.Command(python, "-c", "from transformers.utils.chat_template_utils import _compile_jinja_template")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("VERIFY_JINJA2=1 requires transformers chat template support in %s: %v\n%s", python, err, out)
	}
	return true
}

func lagunaJinjaPython(t *testing.T) string {
	t.Helper()
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Fatal("VERIFY_JINJA2=1 requires python3 on PATH")
	}
	return python
}

func renderLagunaJinja2Template(t *testing.T, templateRelPath string, messages []api.Message, tools []api.Tool, think *api.ThinkValue) string {
	t.Helper()

	templatePath, err := filepath.Abs(templateRelPath)
	if err != nil {
		t.Fatalf("failed to get template path: %v", err)
	}

	type jinjaToolCall struct {
		Function struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		} `json:"function"`
	}
	type jinjaMessage struct {
		Role             string          `json:"role"`
		Content          string          `json:"content"`
		Reasoning        string          `json:"reasoning,omitempty"`
		ReasoningContent string          `json:"reasoning_content,omitempty"`
		ToolCalls        []jinjaToolCall `json:"tool_calls,omitempty"`
	}

	jinjaMessages := make([]jinjaMessage, 0, len(messages))
	for _, msg := range messages {
		jm := jinjaMessage{
			Role:             msg.Role,
			Content:          msg.Content,
			Reasoning:        msg.Thinking,
			ReasoningContent: msg.Thinking,
		}
		for _, call := range msg.ToolCalls {
			jc := jinjaToolCall{}
			jc.Function.Name = call.Function.Name
			raw, err := call.Function.Arguments.MarshalJSON()
			if err != nil {
				t.Fatalf("failed to marshal tool args: %v", err)
			}
			jc.Function.Arguments = json.RawMessage(raw)
			jm.ToolCalls = append(jm.ToolCalls, jc)
		}
		jinjaMessages = append(jinjaMessages, jm)
	}

	messagesJSON, err := json.Marshal(jinjaMessages)
	if err != nil {
		t.Fatalf("failed to marshal messages: %v", err)
	}

	toolsJSON := "None"
	if len(tools) > 0 {
		b, err := json.Marshal(tools)
		if err != nil {
			t.Fatalf("failed to marshal tools: %v", err)
		}
		toolsJSON = string(b)
	}

	enableThinking := "unset"
	if think != nil {
		if think.Bool() {
			enableThinking = "true"
		} else {
			enableThinking = "false"
		}
	}

	script := `
import json
import sys
from pathlib import Path
from transformers.utils.chat_template_utils import _compile_jinja_template

template_path, messages_json, tools_json, enable_thinking = sys.argv[1:5]
tmpl = _compile_jinja_template(Path(template_path).read_text())
kwargs = {
    "messages": json.loads(messages_json),
    "add_generation_prompt": True,
}
if tools_json != "None":
    kwargs["tools"] = json.loads(tools_json)
if enable_thinking == "true":
    kwargs["enable_thinking"] = True
elif enable_thinking == "false":
    kwargs["enable_thinking"] = False
print(tmpl.render(**kwargs), end="")
`
	cmd := exec.Command(lagunaJinjaPython(t), "-c", script, templatePath, string(messagesJSON), toolsJSON, enableThinking)
	var stdout, stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("python render failed: %v\nstderr: %s", err, stderr.String())
	}
	return stdout.String()
}

func TestLagunaTemplateFixturesMatchExpectedVersions(t *testing.T) {
	v2, err := os.ReadFile(lagunaV2Template)
	if err != nil {
		t.Fatalf("failed to read %s: %v", lagunaV2Template, err)
	}

	v8, err := os.ReadFile(lagunaV8Template)
	if err != nil {
		t.Fatalf("failed to read %s: %v", lagunaV8Template, err)
	}

	if !strings.Contains(string(v2), "laguna_glm_thinking_v5/chat_template.jinja") {
		t.Fatalf("%s does not look like the v2 Laguna template fixture", lagunaV2Template)
	}
	if !strings.Contains(string(v8), "laguna_glm_thinking_v8/chat_template.jinja") {
		t.Fatalf("%s does not look like the v8 Laguna template fixture", lagunaV8Template)
	}
	if !strings.Contains(string(v2), "render_assistant_messages_raw") {
		t.Fatalf("%s should retain the v2 raw assistant branch", lagunaV2Template)
	}
	if strings.Contains(string(v8), "render_assistant_messages_raw") {
		t.Fatalf("%s unexpectedly contains the v2 raw assistant branch", lagunaV8Template)
	}
	if diff := cmp.Diff(string(v2), string(v8)); diff == "" {
		t.Fatal("Laguna v2 and v8 template fixtures unexpectedly match")
	}
}

func lagunaWeatherTool() []api.Tool {
	return []api.Tool{{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "get_weather",
			Description: "Get weather",
			Parameters: api.ToolFunctionParameters{
				Type:     "object",
				Required: []string{"location"},
				Properties: testPropsOrdered([]orderedProp{{
					Key: "location",
					Value: api.ToolProperty{
						Type:        api.PropertyType{"string"},
						Description: "City",
					},
				}}),
			},
		},
	}}
}

func lagunaMathTool() []api.Tool {
	return []api.Tool{{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "add",
			Description: "Add numbers",
			Parameters: api.ToolFunctionParameters{
				Type:     "object",
				Required: []string{"a", "b"},
				Properties: testPropsOrdered([]orderedProp{
					{
						Key: "a",
						Value: api.ToolProperty{
							Type:        api.PropertyType{"number"},
							Description: "First number",
						},
					},
					{
						Key: "b",
						Value: api.ToolProperty{
							Type:        api.PropertyType{"number"},
							Description: "Second number",
						},
					},
				}),
			},
		},
	}}
}
