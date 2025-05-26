package tools

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/template"
)

func readFile(t *testing.T, base, name string) *bytes.Buffer {
	t.Helper()

	bts, err := os.ReadFile(filepath.Join(base, name))
	if err != nil {
		t.Fatal(err)
	}

	return bytes.NewBuffer(bts)
}

func TestParseJSONToolCalls(t *testing.T) {
	tests := []struct {
		name          string
		input         string
		nameField     string
		argsField     string
		wantToolCalls []api.ToolCall
		wantErr       error
		prefix        string
	}{
		{
			name:      "valid single tool call",
			input:     `{"name": "test_tool", "arguments": {"arg1": "value1"}}`,
			nameField: "name",
			argsField: "arguments",
			wantToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "test_tool",
						Arguments: map[string]any{
							"arg1": "value1",
						},
					},
				},
			},
			wantErr: nil,
			prefix:  "",
		},
		{
			name:          "incomplete JSON",
			input:         `{"name": "test_tool", "arguments": {"arg1": `,
			nameField:     "name",
			argsField:     "arguments",
			wantToolCalls: nil,
			wantErr:       errAccumulateMore,
			prefix:        "",
		},
		{
			name:          "invalid JSON",
			input:         `not json at all`,
			nameField:     "name",
			argsField:     "arguments",
			wantToolCalls: nil,
			wantErr:       errInvalidToolCall,
			prefix:        "",
		},
		{
			name:          "missing required fields",
			input:         `{"other": "field"}`,
			nameField:     "name",
			argsField:     "arguments",
			wantToolCalls: nil,
			wantErr:       errInvalidToolCall,
			prefix:        "",
		},
		{
			name: "multiple tool calls in array",
			input: `[
				{"name": "tool1", "arguments": {"arg1": 1}},
				{"name": "tool2", "arguments": {"arg2": "value"}}
			]`,
			nameField: "name",
			argsField: "arguments",
			wantToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "tool1",
						Arguments: map[string]any{
							"arg1": float64(1),
						},
					},
				},
				{
					Function: api.ToolCallFunction{
						Name: "tool2",
						Arguments: map[string]any{
							"arg2": "value",
						},
					},
				},
			},
			wantErr: nil,
			prefix:  "",
		},
		{
			name: "multiple tool calls without array",
			input: `
				{"name": "tool1", "arguments": {"arg1": 1}},
				{"name": "tool2", "arguments": {"arg2": "value"}}
			`,
			nameField: "name",
			argsField: "arguments",
			wantToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "tool1",
						Arguments: map[string]any{
							"arg1": float64(1),
						},
					},
				},
				{
					Function: api.ToolCallFunction{
						Name: "tool2",
						Arguments: map[string]any{
							"arg2": "value",
						},
					},
				},
			},
			wantErr: nil,
			prefix:  "",
		},
		{
			name: "multiple tool calls with text after",
			input: `
				{"name": "tool1", "arguments": {"arg1": 1}} text
				{"name": "tool2", "arguments": {"arg2": "value"}} text
			`,
			nameField: "name",
			argsField: "arguments",
			wantToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "tool1",
						Arguments: map[string]any{
							"arg1": float64(1),
						},
					},
				},
				{
					Function: api.ToolCallFunction{
						Name: "tool2",
						Arguments: map[string]any{
							"arg2": "value",
						},
					},
				},
			},
			wantErr: nil,
			prefix:  "",
		},
		{
			name: "second tool call in array",
			input: `
				, {"name": "tool2", "arguments": {"arg2": "value"}}
			`,
			nameField: "name",
			argsField: "arguments",
			wantToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "tool2",
						Arguments: map[string]any{
							"arg2": "value",
						},
					},
				},
			},
			wantErr: nil,
			prefix:  "",
		},
		// a bad JSON would not return any tool calls or content as it would always accumulate more
		{
			name:          "unbalanced square brackets",
			input:         `[{"name": "tool1", "arguments": {"arg1": [1, 2}]`,
			nameField:     "name",
			argsField:     "arguments",
			wantToolCalls: nil,
			wantErr:       errAccumulateMore,
			prefix:        "",
		},
		{
			name:          "incomplete square brackets",
			input:         `[{"name": "tool1", "arguments": {"arg1": [1, 2, 3`,
			nameField:     "name",
			argsField:     "arguments",
			wantToolCalls: nil,
			wantErr:       errAccumulateMore,
			prefix:        "",
		},
		{
			name:      "nested arrays in arguments",
			input:     `{"name": "tool1", "arguments": {"arg1": [1, 2, ["nested", "array"]]}}`,
			nameField: "name",
			argsField: "arguments",
			wantToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "tool1",
						Arguments: map[string]any{
							"arg1": []any{float64(1), float64(2), []any{"nested", "array"}},
						},
					},
				},
			},
			wantErr: nil,
			prefix:  "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotCalls, err := parseJSONToolCalls(tt.input, tt.nameField, tt.argsField, tt.prefix)

			if err != tt.wantErr {
				t.Errorf("parseJSONToolCalls() error = %v, want %v", err, tt.wantErr)
			}

			if len(gotCalls) != 0 && tt.wantErr != nil {
				t.Errorf("parseJSONToolCalls() valid = %v, want %v", len(gotCalls) == 0, tt.wantErr == nil)
			}

			if diff := cmp.Diff(gotCalls, tt.wantToolCalls); diff != "" {
				t.Errorf("parseJSONToolCalls() tool calls mismatch (-got +want):\n%s", diff)
			}
		})
	}
}

func TestParseToolCalls(t *testing.T) {
	p := filepath.Join("testdata")
	t1 := api.ToolCall{
		Function: api.ToolCallFunction{
			Name: "get_current_weather",
			Arguments: api.ToolCallFunctionArguments{
				"format":   "fahrenheit",
				"location": "San Francisco, CA",
			},
		},
	}
	t2 := api.ToolCall{
		Function: api.ToolCallFunction{
			Name: "get_current_weather",
			Arguments: api.ToolCallFunctionArguments{
				"format":   "celsius",
				"location": "Toronto, Canada",
			},
		},
	}

	cases := []struct {
		name             string
		model            string
		output           string
		expectedToolCall []api.ToolCall
		expectedTokens   string
	}{
		{
			name:             "mistral malformed json with tool calls prefix",
			model:            "mistral",
			output:           `[TOOL_CALLS]  [{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_curren}]`,
			expectedToolCall: []api.ToolCall{t1},
			expectedTokens:   "",
		},
		{
			name:             "mistral multiple tool calls without prefix",
			model:            "mistral",
			output:           `[{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}} ]`,
			expectedToolCall: []api.ToolCall{t1, t2},
			expectedTokens:   "",
		},
		{
			name:  "mistral tool calls with text between no prefix",
			model: "mistral",
			output: `[{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}] 
			model outputs more tokens here and then [{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]`,
			expectedToolCall: []api.ToolCall{t1, t2},
			expectedTokens:   `model outputs more tokens here and then [{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]`,
		},
		{
			name:             "mistral valid json with tool calls prefix",
			model:            "mistral",
			output:           `[TOOL_CALLS]  [{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]`,
			expectedToolCall: []api.ToolCall{t1, t2},
			expectedTokens:   "",
		},
		{
			name:  "mistral multiple tool calls with text between and prefix",
			model: "mistral",
			output: `[TOOL_CALLS]  [{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]
			model outputs more tokens here and then [{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]`,
			expectedToolCall: []api.ToolCall{t1, t2, t1, t2},
			expectedTokens:   "",
		},
		{
			name:             "mistral incomplete json with tool calls prefix",
			model:            "mistral",
			output:           `[TOOL_CALLS]  [{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, `,
			expectedToolCall: []api.ToolCall{},
			expectedTokens:   "",
		},
		{
			name:  "mistral invalid tool call with explanatory text no prefix",
			model: "mistral",
			output: `I'm not aware of that information. However, I can suggest searching for the weather using the "get_current_weather" function:

		[{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]`,
			expectedToolCall: []api.ToolCall{},
			expectedTokens:   `I'm not aware of that information. However, I can suggest searching for the weather using the "get_current_weather" function: [{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]`,
		},
		{
			name:             "mistral tool calls without prefix",
			model:            "mistral",
			output:           `[{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]`,
			expectedToolCall: []api.ToolCall{t1, t2},
			expectedTokens:   "",
		},
		{
			name:  "command r plus tool calls with json block format",
			model: "command-r-plus",
			output: "Action: ```json" + `
		[
		    {
		        "tool_name": "get_current_weather",
		        "parameters": {
		            "format": "fahrenheit",
		            "location": "San Francisco, CA"
		        }
		    },
		    {
		        "tool_name": "get_current_weather",
		        "parameters": {
		            "format": "celsius",
		            "location": "Toronto, Canada"
		        }
		    }
		]
		` + "```",
			expectedToolCall: []api.ToolCall{t1, t2},
			expectedTokens:   "",
		},
		{
			name:             "firefunction tool calls with functools prefix",
			model:            "firefunction",
			output:           ` functools[{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]`,
			expectedToolCall: []api.ToolCall{t1, t2},
			expectedTokens:   "",
		},
		{
			name:  "llama3 groq single tool call with xml tags",
			model: "llama3-groq-tool-use",
			output: `<tool_call>
		{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}}
		</tool_call>`,
			expectedToolCall: []api.ToolCall{t1},
			expectedTokens:   "",
		},
		{
			name:             "xlam tool calls with wrapper object",
			model:            "xlam",
			output:           `{"tool_calls": [{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]}`,
			expectedToolCall: []api.ToolCall{t1, t2},
			expectedTokens:   "",
		},
		{
			name:             "qwen2.5 single tool call with prefix",
			model:            "qwen2.5",
			output:           `<tool_call>{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}}</tool_call>`,
			expectedToolCall: []api.ToolCall{t1},
			expectedTokens:   "",
		},
		{
			name:             "qwen2.5 multiple tool calls with and without prefix",
			model:            "qwen2.5",
			output:           `{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}} <tool_call>{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}}</tool_call> <tool_call>{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}</tool_call>`,
			expectedToolCall: []api.ToolCall{t1, t1, t2},
			expectedTokens:   "",
		},
		{
			name:             "qwen2.5 plain text response no tool calls",
			model:            "qwen2.5",
			output:           "The weather in San Francisco, CA is 70°F and in Toronto, Canada is 20°C.",
			expectedToolCall: []api.ToolCall{},
			expectedTokens:   "The weather in San Francisco, CA is 70°F and in Toronto, Canada is 20°C.",
		},
		{
			name:             "qwen2.5 tool calls with trailing text",
			model:            "qwen2.5",
			output:           `[{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}}, {"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}] some tokens after call`,
			expectedToolCall: []api.ToolCall{t1, t2},
			expectedTokens:   "some tokens after call",
		},
		{
			name:             "qwen2.5 tool calls with initial text",
			model:            "qwen2.5",
			output:           `some tokens before call [{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}}, {"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]`,
			expectedToolCall: []api.ToolCall{},
			expectedTokens:   `some tokens before call [{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}}, {"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]`,
		},
		{
			name:             "qwen2.5 tool calls with prefix and trailing text",
			model:            "qwen2.5",
			output:           `<tool_call> [{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}}, {"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}] </tool_call> some tokens after call`,
			expectedToolCall: []api.ToolCall{t1, t2},
			expectedTokens:   "",
		},
		{
			name:             "qwen2.5 tool calls with prefix and initial text",
			model:            "qwen2.5",
			output:           `some tokens before call <tool_call> [{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}}, {"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}] </tool_call>`,
			expectedToolCall: []api.ToolCall{t1, t2},
			expectedTokens:   "some tokens before call",
		},
		{
			name:             "qwen2.5 tool calls without and with prefix",
			model:            "qwen2.5",
			output:           `{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}} <tool_call>{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}</tool_call>`,
			expectedToolCall: []api.ToolCall{t1, t2},
			expectedTokens:   "",
		},
		{
			name:             "qwen2.5 tool calls without and with prefix and text between",
			model:            "qwen2.5",
			output:           `{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}} some tokens between <tool_call>{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}</tool_call> some tokens after call`,
			expectedToolCall: []api.ToolCall{t1, t2},
			expectedTokens:   "some tokens between",
		},
		{
			name:             "qwen2.5 tool calls without prefix and invalid tool call with other tokens",
			model:            "qwen2.5",
			output:           `hi [{"options": "foo"}]`,
			expectedToolCall: []api.ToolCall{},
			expectedTokens:   `hi [{"options": "foo"}]`,
		},
		{
			name:             "qwen2.5 tool calls with prefix and invalid tool call",
			model:            "qwen2.5",
			output:           `<tool_call> [{"options": "foo"}] </tool_call> `,
			expectedToolCall: []api.ToolCall{},
			expectedTokens:   ``,
		},
		{
			name:             "qwen3 tool call with think prefix and tool prefix (sent as a single token)",
			model:            "qwen3",
			output:           `<think>Okay, let me think what tool we should use...</think><tool_call>{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}}</tool_call>`,
			expectedToolCall: []api.ToolCall{t1},
			expectedTokens:   "<think>Okay, let me think what tool we should use...</think>",
		},
		{
			name:             "qwen3 tool call with think prefix, tool prefix, and whitespace (sent as separate tokens)",
			model:            "qwen3",
			output:           `<think>Okay, let me think what tool we should use...</think> <tool_call>{ "name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}} </tool_call>`,
			expectedToolCall: []api.ToolCall{t1},
			expectedTokens:   "<think>Okay, let me think what tool we should use...</think>",
		},
		{
			name:             "qwen3 empty think prefix without tool prefix and invalid tool call",
			model:            "qwen3",
			output:           `<think></think> {"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}} </tool_call>`,
			expectedToolCall: []api.ToolCall{},
			expectedTokens:   `<think></think> {"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}} </tool_call>`,
		},
		{
			name:             "qwen3 empty think prefix with tool prefix and valid tool call",
			model:            "qwen3",
			output:           `<think></think><tool_call>{ "name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}}  </tool_call>`,
			expectedToolCall: []api.ToolCall{t1},
			expectedTokens:   `<think></think>`,
		},
		{
			name:             "qwen3 invalid tool call with fake tool prefix (single rune suffix match)",
			model:            "qwen3",
			output:           `<think></think>< fakeout {"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}} </tool_call>`,
			expectedToolCall: []api.ToolCall{},
			expectedTokens:   `<think></think>< fakeout {"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}} </tool_call>`,
		},
		{
			name:             "qwen3 invalid tool call with partial tool prefix (multiple rune suffix match)",
			model:            "qwen3",
			output:           `<think></think><tool_c fakeout {"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}} </tool_call>`,
			expectedToolCall: []api.ToolCall{},
			expectedTokens:   `<think></think><tool_c fakeout {"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}} </tool_call>`,
		},
		{
			name:             "qwen3 invalid tool call with malformed tool prefix",
			model:            "qwen3",
			output:           `<think></think><tool_cfakeout {"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}} </tool_call>`,
			expectedToolCall: []api.ToolCall{},
			expectedTokens:   `<think></think><tool_cfakeout {"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}} </tool_call>`,
		},
		{
			name:             "model with prefix in template, no prefix in output",
			model:            "qwen2.5",
			output:           `[{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}} {"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]`,
			expectedToolCall: []api.ToolCall{t1, t2},
			expectedTokens:   "",
		},
		{
			name:             "model with prefix in template, prefix in output",
			model:            "qwen2.5",
			output:           `<tool_call>[{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}} {"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]</tool_call>`,
			expectedToolCall: []api.ToolCall{t1, t2},
			expectedTokens:   "",
		},
		{
			name:             "model without prefix in template, no prefix in output",
			model:            "llama3.2",
			output:           `[{"name": "get_current_weather", "parameters": {"format":"fahrenheit","location":"San Francisco, CA"}} {"name": "get_current_weather", "parameters": {"format":"celsius","location":"Toronto, Canada"}}]`,
			expectedToolCall: []api.ToolCall{t1, t2},
			expectedTokens:   "",
		},
		{
			name:             "model without prefix in template, no prefix in output, single tool call",
			model:            "llama3.2",
			output:           `{"name": "get_current_weather", "parameters": {"format":"fahrenheit","location":"San Francisco, CA"}}`,
			expectedToolCall: []api.ToolCall{t1},
			expectedTokens:   "",
		},
		{
			name:             "model without prefix in template, prefix in output",
			model:            "llama3.2",
			output:           `<tool_call> [{"name": "get_current_weather", "parameters": {"format":"fahrenheit","location":"San Francisco, CA"}} {"name": "get_current_weather", "parameters": {"format":"celsius","location":"Toronto, Canada"}}]</tool_call>`,
			expectedToolCall: []api.ToolCall{},
			expectedTokens:   `<tool_call> [{"name": "get_current_weather", "parameters": {"format":"fahrenheit","location":"San Francisco, CA"}} {"name": "get_current_weather", "parameters": {"format":"celsius","location":"Toronto, Canada"}}]</tool_call>`,
		},
		{
			name:             "model with prefix in template, no prefix in output, tokens before",
			model:            "qwen2.5",
			output:           `some tokens before [{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}} {"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]`,
			expectedToolCall: []api.ToolCall{},
			expectedTokens:   `some tokens before [{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}} {"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]`,
		},
		{
			name:             "model with prefix in template, prefix in output, tokens after",
			model:            "qwen2.5",
			output:           `<tool_call>[{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}} {"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]</tool_call> some tokens after`,
			expectedToolCall: []api.ToolCall{t1, t2},
			expectedTokens:   "",
		},
		{
			name:             "model without prefix in template, no prefix in output, tokens after",
			model:            "llama3.2",
			output:           `[{"name": "get_current_weather", "parameters": {"format":"fahrenheit","location":"San Francisco, CA"}} {"name": "get_current_weather", "parameters": {"format":"celsius","location":"Toronto, Canada"}}]</tool_call> some tokens after`,
			expectedToolCall: []api.ToolCall{t1, t2},
			expectedTokens:   "",
		},
		{
			name:             "model without prefix in template, no prefix in output, tokens before",
			model:            "llama3.2",
			output:           `some tokens before [{"name": "get_current_weather", "parameters": {"format":"fahrenheit","location":"San Francisco, CA"}} {"name": "get_current_weather", "parameters": {"format":"celsius","location":"Toronto, Canada"}}]`,
			expectedToolCall: []api.ToolCall{},
			expectedTokens:   `some tokens before [{"name": "get_current_weather", "parameters": {"format":"fahrenheit","location":"San Francisco, CA"}} {"name": "get_current_weather", "parameters": {"format":"celsius","location":"Toronto, Canada"}}]`,
		},
		{
			name:             "model without prefix in template, prefix in output, tokens after",
			model:            "llama3.2",
			output:           `<tool_call> [{"name": "get_current_weather", "parameters": {"format":"fahrenheit","location":"San Francisco, CA"}} {"name": "get_current_weather", "parameters": {"format":"celsius","location":"Toronto, Canada"}}]</tool_call> some tokens after`,
			expectedToolCall: []api.ToolCall{},
			expectedTokens:   `<tool_call> [{"name": "get_current_weather", "parameters": {"format":"fahrenheit","location":"San Francisco, CA"}} {"name": "get_current_weather", "parameters": {"format":"celsius","location":"Toronto, Canada"}}]</tool_call> some tokens after`,
		},
	}

	var tools []api.Tool
	if err := json.Unmarshal(readFile(t, p, "tools.json").Bytes(), &tools); err != nil {
		t.Fatal(err)
	}

	var messages []api.Message
	if err := json.Unmarshal(readFile(t, p, "messages.json").Bytes(), &messages); err != nil {
		t.Fatal(err)
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			tmpl, err := template.Parse(readFile(t, p, fmt.Sprintf("%s.gotmpl", tt.model)).String())
			if err != nil {
				t.Fatal(err)
			}

			t.Run("template", func(t *testing.T) {
				actual := &bytes.Buffer{} // Create new buffer for each test
				if err := tmpl.Execute(actual, template.Values{Tools: tools, Messages: messages}); err != nil {
					t.Fatal(err)
				}

				if diff := cmp.Diff(actual.String(), readFile(t, p, fmt.Sprintf("%s.out", tt.model)).String()); diff != "" {
					t.Errorf("mismatch (-got +want):\n%s", diff)
				}
			})

			t.Run("parse", func(t *testing.T) {
				tp, err := NewParser(tmpl.Template)
				if err != nil {
					t.Fatal(err)
				}
				got := []api.ToolCall{}
				var gotTokens strings.Builder

				tokens := strings.Fields(tt.output)
				for _, tok := range tokens {
					s := " " + tok

					toolCalls, content := tp.Add(s)
					if len(content) > 0 {
						gotTokens.WriteString(content)
					} else if len(toolCalls) > 0 {
						got = append(got, toolCalls...)
					}
				}

				// Compare tool calls if we expect any
				if diff := cmp.Diff(got, tt.expectedToolCall); diff != "" {
					t.Errorf("tool calls mismatch (-got +want):\n%s", diff)
				}

				// Compare tokens if we expect any
				stripped := strings.TrimSpace(gotTokens.String())
				if diff := cmp.Diff(stripped, tt.expectedTokens); diff != "" {
					t.Log("actualTokens", stripped, "expectedTokens", tt.expectedTokens)
					t.Errorf("tokens mismatch (-got +want):\n%s", diff)
				}
			})
		})
	}
}
