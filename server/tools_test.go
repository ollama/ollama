package server

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

func TestParseToolCalls(t *testing.T) {
	p := filepath.Join("testdata", "tools")
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
		wantErr          bool
	}{
		{
			name:             "mistral invalid json",
			model:            "mistral",
			output:           `[TOOL_CALLS]  [{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_curren}]`,
			expectedToolCall: []api.ToolCall{},
			expectedTokens:   "",
			wantErr:          true,
		},
		{
			name:             "mistral multiple tool calls - no prefix",
			model:            "mistral",
			output:           `[{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]`,
			expectedToolCall: []api.ToolCall{t1, t2},
			expectedTokens:   "",
			wantErr:          false,
		},
		{
			name:  "mistral tool calls with text in between - no prefix",
			model: "mistral",
			output: `[{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}] 
			model outputs more tokens here and then [{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]`,
			expectedToolCall: []api.ToolCall{t1, t2},
			expectedTokens:   `model outputs more tokens here and then [{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]`,
			wantErr:          false,
		},
		{
			name:             "mistral valid json - with prefix",
			model:            "mistral",
			output:           `[TOOL_CALLS]  [{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]`,
			expectedToolCall: []api.ToolCall{t1, t2},
			expectedTokens:   "",
			wantErr:          false,
		},
		{
			// In this case we'd be ignoring the text in between and just returning the tool calls
			name:  "mistral valid json with text in between - with prefix",
			model: "mistral",
			output: `[TOOL_CALLS]  [{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]
			model outputs more tokens here and then [{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]`,
			expectedToolCall: []api.ToolCall{t1, t2, t1, t2},
			expectedTokens:   "",
			wantErr:          false,
		},
		{
			name:             "mistral incomplete json",
			model:            "mistral",
			output:           `[TOOL_CALLS]  [{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, `,
			expectedToolCall: []api.ToolCall{},
			expectedTokens:   "",
			wantErr:          true,
		},
		{
			name:  "mistral without tool token",
			model: "mistral",
			output: `I'm not aware of that information. However, I can suggest searching for the weather using the "get_current_weather" function:

		[{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]`,
			expectedToolCall: []api.ToolCall{},
			expectedTokens:   `I'm not aware of that information. However, I can suggest searching for the weather using the "get_current_weather" function: [{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]`,
			wantErr:          true,
		},
		{
			name:             "mistral without tool token - tool first",
			model:            "mistral",
			output:           `[{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]`,
			expectedToolCall: []api.ToolCall{t1, t2},
			expectedTokens:   "",
			wantErr:          false,
		},
		{
			name:  "command-r-plus with json block",
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
			wantErr:          false,
		},
		{
			name:             "firefunction with functools",
			model:            "firefunction",
			output:           ` functools[{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]`,
			expectedToolCall: []api.ToolCall{t1, t2},
			expectedTokens:   "",
			wantErr:          false,
		},
		{
			name:  "llama3 with tool call tags",
			model: "llama3-groq-tool-use",
			output: `<tool_call>
		{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}}
		</tool_call>`,
			expectedToolCall: []api.ToolCall{t1},
			expectedTokens:   "",
			wantErr:          false,
		},
		{
			name:             "xlam with tool_calls wrapper",
			model:            "xlam",
			output:           `{"tool_calls": [{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]}`,
			expectedToolCall: []api.ToolCall{t1, t2},
			expectedTokens:   "",
			wantErr:          false,
		},
		{
			name:             "qwen2.5 with single tool call",
			model:            "qwen2.5-coder",
			output:           `<tool_call>{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}}</tool_call>`,
			expectedToolCall: []api.ToolCall{t1},
			expectedTokens:   "",
			wantErr:          false,
		},
		{
			name:             "qwen with invalid tool token",
			model:            "qwen2.5-coder",
			output:           `[{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}}, {"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]`,
			expectedToolCall: []api.ToolCall{t1, t2},
			expectedTokens:   "",
			wantErr:          false,
		},
		{
			// tests the leftover logic as well
			name:             "qwen3 with single tool call and thinking",
			model:            "qwen3",
			output:           `<think>Okay, let me think what tool we should use...</think><tool_call>{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}}</tool_call>`,
			expectedToolCall: []api.ToolCall{t1},
			expectedTokens:   "<think>Okay, let me think what tool we should use...</think>",
			wantErr:          false,
		},
		{
			name:             "qwen3 with single tool call and thinking spaces",
			model:            "qwen3",
			output:           `<think>Okay, let me think what tool we should use...</think> <tool_call> {"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}} </tool_call>`,
			expectedToolCall: []api.ToolCall{t1},
			expectedTokens:   "<think>Okay, let me think what tool we should use...</think>",
			wantErr:          false,
		},
		// {
		// 	name:             "qwen3 testing",
		// 	model:            "qwen3",
		// 	output:           `<think></think>{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}} </tool_call>`,
		// 	expectedToolCall: []api.ToolCall{},
		// 	expectedTokens:   `<think></think>{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}} </tool_call>`,
		// 	wantErr:          true,
		// },
		// {
		// 	name:             "qwen3 testing 2",
		// 	model:            "qwen3",
		// 	output:           `<think></think><tool_call>{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}}  </tool_call>`,
		// 	expectedToolCall: []api.ToolCall{t1},
		// 	expectedTokens:   `<think></think>`,
		// 	wantErr:          true,
		// },
		{
			name:             "qwen with no tool calls",
			model:            "qwen2.5-coder",
			output:           "The weather in San Francisco, CA is 70°F and in Toronto, Canada is 20°C.",
			expectedToolCall: []api.ToolCall{},
			expectedTokens:   "The weather in San Francisco, CA is 70°F and in Toronto, Canada is 20°C.",
			wantErr:          true,
		},
		{
			name:             "llama3.2 with tool call - no prefix",
			model:            "llama3.2",
			output:           `{"name": "get_current_weather", "parameters": {"format":"fahrenheit","location":"San Francisco, CA"}}`,
			expectedToolCall: []api.ToolCall{t1},
			expectedTokens:   "",
			wantErr:          false,
		},
		{
			name:             "llama3.2 with incomplete tool call - no prefix",
			model:            "llama3.2",
			output:           `{"name": "get_current_weather", "parameters": {"format":"fahrenheit","location":"San Francisco, `,
			expectedToolCall: []api.ToolCall{},
			expectedTokens:   "",
			wantErr:          true,
		},
		{
			name:             "llama3.2 with tool call - in middle",
			model:            "llama3.2",
			output:           `some non json text{"name": "get_current_weather", "parameters": {"format":"fahrenheit","location":"San Francisco, CA"}}`,
			expectedToolCall: []api.ToolCall{},
			expectedTokens:   `some non json text{"name": "get_current_weather", "parameters": {"format":"fahrenheit","location":"San Francisco, CA"}}`,
			wantErr:          true,
		},
		{
			name:             "llama3.2 - fake tool prefix",
			model:            "llama3.2",
			output:           `<tool_call>{"name": "get_current_weather", "parameters": {"format":"fahrenheit","location":"San Francisco, CA"}}`,
			expectedToolCall: []api.ToolCall{},
			expectedTokens:   `<tool_call>{"name": "get_current_weather", "parameters": {"format":"fahrenheit","location":"San Francisco, CA"}}`,
			wantErr:          true,
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
				m := &Model{Template: tmpl}
				tp := NewToolParser(m)
				got := []api.ToolCall{}
				success := false
				var actualTokens strings.Builder

				tokens := strings.Fields(tt.output)
				for _, tok := range tokens {
					add := true
					s := " " + tok

					// TODO(parthsareen): This logic is brittle as it mocks the logic in route, however can
					if tp.state != Done {
						toolCalls, leftover, ok := tp.ParseToolCalls(s)
						if (tp.state == GreedyToolWithPrefix || tp.state == GreedyToolNoPrefix || tp.state == ToolSuffix) || (tp.state == ForceTools && len(toolCalls) == 0) {
							continue
						}
						if tp.state == ContainsPartialPrefix {
							// actualTokens.Reset()
							actualTokens.WriteString(leftover)
							t.Log("leftover", leftover)
							add = false
							// continue
						}
						if ok && len(toolCalls) > 0 {
							success = true
							got = append(got, toolCalls...)
							add = false
							// actualTokens.Reset()
						}
					}
					// s = strings.TrimSpace(s)
					if add {
						actualTokens.WriteString(s)
					}
				}

				if !tt.wantErr {
					if diff := cmp.Diff(got, tt.expectedToolCall); diff != "" {
						t.Errorf("mismatch (-got +want):\n%s", diff)
					}
				}
				if !success && !tt.wantErr {
					t.Errorf("expected success but got errors")
				}
				stripped := strings.TrimSpace(actualTokens.String())
				if diff := cmp.Diff(stripped, tt.expectedTokens); diff != "" {
					t.Log("actualTokens", stripped, "expectedTokens", tt.expectedTokens)
					t.Errorf("tokens mismatch (-got +want):\n%s", diff)
				}
			})
		})
	}
}
