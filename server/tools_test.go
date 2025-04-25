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
		name     string
		model    string
		output   string
		token    string
		expected []api.ToolCall
		wantErr  bool
	}{
		{
			name:     "mistral invalid json",
			model:    "mistral",
			output:   `[TOOL_CALLS]  [{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_curren}]`,
			token:    "[TOOL_CALLS]",
			expected: []api.ToolCall{},
			wantErr:  true,
		},
		{
			name:     "mistral valid json",
			model:    "mistral",
			output:   `[TOOL_CALLS]  [{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]`,
			token:    "[TOOL_CALLS]",
			expected: []api.ToolCall{t1, t2},
			wantErr:  false,
		},
		{
			name:     "mistral incomplete json",
			model:    "mistral",
			output:   `[TOOL_CALLS]  [{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, `,
			token:    "[TOOL_CALLS]",
			expected: []api.ToolCall{},
			wantErr:  true,
		},
		{
			name:  "mistral without tool token",
			model: "mistral",
			output: `I'm not aware of that information. However, I can suggest searching for the weather using the "get_current_weather" function:

		[{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]`,
			token:    "[TOOL_CALLS]",
			expected: []api.ToolCall{},
			wantErr:  true,
		},
		{
			name:     "mistral without tool token - tool first",
			model:    "mistral",
			output:   `[{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]`,
			token:    "[TOOL_CALLS]",
			expected: []api.ToolCall{t1, t2},
			wantErr:  false,
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
			token:    "Action:",
			expected: []api.ToolCall{t1, t2},
			wantErr:  false,
		},
		{
			name:     "firefunction with functools",
			model:    "firefunction",
			output:   ` functools[{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]`,
			token:    "functools",
			expected: []api.ToolCall{t1, t2},
			wantErr:  false,
		},
		{
			name:  "llama3 with tool call tags",
			model: "llama3-groq-tool-use",
			output: `<tool_call>
		{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}}
		</tool_call>`,
			token:    "<tool_call>",
			expected: []api.ToolCall{t1},
			wantErr:  false,
		},
		{
			name:     "xlam with tool_calls wrapper",
			model:    "xlam",
			output:   `{"tool_calls": [{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]}`,
			token:    "",
			expected: []api.ToolCall{t1, t2},
			wantErr:  false,
		},
		{
			name:     "qwen with single tool call",
			model:    "qwen2.5-coder",
			output:   `<tool_call>{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}}</tool_call>`,
			token:    "<tool_call>",
			expected: []api.ToolCall{t1},
			wantErr:  false,
		},
		{
			name:     "qwen with invalid tool token",
			model:    "qwen2.5-coder",
			output:   `[{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}}, {"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]`,
			token:    "[TOOL_CALLS]",
			expected: []api.ToolCall{t1, t2},
			wantErr:  false,
		},
		{
			name:     "qwen with no tool calls",
			model:    "qwen2.5-coder",
			output:   " The weather in San Francisco, CA is 70°F and in Toronto, Canada is 20°C.",
			token:    "",
			expected: []api.ToolCall{},
			wantErr:  true,
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
		t.Run(tt.model, func(t *testing.T) {
			tmpl, err := template.Parse(readFile(t, p, fmt.Sprintf("%s.gotmpl", tt.model)).String())
			if err != nil {
				t.Fatal(err)
			}

			t.Run("template", func(t *testing.T) {
				var actual bytes.Buffer
				if err := tmpl.Execute(&actual, template.Values{Tools: tools, Messages: messages}); err != nil {
					t.Fatal(err)
				}

				if diff := cmp.Diff(actual.String(), readFile(t, p, fmt.Sprintf("%s.out", tt.model)).String()); diff != "" {
					t.Errorf("mismatch (-got +want):\n%s", diff)
				}
			})

			t.Run("parse", func(t *testing.T) {
				m := &Model{Template: tmpl}
				tmpl, ok := ToolTemplate(m)
				if !ok {
					t.Fatal("no tool template found")
				}
				got := []api.ToolCall{}
				tokens := strings.Fields(tt.output)
				sb := strings.Builder{}
				success := false
				for _, tok := range tokens {
					sb.WriteString(" " + tok)
					toolCalls, partial, err := ParseToolCalls(sb.String(), tt.token, tmpl)
					if err == nil {
						success = true
					}
					if partial {
						continue
					}
					got = append(got, toolCalls...)
					sb.Reset()
				}

				if !tt.wantErr {
					if diff := cmp.Diff(got, tt.expected); diff != "" {
						t.Errorf("mismatch (-got +want):\n%s", diff)
					}
				}
				if !success && !tt.wantErr {
					t.Errorf("expected success but got errors")
				}
			})
		})
	}
}

func TestParseObjects(t *testing.T) {
	tests := []struct {
		input string
		want  []map[string]any
	}{
		{
			input: `[{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]`,
			want: []map[string]any{
				{"name": "get_current_weather", "arguments": map[string]any{"format": "fahrenheit", "location": "San Francisco, CA"}},
				{"name": "get_current_weather", "arguments": map[string]any{"format": "celsius", "location": "Toronto, Canada"}},
			},
		},
		{
			input: `<some_token>{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}} </toolcall>`,
			want: []map[string]any{
				{"name": "get_current_weather", "arguments": map[string]any{"format": "fahrenheit", "location": "San Francisco, CA"}},
			},
		},
		{
			input: `<some_token>{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}} </toolcall> <toolcall>{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, ON"}} </toolcall>`,
			want: []map[string]any{
				{"name": "get_current_weather", "arguments": map[string]any{"format": "fahrenheit", "location": "San Francisco, CA"}},
				{"name": "get_current_weather", "arguments": map[string]any{"format": "celsius", "location": "Toronto, ON"}},
			},
		},
		{
			input: `{"name": "get_current_weather", "arguments": `,
			want:  nil,
		},
	}

	for _, tc := range tests {
		t.Run(tc.input, func(t *testing.T) {
			got := parseObjects(tc.input)

			if diff := cmp.Diff(got, tc.want); diff != "" {
				t.Errorf("mismatch (-got +want):\n%s", diff)
			}
		})
	}
}
