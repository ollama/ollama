package server

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/api"
)

func TestParsePythonFunctionCall(t *testing.T) {
	t1 := api.ToolCall{
		Function: api.ToolCallFunction{
			Name: "get_current_weather",
			Arguments: api.ToolCallFunctionArguments{
				"location": "San Francisco, CA",
				"format":   "fahrenheit",
			},
		},
	}

	t2 := api.ToolCall{
		Function: api.ToolCallFunction{
			Name: "get_forecast",
			Arguments: api.ToolCallFunctionArguments{
				"days":     5,
				"location": "Seattle",
			},
		},
	}

	t3 := api.ToolCall{
		Function: api.ToolCallFunction{
			Name: "get_current_weather",
			Arguments: api.ToolCallFunctionArguments{
				"list":   []any{1, 2, 3},
				"int":    -1,
				"float":  1.23,
				"string": "hello",
			},
		},
	}
	t4 := api.ToolCall{
		Function: api.ToolCallFunction{
			Name: "get_current_weather",
		},
	}

	cases := []struct {
		name  string
		input string
		want  []api.ToolCall
		err   bool
	}{
		{
			name:  "malformed function call - missing closing paren",
			input: "get_current_weather(location=\"San Francisco\"",
			err:   true,
		},
		{
			name:  "empty function call",
			input: "get_current_weather()",
			want:  []api.ToolCall{t4},
			err:   false,
		},
		{
			name:  "single valid function call",
			input: "get_current_weather(location=\"San Francisco, CA\", format=\"fahrenheit\")",
			want:  []api.ToolCall{t1},
		},
		{
			name:  "multiple valid function calls",
			input: "get_current_weather(location=\"San Francisco, CA\", format=\"fahrenheit\") get_forecast(days=5, location=\"Seattle\")",
			want:  []api.ToolCall{t1, t2},
		},
		{
			name:  "multiple valid function calls with list",
			input: "get_current_weather(list=[1,2,3], int=-1, float=1.23, string=\"hello\")",
			want:  []api.ToolCall{t3},
		},
		{
			name:  "positional arguments not supported",
			input: "get_current_weather(1, 2, 3)",
			err:   true,
		},
		{
			name:  "invalid argument format without equals",
			input: "get_current_weather(\"San Francisco\")",
			err:   true,
		},
		{
			name:  "nested lists",
			input: "get_current_weather(data=[[1,2],[3,4]])",
			want: []api.ToolCall{{
				Function: api.ToolCallFunction{
					Name: "get_current_weather",
					Arguments: api.ToolCallFunctionArguments{
						"data": []any{[]any{1, 2}, []any{3, 4}},
					},
				},
			}},
		},
		{
			name:  "boolean and none values",
			input: "get_current_weather(active=true, enabled=false, value=None)",
			want: []api.ToolCall{{
				Function: api.ToolCallFunction{
					Name: "get_current_weather",
					Arguments: api.ToolCallFunctionArguments{
						"active":  true,
						"enabled": false,
						"value":   nil,
					},
				},
			}},
		},
		{
			name:  "single vs double quotes",
			input: "get_current_weather(str1='single', str2=\"double\")",
			want: []api.ToolCall{{
				Function: api.ToolCallFunction{
					Name: "get_current_weather",
					Arguments: api.ToolCallFunctionArguments{
						"str1": "single",
						"str2": "double",
					},
				},
			}},
		},
		{
			name:  "whitespace handling",
			input: "get_current_weather( location = \"San Francisco\" , temp = 72 )",
			want: []api.ToolCall{{
				Function: api.ToolCallFunction{
					Name: "get_current_weather",
					Arguments: api.ToolCallFunctionArguments{
						"location": "San Francisco",
						"temp":     72,
					},
				},
			}},
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parsePythonToolCall(tt.input)
			if (err != nil) != tt.err {
				t.Fatalf("expected error: %v, got error: %v", tt.err, err)
			}
			if tt.err {
				return
			}
			if diff := cmp.Diff(got, tt.want); diff != "" {
				t.Errorf("mismatch (-got +want):\n%s", diff)
			}
		})
	}
}

func TestParsePythonValue(t *testing.T) {
	cases := []struct {
		name  string
		input string
		want  any
		err   bool
	}{
		{
			name:  "string with double quotes",
			input: "\"hello\"",
			want:  "hello",
		},
		{
			name:  "string with single quotes",
			input: "'world'",
			want:  "world",
		},
		{
			name:  "integer",
			input: "42",
			want:  42,
		},
		{
			name:  "float",
			input: "3.14",
			want:  3.14,
		},
		{
			name:  "boolean true",
			input: "True",
			want:  true,
		},
		{
			name:  "boolean false",
			input: "False",
			want:  false,
		},
		{
			name:  "none/null",
			input: "None",
			want:  nil,
		},
		{
			name:  "simple list",
			input: "[1, 2, 3]",
			want:  []any{1, 2, 3},
		},
		{
			name:  "nested list",
			input: "[1, [2, 3], 4]",
			want:  []any{1, []any{2, 3}, 4},
		},
		{
			name:  "mixed type list",
			input: "[1, \"two\", 3.0, true]",
			want:  []any{1, "two", 3.0, true},
		},
		{
			name:  "invalid list",
			input: "[1, 2,",
			want:  nil,
			err:   true,
		},
		{
			name:  "dictionaries",
			input: "{'a': 1, 'b': 2}",
			want:  map[any]any{"a": 1, "b": 2},
			err:   false,
		},
		{
			name:  "int dictionary",
			input: "{1: 2}",
			want:  map[any]any{1: 2},
			err:   false,
		},
		{
			name:  "mixed type dictionary",
			input: "{'a': 1, 'b': 2.0, 'c': True}",
			want:  map[any]any{"a": 1, "b": 2.0, "c": true},
			err:   false,
		},
		{
			name:  "invalid dictionary - missing closing brace",
			input: "{'a': 1, 'b': 2",
			want:  nil,
			err:   true,
		},
		{
			name:  "sets",
			input: "{1, 2, 3}",
			want:  []any{1, 2, 3},
			err:   false,
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parsePythonValue(tt.input)
			if (err != nil) != tt.err {
				t.Fatalf("expected error: %v, got error: %v", tt.err, err)
			}
			if tt.err {
				return
			}
			if diff := cmp.Diff(got, tt.want); diff != "" {
				t.Errorf("mismatch (-got +want):\n%s", diff)
			}
		})
	}
}
