package parsers

import (
	"reflect"
	"testing"

	"github.com/ollama/ollama/api"
)

// tool creates a test tool with the given name and properties
func tool(name string, props map[string]api.ToolProperty) api.Tool {
	t := api.Tool{Type: "function", Function: api.ToolFunction{Name: name}}
	t.Function.Parameters.Type = "object"
	t.Function.Parameters.Properties = testPropsMap(props)
	return t
}

func TestQwenParserStreaming(t *testing.T) {
	type step struct {
		input      string
		wantEvents []qwenEvent
	}

	cases := []struct {
		desc  string
		steps []step
		only  bool
	}{
		{
			desc: "simple message streamed word by word",
			steps: []step{
				{
					input:      "hi",
					wantEvents: []qwenEvent{qwenEventContent{content: "hi"}},
				},
				{
					input:      " there",
					wantEvents: []qwenEvent{qwenEventContent{content: " there"}},
				},
			},
		},
		{
			desc: "content before tool call",
			steps: []step{
				{
					input:      "hi there<tool_call>",
					wantEvents: []qwenEvent{qwenEventContent{content: "hi there"}},
				},
			},
		},
		{
			desc: "multiple tool calls in one message",
			steps: []step{
				{
					input: "before1<tool_call>in tool call</tool_call>after1<tool_call>in tool call 2</tool_call>after2",
					wantEvents: []qwenEvent{
						qwenEventContent{content: "before1"},
						qwenEventRawToolCall{raw: "in tool call"},
						qwenEventContent{content: "after1"},
						qwenEventRawToolCall{raw: "in tool call 2"},
						qwenEventContent{content: "after2"},
					},
				},
			},
		},
		{
			desc: "tool calls with split tags",
			steps: []step{
				{
					input: "before<tool",
					wantEvents: []qwenEvent{
						qwenEventContent{content: "before"},
					},
				},
				{
					input:      "_call>in tool call</tool",
					wantEvents: []qwenEvent{},
				},
				{
					input: "_call>af",
					wantEvents: []qwenEvent{
						qwenEventRawToolCall{raw: "in tool call"},
						qwenEventContent{content: "af"},
					},
				},
				{
					input: "ter",
					wantEvents: []qwenEvent{
						qwenEventContent{content: "ter"},
					},
				},
			},
		},
		{
			desc: "tool call tags split character by character",
			steps: []step{
				{input: "<", wantEvents: []qwenEvent{}},
				{input: "t", wantEvents: []qwenEvent{}},
				{input: "o", wantEvents: []qwenEvent{}},
				{input: "o", wantEvents: []qwenEvent{}},
				{input: "l", wantEvents: []qwenEvent{}},
				{input: "_", wantEvents: []qwenEvent{}},
				{input: "c", wantEvents: []qwenEvent{}},
				{input: "a", wantEvents: []qwenEvent{}},
				{input: "l", wantEvents: []qwenEvent{}},
				{input: "l", wantEvents: []qwenEvent{}},
				{input: ">", wantEvents: []qwenEvent{}},
				{input: "a", wantEvents: []qwenEvent{}},
				{input: "b", wantEvents: []qwenEvent{}},
				{input: "c", wantEvents: []qwenEvent{}},
				{input: "<", wantEvents: []qwenEvent{}},
				{input: "/", wantEvents: []qwenEvent{}},
				{input: "t", wantEvents: []qwenEvent{}},
				{input: "o", wantEvents: []qwenEvent{}},
				{input: "o", wantEvents: []qwenEvent{}},
				{input: "l", wantEvents: []qwenEvent{}},
				{input: "_", wantEvents: []qwenEvent{}},
				{input: "c", wantEvents: []qwenEvent{}},
				{input: "a", wantEvents: []qwenEvent{}},
				{input: "l", wantEvents: []qwenEvent{}},
				{input: "l", wantEvents: []qwenEvent{}},
				{input: ">", wantEvents: []qwenEvent{qwenEventRawToolCall{raw: "abc"}}},
			},
		},
		{
			desc: "trailing whitespace between content and tool call",
			steps: []step{
				{
					input: "abc\n<tool_call>def</tool_call>",
					wantEvents: []qwenEvent{
						qwenEventContent{content: "abc"},
						qwenEventRawToolCall{raw: "def"},
					},
				},
			},
		},
		{
			desc: "unambiguous empty: partial tool open at buffer start",
			steps: []step{
				{
					input:      "<tool_ca",
					wantEvents: []qwenEvent{},
				},
				{
					input: "ll>abc</tool_call>",
					wantEvents: []qwenEvent{
						qwenEventRawToolCall{raw: "abc"},
					},
				},
			},
		},
		{
			desc: "trailing whitespace between tool call and content",
			steps: []step{
				{
					input: "<tool_call>abc</tool_call>\ndef",
					wantEvents: []qwenEvent{
						qwenEventRawToolCall{raw: "abc"},
						qwenEventContent{content: "def"},
					},
				},
			},
		},
		{
			desc: "empty content before tool call",
			steps: []step{
				{
					input: "\n<tool_call>abc</tool_call>",
					wantEvents: []qwenEvent{
						qwenEventRawToolCall{raw: "abc"},
					},
				},
			},
		},
		{
			desc: "partial tool open tag fakeout",
			steps: []step{
				{
					input: "abc\n<tool_call",
					wantEvents: []qwenEvent{
						// \n should not be emitted yet because `<tool_call` might be a tool
						// open tag, in which case the whitespace should be trimmed
						qwenEventContent{content: "abc"},
					},
				},
				{
					input: " fakeout",
					wantEvents: []qwenEvent{
						qwenEventContent{content: "\n<tool_call fakeout"},
					},
				},
			},
		},
		{
			desc: "token-by-token whitespace handling",
			steps: []step{
				{
					input: "a",
					wantEvents: []qwenEvent{
						qwenEventContent{content: "a"},
					},
				},
				{
					input:      "\n",
					wantEvents: []qwenEvent{},
				},
				{
					input: "b",
					wantEvents: []qwenEvent{
						qwenEventContent{content: "\nb"},
					},
				},
			},
		},
		{
			desc: "unicode content",
			steps: []step{
				{
					input: "你好 🌍<tool_call>test</tool_call>مرحبا",
					wantEvents: []qwenEvent{
						qwenEventContent{content: "你好 🌍"},
						qwenEventRawToolCall{raw: "test"},
						qwenEventContent{content: "مرحبا"},
					},
				},
			},
		},
		{
			desc: "arabic text handling",
			steps: []step{
				{
					input:      "مرحبا بالعالم",
					wantEvents: []qwenEvent{qwenEventContent{content: "مرحبا بالعالم"}},
				},
			},
		},
		{
			desc: "emoji passthrough",
			steps: []step{
				{
					input:      "✅",
					wantEvents: []qwenEvent{qwenEventContent{content: "✅"}},
				},
			},
		},
		{
			desc: "emoji after tool call",
			steps: []step{
				{
					input: "<tool_call>test</tool_call>完成 ✅",
					wantEvents: []qwenEvent{
						qwenEventRawToolCall{raw: "test"},
						qwenEventContent{content: "完成 ✅"},
					},
				},
			},
		},
		{
			desc: "unicode streaming with whitespace handling",
			steps: []step{
				{
					input: "مرحبا",
					wantEvents: []qwenEvent{
						qwenEventContent{content: "مرحبا"},
					},
				},
				{
					input:      " \n",
					wantEvents: []qwenEvent{},
				},
				{
					input: "世界",
					wantEvents: []qwenEvent{
						qwenEventContent{content: " \n世界"},
					},
				},
			},
		},
		{
			desc: "non-breaking space withheld across chunks",
			steps: []step{
				{
					input: "Hello\u00a0",
					wantEvents: []qwenEvent{
						qwenEventContent{content: "Hello"},
					},
				},
				{
					input: "world",
					wantEvents: []qwenEvent{
						qwenEventContent{content: "\u00a0world"},
					},
				},
			},
		},
		{
			desc: "ideographic space before partial tool",
			steps: []step{
				{
					input: "Hello\u3000<tool",
					wantEvents: []qwenEvent{
						qwenEventContent{content: "Hello"},
					},
				},
				{
					input:      "_call>abc",
					wantEvents: []qwenEvent{},
				},
				{
					input: "</tool_call>def",
					wantEvents: []qwenEvent{
						qwenEventRawToolCall{raw: "abc"},
						qwenEventContent{content: "def"},
					},
				},
			},
		},
		{
			desc: "ideographic space before partial tool fakeout",
			steps: []step{
				{
					input: "Hello\u3000<tool",
					wantEvents: []qwenEvent{
						qwenEventContent{content: "Hello"},
					},
				},
				{
					input: "fakeout>abc",
					wantEvents: []qwenEvent{
						qwenEventContent{content: "\u3000<toolfakeout>abc"},
					},
				},
			},
		},
		{
			desc: "unicode with partial tool tag",
			steps: []step{
				{
					input: "测试🎯 <to",
					wantEvents: []qwenEvent{
						qwenEventContent{content: "测试🎯"},
					},
				},
			},
		},
	}

	anyOnlies := false
	for _, tc := range cases {
		if tc.only {
			anyOnlies = true
		}
	}

	for _, tc := range cases {
		if anyOnlies && !tc.only {
			continue
		}

		t.Run(tc.desc, func(t *testing.T) {
			parser := Qwen3CoderParser{}

			for i, step := range tc.steps {
				parser.acc.WriteString(step.input)
				gotEvents := parser.parseEvents()

				if len(gotEvents) == 0 && len(step.wantEvents) == 0 {
					// avoid deep equal on empty vs. nil slices
					continue
				}

				if !reflect.DeepEqual(gotEvents, step.wantEvents) {
					t.Errorf("step %d: input %q: got events %#v, want %#v", i, step.input, gotEvents, step.wantEvents)
				}
			}
		})
	}
}

func TestQwenToolParser(t *testing.T) {
	type step struct {
		name         string
		rawToolCall  string
		tools        []api.Tool
		wantToolCall api.ToolCall
	}

	steps := []step{
		{
			name:  "simple tool call",
			tools: []api.Tool{},
			rawToolCall: `<function=get_current_temperature>
<parameter=location>
San Francisco
</parameter>
<parameter=unit>
celsius
</parameter>
</function>`,
			wantToolCall: api.ToolCall{
				Function: api.ToolCallFunction{
					Name: "get_current_temperature",
					Arguments: testArgs(map[string]any{
						"location": "San Francisco",
						"unit":     "celsius",
					}),
				},
			},
		},
		{
			name:  "names with spaces",
			tools: []api.Tool{},
			rawToolCall: `<function=get current temperature>
<parameter=location with spaces>
San Francisco
</parameter>
<parameter=unit with spaces>
celsius
</parameter>
</function>`,
			wantToolCall: api.ToolCall{
				Function: api.ToolCallFunction{
					Name: "get current temperature",
					Arguments: testArgs(map[string]any{
						"location with spaces": "San Francisco",
						"unit with spaces":     "celsius",
					}),
				},
			},
		},
		// this mirrors the reference implementation's behavior, but unclear if it
		// ever happens. If so, then we should probably remove them instead, this
		// test is to just document the current behavior and test that we don't get
		// xml errors
		{
			name:  "names with quotes",
			tools: []api.Tool{},
			rawToolCall: `<function="get current temperature">
<parameter="location with spaces">
San Francisco
</parameter>
<parameter="unit with spaces">
"celsius"
</parameter>
</function>`,
			wantToolCall: api.ToolCall{
				Function: api.ToolCallFunction{
					Name: "\"get current temperature\"",
					Arguments: testArgs(map[string]any{
						"\"location with spaces\"": "San Francisco",
						"\"unit with spaces\"":     "\"celsius\"",
					}),
				},
			},
		},
		{
			name: "tool call with typed parameters",
			tools: []api.Tool{
				tool("calculate", map[string]api.ToolProperty{
					"x":       {Type: api.PropertyType{"number"}},
					"y":       {Type: api.PropertyType{"integer"}},
					"enabled": {Type: api.PropertyType{"boolean"}},
					"items":   {Type: api.PropertyType{"array"}},
				}),
			},
			rawToolCall: `<function=calculate>
<parameter=x>
3.14
</parameter>
<parameter=y>
42
</parameter>
<parameter=enabled>
true
</parameter>
<parameter=items>
["a", "b", "c"]
</parameter>
</function>`,
			wantToolCall: api.ToolCall{
				Function: api.ToolCallFunction{
					Name: "calculate",
					Arguments: testArgs(map[string]any{
						"x":       3.14,
						"y":       42,
						"enabled": true,
						"items":   []any{"a", "b", "c"},
					}),
				},
			},
		},
		// regression test for <https://github.com/ollama/ollama/issues/12357>
		{
			name:  "ampersands in parameter values",
			tools: []api.Tool{},
			rawToolCall: `<function=exec>
<parameter=command>
ls && echo "done"
</parameter>
</function>`,
			wantToolCall: api.ToolCall{
				Function: api.ToolCallFunction{
					Name: "exec",
					Arguments: testArgs(map[string]any{
						"command": "ls && echo \"done\"",
					}),
				},
			},
		},
		{
			name:  "angle brackets in parameter values",
			tools: []api.Tool{},
			rawToolCall: `<function=exec>
<parameter=command>
ls && echo "a > b and a < b"
</parameter>
</function>`,
			wantToolCall: api.ToolCall{
				Function: api.ToolCallFunction{
					Name: "exec",
					Arguments: testArgs(map[string]any{
						"command": "ls && echo \"a > b and a < b\"",
					}),
				},
			},
		},
		{
			name:  "unicode in function names and parameters",
			tools: []api.Tool{},
			rawToolCall: `<function=获取天气>
<parameter=城市>
北京
</parameter>
<parameter=message>
Hello! 你好! 🌟 مرحبا
</parameter>
</function>`,
			wantToolCall: api.ToolCall{
				Function: api.ToolCallFunction{
					Name: "获取天气",
					Arguments: testArgs(map[string]any{
						"城市":      "北京",
						"message": "Hello! 你好! 🌟 مرحبا",
					}),
				},
			},
		},
	}

	for i, step := range steps {
		gotToolCall, err := parseToolCall(qwenEventRawToolCall{raw: step.rawToolCall}, step.tools)
		if err != nil {
			t.Errorf("step %d (%s): %v", i, step.name, err)
		}
		if !toolCallEqual(gotToolCall, step.wantToolCall) {
			t.Errorf("step %d (%s): got tool call %#v, want %#v", i, step.name, gotToolCall, step.wantToolCall)
		}
	}
}

func TestTrailingWhitespaceLenUnicode(t *testing.T) {
	cases := []struct {
		name  string
		input string
		want  int
	}{
		{
			name:  "ascii space",
			input: "Hello ",
			want:  1,
		},
		{
			name:  "non-breaking space",
			input: "Hello\u00a0",
			want:  2,
		},
		{
			name:  "ideographic space",
			input: "Hello\u3000",
			want:  3,
		},
		{
			name:  "multiple runes of whitespace",
			input: "Hi\u00a0\u3000",
			want:  5,
		},
	}

	for _, tc := range cases {
		got := trailingWhitespaceLen(tc.input)
		if got != tc.want {
			t.Errorf("%s: trailingWhitespaceLen(%q) = %d, want %d", tc.name, tc.input, got, tc.want)
		}
	}
}

func TestQwenToolCallValueParsing(t *testing.T) {
	cases := []struct {
		desc      string
		raw       string
		paramType api.PropertyType
		want      any
	}{
		{
			desc:      "default string value (no type specified)",
			paramType: api.PropertyType{},
			raw:       "some-string",
			want:      "some-string",
		},
		{
			desc:      "trim a single leading and trailing newline",
			paramType: api.PropertyType{},
			raw:       "\nsome-string\n",
			want:      "some-string",
		},
		{
			desc:      "trim at most one leading and trailing newline",
			paramType: api.PropertyType{},
			raw:       "\n\nsome-string\n\n",
			want:      "\nsome-string\n",
		},
		{
			desc:      "newline really has to be the first character to be trimmed",
			paramType: api.PropertyType{},
			raw:       " \nsome-string\n ",
			want:      " \nsome-string\n ",
		},
		{
			desc:      "numeric type",
			paramType: api.PropertyType{"number"},
			raw:       "123",
			want:      123,
		},
		// Integer parsing tests
		{
			desc:      "integer type",
			paramType: api.PropertyType{"integer"},
			raw:       "42",
			want:      42,
		},
		{
			desc:      "negative integer",
			paramType: api.PropertyType{"integer"},
			raw:       "-100",
			want:      -100,
		},
		{
			desc:      "zero integer",
			paramType: api.PropertyType{"integer"},
			raw:       "0",
			want:      0,
		},
		{
			desc:      "integer with leading zeros",
			paramType: api.PropertyType{"integer"},
			raw:       "007",
			want:      7,
		},
		{
			desc:      "large integer",
			paramType: api.PropertyType{"integer"},
			raw:       "2147483648", // Just beyond int32 max
			want:      int64(2147483648),
		},
		// Float/number parsing tests
		{
			desc:      "float type",
			paramType: api.PropertyType{"number"},
			raw:       "3.14",
			want:      3.14,
		},
		{
			desc:      "negative float",
			paramType: api.PropertyType{"number"},
			raw:       "-273.15",
			want:      -273.15,
		},
		{
			desc:      "float without decimal part",
			paramType: api.PropertyType{"number"},
			raw:       "100.0",
			want:      100,
		},
		{
			desc:      "scientific notation positive",
			paramType: api.PropertyType{"number"},
			raw:       "1.23e5",
			want:      123000, // Will be int since it has no decimal part
		},
		{
			desc:      "scientific notation negative",
			paramType: api.PropertyType{"number"},
			raw:       "1.5e-3",
			want:      0.0015,
		},
		{
			desc:      "very small float",
			paramType: api.PropertyType{"number"},
			raw:       "0.00000001",
			want:      0.00000001,
		},
		// String parsing tests
		{
			desc:      "explicit string type",
			paramType: api.PropertyType{"string"},
			raw:       "hello world",
			want:      "hello world",
		},
		{
			desc:      "string with special characters",
			paramType: api.PropertyType{"string"},
			raw:       "/usr/local/bin/test-file_v2.0.sh",
			want:      "/usr/local/bin/test-file_v2.0.sh",
		},
		{
			desc:      "string with quotes",
			paramType: api.PropertyType{"string"},
			raw:       `He said "hello" to me`,
			want:      `He said "hello" to me`,
		},
		{
			desc:      "multiline string",
			paramType: api.PropertyType{"string"},
			raw:       "line one\nline two\nline three",
			want:      "line one\nline two\nline three",
		},
		{
			desc:      "empty string",
			paramType: api.PropertyType{"string"},
			raw:       "",
			want:      "",
		},
		{
			desc:      "string that looks like a number",
			paramType: api.PropertyType{"string"},
			raw:       "12345",
			want:      "12345",
		},
		// Boolean parsing tests
		{
			desc:      "boolean true",
			paramType: api.PropertyType{"boolean"},
			raw:       "true",
			want:      true,
		},
		{
			desc:      "boolean false",
			paramType: api.PropertyType{"boolean"},
			raw:       "false",
			want:      false,
		},
		{
			desc:      "boolean case insensitive true",
			paramType: api.PropertyType{"boolean"},
			raw:       "True",
			want:      true,
		},
		{
			desc:      "boolean case insensitive false",
			paramType: api.PropertyType{"boolean"},
			raw:       "FALSE",
			want:      false,
		},
		// Null parsing tests
		{
			desc:      "null value lowercase",
			paramType: api.PropertyType{"string"},
			raw:       "null",
			want:      nil,
		},
		{
			desc:      "null value case insensitive",
			paramType: api.PropertyType{"integer"},
			raw:       "NULL",
			want:      nil,
		},
		// Array parsing tests
		{
			desc:      "array of strings",
			paramType: api.PropertyType{"array"},
			raw:       `["foo", "bar", "baz"]`,
			want:      []any{"foo", "bar", "baz"},
		},
		{
			desc:      "array of numbers",
			paramType: api.PropertyType{"array"},
			raw:       `[1, 2.5, 3]`,
			want:      []any{float64(1), 2.5, float64(3)},
		},
		{
			desc:      "array of mixed types",
			paramType: api.PropertyType{"array"},
			raw:       `["string", 123, true, null]`,
			want:      []any{"string", float64(123), true, nil},
		},
		{
			desc:      "empty array",
			paramType: api.PropertyType{"array"},
			raw:       `[]`,
			want:      []any{},
		},
		// Object parsing tests
		{
			desc:      "simple object",
			paramType: api.PropertyType{"object"},
			raw:       `{"key": "value", "number": 42}`,
			want:      map[string]any{"key": "value", "number": float64(42)},
		},
		{
			desc:      "nested object",
			paramType: api.PropertyType{"object"},
			raw:       `{"outer": {"inner": "value"}}`,
			want:      map[string]any{"outer": map[string]any{"inner": "value"}},
		},
		{
			desc:      "empty object",
			paramType: api.PropertyType{"object"},
			raw:       `{}`,
			want:      map[string]any{},
		},
		// Error cases and fallback behavior
		{
			desc:      "invalid integer falls back to string",
			paramType: api.PropertyType{"integer"},
			raw:       "not-a-number",
			want:      "not-a-number",
		},
		{
			desc:      "invalid float falls back to string",
			paramType: api.PropertyType{"number"},
			raw:       "3.14.159",
			want:      "3.14.159",
		},
		{
			desc:      "invalid boolean falls back to false",
			paramType: api.PropertyType{"boolean"},
			raw:       "yes",
			want:      false,
		},
		{
			desc:      "invalid JSON array falls back to string",
			paramType: api.PropertyType{"array"},
			raw:       "[1, 2, unclosed",
			want:      "[1, 2, unclosed",
		},
		{
			desc:      "invalid JSON object falls back to string",
			paramType: api.PropertyType{"object"},
			raw:       `{"key": unclosed`,
			want:      `{"key": unclosed`,
		},
		// Edge cases
		{
			desc:      "integer overflow should use int64",
			paramType: api.PropertyType{"integer"},
			raw:       "2147483648", // Beyond int32 max
			want:      int64(2147483648),
		},
		{
			desc:      "float with many decimal places",
			paramType: api.PropertyType{"number"},
			raw:       "3.141592653589793",
			want:      3.141592653589793,
		},
		{
			desc:      "string with JSON-like content",
			paramType: api.PropertyType{"string"},
			raw:       `{"this": "is", "just": "a string"}`,
			want:      `{"this": "is", "just": "a string"}`,
		},
		{
			desc:      "whitespace-only string",
			paramType: api.PropertyType{"string"},
			raw:       "   ",
			want:      "   ",
		},
		// Unknown parameter (no type specified in tools)
		{
			desc:      "parameter not in tool definition defaults to string",
			paramType: api.PropertyType{},
			raw:       "some value",
			want:      "some value",
		},
		// Union type tests
		{
			desc:      "string or number union - valid number",
			paramType: api.PropertyType{"string", "number"},
			raw:       "42.5",
			want:      42.5,
		},
		{
			desc:      "string or number union - non-numeric string",
			paramType: api.PropertyType{"string", "number"},
			raw:       "hello",
			want:      "hello",
		},
		{
			desc:      "number or string union - valid number (order shouldn't matter)",
			paramType: api.PropertyType{"number", "string"},
			raw:       "42.5",
			want:      42.5,
		},
		{
			desc:      "integer or null union - valid integer",
			paramType: api.PropertyType{"integer", "null"},
			raw:       "123",
			want:      123,
		},
		{
			desc:      "integer or null union - null value",
			paramType: api.PropertyType{"integer", "null"},
			raw:       "null",
			want:      nil,
		},
		{
			desc:      "null or integer union - null value (order shouldn't matter)",
			paramType: api.PropertyType{"null", "integer"},
			raw:       "null",
			want:      nil,
		},
		{
			desc:      "boolean or string union - valid boolean",
			paramType: api.PropertyType{"boolean", "string"},
			raw:       "true",
			want:      true,
		},
		{
			desc:      "boolean or string union - non-boolean becomes string",
			paramType: api.PropertyType{"boolean", "string"},
			raw:       "yes",
			want:      "yes",
		},
		{
			desc:      "string or boolean union - valid boolean (precedence test)",
			paramType: api.PropertyType{"string", "boolean"},
			raw:       "false",
			want:      false, // Should be boolean, not string "false"
		},
		{
			desc:      "integer or number union - integer value",
			paramType: api.PropertyType{"integer", "number"},
			raw:       "42",
			want:      42,
		},
		{
			desc:      "integer or number union - float value",
			paramType: api.PropertyType{"integer", "number"},
			raw:       "42.5",
			want:      42.5,
		},
		{
			desc:      "number or integer union - integer value (precedence test)",
			paramType: api.PropertyType{"number", "integer"},
			raw:       "42",
			want:      42, // Should try integer first due to precedence
		},
		{
			desc:      "array or object union - valid array",
			paramType: api.PropertyType{"array", "object"},
			raw:       `[1, 2, 3]`,
			want:      []any{float64(1), float64(2), float64(3)},
		},
		{
			desc:      "array or object union - valid object",
			paramType: api.PropertyType{"array", "object"},
			raw:       `{"key": "value"}`,
			want:      map[string]any{"key": "value"},
		},
		{
			desc:      "object or array union - valid array (precedence test)",
			paramType: api.PropertyType{"object", "array"},
			raw:       `[1, 2, 3]`,
			want:      []any{float64(1), float64(2), float64(3)},
		},
		{
			desc:      "complex multi-type union - null",
			paramType: api.PropertyType{"string", "number", "boolean", "null"},
			raw:       "null",
			want:      nil,
		},
		{
			desc:      "complex multi-type union - boolean",
			paramType: api.PropertyType{"string", "number", "boolean", "null"},
			raw:       "true",
			want:      true,
		},
		{
			desc:      "complex multi-type union - number",
			paramType: api.PropertyType{"string", "number", "boolean", "null"},
			raw:       "3.14",
			want:      3.14,
		},
		{
			desc:      "complex multi-type union - string",
			paramType: api.PropertyType{"string", "number", "boolean", "null"},
			raw:       "hello",
			want:      "hello",
		},
		{
			desc:      "integer string union - integer string becomes integer",
			paramType: api.PropertyType{"integer", "string"},
			raw:       "123",
			want:      123,
		},
		{
			desc:      "string integer union - integer string becomes integer (precedence)",
			paramType: api.PropertyType{"string", "integer"},
			raw:       "123",
			want:      123, // Integer has higher precedence than string
		},
		{
			desc:      "anyOf array or string - with array of objects",
			paramType: api.PropertyType{"array", "string"},
			raw:       `[{"content": "task 1", "status": "pending", "priority": "high", "id": "1"}, {"content": "task 2", "status": "completed", "priority": "low", "id": "2"}]`,
			want: []any{
				map[string]any{"content": "task 1", "status": "pending", "priority": "high", "id": "1"},
				map[string]any{"content": "task 2", "status": "completed", "priority": "low", "id": "2"},
			},
		},
		{
			desc:      "anyOf array or string - with plain string",
			paramType: api.PropertyType{"array", "string"},
			raw:       "Error: could not load data",
			want:      "Error: could not load data",
		},
	}

	for _, tc := range cases {
		t.Run(tc.desc, func(t *testing.T) {
			got := parseValue(tc.raw, tc.paramType)
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("got %v (type %T), want %v (type %T)", got, got, tc.want, tc.want)
			}
		})
	}
}

func TestQwenXMLTransform(t *testing.T) {
	cases := []struct {
		desc string
		raw  string
		want string
	}{
		{
			desc: "simple example",
			raw: `<function=get_current_temperature>
<parameter=location>
San Francisco
</parameter>
<parameter=unit>
celsius
</parameter>
</function>`,
			want: `<function name="get_current_temperature">
<parameter name="location">
San Francisco
</parameter>
<parameter name="unit">
celsius
</parameter>
</function>`,
		},
		// even though quotes aren't expected in these tags, we have these tests to
		// make sure they're escaped so they don't blow up the xml parser in case
		// they happen
		{
			desc: "names with quotes",
			raw: `<function="get current temperature">
<parameter="location with spaces">
San Francisco
</parameter>
<parameter="unit with spaces">
celsius
</parameter>
</function>`,
			want: `<function name="&#34;get current temperature&#34;">
<parameter name="&#34;location with spaces&#34;">
San Francisco
</parameter>
<parameter name="&#34;unit with spaces&#34;">
celsius
</parameter>
</function>`,
		},
		{
			desc: "ampersands in parameter values",
			raw: `<function=get_current_temperature>
		<parameter=location>
		San Francisco & San Jose
		</parameter>
		</function>`,
			want: `<function name="get_current_temperature">
		<parameter name="location">
		San Francisco &amp; San Jose
		</parameter>
		</function>`,
		},
	}

	for _, tc := range cases {
		got := transformToXML(tc.raw)
		if got != tc.want {
			t.Errorf("got %q, want %q", got, tc.want)
		}
	}
}

func TestTrailingWhitespaceLen(t *testing.T) {
	cases := []struct {
		desc string
		s    string
		want int
	}{
		{desc: "no whitespace", s: "abc", want: 0},
		{desc: "trailing whitespace", s: "abc ", want: 1},
		{desc: "trailing whitespace with newlines", s: "abc \n", want: 2},
		{desc: "only whitespace", s: " \n  ", want: 4},
		{desc: "leading whitespace doesn't count", s: " \n abc", want: 0},
		{desc: "unicode with trailing space", s: "测试🎯 ", want: 1},
		{desc: "unicode with trailing tab and newline", s: "مرحبا\t\n", want: 2},
	}

	for _, tc := range cases {
		got := trailingWhitespaceLen(tc.s)
		if got != tc.want {
			t.Errorf("got %d, want %d", got, tc.want)
		}
	}
}

func TestOverlapFunction(t *testing.T) {
	cases := []struct {
		desc  string
		s     string
		delim string
		want  int
	}{
		{desc: "no overlap", s: "hello", delim: "<tool", want: 0},
		{desc: "full overlap", s: "hello<tool", delim: "<tool>", want: 5},
		{desc: "partial overlap", s: "hello<to", delim: "<tool>", want: 3},
		{desc: "unicode with partial overlap", s: "测试🎯<to", delim: "<tool>", want: 3},
		{desc: "unicode string with no overlap", s: "مرحبا", delim: "<tool>", want: 0},
		{desc: "unicode at boundary", s: "世界<", delim: "<tool>", want: 1},
		{desc: "unicode delimiter single rune", s: "hello🔧", delim: "🔧工具", want: len("🔧")},
		{desc: "unicode delimiter multiple runes", s: "hello🔧工", delim: "🔧工具", want: len("🔧工")},
	}

	for _, tc := range cases {
		t.Run(tc.desc, func(t *testing.T) {
			got := overlap(tc.s, tc.delim)
			if got != tc.want {
				t.Errorf("overlap(%q, %q) = %d, want %d", tc.s, tc.delim, got, tc.want)
			}
		})
	}
}
