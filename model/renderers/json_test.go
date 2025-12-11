package renderers

import (
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestMarshalWithSpaces(t *testing.T) {
	tests := []struct {
		name     string
		input    any
		expected string
	}{
		// basic formatting tests
		{
			name:     "simple object",
			input:    map[string]any{"key": "value"},
			expected: `{"key": "value"}`,
		},
		{
			name:     "simple array",
			input:    []any{"a", "b", "c"},
			expected: `["a", "b", "c"]`,
		},
		// escaped quotes
		{
			name:     "escaped quote in string",
			input:    map[string]any{"text": `quote"inside`},
			expected: `{"text": "quote\"inside"}`,
		},
		{
			name:     "multiple escaped quotes",
			input:    map[string]any{"text": `say "hello" and "goodbye"`},
			expected: `{"text": "say \"hello\" and \"goodbye\""}`,
		},
		// escaped backslashes
		{
			name:     "escaped backslash",
			input:    map[string]any{"path": `C:\windows\system32`},
			expected: `{"path": "C:\\windows\\system32"}`,
		},
		{
			name:     "double backslash",
			input:    map[string]any{"text": `test\\more`},
			expected: `{"text": "test\\\\more"}`,
		},
		{
			name:     "backslash before quote",
			input:    map[string]any{"text": `end with \"`},
			expected: `{"text": "end with \\\""}`,
		},
		// standard JSON escape sequences
		{
			name:     "newline in string",
			input:    map[string]any{"text": "line1\nline2"},
			expected: `{"text": "line1\nline2"}`,
		},
		{
			name:     "tab in string",
			input:    map[string]any{"text": "before\tafter"},
			expected: `{"text": "before\tafter"}`,
		},
		{
			name:     "carriage return",
			input:    map[string]any{"text": "before\rafter"},
			expected: `{"text": "before\rafter"}`,
		},
		{
			name:     "multiple escape sequences",
			input:    map[string]any{"text": "line1\nline2\ttab\rcarriage"},
			expected: `{"text": "line1\nline2\ttab\rcarriage"}`,
		},
		// strings containing colons and commas (no spaces should be added inside)
		{
			name:     "colon in string",
			input:    map[string]any{"url": "http://example.com"},
			expected: `{"url": "http://example.com"}`,
		},
		{
			name:     "comma in string",
			input:    map[string]any{"list": "apple, banana, cherry"},
			expected: `{"list": "apple, banana, cherry"}`,
		},
		{
			name:     "colon and comma in string",
			input:    map[string]any{"data": "key:value, key2:value2"},
			expected: `{"data": "key:value, key2:value2"}`,
		},
		// unicode characters
		{
			name:     "emoji",
			input:    map[string]any{"emoji": "😀🎉✨"},
			expected: `{"emoji": "😀🎉✨"}`,
		},
		{
			name:     "chinese characters",
			input:    map[string]any{"text": "你好世界"},
			expected: `{"text": "你好世界"}`,
		},
		{
			name:     "arabic characters",
			input:    map[string]any{"text": "مرحبا"},
			expected: `{"text": "مرحبا"}`,
		},
		{
			name:     "mixed unicode and ascii",
			input:    map[string]any{"text": "Hello 世界! 😀"},
			expected: `{"text": "Hello 世界! 😀"}`,
		},
		{
			name:     "unicode with special symbols",
			input:    map[string]any{"text": "®©™€£¥"},
			expected: `{"text": "®©™€£¥"}`,
		},
		// complex combinations - strings that look like JSON
		{
			name:     "json string inside value",
			input:    map[string]any{"nested": `{"key":"value"}`},
			expected: `{"nested": "{\"key\":\"value\"}"}`,
		},
		{
			name:     "json array inside value",
			input:    map[string]any{"array": `["a","b","c"]`},
			expected: `{"array": "[\"a\",\"b\",\"c\"]"}`,
		},
		// edge cases
		{
			name:     "empty string",
			input:    map[string]any{"empty": ""},
			expected: `{"empty": ""}`,
		},
		{
			name:     "empty object",
			input:    map[string]any{},
			expected: `{}`,
		},
		{
			name:     "empty array",
			input:    []any{},
			expected: `[]`,
		},
		{
			name:     "numbers",
			input:    map[string]any{"int": 42, "float": 3.14},
			expected: `{"float": 3.14, "int": 42}`,
		},
		{
			name:     "boolean",
			input:    map[string]any{"bool": true, "other": false},
			expected: `{"bool": true, "other": false}`,
		},
		{
			name:     "null value",
			input:    map[string]any{"value": nil},
			expected: `{"value": null}`,
		},
		// nested structures with complex strings
		{
			name: "nested object with escapes",
			input: map[string]any{
				"outer": map[string]any{
					"path":  `C:\folder\file.txt`,
					"quote": `He said "hi"`,
				},
			},
			expected: `{"outer": {"path": "C:\\folder\\file.txt", "quote": "He said \"hi\""}}`,
		},
		{
			name: "array with unicode and escapes",
			input: []any{
				"normal",
				"with\nnewline",
				"with\"quote",
				"emoji😀",
				"colon:comma,",
			},
			expected: `["normal", "with\nnewline", "with\"quote", "emoji😀", "colon:comma,"]`,
		},
		{
			name:     "backslash at positions before special chars",
			input:    map[string]any{"text": `a\b:c\d,e`},
			expected: `{"text": "a\\b:c\\d,e"}`,
		},
		{
			name:     "multiple backslashes before quote",
			input:    map[string]any{"text": `ends\\"`},
			expected: `{"text": "ends\\\\\""}`,
		},
		{
			name:     "unicode with escapes",
			input:    map[string]any{"text": "Hello\n世界\t😀"},
			expected: `{"text": "Hello\n世界\t😀"}`,
		},

		// Real-world tool call example
		{
			name: "tool call arguments",
			input: map[string]any{
				"location": "San Francisco, CA",
				"unit":     "fahrenheit",
				"format":   "json",
			},
			expected: `{"format": "json", "location": "San Francisco, CA", "unit": "fahrenheit"}`,
		},
		{
			name: "complex tool arguments with escapes",
			input: map[string]any{
				"query":       `SELECT * FROM "users" WHERE name = 'O'Brien'`,
				"description": "Fetch user\ndata from DB",
				"path":        `C:\data\users.db`,
			},
			expected: `{"description": "Fetch user\ndata from DB", "path": "C:\\data\\users.db", "query": "SELECT * FROM \"users\" WHERE name = 'O'Brien'"}`,
		},
		{
			name:     "unicode immediately adjacent to JSON structure chars",
			input:    map[string]any{"😀key": "😀value", "test": "😀:😀,😀"},
			expected: `{"test": "😀:😀,😀", "😀key": "😀value"}`,
		},
		{
			name:     "long unicode string stress test",
			input:    map[string]any{"text": "😀😁😂😃😄😅😆😇😈😉😊😋😌😍😎😏😐😑😒😓😔😕😖😗😘😙😚😛😜😝😞😟"},
			expected: `{"text": "😀😁😂😃😄😅😆😇😈😉😊😋😌😍😎😏😐😑😒😓😔😕😖😗😘😙😚😛😜😝😞😟"}`,
		},
		{
			name: "deeply nested with unicode everywhere",
			input: map[string]any{
				"😀": map[string]any{
					"你好": []any{"مرحبا", "®©™", "∑∫∂√"},
				},
			},
			expected: `{"😀": {"你好": ["مرحبا", "®©™", "∑∫∂√"]}}`,
		},
		{
			name:     "unicode with all JSON special chars interleaved",
			input:    map[string]any{"k😀:k": "v😀,v", "a:😀": "b,😀", "😀": ":,😀,:"},
			expected: `{"a:😀": "b,😀", "k😀:k": "v😀,v", "😀": ":,😀,:"}`,
		},
		{
			name:     "combining diacritics and RTL text",
			input:    map[string]any{"hebrew": "עִבְרִית", "combined": "é̀ñ", "mixed": "test:עִבְרִית,é̀ñ"},
			expected: `{"combined": "é̀ñ", "hebrew": "עִבְרִית", "mixed": "test:עִבְרִית,é̀ñ"}`,
		},
		{
			name:     "pathological case: unicode + escapes + special chars",
			input:    map[string]any{"😀": "test\n😀\"quote😀\\backslash😀:colon😀,comma😀"},
			expected: `{"😀": "test\n😀\"quote😀\\backslash😀:colon😀,comma😀"}`,
		},

		// all JSON structural characters inside strings
		{
			name:     "braces and brackets in strings",
			input:    map[string]any{"text": "test{with}braces[and]brackets"},
			expected: `{"text": "test{with}braces[and]brackets"}`,
		},
		{
			name:     "braces and brackets with colons and commas",
			input:    map[string]any{"code": "{key:value,[1,2,3]}"},
			expected: `{"code": "{key:value,[1,2,3]}"}`,
		},
		{
			name:     "json-like string with all structural chars",
			input:    map[string]any{"schema": `{"type":"object","properties":{"name":{"type":"string"},"items":{"type":"array"}}}`},
			expected: `{"schema": "{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"},\"items\":{\"type\":\"array\"}}}"}`,
		},

		// forward slash tests (JSON allows \/ as an escape sequence)
		{
			name:     "forward slash in URL",
			input:    map[string]any{"url": "https://example.com/path/to/resource"},
			expected: `{"url": "https://example.com/path/to/resource"}`,
		},
		{
			name:     "regex pattern with slashes",
			input:    map[string]any{"regex": "/[a-z]+/gi"},
			expected: `{"regex": "/[a-z]+/gi"}`,
		},

		// all JSON escape sequences
		{
			name:     "backspace escape",
			input:    map[string]any{"text": "before\bafter"},
			expected: `{"text": "before\bafter"}`,
		},
		{
			name:     "form feed escape",
			input:    map[string]any{"text": "before\fafter"},
			expected: `{"text": "before\fafter"}`,
		},
		{
			name:     "all standard escapes combined",
			input:    map[string]any{"text": "\"\\\b\f\n\r\t"},
			expected: `{"text": "\"\\\b\f\n\r\t"}`,
		},

		// unicode escape sequences
		{
			name:     "string that forces unicode escapes",
			input:    map[string]any{"control": "\u0000\u0001\u001f"},
			expected: `{"control": "\u0000\u0001\u001f"}`,
		},

		// empty objects and arrays nested with strings
		{
			name:     "nested empty structures with string values",
			input:    map[string]any{"empty_obj": map[string]any{}, "empty_arr": []any{}, "text": "{}[]"},
			expected: `{"empty_arr": [], "empty_obj": {}, "text": "{}[]"}`,
		},

		// complex nesting with all structural characters
		{
			name: "deeply nested with all char types",
			input: map[string]any{
				"level1": map[string]any{
					"array": []any{
						map[string]any{"nested": "value:with,special{chars}[here]"},
						[]any{"a", "b", "c"},
					},
				},
			},
			expected: `{"level1": {"array": [{"nested": "value:with,special{chars}[here]"}, ["a", "b", "c"]]}}`,
		},

		// string containing escaped structural characters
		{
			name:     "string with multiple escape sequences and structural chars",
			input:    map[string]any{"data": "test\"quote\"{brace}[bracket]:colon,comma\\backslash/slash"},
			expected: `{"data": "test\"quote\"{brace}[bracket]:colon,comma\\backslash/slash"}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := marshalWithSpaces(tt.input)
			if err != nil {
				t.Fatalf("marshalWithSpaces failed: %v", err)
			}

			resultStr := string(result)
			if diff := cmp.Diff(resultStr, tt.expected); diff != "" {
				t.Errorf("mismatch (-got +want):\n%s", diff)
			}
		})
	}
}
