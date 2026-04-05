package parsers

import (
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
)

func TestGemma4Parser(t *testing.T) {
	tests := []struct {
		name              string
		input             string
		expectedContent   string
		expectedThinking  string
		expectedToolCalls []api.ToolCall
		thinkingEnabled   bool
		lastMessage       *api.Message
	}{
		{
			name:            "simple_content",
			input:           "This is a simple response.",
			expectedContent: "This is a simple response.",
		},
		{
			name:             "thinking_then_content",
			input:            "<|channel>thought\nLet me think about this...<channel|>The answer is 42.",
			expectedContent:  "The answer is 42.",
			expectedThinking: "Let me think about this...",
			thinkingEnabled:  true,
		},
		{
			name:             "multiple_thinking_blocks",
			input:            "<|channel>first thought<channel|><|channel>second thought<channel|>Final answer.",
			expectedContent:  "Final answer.",
			expectedThinking: "first thoughtsecond thought",
			thinkingEnabled:  true,
		},
		{
			name:             "thinking_only_no_content",
			input:            "<|channel>just thinking<channel|>",
			expectedContent:  "",
			expectedThinking: "just thinking",
			thinkingEnabled:  true,
		},
		{
			name:  "tool_call_simple",
			input: `<|tool_call>call:get_weather{location:<|"|>Paris<|"|>}<tool_call|>`,
			expectedToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: testArgs(map[string]any{
							"location": "Paris",
						}),
					},
				},
			},
		},
		{
			name:  "tool_call_with_multiple_args",
			input: `<|tool_call>call:get_weather{location:<|"|>Paris<|"|>,units:<|"|>metric<|"|>}<tool_call|>`,
			expectedToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: testArgs(map[string]any{
							"location": "Paris",
							"units":    "metric",
						}),
					},
				},
			},
		},
		{
			name:  "tool_call_with_number_arg",
			input: `<|tool_call>call:set_temp{value:42}<tool_call|>`,
			expectedToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "set_temp",
						Arguments: testArgs(map[string]any{
							"value": 42.0,
						}),
					},
				},
			},
		},
		{
			name:  "tool_call_with_boolean_arg",
			input: `<|tool_call>call:toggle{enabled:true}<tool_call|>`,
			expectedToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "toggle",
						Arguments: testArgs(map[string]any{
							"enabled": true,
						}),
					},
				},
			},
		},
		{
			name:  "tool_call_with_nested_object",
			input: `<|tool_call>call:process{config:{enabled:true,name:<|"|>test<|"|>}}<tool_call|>`,
			expectedToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "process",
						Arguments: testArgs(map[string]any{
							"config": map[string]any{
								"enabled": true,
								"name":    "test",
							},
						}),
					},
				},
			},
		},
		{
			name:  "tool_call_with_array",
			input: `<|tool_call>call:process{items:[<|"|>a<|"|>,<|"|>b<|"|>]}<tool_call|>`,
			expectedToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "process",
						Arguments: testArgs(map[string]any{
							"items": []any{"a", "b"},
						}),
					},
				},
			},
		},
		{
			name:  "tool_call_with_array_of_multiple_gemma_quoted_strings",
			input: `<|tool_call>call:process{items:[<|"|>a<|"|>,<|"|>b "quoted"<|"|>,<|"|>c<|"|>]}<tool_call|>`,
			expectedToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "process",
						Arguments: testArgs(map[string]any{
							"items": []any{"a", `b "quoted"`, "c"},
						}),
					},
				},
			},
		},
		{
			name: "tool_call_with_multiline_string_arg",
			input: `<|tool_call>call:bash{command:<|"|>date
<|"|>}<tool_call|>`,
			expectedToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "bash",
						Arguments: testArgs(map[string]any{
							"command": "date\n",
						}),
					},
				},
			},
		},
		{
			name:  "tool_call_with_escaped_double_quotes_in_string_arg",
			input: `<|tool_call>call:search{query:<|"|>say \"hello\"<|"|>}<tool_call|>`,
			expectedToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "search",
						Arguments: testArgs(map[string]any{
							"query": `say \"hello\"`,
						}),
					},
				},
			},
		},
		{
			name:  "tool_call_with_unescaped_double_quotes_in_string_arg",
			input: `<|tool_call>call:search{query:<|"|>say "hello"<|"|>}<tool_call|>`,
			expectedToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "search",
						Arguments: testArgs(map[string]any{
							"query": `say "hello"`,
						}),
					},
				},
			},
		},
		{
			name:  "tool_call_with_multiple_unescaped_double_quote_segments",
			input: `<|tool_call>call:search{query:<|"|>say "hello", then "goodbye"<|"|>}<tool_call|>`,
			expectedToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "search",
						Arguments: testArgs(map[string]any{
							"query": `say "hello", then "goodbye"`,
						}),
					},
				},
			},
		},
		{
			name:  "tool_call_with_mixed_escaped_and_unescaped_double_quotes",
			input: `<|tool_call>call:search{query:<|"|>first \"quoted\" then "raw"<|"|>}<tool_call|>`,
			expectedToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "search",
						Arguments: testArgs(map[string]any{
							"query": `first \"quoted\" then "raw"`,
						}),
					},
				},
			},
		},
		{
			name:  "tool_call_done_flush_without_close_tag_with_unescaped_double_quotes",
			input: `<|tool_call>call:search{query:<|"|>say "hello" and "bye"<|"|>}`,
			expectedToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "search",
						Arguments: testArgs(map[string]any{
							"query": `say "hello" and "bye"`,
						}),
					},
				},
			},
		},
		{
			name:  "tool_call_with_mixed_raw_and_gemma_quoted_values",
			input: `<|tool_call>call:search{query:"raw \"quoted\"",note:<|"|>gemma "quoted"<|"|>}<tool_call|>`,
			expectedToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "search",
						Arguments: testArgs(map[string]any{
							"query": `raw "quoted"`,
							"note":  `gemma "quoted"`,
						}),
					},
				},
			},
		},
		{
			name:  "tool_call_with_array_of_objects_and_mixed_quotes",
			input: `<|tool_call>call:plan{steps:[{title:<|"|>step "one"<|"|>,done:false},{title:<|"|>step \"two\"<|"|>,done:true}]}<tool_call|>`,
			expectedToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "plan",
						Arguments: testArgs(map[string]any{
							"steps": []any{
								map[string]any{
									"title": `step "one"`,
									"done":  false,
								},
								map[string]any{
									"title": `step \"two\"`,
									"done":  true,
								},
							},
						}),
					},
				},
			},
		},
		{
			name:  "tool_call_with_windows_path_single_backslashes",
			input: `<|tool_call>call:open_file{path:<|"|>C:\users\bob\file.txt<|"|>}<tool_call|>`,
			expectedToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "open_file",
						Arguments: testArgs(map[string]any{
							"path": `C:\users\bob\file.txt`,
						}),
					},
				},
			},
		},
		{
			name:  "multiple_tool_calls",
			input: `<|tool_call>call:get_weather{location:<|"|>Paris<|"|>}<tool_call|><|tool_call>call:get_weather{location:<|"|>London<|"|>}<tool_call|>`,
			expectedToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: testArgs(map[string]any{
							"location": "Paris",
						}),
					},
				},
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: testArgs(map[string]any{
							"location": "London",
						}),
					},
				},
			},
		},
		{
			name:             "thinking_then_tool_call",
			input:            "<|channel>thought\nI need to check the weather<channel|><|tool_call>call:get_weather{location:<|\"|>Paris<|\"|>}<tool_call|>",
			expectedThinking: "I need to check the weather",
			expectedToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: testArgs(map[string]any{
							"location": "Paris",
						}),
					},
				},
			},
			thinkingEnabled: true,
		},
		{
			name:            "content_then_tool_call",
			input:           `Let me check that for you.<|tool_call>call:get_weather{location:<|"|>Paris<|"|>}<tool_call|>`,
			expectedContent: "Let me check that for you.",
			expectedToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "get_weather",
						Arguments: testArgs(map[string]any{
							"location": "Paris",
						}),
					},
				},
			},
		},
		{
			name:            "thinking_disabled_channel_tags_as_content",
			input:           "<|channel>this is not thinking<channel|>actual content",
			expectedContent: "actual content",
			thinkingEnabled: false,
		},
		{
			name:            "prefill_content_only",
			input:           "Continuing content.",
			expectedContent: "Continuing content.",
			lastMessage: &api.Message{
				Role:    "assistant",
				Content: "Previous content",
			},
			thinkingEnabled: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := &Gemma4Parser{hasThinkingSupport: true}
			parser.Init(nil, tt.lastMessage, &api.ThinkValue{Value: tt.thinkingEnabled})

			content, thinking, toolCalls, err := parser.Add(tt.input, true)
			if err != nil {
				t.Fatalf("Add() error = %v", err)
			}

			if diff := cmp.Diff(tt.expectedContent, content); diff != "" {
				t.Errorf("content mismatch (-want +got):\n%s", diff)
			}

			if diff := cmp.Diff(tt.expectedThinking, thinking); diff != "" {
				t.Errorf("thinking mismatch (-want +got):\n%s", diff)
			}

			if diff := cmp.Diff(tt.expectedToolCalls, toolCalls, argsComparer); diff != "" {
				t.Errorf("tool calls mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestGemma4Parser_Streaming(t *testing.T) {
	parser := &Gemma4Parser{hasThinkingSupport: true}
	parser.Init(nil, nil, &api.ThinkValue{Value: true})

	chunks := []string{
		"<|channel>thought",
		"\nLet me think",
		"...<channel|>The answer",
		" is 42.",
	}

	var finalContent, finalThinking strings.Builder

	for i, chunk := range chunks {
		done := i == len(chunks)-1
		content, thinking, _, err := parser.Add(chunk, done)
		if err != nil {
			t.Fatalf("Add() error on chunk %d: %v", i, err)
		}

		finalContent.WriteString(content)
		finalThinking.WriteString(thinking)
	}

	if finalContent.String() != "The answer is 42." {
		t.Errorf("expected content %q, got %q", "The answer is 42.", finalContent.String())
	}

	if finalThinking.String() != "Let me think..." {
		t.Errorf("expected thinking %q, got %q", "Let me think...", finalThinking.String())
	}
}

func TestGemma4Parser_StreamingToolCall(t *testing.T) {
	parser := &Gemma4Parser{hasThinkingSupport: false}
	parser.Init(nil, nil, nil)

	chunks := []string{
		`<|tool_call>call:get_`,
		`weather{location:<|"|>Par`,
		`is<|"|>}<tool_call|>`,
	}

	var finalContent strings.Builder
	var finalToolCalls []api.ToolCall

	for i, chunk := range chunks {
		done := i == len(chunks)-1
		content, _, toolCalls, err := parser.Add(chunk, done)
		if err != nil {
			t.Fatalf("Add() error on chunk %d: %v", i, err)
		}

		finalContent.WriteString(content)
		finalToolCalls = append(finalToolCalls, toolCalls...)
	}

	if finalContent.String() != "" {
		t.Errorf("expected no content, got %q", finalContent.String())
	}

	expectedToolCalls := []api.ToolCall{
		{
			Function: api.ToolCallFunction{
				Name: "get_weather",
				Arguments: testArgs(map[string]any{
					"location": "Paris",
				}),
			},
		},
	}

	if diff := cmp.Diff(expectedToolCalls, finalToolCalls, argsComparer); diff != "" {
		t.Errorf("tool calls mismatch (-want +got):\n%s", diff)
	}
}

func TestGemma4Parser_StreamingSplitThinkingTag(t *testing.T) {
	tests := []struct {
		name             string
		chunks           []string
		expectedContent  string
		expectedThinking string
	}{
		{
			name: "split_channel_open_tag",
			chunks: []string{
				"<|chan",
				"nel>thinking here<channel|>content",
			},
			expectedContent:  "content",
			expectedThinking: "thinking here",
		},
		{
			name: "split_channel_close_tag",
			chunks: []string{
				"<|channel>thinking here<chan",
				"nel|>content",
			},
			expectedContent:  "content",
			expectedThinking: "thinking here",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := &Gemma4Parser{hasThinkingSupport: true}
			parser.Init(nil, nil, &api.ThinkValue{Value: true})

			var finalContent, finalThinking strings.Builder
			for i, chunk := range tt.chunks {
				done := i == len(tt.chunks)-1
				content, thinking, _, err := parser.Add(chunk, done)
				if err != nil {
					t.Fatalf("Add() error on chunk %d: %v", i, err)
				}
				finalContent.WriteString(content)
				finalThinking.WriteString(thinking)
			}

			if finalContent.String() != tt.expectedContent {
				t.Errorf("expected content %q, got %q", tt.expectedContent, finalContent.String())
			}
			if finalThinking.String() != tt.expectedThinking {
				t.Errorf("expected thinking %q, got %q", tt.expectedThinking, finalThinking.String())
			}
		})
	}
}

func TestGemma4ArgsToJSON(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "simple_string",
			input:    `{location:<|"|>Paris<|"|>}`,
			expected: `{"location":"Paris"}`,
		},
		{
			name:     "multiple_args",
			input:    `{location:<|"|>Paris<|"|>,units:<|"|>metric<|"|>}`,
			expected: `{"location":"Paris","units":"metric"}`,
		},
		{
			name:     "number_value",
			input:    `{value:42}`,
			expected: `{"value":42}`,
		},
		{
			name:     "boolean_value",
			input:    `{enabled:true}`,
			expected: `{"enabled":true}`,
		},
		{
			name:     "nested_object",
			input:    `{config:{enabled:true,name:<|"|>test<|"|>}}`,
			expected: `{"config":{"enabled":true,"name":"test"}}`,
		},
		{
			name:     "array_value",
			input:    `{items:[<|"|>a<|"|>,<|"|>b<|"|>]}`,
			expected: `{"items":["a","b"]}`,
		},
		{
			name:     "array_value_with_multiple_gemma_quoted_strings",
			input:    `{items:[<|"|>a<|"|>,<|"|>b "quoted"<|"|>,<|"|>c<|"|>]}`,
			expected: `{"items":["a","b \"quoted\"","c"]}`,
		},
		{
			name:     "empty_object",
			input:    `{}`,
			expected: `{}`,
		},
		{
			name:     "mixed_types",
			input:    `{name:<|"|>test<|"|>,count:5,active:true,tags:[<|"|>a<|"|>]}`,
			expected: `{"name":"test","count":5,"active":true,"tags":["a"]}`,
		},
		{
			name:     "null_value",
			input:    `{value:null}`,
			expected: `{"value":null}`,
		},
		{
			name: "multiline_string_value",
			input: `{command:<|"|>date
<|"|>}`,
			expected: `{"command":"date\n"}`,
		},
		{
			name:     "string_value_with_escaped_double_quotes",
			input:    `{query:<|"|>say \"hello\"<|"|>}`,
			expected: `{"query":"say \\\"hello\\\""}`,
		},
		{
			name:     "string_value_with_unescaped_double_quotes",
			input:    `{query:<|"|>say "hello"<|"|>}`,
			expected: `{"query":"say \"hello\""}`,
		},
		{
			name:     "string_value_with_multiple_unescaped_double_quote_segments",
			input:    `{query:<|"|>say "hello", then "goodbye"<|"|>}`,
			expected: `{"query":"say \"hello\", then \"goodbye\""}`,
		},
		{
			name:     "string_value_with_mixed_escaped_and_unescaped_double_quotes",
			input:    `{query:<|"|>first \"quoted\" then "raw"<|"|>}`,
			expected: `{"query":"first \\\"quoted\\\" then \"raw\""}`,
		},
		{
			name:     "string_value_with_punctuation_and_structural_chars",
			input:    `{query:<|"|>a,b:{c}[d]<|"|>}`,
			expected: `{"query":"a,b:{c}[d]"}`,
		},
		{
			name:     "string_value_with_windows_path_backslashes",
			input:    `{path:<|"|>C:\\Temp\\file.txt<|"|>}`,
			expected: `{"path":"C:\\\\Temp\\\\file.txt"}`,
		},
		{
			name:     "string_value_with_windows_path_single_backslashes",
			input:    `{path:<|"|>C:\users\bob<|"|>}`,
			expected: `{"path":"C:\\users\\bob"}`,
		},
		{
			name:     "string_value_with_escaped_forward_slashes",
			input:    `{url:<|"|>https:\/\/example.com\/a<|"|>}`,
			expected: `{"url":"https:\\/\\/example.com\\/a"}`,
		},
		{
			name:     "string_value_with_unicode_escape_sequence",
			input:    `{s:<|"|>snowman:\u2603<|"|>}`,
			expected: `{"s":"snowman:\\u2603"}`,
		},
		{
			name:     "string_value_with_unknown_escape_sequence",
			input:    `{s:<|"|>bad \x escape<|"|>}`,
			expected: `{"s":"bad \\x escape"}`,
		},
		{
			name:     "string_value_with_invalid_unicode_escape_sequence",
			input:    `{s:<|"|>bad \uZZZZ escape<|"|>}`,
			expected: `{"s":"bad \\uZZZZ escape"}`,
		},
		{
			name:     "raw_quoted_string_with_escaped_quotes",
			input:    `{q:"say \"hi\" and \"bye\""}`,
			expected: `{"q":"say \"hi\" and \"bye\""}`,
		},
		{
			name:     "nested_mixed_raw_and_gemma_quoted_values",
			input:    `{meta:{title:<|"|>t "1"<|"|>,note:"n \"2\""},items:[<|"|>x "3"<|"|>,"y \"4\""]}`,
			expected: `{"meta":{"title":"t \"1\"","note":"n \"2\""},"items":["x \"3\"","y \"4\""]}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := gemma4ArgsToJSON(tt.input)
			if result != tt.expected {
				t.Errorf("expected %q, got %q", tt.expected, result)
			}
		})
	}
}

func TestGemma4Parser_HasToolSupport(t *testing.T) {
	parser := &Gemma4Parser{}
	if !parser.HasToolSupport() {
		t.Error("Gemma4Parser should support tools")
	}
}

func TestGemma4Parser_HasThinkingSupport(t *testing.T) {
	parser := &Gemma4Parser{hasThinkingSupport: true}
	if !parser.HasThinkingSupport() {
		t.Error("Gemma4Parser with thinking support should report it")
	}

	parser2 := &Gemma4Parser{hasThinkingSupport: false}
	if parser2.HasThinkingSupport() {
		t.Error("Gemma4Parser without thinking support should not report it")
	}
}

func TestParseGemma4ToolCall_InvalidRawQuotedEscape(t *testing.T) {
	_, err := parseGemma4ToolCall(`call:open_file{path:"C:\users\bob\file.txt"}`)
	if err == nil {
		t.Fatal("expected parseGemma4ToolCall to reject malformed raw-quoted JSON escapes")
	}
}

func TestParseGemma4ToolCall_QuotedScalarsStayStrings(t *testing.T) {
	toolCall, err := parseGemma4ToolCall(`call:foo{n:<|"|>1<|"|>,b:<|"|>true<|"|>,z:<|"|>null<|"|>}`)
	if err != nil {
		t.Fatalf("parseGemma4ToolCall returned error: %v", err)
	}

	want := api.ToolCall{
		Function: api.ToolCallFunction{
			Name: "foo",
			Arguments: testArgs(map[string]any{
				"n": "1",
				"b": "true",
				"z": "null",
			}),
		},
	}

	if diff := cmp.Diff(want, toolCall, argsComparer); diff != "" {
		t.Fatalf("quoted scalar handling differed from the reference implementation (-want +got):\n%s", diff)
	}
}

func TestParseGemma4ToolCall_UnquotedScalarsKeepStructuredTypes(t *testing.T) {
	toolCall, err := parseGemma4ToolCall(`call:foo{n:1,b:true,z:null}`)
	if err != nil {
		t.Fatalf("parseGemma4ToolCall returned error: %v", err)
	}

	want := api.ToolCall{
		Function: api.ToolCallFunction{
			Name: "foo",
			Arguments: testArgs(map[string]any{
				"n": 1.0,
				"b": true,
				"z": nil,
			}),
		},
	}

	if diff := cmp.Diff(want, toolCall, argsComparer); diff != "" {
		t.Fatalf("unquoted scalar handling differed from the reference implementation (-want +got):\n%s", diff)
	}
}

func TestParseGemma4ToolCall_ReferenceImplementationExample(t *testing.T) {
	toolCall, err := parseGemma4ToolCall(`call:get_current_temperature{detail_level:0,location:<|"|>Paris, France<|"|>,unit:<|"|>celsius<|"|>}`)
	if err != nil {
		t.Fatalf("parseGemma4ToolCall returned error: %v", err)
	}

	want := api.ToolCall{
		Function: api.ToolCallFunction{
			Name: "get_current_temperature",
			Arguments: testArgs(map[string]any{
				"detail_level": 0.0,
				"location":     "Paris, France",
				"unit":         "celsius",
			}),
		},
	}

	if diff := cmp.Diff(want, toolCall, argsComparer); diff != "" {
		t.Fatalf("tool call handling differed from the reference implementation (-want +got):\n%s", diff)
	}
}

func TestParseGemma4ToolCall_InvalidRawQuotedStructuralString(t *testing.T) {
	_, err := parseGemma4ToolCall(`call:foo{q:"a,b:c"}`)
	if err == nil {
		t.Fatal("expected parseGemma4ToolCall to reject raw-quoted strings with structural text that the reference implementation does not support")
	}
}
