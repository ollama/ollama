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
			name:  "tool_call_exec_with_embedded_quoted_path",
			input: `<|tool_call>call:exec{command:<|"|>ls -F "vault/"<|"|>}<tool_call|>`,
			expectedToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "exec",
						Arguments: testArgs(map[string]any{
							"command": `ls -F "vault/"`,
						}),
					},
				},
			},
		},
		{
			name:  "tool_call_exec_with_embedded_quoted_url",
			input: `<|tool_call>call:exec{command:<|"|>fetch "https://ollama.com/library/gemma4" --extract<|"|>}<tool_call|>`,
			expectedToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "exec",
						Arguments: testArgs(map[string]any{
							"command": `fetch "https://ollama.com/library/gemma4" --extract`,
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
			name: "tool_call_edit_with_array_of_objects_multiline_markdown_and_quotes",
			input: `<|tool_call>call:edit{edits:[
  {newText:<|"|># Gemma4 + openclaude speed optimization guide

Use "nvfp4" only after validating tool calls.
<|"|>,oldText:<|"|># Gemma4 + openclaude speed optimization guide

Use the default quantization.
<|"|>},
  {newText:<|"|>## 14. Methods tried but not adopted

### 1. Model quantization
Do not enable "mxfp8" by default.
<|"|>,oldText:<|"|>## 14. Methods tried but not adopted
<|"|>}
]}<tool_call|>`,
			expectedToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "edit",
						Arguments: testArgs(map[string]any{
							"edits": []any{
								map[string]any{
									"newText": "# Gemma4 + openclaude speed optimization guide\n\nUse \"nvfp4\" only after validating tool calls.\n",
									"oldText": "# Gemma4 + openclaude speed optimization guide\n\nUse the default quantization.\n",
								},
								map[string]any{
									"newText": "## 14. Methods tried but not adopted\n\n### 1. Model quantization\nDo not enable \"mxfp8\" by default.\n",
									"oldText": "## 14. Methods tried but not adopted\n",
								},
							},
						}),
					},
				},
			},
		},
		{
			name: "tool_call_edit_with_array_of_objects_issue_comment_spacing",
			input: `<|tool_call>call:edit{edits:[
  {newText:<|"|># Gemma4 + openclaude speed optimization guide...<|"|>,
   oldText:<|"|># Gemma4 + openclaude speed optimization guide...<|"|>},
  {newText:<|"|>## 14. Methods tried but not adopted\n\n### 1. Model quantization...<|"|>,
   oldText:<|"|>## 14. Methods tried but not adopted\n<|"|>}
]}<tool_call|>`,
			expectedToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "edit",
						Arguments: testArgs(map[string]any{
							"edits": []any{
								map[string]any{
									"newText": "# Gemma4 + openclaude speed optimization guide...",
									"oldText": "# Gemma4 + openclaude speed optimization guide...",
								},
								map[string]any{
									"newText": "## 14. Methods tried but not adopted\\n\\n### 1. Model quantization...",
									"oldText": "## 14. Methods tried but not adopted\\n",
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
						Index: 0,
						Name:  "get_weather",
						Arguments: testArgs(map[string]any{
							"location": "Paris",
						}),
					},
				},
				{
					Function: api.ToolCallFunction{
						Index: 1,
						Name:  "get_weather",
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

func TestGemma4Parser_IgnoresExtraToolCallCloseTags(t *testing.T) {
	tests := []struct {
		name            string
		chunks          []string
		expectedContent string
	}{
		{
			name: "same_chunk_without_trailing_content",
			chunks: []string{
				`<|tool_call>call:get_weather{location:<|"|>Paris<|"|>}<tool_call|><tool_call|>`,
			},
			expectedContent: "",
		},
		{
			name: "same_chunk_before_real_content",
			chunks: []string{
				`<|tool_call>call:get_weather{location:<|"|>Paris<|"|>}<tool_call|><tool_call|>Done.`,
			},
			expectedContent: "Done.",
		},
		{
			name: "split_across_chunks_before_real_content",
			chunks: []string{
				`<|tool_call>call:get_weather{location:<|"|>Paris<|"|>}<tool_call|><tool_`,
				`call|>Done.`,
			},
			expectedContent: "Done.",
		},
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

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := &Gemma4Parser{hasThinkingSupport: false}
			parser.Init(nil, nil, nil)

			var finalContent strings.Builder
			var finalToolCalls []api.ToolCall

			for i, chunk := range tt.chunks {
				done := i == len(tt.chunks)-1
				content, _, toolCalls, err := parser.Add(chunk, done)
				if err != nil {
					t.Fatalf("Add() error on chunk %d: %v", i, err)
				}

				finalContent.WriteString(content)
				finalToolCalls = append(finalToolCalls, toolCalls...)
			}

			if diff := cmp.Diff(tt.expectedContent, finalContent.String()); diff != "" {
				t.Errorf("content mismatch (-want +got):\n%s", diff)
			}

			if diff := cmp.Diff(expectedToolCalls, finalToolCalls, argsComparer); diff != "" {
				t.Errorf("tool calls mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestGemma4Parser_IgnoresToolResponseBoundaryAfterToolCall(t *testing.T) {
	tests := []struct {
		name            string
		chunks          []string
		expectedContent string
	}{
		{
			name: "same_chunk_without_trailing_content",
			chunks: []string{
				`<|tool_call>call:get_weather{location:<|"|>Paris<|"|>}<tool_call|><|tool_response>`,
			},
			expectedContent: "",
		},
		{
			name: "same_chunk_before_real_content",
			chunks: []string{
				`<|tool_call>call:get_weather{location:<|"|>Paris<|"|>}<tool_call|><|tool_response>Done.`,
			},
			expectedContent: "Done.",
		},
		{
			name: "split_across_chunks_before_real_content",
			chunks: []string{
				`<|tool_call>call:get_weather{location:<|"|>Paris<|"|>}<tool_call|><|tool_res`,
				`ponse>Done.`,
			},
			expectedContent: "Done.",
		},
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

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := &Gemma4Parser{hasThinkingSupport: false}
			parser.Init(nil, nil, nil)

			var finalContent strings.Builder
			var finalToolCalls []api.ToolCall

			for i, chunk := range tt.chunks {
				done := i == len(tt.chunks)-1
				content, _, toolCalls, err := parser.Add(chunk, done)
				if err != nil {
					t.Fatalf("Add() error on chunk %d: %v", i, err)
				}

				finalContent.WriteString(content)
				finalToolCalls = append(finalToolCalls, toolCalls...)
			}

			if diff := cmp.Diff(tt.expectedContent, finalContent.String()); diff != "" {
				t.Errorf("content mismatch (-want +got):\n%s", diff)
			}

			if diff := cmp.Diff(expectedToolCalls, finalToolCalls, argsComparer); diff != "" {
				t.Errorf("tool calls mismatch (-want +got):\n%s", diff)
			}
		})
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
			name:     "nested_object_with_space_after_object_open_and_before_bare_key",
			input:    `{edits:[{ newText:<|"|>a<|"|>, oldText:<|"|>b<|"|>} ]}`,
			expected: `{"edits":[{ "newText":"a", "oldText":"b"} ]}`,
		},
		{
			name:     "nested_object_with_space_before_bare_key",
			input:    `{edits:[{newText:<|"|>a<|"|>, oldText:<|"|>b<|"|>} ]}`,
			expected: `{"edits":[{"newText":"a", "oldText":"b"} ]}`,
		},
		{
			name: "nested_object_with_newline_before_bare_key",
			input: `{edits:[
  {newText:<|"|>a<|"|>,
   oldText:<|"|>b<|"|>}
]}`,
			expected: `{"edits":[
  {"newText":"a",
   "oldText":"b"}
]}`,
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
			name:     "raw_quoted_string_containing_key_like_text",
			input:    `{q:"keep , oldText: literal",note:<|"|>ok<|"|>}`,
			expected: `{"q":"keep , oldText: literal","note":"ok"}`,
		},
		{
			name:     "raw_quoted_string_with_escaped_quotes_and_key_like_text",
			input:    `{q:"keep , oldText: and \"quoted\" text",note:<|"|>ok<|"|>}`,
			expected: `{"q":"keep , oldText: and \"quoted\" text","note":"ok"}`,
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

func TestRepairGemma4MissingStringDelimiter(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "closes_before_object_close",
			input:    `{command:<|"|>ls}`,
			expected: `{command:<|"|>ls<|"|>}`,
		},
		{
			name:     "closes_terminal_value_after_previous_property",
			input:    `{path:<|"|>/tmp<|"|>,command:<|"|>ls}`,
			expected: `{path:<|"|>/tmp<|"|>,command:<|"|>ls<|"|>}`,
		},
		{
			name:     "closes_at_end",
			input:    `{command:<|"|>ls`,
			expected: `{command:<|"|>ls<|"|>`,
		},
		{
			name:     "preserves_valid_gemma_quoted_string",
			input:    `{command:<|"|>ls<|"|>}`,
			expected: `{command:<|"|>ls<|"|>}`,
		},
		{
			name:     "preserves_input_without_gemma_delimiter",
			input:    `{command:ls}`,
			expected: `{command:ls}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := repairGemma4MissingStringDelimiter(tt.input)
			if got != tt.expected {
				t.Fatalf("repairGemma4MissingStringDelimiter(%q) = %q, want %q", tt.input, got, tt.expected)
			}
		})
	}
}

func TestRepairGemma4MissingObjectClose(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "adds_object_close",
			input:    `{content:<|"|>hello<|"|>`,
			expected: `{content:<|"|>hello<|"|>}`,
		},
		{
			name:     "adds_object_close_before_trailing_space",
			input:    "{content:<|\"|>hello<|\"|>  ",
			expected: "{content:<|\"|>hello<|\"|>}  ",
		},
		{
			name:     "preserves_existing_object_close",
			input:    `{content:<|"|>hello<|"|>}`,
			expected: `{content:<|"|>hello<|"|>}`,
		},
		{
			name:     "preserves_non_object_input",
			input:    `content:<|"|>hello<|"|>`,
			expected: `content:<|"|>hello<|"|>`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := repairGemma4MissingObjectClose(tt.input)
			if got != tt.expected {
				t.Fatalf("repairGemma4MissingObjectClose(%q) = %q, want %q", tt.input, got, tt.expected)
			}
		})
	}
}

func TestRepairGemma4SingleQuotedValues(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "converts_single_quoted_value",
			input:    `{pattern:':\s*\w+'}`,
			expected: `{pattern:<|"|>:\s*\w+<|"|>}`,
		},
		{
			name:     "converts_middle_single_quoted_value",
			input:    `{include:<|"|>*.py<|"|>,pattern:'abc',path:<|"|>/tmp<|"|>}`,
			expected: `{include:<|"|>*.py<|"|>,pattern:<|"|>abc<|"|>,path:<|"|>/tmp<|"|>}`,
		},
		{
			name:     "drops_dangling_gemma_delimiter_after_single_quoted_value",
			input:    `{pattern:'abc'<|"|>}`,
			expected: `{pattern:<|"|>abc<|"|>}`,
		},
		{
			name:     "preserves_gemma_quoted_value",
			input:    `{pattern:<|"|>abc<|"|>}`,
			expected: `{pattern:<|"|>abc<|"|>}`,
		},
		{
			name:     "preserves_json_quoted_value",
			input:    `{pattern:"abc"}`,
			expected: `{pattern:"abc"}`,
		},
		{
			name:     "preserves_unterminated_single_quote",
			input:    `{pattern:'abc}`,
			expected: `{pattern:'abc}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := repairGemma4SingleQuotedValues(tt.input)
			if got != tt.expected {
				t.Fatalf("repairGemma4SingleQuotedValues(%q) = %q, want %q", tt.input, got, tt.expected)
			}
		})
	}
}

func TestRepairGemma4RawTerminalStringValue(t *testing.T) {
	numberProps := api.NewToolPropertiesMap()
	numberProps.Set("content", api.ToolProperty{Type: api.PropertyType{"number"}})
	numberTool := api.Tool{
		Type: "function",
		Function: api.ToolFunction{
			Name: "write",
			Parameters: api.ToolFunctionParameters{
				Type:       "object",
				Properties: numberProps,
			},
		},
	}

	tests := []struct {
		name       string
		input      string
		toolName   string
		tools      []api.Tool
		expected   string
		expectedOK bool
	}{
		{
			name:       "wraps_known_string_property_terminal_value",
			input:      "{content:\n\n# Title",
			toolName:   "write",
			tools:      []api.Tool{gemma4TestStringTool("write", "content")},
			expected:   "{content:<|\"|>\n\n# Title<|\"|>",
			expectedOK: true,
		},
		{
			name:       "stops_before_next_known_property",
			input:      `{content:hello,mode:<|"|>fast<|"|>}`,
			toolName:   "write",
			tools:      []api.Tool{gemma4TestStringTool("write", "content", "mode")},
			expected:   `{content:<|"|>hello<|"|>,mode:<|"|>fast<|"|>}`,
			expectedOK: true,
		},
		{
			name:       "does_not_repair_without_schema",
			input:      `{content:hello`,
			toolName:   "write",
			tools:      nil,
			expectedOK: false,
		},
		{
			name:       "does_not_repair_non_string_property",
			input:      `{content:hello`,
			toolName:   "write",
			tools:      []api.Tool{numberTool},
			expectedOK: false,
		},
		{
			name:       "does_not_repair_already_structured_value",
			input:      `{content:<|"|>hello<|"|>}`,
			toolName:   "write",
			tools:      []api.Tool{gemma4TestStringTool("write", "content")},
			expectedOK: false,
		},
		{
			name:       "does_not_repair_json_literal_start",
			input:      `{content:123}`,
			toolName:   "write",
			tools:      []api.Tool{gemma4TestStringTool("write", "content")},
			expectedOK: false,
		},
		{
			name:       "does_not_repair_unknown_tool",
			input:      `{content:hello`,
			toolName:   "missing",
			tools:      []api.Tool{gemma4TestStringTool("write", "content")},
			expectedOK: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := repairGemma4RawTerminalStringValue(tt.input, tt.toolName, tt.tools)
			if ok != tt.expectedOK {
				t.Fatalf("repairGemma4RawTerminalStringValue ok = %t, want %t", ok, tt.expectedOK)
			}
			if got != tt.expected {
				t.Fatalf("repairGemma4RawTerminalStringValue got %q, want %q", got, tt.expected)
			}
		})
	}
}

func TestGemma4RepairCandidates(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		toolName string
		tools    []api.Tool
		expected []string
	}{
		{
			name:     "missing_string_delimiter_candidate",
			input:    `{command:<|"|>ls}`,
			toolName: "bash",
			tools:    []api.Tool{gemma4TestStringTool("bash", "command")},
			expected: []string{`{command:<|"|>ls<|"|>}`},
		},
		{
			name:     "single_quoted_value_candidate",
			input:    `{pattern:'abc'<|"|>}`,
			toolName: "grep",
			tools:    []api.Tool{gemma4TestStringTool("grep", "pattern")},
			expected: []string{`{pattern:<|"|>abc<|"|>}`},
		},
		{
			name:     "raw_string_candidate_also_gets_missing_object_close",
			input:    `{content:hello`,
			toolName: "write",
			tools:    []api.Tool{gemma4TestStringTool("write", "content")},
			expected: []string{
				`{content:hello`,
				`{content:<|"|>hello<|"|>}`,
			},
		},
		{
			name:     "does_not_add_missing_object_close_without_another_repair",
			input:    `{n:1`,
			toolName: "count",
			tools:    []api.Tool{gemma4TestStringTool("count", "name")},
			expected: []string{`{n:1`},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := gemma4RepairCandidates(tt.input, tt.toolName, tt.tools)
			if diff := cmp.Diff(tt.expected, got); diff != "" {
				t.Fatalf("gemma4RepairCandidates mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func gemma4TestStringTool(name string, argNames ...string) api.Tool {
	props := api.NewToolPropertiesMap()
	for _, argName := range argNames {
		props.Set(argName, api.ToolProperty{Type: api.PropertyType{"string"}})
	}

	return api.Tool{
		Type: "function",
		Function: api.ToolFunction{
			Name: name,
			Parameters: api.ToolFunctionParameters{
				Type:       "object",
				Properties: props,
			},
		},
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
	_, err := parseGemma4ToolCall(`call:open_file{path:"C:\users\bob\file.txt"}`, nil)
	if err == nil {
		t.Fatal("expected parseGemma4ToolCall to reject malformed raw-quoted JSON escapes")
	}
}

func TestParseGemma4ToolCall_QuotedScalarsStayStrings(t *testing.T) {
	toolCall, err := parseGemma4ToolCall(`call:foo{n:<|"|>1<|"|>,b:<|"|>true<|"|>,z:<|"|>null<|"|>}`, nil)
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
	toolCall, err := parseGemma4ToolCall(`call:foo{n:1,b:true,z:null}`, nil)
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
	toolCall, err := parseGemma4ToolCall(`call:get_current_temperature{detail_level:0,location:<|"|>Paris, France<|"|>,unit:<|"|>celsius<|"|>}`, nil)
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

func TestParseGemma4ToolCall_RepairsIssue15315Examples(t *testing.T) {
	writeContent := "\n\n# Project Style Guide for Autonomous Agents' Code Generation (AGENTS.md)\n\n" +
		"This document captures the *de facto* coding standards observed across the `src/` and `components/` source code, designed to ensure consistency for all generated code and modules consumed by the agent system."

	tests := []struct {
		name    string
		content string
		tools   []api.Tool
		want    api.ToolCall
	}{
		{
			name: "raw multiline string",
			// Source: https://github.com/ollama/ollama/issues/15315#issue-4203625511
			content: "call:write{content:" + writeContent,
			tools:   []api.Tool{gemma4TestStringTool("write", "content")},
			want: api.ToolCall{
				Function: api.ToolCallFunction{
					Name: "write",
					Arguments: testArgs(map[string]any{
						"content": writeContent,
					}),
				},
			},
		},
		{
			name:    "single quoted value with dangling gemma string delimiter",
			content: `call:grep{include:<|"|>*.py<|"|>,output_mode:<|"|>content<|"|>,path:<|"|>/data/robotics/experiment1<|"|>,pattern:':\s*\w+'<|"|>}`,
			tools:   []api.Tool{gemma4TestStringTool("grep", "include", "output_mode", "path", "pattern")},
			// Source: https://github.com/ollama/ollama/issues/15315#issue-4203625511
			want: api.ToolCall{
				Function: api.ToolCallFunction{
					Name: "grep",
					Arguments: testArgs(map[string]any{
						"include":     "*.py",
						"output_mode": "content",
						"path":        "/data/robotics/experiment1",
						"pattern":     `:\s*\w+`,
					}),
				},
			},
		},
		{
			name:    "unclosed gemma string before object close",
			content: `call:bash{command:<|"|>ls}`,
			tools:   []api.Tool{gemma4TestStringTool("bash", "command")},
			// Source: https://github.com/ollama/ollama/issues/15315#issuecomment-4194547092
			want: api.ToolCall{
				Function: api.ToolCallFunction{
					Name: "bash",
					Arguments: testArgs(map[string]any{
						"command": "ls",
					}),
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseGemma4ToolCall(tt.content, tt.tools)
			if err != nil {
				t.Fatalf("parseGemma4ToolCall returned error: %v", err)
			}

			if diff := cmp.Diff(tt.want, got, argsComparer); diff != "" {
				t.Fatalf("tool call mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestParseGemma4ToolCall_RepairsMultipleProperties(t *testing.T) {
	tests := []struct {
		name    string
		content string
		tools   []api.Tool
		want    api.ToolCall
	}{
		{
			name:    "single_quoted_middle_property",
			content: `call:grep{include:<|"|>*.py<|"|>,pattern:'abc',path:<|"|>/tmp<|"|>}`,
			tools:   []api.Tool{gemma4TestStringTool("grep", "include", "pattern", "path")},
			want: api.ToolCall{
				Function: api.ToolCallFunction{
					Name: "grep",
					Arguments: testArgs(map[string]any{
						"include": "*.py",
						"pattern": "abc",
						"path":    "/tmp",
					}),
				},
			},
		},
		{
			name:    "unclosed_gemma_string_terminal_property",
			content: `call:bash{path:<|"|>/tmp<|"|>,command:<|"|>ls}`,
			tools:   []api.Tool{gemma4TestStringTool("bash", "path", "command")},
			want: api.ToolCall{
				Function: api.ToolCallFunction{
					Name: "bash",
					Arguments: testArgs(map[string]any{
						"path":    "/tmp",
						"command": "ls",
					}),
				},
			},
		},
		{
			name:    "raw_string_before_next_property",
			content: `call:write{content:hello,mode:<|"|>fast<|"|>}`,
			tools:   []api.Tool{gemma4TestStringTool("write", "content", "mode")},
			want: api.ToolCall{
				Function: api.ToolCallFunction{
					Name: "write",
					Arguments: testArgs(map[string]any{
						"content": "hello",
						"mode":    "fast",
					}),
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseGemma4ToolCall(tt.content, tt.tools)
			if err != nil {
				t.Fatalf("parseGemma4ToolCall returned error: %v", err)
			}

			if diff := cmp.Diff(tt.want, got, argsComparer); diff != "" {
				t.Fatalf("tool call mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestParseGemma4ToolCall_DoesNotRepairNonTerminalUnclosedGemmaString(t *testing.T) {
	// TODO(drifkin): our current examples show unclosed gemma strings as the last
	// values, but if we find examples where there's an unclosed non-last value,
	// we should consider repairing it. This test shows that we don't yet repair
	// this type (the heuristics of where to close are much more complicated)
	_, err := parseGemma4ToolCall(`call:example{first:<|"|>one,second:<|"|>two<|"|>}`, []api.Tool{
		gemma4TestStringTool("example", "first", "second"),
	})
	if err == nil {
		t.Fatal("expected non-terminal unclosed Gemma string to remain unsupported")
	}
}

func TestParseGemma4ToolCall_RawQuotedStructuralString(t *testing.T) {
	got, err := parseGemma4ToolCall(`call:foo{q:"a,b:c"}`, nil)
	if err != nil {
		t.Fatalf("parseGemma4ToolCall returned error: %v", err)
	}

	want := api.ToolCall{
		Function: api.ToolCallFunction{
			Name: "foo",
			Arguments: testArgs(map[string]any{
				"q": "a,b:c",
			}),
		},
	}

	if diff := cmp.Diff(want, got, argsComparer); diff != "" {
		t.Fatalf("tool call mismatch (-want +got):\n%s", diff)
	}
}
