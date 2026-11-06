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
			name:  "tool_call_with_null_arg",
			input: `<|tool_call>call:set_optional{value:null,nested:{value:null}}<tool_call|>`,
			expectedToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "set_optional",
						Arguments: testArgs(map[string]any{
							"value": nil,
							"nested": map[string]any{
								"value": nil,
							},
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

func TestGemma4Parser_ToolResponseContinuationStartsInThinking(t *testing.T) {
	parser := &Gemma4Parser{hasThinkingSupport: true}
	parser.Init(nil, &api.Message{Role: "tool", Content: "go.mod\ngo.sum\n"}, &api.ThinkValue{Value: true})

	content, thinking, toolCalls, err := parser.Add("The user asked for files.<channel|>go.mod and go.sum", true)
	if err != nil {
		t.Fatalf("Add() error = %v", err)
	}

	if diff := cmp.Diff("go.mod and go.sum", content); diff != "" {
		t.Errorf("content mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff("The user asked for files.", thinking); diff != "" {
		t.Errorf("thinking mismatch (-want +got):\n%s", diff)
	}
	if len(toolCalls) != 0 {
		t.Fatalf("expected no tool calls, got %+v", toolCalls)
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
		{
			name:     "bare_word_candidate_comes_last",
			input:    `{content:hello,mode:Fast}`,
			toolName: "write",
			tools:    []api.Tool{gemma4TestStringTool("write", "content")},
			expected: []string{
				`{content:hello,mode:Fast}`,
				`{content:<|"|>hello,mode:Fast<|"|>}`,
				`{content:<|"|>hello<|"|>,mode:<|"|>Fast<|"|>}`,
			},
		},
		{
			name:     "bare_word_candidate_keeps_single_quote_repair",
			input:    `{pattern:'a: b',mode:Fast}`,
			toolName: "grep",
			expected: []string{
				`{pattern:<|"|>a: b<|"|>,mode:Fast}`,
				`{pattern:<|"|>a: b<|"|>,mode:<|"|>Fast<|"|>}`,
			},
		},
		{
			name:     "hex_number_candidate_comes_last",
			input:    `{flags:0x0100,n:1}`,
			toolName: "set",
			expected: []string{
				`{flags:0x0100,n:1}`,
				`{flags:<|"|>0x0100<|"|>,n:1}`,
			},
		},
		{
			name:     "colon_token_candidate_comes_after_trailing_commas",
			input:    `{ref:01A2B3:Base.esm,mode:Fast,}`,
			toolName: "set",
			expected: []string{
				`{ref:01A2B3:Base.esm,mode:Fast,}`,
				`{ref:01A2B3:Base.esm,mode:<|"|>Fast<|"|>,}`,
				`{ref:01A2B3:Base.esm,mode:Fast}`,
				`{ref:01A2B3:Base.esm,mode:<|"|>Fast<|"|>}`,
				`{ref:<|"|>01A2B3:Base.esm<|"|>,mode:<|"|>Fast<|"|>,}`,
				`{ref:<|"|>01A2B3:Base.esm<|"|>,mode:<|"|>Fast<|"|>}`,
			},
		},
		{
			name:     "fenced_value_candidate_comes_last",
			input:    "{content:---\nA: 'x'\n---\n,mode:Fast,}",
			toolName: "set",
			expected: []string{
				"{content:---\nA: <|\"|>x<|\"|>\n---\n,mode:Fast,}",
				"{content:---\nA: <|\"|>x<|\"|>\n---\n,mode:<|\"|>Fast<|\"|>,}",
				"{content:---\nA: <|\"|>x<|\"|>\n---\n,mode:Fast}",
				"{content:---\nA: <|\"|>x<|\"|>\n---\n,mode:<|\"|>Fast<|\"|>}",
				"{content:<|\"|>A: 'x'<|\"|>\n,mode:<|\"|>Fast<|\"|>,}",
				"{content:<|\"|>A: 'x'<|\"|>\n,mode:<|\"|>Fast<|\"|>}",
			},
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

func TestGemma4PlainQuoteStringClose(t *testing.T) {
	write := []api.Tool{gemma4TestStringTool("write", "content", "path")}
	tests := []struct {
		name    string
		content string
		tools   []api.Tool
		want    map[string]any // nil: the call is not repaired
	}{
		{
			// A whole file's text closed with a plain quote on a line of its own, and
			// no comma before the next argument.
			name:    "plain_quote_then_key",
			content: "call:write{content:<|\"|>EditorID: FlameBlade\nDescription:\n  Value: ''\nObjectEffect: 049BB7:Skyrim.esm  # EnchWeaponFireDamage01\n\"\npath:<|\"|>FlameBlade/plugin/Weapons/FlameBlade.yaml<|\"|>}",
			tools:   write,
			want: map[string]any{
				"content": "EditorID: FlameBlade\nDescription:\n  Value: ''\nObjectEffect: 049BB7:Skyrim.esm  # EnchWeaponFireDamage01\n",
				"path":    "FlameBlade/plugin/Weapons/FlameBlade.yaml",
			},
		},
		{
			name:    "plain_quote_comma_then_key",
			content: `call:write{content:<|"|>a: 1",path:<|"|>x.yaml<|"|>}`,
			tools:   write,
			want:    map[string]any{"content": "a: 1", "path": "x.yaml"},
		},
		{
			name:    "quote_before_a_word_that_is_no_parameter",
			content: "call:write{content:<|\"|>a: \"b\"\nnote:<|\"|>x.yaml<|\"|>}",
			tools:   write,
		},
		{
			name:    "no_tool_to_name_the_parameters",
			content: "call:write{content:<|\"|>a: 1\"\npath:<|\"|>x.yaml<|\"|>}",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseGemma4ToolCall(tt.content, tt.tools)
			if tt.want == nil {
				if err == nil {
					if _, split := got.Function.Arguments.Get("path"); split {
						t.Fatalf("expected no repair at the quote, got %v", got.Function.Arguments)
					}
				}
				return
			}
			if err != nil {
				t.Fatalf("parseGemma4ToolCall returned error: %v", err)
			}
			want := api.ToolCall{Function: api.ToolCallFunction{Name: "write", Arguments: testArgs(tt.want)}}
			if diff := cmp.Diff(want, got, argsComparer); diff != "" {
				t.Fatalf("tool call mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestGemma4Parser_PlainQuoteClosedStringInFinishedToolCall(t *testing.T) {
	parser := &Gemma4Parser{hasThinkingSupport: true}
	parser.Init([]api.Tool{gemma4TestStringTool("write", "content", "path")}, nil, &api.ThinkValue{Value: true})

	content, _, toolCalls, err := parser.Add("<|tool_call>call:write{content:<|\"|>a: 1\n\"\npath:<|\"|>x.yaml<|\"|>}<tool_call|>", true)
	if err != nil {
		t.Fatalf("Add() error = %v", err)
	}
	if content != "" {
		t.Fatalf("expected no content, got %q", content)
	}
	want := []api.ToolCall{{Function: api.ToolCallFunction{Name: "write", Arguments: testArgs(map[string]any{
		"content": "a: 1\n",
		"path":    "x.yaml",
	})}}}
	if diff := cmp.Diff(want, toolCalls, argsComparer); diff != "" {
		t.Fatalf("tool calls mismatch (-want +got):\n%s", diff)
	}
}

func TestRepairGemma4BareWordValues(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "word_after_a_key",
			input:    `{mode:Fast}`,
			expected: `{mode:<|"|>Fast<|"|>}`,
		},
		{
			name:     "word_in_an_object_in_a_list",
			input:    `{set:[{path:<|"|>AIData.Aggression<|"|>,value:Unaggressive}]}`,
			expected: `{set:[{path:<|"|>AIData.Aggression<|"|>,value:<|"|>Unaggressive<|"|>}]}`,
		},
		{
			name:     "words_as_list_items",
			input:    `{flags:[Female, Unique ],n:[1,2]}`,
			expected: `{flags:[<|"|>Female<|"|>, <|"|>Unique<|"|> ],n:[1,2]}`,
		},
		{
			name:     "words_with_dots_digits_and_underscores",
			input:    `{"path" : AIData.Aggression,a:_x1,b:x_2.y}`,
			expected: `{"path" : <|"|>AIData.Aggression<|"|>,a:<|"|>_x1<|"|>,b:<|"|>x_2.y<|"|>}`,
		},
		{
			name:     "keeps_words_in_strings",
			input:    `{note:<|"|>mode:Fast}<|"|>,raw:"mode:Fast]"}`,
			expected: `{note:<|"|>mode:Fast}<|"|>,raw:"mode:Fast]"}`,
		},
		{
			name:     "keeps_json_literals",
			input:    `{a:true,b:false,c:null,d:[true,null]}`,
			expected: `{a:true,b:false,c:null,d:[true,null]}`,
		},
		{
			name:     "keeps_literals_written_in_other_cases",
			input:    `{a:True,b:NULL,c:None}`,
			expected: `{a:True,b:NULL,c:None}`,
		},
		{
			name:     "keeps_numbers",
			input:    `{a:5,b:-1.5,c:1e3,d:0x1F,e:.5}`,
			expected: `{a:5,b:-1.5,c:1e3,d:0x1F,e:.5}`,
		},
		{
			name:     "keeps_words_that_are_not_all_of_the_value",
			input:    `{name:Iron Sword,path:a/b,c:x-y,d:Fast<|"|>x<|"|>}`,
			expected: `{name:Iron Sword,path:a/b,c:x-y,d:Fast<|"|>x<|"|>}`,
		},
		{
			name:     "keeps_keys",
			input:    `{a:b:c,Fast}`,
			expected: `{a:b:c,Fast}`,
		},
		{
			name:     "keeps_a_word_with_nothing_after_it",
			input:    `{mode:Fast`,
			expected: `{mode:Fast`,
		},
		{
			name:     "keeps_unbalanced_delimiters",
			input:    `{mode:Fast,a:<|"|>x}`,
			expected: `{mode:Fast,a:<|"|>x}`,
		},
		{
			name:     "keeps_a_delimiter_in_a_json_quoted_string",
			input:    `{raw:"<|"|>",mode:Fast,b:<|"|>}`,
			expected: `{raw:"<|"|>",mode:Fast,b:<|"|>}`,
		},
		{
			name:     "keeps_brackets_that_do_not_nest",
			input:    `{mode:Fast,a:[x}]`,
			expected: `{mode:Fast,a:[x}]`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := repairGemma4BareWordValues(tt.input); got != tt.expected {
				t.Fatalf("repairGemma4BareWordValues(%q) = %q, want %q", tt.input, got, tt.expected)
			}
		})
	}
}

// gemma4BareWordEditRecordCall sets a list of paths, one to a single-quoted
// value and one to a bare word.
const gemma4BareWordEditRecordCall = `call:editRecord{project:<|"|>HildaTheNord<|"|>,record:<|"|>Hilda<|"|>,set:[{path:<|"|>Configuration.Level<|"|>,value:'{MutagenObjectType: PcLevelMult, LevelMult: 1}'},{path:<|"|>Configuration.CalcMinLevel<|"|>,value:5},{path:<|"|>Configuration.CalcMaxLevel<|"|>,value:30},{path:<|"|>Configuration.SpeedMultiplier<|"|>,value:100},{path:<|"|>AIData.Aggression<|"|>,value:Unaggressive}]}`

var gemma4BareWordEditRecordArgs = map[string]any{
	"project": "HildaTheNord",
	"record":  "Hilda",
	"set": []any{
		map[string]any{"path": "Configuration.Level", "value": "{MutagenObjectType: PcLevelMult, LevelMult: 1}"},
		map[string]any{"path": "Configuration.CalcMinLevel", "value": 5.0},
		map[string]any{"path": "Configuration.CalcMaxLevel", "value": 30.0},
		map[string]any{"path": "Configuration.SpeedMultiplier", "value": 100.0},
		map[string]any{"path": "AIData.Aggression", "value": "Unaggressive"},
	},
}

func TestParseGemma4ClosedToolCall_BareWordValues(t *testing.T) {
	tests := []struct {
		name    string
		content string
		want    map[string]any // nil: the call still fails
	}{
		{
			name:    "word_in_a_list_of_objects",
			content: gemma4BareWordEditRecordCall,
			want:    gemma4BareWordEditRecordArgs,
		},
		{
			name:    "words_as_list_items",
			content: `call:set{flags:[Female,Unique],level:5}`,
			want:    map[string]any{"flags": []any{"Female", "Unique"}, "level": 5.0},
		},
		{
			name:    "literals_numbers_and_strings_keep_their_reading",
			content: `call:set{a:true,b:false,c:null,d:-1.5,note:<|"|>mode:Fast,<|"|>,mode:Fast}`,
			want:    map[string]any{"a": true, "b": false, "c": nil, "d": -1.5, "note": "mode:Fast,", "mode": "Fast"},
		},
		{
			name:    "word_in_an_unclosed_list",
			content: `call:set{flags:[Female,Unique,level:5}`,
			want:    map[string]any{"flags": []any{"Female", "Unique"}, "level": 5.0},
		},
		{
			// A finished call's last string left open is closed at its end first, as
			// it is without a bare word; the delimiters are then balanced.
			name:    "word_before_a_last_string_left_open",
			content: `call:set{mode:Fast,a:<|"|>x<|"|>,b:<|"|>}`,
			want:    map[string]any{"mode": "Fast", "a": "x", "b": ""},
		},
		{
			name:    "literal_written_in_another_case",
			content: `call:set{a:True}`,
		},
		{
			name:    "none",
			content: `call:set{a:None}`,
		},
		{
			name:    "two_words",
			content: `call:set{name:Iron Sword}`,
		},
		{
			name:    "word_followed_by_a_colon",
			content: `call:set{a:b:c}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseGemma4ClosedToolCall(tt.content, nil)
			if tt.want == nil {
				if err == nil {
					t.Fatalf("expected the call to fail, got %v", got.Function.Arguments)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseGemma4ClosedToolCall returned error: %v", err)
			}
			name := tt.content[len("call:"):strings.Index(tt.content, "{")]
			want := api.ToolCall{Function: api.ToolCallFunction{Name: name, Arguments: testArgs(tt.want)}}
			if diff := cmp.Diff(want, got, argsComparer); diff != "" {
				t.Fatalf("tool call mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestParseGemma4ToolCall_DoesNotRepairBareWordWithUnbalancedDelimiters(t *testing.T) {
	// Flushed on done, the call may have been cut off, and with a string left open
	// which words are outside every string is not clear.
	if got, err := parseGemma4ToolCall(`call:set{mode:Fast,a:<|"|>x<|"|>,b:<|"|>}`, nil); err == nil {
		t.Fatalf("expected the call to fail, got %v", got.Function.Arguments)
	}
}

func TestGemma4Parser_BareWordValueInFinishedToolCall(t *testing.T) {
	parser := &Gemma4Parser{hasThinkingSupport: true}
	parser.Init(nil, nil, &api.ThinkValue{Value: true})

	content, _, toolCalls, err := parser.Add("<|tool_call>"+gemma4BareWordEditRecordCall+"<tool_call|>", true)
	if err != nil {
		t.Fatalf("Add() error = %v", err)
	}
	if content != "" {
		t.Fatalf("expected no content, got %q", content)
	}
	want := []api.ToolCall{{Function: api.ToolCallFunction{Name: "editRecord", Arguments: testArgs(gemma4BareWordEditRecordArgs)}}}
	if diff := cmp.Diff(want, toolCalls, argsComparer); diff != "" {
		t.Fatalf("tool calls mismatch (-want +got):\n%s", diff)
	}
}

func TestRepairGemma4BarePhraseValues(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "phrase_after_a_key",
			input:    `{name:Iron Sword}`,
			expected: `{name:<|"|>Iron Sword<|"|>}`,
		},
		{
			name:     "phrase_and_word_in_a_list_of_objects",
			input:    `{set:[{path:<|"|>Name.Value<|"|>,value:Flame Blade},{path:<|"|>ObjectEffect<|"|>,value:FlameBladeBurn}]}`,
			expected: `{set:[{path:<|"|>Name.Value<|"|>,value:<|"|>Flame Blade<|"|>},{path:<|"|>ObjectEffect<|"|>,value:<|"|>FlameBladeBurn<|"|>}]}`,
		},
		{
			name:     "phrases_as_list_items",
			input:    `{names:[Iron Sword, Steel Sword ],n:[1,2]}`,
			expected: `{names:[<|"|>Iron Sword<|"|>, <|"|>Steel Sword<|"|> ],n:[1,2]}`,
		},
		{
			name:     "words_after_the_first_from_a_digit",
			input:    `{name:Flame Blade 2,n:3}`,
			expected: `{name:<|"|>Flame Blade 2<|"|>,n:3}`,
		},
		{
			name:     "spaces_and_tabs_between_words_kept",
			input:    "{name:Iron  \tSword ,n:1}",
			expected: "{name:<|\"|>Iron  \tSword<|\"|> ,n:1}",
		},
		{
			name:     "literal_read_in_a_phrase_only",
			input:    `{a:None,b:None of these,c:true}`,
			expected: `{a:None,b:<|"|>None of these<|"|>,c:true}`,
		},
		{
			name:     "keeps_phrases_that_are_not_all_of_the_value",
			input:    `{a:Iron Sword: x,b:a b/c,c:x y-z,d:Iron Sword<|"|>x<|"|>}`,
			expected: `{a:Iron Sword: x,b:a b/c,c:x y-z,d:Iron Sword<|"|>x<|"|>}`,
		},
		{
			name:     "keeps_words_on_two_lines",
			input:    "{name:Iron\nSword}",
			expected: "{name:Iron\nSword}",
		},
		{
			name:     "keeps_a_phrase_from_a_number",
			input:    `{a:5 swords}`,
			expected: `{a:5 swords}`,
		},
		{
			name:     "keeps_words_in_strings",
			input:    `{note:<|"|>name:Iron Sword}<|"|>,raw:"name:Iron Sword]"}`,
			expected: `{note:<|"|>name:Iron Sword}<|"|>,raw:"name:Iron Sword]"}`,
		},
		{
			name:     "keeps_unbalanced_delimiters",
			input:    `{name:Iron Sword,a:<|"|>x}`,
			expected: `{name:Iron Sword,a:<|"|>x}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := repairGemma4BarePhraseValues(tt.input); got != tt.expected {
				t.Fatalf("repairGemma4BarePhraseValues(%q) = %q, want %q", tt.input, got, tt.expected)
			}
		})
	}
}

// gemma4BarePhraseEditRecordCall sets a name to two words and an enchantment to
// one, both bare.
const gemma4BarePhraseEditRecordCall = `call:editRecord{project:<|"|>FlameBlade<|"|>,record:<|"|>Weapons/FlameBlade.yaml<|"|>,set:[{path:<|"|>BasicStats.Damage<|"|>,value:12},{path:<|"|>BasicStats.Weight<|"|>,value:10},{path:<|"|>Name.Value<|"|>,value:Flame Blade},{path:<|"|>ObjectEffect<|"|>,value:FlameBladeBurn},{path:<|"|>EnchantmentAmount<|"|>,value:500}]}`

var gemma4BarePhraseEditRecordArgs = map[string]any{
	"project": "FlameBlade",
	"record":  "Weapons/FlameBlade.yaml",
	"set": []any{
		map[string]any{"path": "BasicStats.Damage", "value": 12.0},
		map[string]any{"path": "BasicStats.Weight", "value": 10.0},
		map[string]any{"path": "Name.Value", "value": "Flame Blade"},
		map[string]any{"path": "ObjectEffect", "value": "FlameBladeBurn"},
		map[string]any{"path": "EnchantmentAmount", "value": 500.0},
	},
}

// gemma4TestEditRecordTool has editRecord's parameters as their types, its list
// of changes naming each change's keys.
func gemma4TestEditRecordTool() api.Tool {
	props := api.NewToolPropertiesMap()
	props.Set("project", api.ToolProperty{Type: api.PropertyType{"string"}})
	props.Set("record", api.ToolProperty{Type: api.PropertyType{"string"}})
	props.Set("set", api.ToolProperty{
		Type: api.PropertyType{"array"},
		Items: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"path":  map[string]any{"type": "string"},
				"value": map[string]any{},
			},
		},
	})

	return api.Tool{
		Type: "function",
		Function: api.ToolFunction{
			Name:       "editRecord",
			Parameters: api.ToolFunctionParameters{Type: "object", Properties: props},
		},
	}
}

func TestParseGemma4ClosedToolCall_BarePhraseValues(t *testing.T) {
	editRecord := []api.Tool{gemma4TestEditRecordTool()}
	tests := []struct {
		name    string
		content string
		tools   []api.Tool
		want    map[string]any // nil: the call still fails
	}{
		{
			name:    "phrase_in_a_list_of_objects",
			content: gemma4BarePhraseEditRecordCall,
			tools:   editRecord,
			want:    gemma4BarePhraseEditRecordArgs,
		},
		{
			name:    "phrase_in_an_unclosed_list",
			content: `call:editRecord{project:<|"|>FlameBlade<|"|>,set:[{path:<|"|>Name.Value<|"|>,value:Flame Blade}}`,
			tools:   editRecord,
			want:    map[string]any{"project": "FlameBlade", "set": []any{map[string]any{"path": "Name.Value", "value": "Flame Blade"}}},
		},
		{
			name:    "phrase_beside_an_escaped_quote_and_a_hex_number",
			content: `call:editRecord{project:\"FlameBlade\",set:[{path:<|"|>Name.Value<|"|>,value:Flame Blade},{path:<|"|>Flags<|"|>,value:0x0100}]}`,
			tools:   editRecord,
			want: map[string]any{"project": "FlameBlade", "set": []any{
				map[string]any{"path": "Name.Value", "value": "Flame Blade"},
				map[string]any{"path": "Flags", "value": "0x0100"},
			}},
		},
		{
			name:    "phrase_beside_an_object_of_fields",
			content: `call:editRecord{set:[{path:<|"|>Name.Value<|"|>,value:Flame Blade},{path:<|"|>Level<|"|>,value:{MutagenObjectType:PcLevelMult,LevelMult:1}}]}`,
			tools:   editRecord,
			want: map[string]any{"set": []any{
				map[string]any{"path": "Name.Value", "value": "Flame Blade"},
				map[string]any{"path": "Level", "value": map[string]any{"MutagenObjectType": "PcLevelMult", "LevelMult": 1.0}},
			}},
		},
		{
			name:    "phrase_for_an_unknown_tool",
			content: gemma4BarePhraseEditRecordCall,
		},
		{
			name:    "phrase_running_on_past_a_comma",
			content: `call:editRecord{set:[{path:<|"|>Note<|"|>,value:Hello there, how: are you}]}`,
			tools:   editRecord,
		},
		{
			name:    "phrase_beside_a_parameter_the_tool_lacks",
			content: `call:editRecord{set:[{path:<|"|>Name.Value<|"|>,value:Flame Blade}],how:are you}`,
			tools:   editRecord,
		},
		{
			name:    "phrase_before_a_colon",
			content: `call:editRecord{set:[{path:<|"|>Name.Value<|"|>,value:Flame Blade: new}]}`,
			tools:   editRecord,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseGemma4ClosedToolCall(tt.content, tt.tools)
			if tt.want == nil {
				if err == nil {
					t.Fatalf("expected the call to fail, got %v", got.Function.Arguments)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseGemma4ClosedToolCall returned error: %v", err)
			}
			name := tt.content[len("call:"):strings.Index(tt.content, "{")]
			want := api.ToolCall{Function: api.ToolCallFunction{Name: name, Arguments: testArgs(tt.want)}}
			if diff := cmp.Diff(want, got, argsComparer); diff != "" {
				t.Fatalf("tool call mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestGemma4Parser_BarePhraseValueInFinishedToolCall(t *testing.T) {
	parser := &Gemma4Parser{hasThinkingSupport: true}
	parser.Init([]api.Tool{gemma4TestEditRecordTool()}, nil, &api.ThinkValue{Value: true})

	content, _, toolCalls, err := parser.Add("<|tool_call>"+gemma4BarePhraseEditRecordCall+"<tool_call|>", true)
	if err != nil {
		t.Fatalf("Add() error = %v", err)
	}
	if content != "" {
		t.Fatalf("expected no content, got %q", content)
	}
	want := []api.ToolCall{{Function: api.ToolCallFunction{Name: "editRecord", Arguments: testArgs(gemma4BarePhraseEditRecordArgs)}}}
	if diff := cmp.Diff(want, toolCalls, argsComparer); diff != "" {
		t.Fatalf("tool calls mismatch (-want +got):\n%s", diff)
	}
}

func TestRepairGemma4EscapedQuoteValues(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "value_after_a_key",
			input:    `{level:\"5\"}`,
			expected: `{level:<|"|>5<|"|>}`,
		},
		{
			name:     "values_in_objects_in_a_list",
			input:    `{set:[{path:<|"|>Level<|"|>,value:\"{Type: PcLevelMult, Mult: 1}\"},{path:<|"|>Min<|"|>,value:\"5\"}]}`,
			expected: `{set:[{path:<|"|>Level<|"|>,value:<|"|>{Type: PcLevelMult, Mult: 1}<|"|>},{path:<|"|>Min<|"|>,value:<|"|>5<|"|>}]}`,
		},
		{
			name:     "values_as_list_items_and_with_spaces",
			input:    `{flags:[\"Female\", \"Unique\" ],"mode" : \"\" ,n:1}`,
			expected: `{flags:[<|"|>Female<|"|>, <|"|>Unique<|"|> ],"mode" : <|"|><|"|> ,n:1}`,
		},
		{
			name:     "keeps_escaped_quotes_in_strings",
			input:    `{note:<|"|>say \"hi\"<|"|>,raw:"say \"hi\""}`,
			expected: `{note:<|"|>say \"hi\"<|"|>,raw:"say \"hi\""}`,
		},
		{
			name:     "keeps_text_with_a_backslash_or_a_quote",
			input:    `{path:\"a\b\",say:\"a"b\"}`,
			expected: `{path:\"a\b\",say:\"a"b\"}`,
		},
		{
			name:     "reads_an_escaped_backslash_as_one",
			input:    `{value:\"- Data:\\n    Level: 1\",end:\"a\\\",n:1}`,
			expected: `{value:<|"|>- Data:\n    Level: 1<|"|>,end:<|"|>a\<|"|>,n:1}`,
		},
		{
			name:     "keeps_a_backslash_beside_an_escaped_one",
			input:    `{value:\"a\\\b\"}`,
			expected: `{value:\"a\\\b\"}`,
		},
		{
			name:     "keeps_text_that_is_not_all_of_the_value",
			input:    `{a:\"x\"y,b:1}`,
			expected: `{a:\"x\"y,b:1}`,
		},
		{
			name:     "keeps_a_value_left_open",
			input:    `{a:\"x}`,
			expected: `{a:\"x}`,
		},
		{
			name:     "keeps_unbalanced_delimiters",
			input:    `{a:\"x\",b:<|"|>y}`,
			expected: `{a:\"x\",b:<|"|>y}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := repairGemma4EscapedQuoteValues(tt.input); got != tt.expected {
				t.Fatalf("repairGemma4EscapedQuoteValues(%q) = %q, want %q", tt.input, got, tt.expected)
			}
		})
	}
}

// gemma4EscapedQuoteEditRecordCall sets a list of paths to values written between
// escaped quotes.
const gemma4EscapedQuoteEditRecordCall = `call:editRecord{build:true,project:<|"|>HildaTheNord<|"|>,record:<|"|>Hilda<|"|>,set:[{path:<|"|>Configuration.Level<|"|>,value:\"{MutagenObjectType: PcLevelMult, LevelMult: 1}\"},{path:<|"|>Configuration.CalcMinLevel<|"|>,value:\"5\"},{path:<|"|>Configuration.CalcMaxLevel<|"|>,value:\"30\"},{path:<|"|>AIData.Aggression<|"|>,value:\"Unaggressive\"}]}`

var gemma4EscapedQuoteEditRecordArgs = map[string]any{
	"build":   true,
	"project": "HildaTheNord",
	"record":  "Hilda",
	"set": []any{
		map[string]any{"path": "Configuration.Level", "value": "{MutagenObjectType: PcLevelMult, LevelMult: 1}"},
		map[string]any{"path": "Configuration.CalcMinLevel", "value": "5"},
		map[string]any{"path": "Configuration.CalcMaxLevel", "value": "30"},
		map[string]any{"path": "AIData.Aggression", "value": "Unaggressive"},
	},
}

func TestParseGemma4ClosedToolCall_EscapedQuoteValues(t *testing.T) {
	tests := []struct {
		name    string
		content string
		want    map[string]any // nil: the call still fails
	}{
		{
			name:    "values_in_a_list_of_objects",
			content: gemma4EscapedQuoteEditRecordCall,
			want:    gemma4EscapedQuoteEditRecordArgs,
		},
		{
			name:    "beside_a_bare_word_and_a_single_quoted_value",
			content: `call:set{a:\"x\",b:Fast,c:'y z',n:5}`,
			want:    map[string]any{"a": "x", "b": "Fast", "c": "y z", "n": 5.0},
		},
		{
			name:    "in_an_unclosed_list",
			content: `call:set{flags:[\"Female\",\"Unique\",level:5}`,
			want:    map[string]any{"flags": []any{"Female", "Unique"}, "level": 5.0},
		},
		{
			// A list's items as the lines of a file, their line ends escaped once more.
			name:    "lines_with_escaped_line_ends",
			content: `call:editRecord{build:true,project:<|"|>BanditSilver<|"|>,set:[{path:<|"|>Entries<|"|>,value:\"- Data:\\n    Level: 1\\n    Reference: 039D26:Skyrim.esm  # SubCharBandit01Melee1H\\n    Count: 1\"}]}`,
			want: map[string]any{"build": true, "project": "BanditSilver", "set": []any{
				map[string]any{"path": "Entries", "value": "- Data:\\n    Level: 1\\n    Reference: 039D26:Skyrim.esm  # SubCharBandit01Melee1H\\n    Count: 1"},
			}},
		},
		{
			name:    "text_with_a_backslash",
			content: `call:set{path:\"a\b\"}`,
		},
		{
			name:    "text_that_is_not_all_of_the_value",
			content: `call:set{a:\"x\"y}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseGemma4ClosedToolCall(tt.content, nil)
			if tt.want == nil {
				if err == nil {
					t.Fatalf("expected the call to fail, got %v", got.Function.Arguments)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseGemma4ClosedToolCall returned error: %v", err)
			}
			name := tt.content[len("call:"):strings.Index(tt.content, "{")]
			want := api.ToolCall{Function: api.ToolCallFunction{Name: name, Arguments: testArgs(tt.want)}}
			if diff := cmp.Diff(want, got, argsComparer); diff != "" {
				t.Fatalf("tool call mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestGemma4Parser_EscapedQuoteValueInFinishedToolCall(t *testing.T) {
	parser := &Gemma4Parser{hasThinkingSupport: true}
	parser.Init(nil, nil, &api.ThinkValue{Value: true})

	content, _, toolCalls, err := parser.Add("<|tool_call>"+gemma4EscapedQuoteEditRecordCall+"<tool_call|>", true)
	if err != nil {
		t.Fatalf("Add() error = %v", err)
	}
	if content != "" {
		t.Fatalf("expected no content, got %q", content)
	}
	want := []api.ToolCall{{Function: api.ToolCallFunction{Name: "editRecord", Arguments: testArgs(gemma4EscapedQuoteEditRecordArgs)}}}
	if diff := cmp.Diff(want, toolCalls, argsComparer); diff != "" {
		t.Fatalf("tool calls mismatch (-want +got):\n%s", diff)
	}
}

func TestRepairGemma4HexNumberValues(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "value_after_a_key",
			input:    `{flags:0x000100}`,
			expected: `{flags:<|"|>0x000100<|"|>}`,
		},
		{
			name:     "values_in_objects_in_a_list",
			input:    `{set:[{path:<|"|>A<|"|>,value:0x1F},{path:<|"|>B<|"|>,value:0XaB}]}`,
			expected: `{set:[{path:<|"|>A<|"|>,value:<|"|>0x1F<|"|>},{path:<|"|>B<|"|>,value:<|"|>0XaB<|"|>}]}`,
		},
		{
			name:     "values_as_list_items_and_with_spaces",
			input:    `{masks:[0x01, 0x02 ],"mode" : 0x3 ,n:1}`,
			expected: `{masks:[<|"|>0x01<|"|>, <|"|>0x02<|"|> ],"mode" : <|"|>0x3<|"|> ,n:1}`,
		},
		{
			name:     "keeps_hex_numbers_in_strings",
			input:    `{note:<|"|>0x10<|"|>,raw:"0x10"}`,
			expected: `{note:<|"|>0x10<|"|>,raw:"0x10"}`,
		},
		{
			name:     "keeps_text_that_is_not_all_of_a_hex_number",
			input:    `{a:0x10g,b:0x,c:0x 1,d:00x1}`,
			expected: `{a:0x10g,b:0x,c:0x 1,d:00x1}`,
		},
		{
			name:     "keeps_decimal_numbers",
			input:    `{a:100,b:0,c:[1, 2]}`,
			expected: `{a:100,b:0,c:[1, 2]}`,
		},
		{
			name:     "keeps_unbalanced_delimiters",
			input:    `{a:0x1,b:<|"|>y}`,
			expected: `{a:0x1,b:<|"|>y}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := repairGemma4HexNumberValues(tt.input); got != tt.expected {
				t.Fatalf("repairGemma4HexNumberValues(%q) = %q, want %q", tt.input, got, tt.expected)
			}
		})
	}
}

// gemma4HexNumberEditRecordCall sets a list of paths, one to a hex number written
// bare, beside an object written with JSON's quotes.
const gemma4HexNumberEditRecordCall = `call:editRecord{build:true,project:<|"|>HildaTheNord<|"|>,record:<|"|>Npcs/Hilda.yaml<|"|>,remove:[<|"|>Health<|"|>,<|"|>Magicka<|"|>,<|"|>Stamina<|"|>,<|"|>GearedUpWeapons<|"|>,<|"|>Unused2<|"|>],set:[{path:<|"|>PlayerSkills.Health<|"|>,value:100},{path:<|"|>PlayerSkills.Magicka<|"|>,value:100},{path:<|"|>PlayerSkills.Stamina<|"|>,value:100},{path:<|"|>PlayerSkills.GearedUpWeapons<|"|>,value:1},{path:<|"|>PlayerSkills.Unused2<|"|>,value:0x000100},{path:<|"|>Configuration.Level<|"|>,value:{"MutagenObjectType":"PcLevelMult","LevelMult":1}}]}`

var gemma4HexNumberEditRecordArgs = map[string]any{
	"build":   true,
	"project": "HildaTheNord",
	"record":  "Npcs/Hilda.yaml",
	"remove":  []any{"Health", "Magicka", "Stamina", "GearedUpWeapons", "Unused2"},
	"set": []any{
		map[string]any{"path": "PlayerSkills.Health", "value": 100.0},
		map[string]any{"path": "PlayerSkills.Magicka", "value": 100.0},
		map[string]any{"path": "PlayerSkills.Stamina", "value": 100.0},
		map[string]any{"path": "PlayerSkills.GearedUpWeapons", "value": 1.0},
		map[string]any{"path": "PlayerSkills.Unused2", "value": "0x000100"},
		map[string]any{"path": "Configuration.Level", "value": map[string]any{"MutagenObjectType": "PcLevelMult", "LevelMult": 1.0}},
	},
}

func TestParseGemma4ClosedToolCall_HexNumberValues(t *testing.T) {
	tests := []struct {
		name    string
		content string
		want    map[string]any // nil: the call still fails
	}{
		{
			name:    "value_in_a_list_of_objects",
			content: gemma4HexNumberEditRecordCall,
			want:    gemma4HexNumberEditRecordArgs,
		},
		{
			name:    "beside_a_bare_word_and_a_single_quoted_value",
			content: `call:set{a:0x0F,b:Fast,c:'y z',n:5}`,
			want:    map[string]any{"a": "0x0F", "b": "Fast", "c": "y z", "n": 5.0},
		},
		{
			name:    "in_an_unclosed_list",
			content: `call:set{masks:[0x01,0x02,level:5}`,
			want:    map[string]any{"masks": []any{"0x01", "0x02"}, "level": 5.0},
		},
		{
			name:    "text_that_is_not_a_hex_number",
			content: `call:set{a:0x0G}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseGemma4ClosedToolCall(tt.content, nil)
			if tt.want == nil {
				if err == nil {
					t.Fatalf("expected the call to fail, got %v", got.Function.Arguments)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseGemma4ClosedToolCall returned error: %v", err)
			}
			name := tt.content[len("call:"):strings.Index(tt.content, "{")]
			want := api.ToolCall{Function: api.ToolCallFunction{Name: name, Arguments: testArgs(tt.want)}}
			if diff := cmp.Diff(want, got, argsComparer); diff != "" {
				t.Fatalf("tool call mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestGemma4Parser_HexNumberValueInFinishedToolCall(t *testing.T) {
	parser := &Gemma4Parser{hasThinkingSupport: true}
	parser.Init(nil, nil, &api.ThinkValue{Value: true})

	content, _, toolCalls, err := parser.Add("<|tool_call>"+gemma4HexNumberEditRecordCall+"<tool_call|>", true)
	if err != nil {
		t.Fatalf("Add() error = %v", err)
	}
	if content != "" {
		t.Fatalf("expected no content, got %q", content)
	}
	want := []api.ToolCall{{Function: api.ToolCallFunction{Name: "editRecord", Arguments: testArgs(gemma4HexNumberEditRecordArgs)}}}
	if diff := cmp.Diff(want, toolCalls, argsComparer); diff != "" {
		t.Fatalf("tool calls mismatch (-want +got):\n%s", diff)
	}
}

func TestRepairGemma4ColonTokenValues(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "value_after_a_key",
			input:    `{ref:01A2B3:Base.esm}`,
			expected: `{ref:<|"|>01A2B3:Base.esm<|"|>}`,
		},
		{
			name:     "values_in_objects_in_a_list",
			input:    `{value:[{Data:{Level:1,Reference:037C18:Skyrim.esm,Count:1}},{Data:{Reference:2a:b,Count:2}}]}`,
			expected: `{value:[{Data:{Level:1,Reference:<|"|>037C18:Skyrim.esm<|"|>,Count:1}},{Data:{Reference:<|"|>2a:b<|"|>,Count:2}}]}`,
		},
		{
			name:     "values_as_list_items_and_with_spaces",
			input:    `{refs:[1x:y, 0A:B.esp ],"mode" : 9:Name ,n:1}`,
			expected: `{refs:[<|"|>1x:y<|"|>, <|"|>0A:B.esp<|"|> ],"mode" : <|"|>9:Name<|"|> ,n:1}`,
		},
		{
			name:     "keeps_tokens_in_strings",
			input:    `{note:<|"|>0a:b<|"|>,raw:"0a:b"}`,
			expected: `{note:<|"|>0a:b<|"|>,raw:"0a:b"}`,
		},
		{
			name:     "keeps_text_that_is_not_one_token",
			input:    `{a:0b c:d,e:0f::g,h:0i:,j:0k:{l:1},m:12:30,o:0p.:q,r:0s:t u,v:x:y}`,
			expected: `{a:0b c:d,e:0f::g,h:0i:,j:0k:{l:1},m:12:30,o:0p.:q,r:0s:t u,v:x:y}`,
		},
		{
			name:     "keeps_bare_words_and_numbers",
			input:    `{a:Fast,b:12,c:[1, 2]}`,
			expected: `{a:Fast,b:12,c:[1, 2]}`,
		},
		{
			name:     "keeps_unbalanced_delimiters",
			input:    `{a:0x:y,b:<|"|>y}`,
			expected: `{a:0x:y,b:<|"|>y}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := repairGemma4ColonTokenValues(tt.input); got != tt.expected {
				t.Fatalf("repairGemma4ColonTokenValues(%q) = %q, want %q", tt.input, got, tt.expected)
			}
		})
	}
}

// gemma4ColonTokenEditRecordCall sets a list's entries whole, each naming a
// record by a token with colons inside, written bare, beside bare words.
const gemma4ColonTokenEditRecordCall = `call:editRecord{build:true,project:<|"|>BanditSilver<|"|>,record:<|"|>LItemBanditWeapon1H<|"|>,set:[{path:<|"|>Entries<|"|>,value:[{Data:{Level:1,Reference:037C18:Skyrim.esm,Count:1}},{Data:{Level:1,Reference:037C19:Skyrim.esm,Count:1}},{Data:{Level:1,Reference:SilverSword,Count:1}}]}]}`

var gemma4ColonTokenEditRecordArgs = map[string]any{
	"build":   true,
	"project": "BanditSilver",
	"record":  "LItemBanditWeapon1H",
	"set": []any{
		map[string]any{"path": "Entries", "value": []any{
			map[string]any{"Data": map[string]any{"Level": 1.0, "Reference": "037C18:Skyrim.esm", "Count": 1.0}},
			map[string]any{"Data": map[string]any{"Level": 1.0, "Reference": "037C19:Skyrim.esm", "Count": 1.0}},
			map[string]any{"Data": map[string]any{"Level": 1.0, "Reference": "SilverSword", "Count": 1.0}},
		}},
	},
}

func TestParseGemma4ClosedToolCall_ColonTokenValues(t *testing.T) {
	tests := []struct {
		name    string
		content string
		want    map[string]any // nil: the call still fails
	}{
		{
			name:    "values_in_a_list_of_objects",
			content: gemma4ColonTokenEditRecordCall,
			want:    gemma4ColonTokenEditRecordArgs,
		},
		{
			name:    "beside_a_hex_number_a_trailing_comma_and_a_single_quoted_value",
			content: `call:set{a:01:B.esm,b:0x0F,c:'y z',n:5,}`,
			want:    map[string]any{"a": "01:B.esm", "b": "0x0F", "c": "y z", "n": 5.0},
		},
		{
			name:    "a_token_from_a_letter",
			content: `call:set{a:b:c.esm}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseGemma4ClosedToolCall(tt.content, nil)
			if tt.want == nil {
				if err == nil {
					t.Fatalf("expected the call to fail, got %v", got.Function.Arguments)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseGemma4ClosedToolCall returned error: %v", err)
			}
			name := tt.content[len("call:"):strings.Index(tt.content, "{")]
			want := api.ToolCall{Function: api.ToolCallFunction{Name: name, Arguments: testArgs(tt.want)}}
			if diff := cmp.Diff(want, got, argsComparer); diff != "" {
				t.Fatalf("tool call mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestGemma4Parser_ColonTokenValueInFinishedToolCall(t *testing.T) {
	parser := &Gemma4Parser{hasThinkingSupport: true}
	parser.Init(nil, nil, &api.ThinkValue{Value: true})

	content, _, toolCalls, err := parser.Add("<|tool_call>"+gemma4ColonTokenEditRecordCall+"<tool_call|>", true)
	if err != nil {
		t.Fatalf("Add() error = %v", err)
	}
	if content != "" {
		t.Fatalf("expected no content, got %q", content)
	}
	want := []api.ToolCall{{Function: api.ToolCallFunction{Name: "editRecord", Arguments: testArgs(gemma4ColonTokenEditRecordArgs)}}}
	if diff := cmp.Diff(want, toolCalls, argsComparer); diff != "" {
		t.Fatalf("tool calls mismatch (-want +got):\n%s", diff)
	}
}

func TestRepairGemma4FencedValues(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "value_before_the_next_key",
			input:    "{content:---\nEditorID: A\nName: 'B'\n---\n,path:<|\"|>A.yaml<|\"|>}",
			expected: "{content:<|\"|>EditorID: A\nName: 'B'<|\"|>\n,path:<|\"|>A.yaml<|\"|>}",
		},
		{
			name:     "value_before_the_close",
			input:    "{content:---\nA: 1\n---}",
			expected: "{content:<|\"|>A: 1<|\"|>}",
		},
		{
			name:     "line_ends_of_a_return_and_a_newline_and_spaces_after_the_fences",
			input:    "{content:--- \r\nA: {b: c}, d\r\n---  \r\n,path:x}",
			expected: "{content:<|\"|>A: {b: c}, d<|\"|>  \r\n,path:x}",
		},
		{
			name:     "a_fence_line_inside_the_text",
			input:    "{content:---\nA: 1\n---\nB: 2\n---\n,path:x}",
			expected: "{content:<|\"|>A: 1\n---\nB: 2<|\"|>\n,path:x}",
		},
		{
			name:     "no_text",
			input:    "{content:---\n---\n,path:x}",
			expected: "{content:<|\"|><|\"|>\n,path:x}",
		},
		{
			name:     "keeps_values_that_are_not_fenced",
			input:    "{a:---,b:1}",
			expected: "{a:---,b:1}",
		},
		{
			name:     "keeps_text_with_no_closing_fence",
			input:    "{a:---\nb\n----\n,c:1}",
			expected: "{a:---\nb\n----\n,c:1}",
		},
		{
			name:     "keeps_a_fence_followed_by_more_text",
			input:    "{a:---\nb\n--- c\n}",
			expected: "{a:---\nb\n--- c\n}",
		},
		{
			name:     "keeps_a_fence_followed_by_no_key",
			input:    "{a:---\nb\n---\n,c}",
			expected: "{a:---\nb\n---\n,c}",
		},
		{
			name:     "keeps_fences_in_strings",
			input:    "{a:<|\"|>---\nb\n---<|\"|>}",
			expected: "{a:<|\"|>---\nb\n---<|\"|>}",
		},
		{
			name:     "keeps_text_holding_a_delimiter",
			input:    "{a:---\nb <|\"|>c<|\"|>\n---\n,d:1}",
			expected: "{a:---\nb <|\"|>c<|\"|>\n---\n,d:1}",
		},
		{
			name:     "keeps_unbalanced_delimiters",
			input:    "{a:---\nb\n---\n,c:<|\"|>d}",
			expected: "{a:---\nb\n---\n,c:<|\"|>d}",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := repairGemma4FencedValues(tt.input); got != tt.expected {
				t.Fatalf("repairGemma4FencedValues(%q) = %q, want %q", tt.input, got, tt.expected)
			}
		})
	}
}

// gemma4FencedWriteText is a record file's text, with quotes, colons, braces and
// list items among its lines.
const gemma4FencedWriteText = "EditorID: SteadyAim\nName:\n  TargetLanguage: English\n  Value: Steady Aim\nDescription:\n  TargetLanguage: English\n  Value: 'Bows deal 30% more damage while you sneak.'\nLevel: {MutagenObjectType: PcLevelMult, LevelMult: 1}\nEffects:\n- MutagenObjectType: PerkEntryPointModifyValue\n  EntryPoint: ModAttackDamage\n  Modification: Multiply\n  Value: 1.3"

// gemma4FencedWriteCall writes a file whose text is fenced by two `---` lines
// for its delimiters.
const gemma4FencedWriteCall = "call:write{content:---\n" + gemma4FencedWriteText + "\n---\n,path:<|\"|>Perks/SteadyAim.yaml<|\"|>}"

var gemma4FencedWriteArgs = map[string]any{
	"content": gemma4FencedWriteText,
	"path":    "Perks/SteadyAim.yaml",
}

func TestParseGemma4ClosedToolCall_FencedValues(t *testing.T) {
	writeTool := []api.Tool{gemma4TestStringTool("write", "content", "path")}
	tests := []struct {
		name    string
		content string
		tools   []api.Tool
		want    map[string]any // nil: the call still fails
	}{
		{
			name:    "a_file_text_with_no_tools",
			content: gemma4FencedWriteCall,
			want:    gemma4FencedWriteArgs,
		},
		{
			name:    "a_file_text_with_the_tool",
			content: gemma4FencedWriteCall,
			tools:   writeTool,
			want:    gemma4FencedWriteArgs,
		},
		{
			name:    "beside_a_bare_word_and_a_trailing_comma",
			content: "call:set{text:---\nA: 'b'\n---\n,mode:Fast,}",
			want:    map[string]any{"text": "A: 'b'", "mode": "Fast"},
		},
		{
			name:    "a_fence_never_closed",
			content: "call:write{content:---\nA: 1\n,path:<|\"|>x<|\"|>}",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseGemma4ClosedToolCall(tt.content, tt.tools)
			if tt.want == nil {
				if err == nil {
					t.Fatalf("expected the call to fail, got %v", got.Function.Arguments)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseGemma4ClosedToolCall returned error: %v", err)
			}
			name := tt.content[len("call:"):strings.Index(tt.content, "{")]
			want := api.ToolCall{Function: api.ToolCallFunction{Name: name, Arguments: testArgs(tt.want)}}
			if diff := cmp.Diff(want, got, argsComparer); diff != "" {
				t.Fatalf("tool call mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestGemma4Parser_FencedValueInFinishedToolCall(t *testing.T) {
	parser := &Gemma4Parser{hasThinkingSupport: true}
	parser.Init([]api.Tool{gemma4TestStringTool("write", "content", "path")}, nil, &api.ThinkValue{Value: true})

	content, _, toolCalls, err := parser.Add("<|tool_call>"+gemma4FencedWriteCall+"<tool_call|>", true)
	if err != nil {
		t.Fatalf("Add() error = %v", err)
	}
	if content != "" {
		t.Fatalf("expected no content, got %q", content)
	}
	want := []api.ToolCall{{Function: api.ToolCallFunction{Name: "write", Arguments: testArgs(gemma4FencedWriteArgs)}}}
	if diff := cmp.Diff(want, toolCalls, argsComparer); diff != "" {
		t.Fatalf("tool calls mismatch (-want +got):\n%s", diff)
	}
}

func TestRepairGemma4TrailingCommas(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "before_an_object_close",
			input:    `{a:1,set:[],}`,
			expected: `{a:1,set:[]}`,
		},
		{
			name:     "before_a_list_close_and_with_spaces",
			input:    `{a:[1, 2 , ],b:{c:<|"|>x<|"|> , } }`,
			expected: `{a:[1, 2  ],b:{c:<|"|>x<|"|>  } }`,
		},
		{
			name:     "keeps_commas_that_separate",
			input:    `{a:1,b:[2,3],c:{d:4,e:5}}`,
			expected: `{a:1,b:[2,3],c:{d:4,e:5}}`,
		},
		{
			name:     "keeps_commas_in_strings",
			input:    `{a:<|"|>x,}<|"|>,b:"y,]",c:2,}`,
			expected: `{a:<|"|>x,}<|"|>,b:"y,]",c:2}`,
		},
		{
			name:     "keeps_unbalanced_delimiters",
			input:    `{a:1,b:<|"|>y,}`,
			expected: `{a:1,b:<|"|>y,}`,
		},
		{
			name:     "keeps_an_unclosed_json_string",
			input:    `{a:1,b:"y,}`,
			expected: `{a:1,b:"y,}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := repairGemma4TrailingCommas(tt.input); got != tt.expected {
				t.Fatalf("repairGemma4TrailingCommas(%q) = %q, want %q", tt.input, got, tt.expected)
			}
		})
	}
}

// gemma4TrailingCommaEditRecordCall removes fields and sets none, with a comma
// after its last argument.
const gemma4TrailingCommaEditRecordCall = `call:editRecord{build:true,project:<|"|>FeatherStep<|"|>,record:<|"|>MagicEffects/FeatherStepEffect.yaml<|"|>,remove:[<|"|>EffectType<|"|>,<|"|>BaseValue<|"|>,<|"|>Unit<|"|>],set:[],}`

var gemma4TrailingCommaEditRecordArgs = map[string]any{
	"build":   true,
	"project": "FeatherStep",
	"record":  "MagicEffects/FeatherStepEffect.yaml",
	"remove":  []any{"EffectType", "BaseValue", "Unit"},
	"set":     []any{},
}

func TestParseGemma4ClosedToolCall_TrailingCommas(t *testing.T) {
	tests := []struct {
		name    string
		content string
		want    map[string]any // nil: the call still fails
	}{
		{
			name:    "after_the_last_argument",
			content: gemma4TrailingCommaEditRecordCall,
			want:    gemma4TrailingCommaEditRecordArgs,
		},
		{
			name:    "in_lists_and_beside_other_repairs",
			content: `call:set{a:[1,2,],b:'y z',c:Fast,d:0x0F,}`,
			want:    map[string]any{"a": []any{1.0, 2.0}, "b": "y z", "c": "Fast", "d": "0x0F"},
		},
		{
			name:    "in_an_unclosed_list",
			content: `call:set{items:[1,2,`,
			want:    map[string]any{"items": []any{1.0, 2.0}},
		},
		{
			name:    "a_comma_that_is_all_of_a_value",
			content: `call:set{a:,}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseGemma4ClosedToolCall(tt.content, nil)
			if tt.want == nil {
				if err == nil {
					t.Fatalf("expected the call to fail, got %v", got.Function.Arguments)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseGemma4ClosedToolCall returned error: %v", err)
			}
			name := tt.content[len("call:"):strings.Index(tt.content, "{")]
			want := api.ToolCall{Function: api.ToolCallFunction{Name: name, Arguments: testArgs(tt.want)}}
			if diff := cmp.Diff(want, got, argsComparer); diff != "" {
				t.Fatalf("tool call mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestGemma4Parser_TrailingCommaInFinishedToolCall(t *testing.T) {
	parser := &Gemma4Parser{hasThinkingSupport: true}
	parser.Init(nil, nil, &api.ThinkValue{Value: true})

	content, _, toolCalls, err := parser.Add("<|tool_call>"+gemma4TrailingCommaEditRecordCall+"<tool_call|>", true)
	if err != nil {
		t.Fatalf("Add() error = %v", err)
	}
	if content != "" {
		t.Fatalf("expected no content, got %q", content)
	}
	want := []api.ToolCall{{Function: api.ToolCallFunction{Name: "editRecord", Arguments: testArgs(gemma4TrailingCommaEditRecordArgs)}}}
	if diff := cmp.Diff(want, toolCalls, argsComparer); diff != "" {
		t.Fatalf("tool calls mismatch (-want +got):\n%s", diff)
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

func TestRepairGemma4UnclosedBrackets(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "closes_list_and_object_at_end",
			input:    `{edits:[{newText:<|"|>a<|"|>,oldText:<|"|>b<|"|>,path:<|"|>x<|"|>}`,
			expected: `{edits:[{newText:<|"|>a<|"|>,oldText:<|"|>b<|"|>,path:<|"|>x<|"|>}]}`,
		},
		{
			name:     "closes_list_before_a_key",
			input:    `{edits:[{newText:<|"|>a<|"|>,oldText:<|"|>b<|"|>},path:<|"|>x<|"|>}`,
			expected: `{edits:[{newText:<|"|>a<|"|>,oldText:<|"|>b<|"|>}],path:<|"|>x<|"|>}`,
		},
		{
			name:     "closes_list_before_a_quoted_key",
			input:    `{edits:[{a:1},"path":<|"|>x<|"|>}`,
			expected: `{edits:[{a:1}],"path":<|"|>x<|"|>}`,
		},
		{
			name:     "closes_list_where_its_object_closes",
			input:    `{items:[1,2}`,
			expected: `{items:[1,2]}`,
		},
		{
			name:     "keeps_items_of_a_list",
			input:    `{edits:[{a:1},{b:2}`,
			expected: `{edits:[{a:1},{b:2}]}`,
		},
		{
			name:     "keeps_strings_of_a_list",
			input:    `{names:[<|"|>a<|"|>,<|"|>b<|"|>`,
			expected: `{names:[<|"|>a<|"|>,<|"|>b<|"|>]}`,
		},
		{
			name:     "ignores_brackets_in_strings",
			input:    `{text:<|"|>[{,key:<|"|>,raw:"[x:{"`,
			expected: `{text:<|"|>[{,key:<|"|>,raw:"[x:{"}`,
		},
		{
			name:     "drops_a_brace_too_many_before_a_list_close",
			input:    `{set:[{path:<|"|>a<|"|>,value:{b:1}}}]}`,
			expected: `{set:[{path:<|"|>a<|"|>,value:{b:1}}]}`,
		},
		{
			name:     "leaves_balanced_input",
			input:    `{edits:[{a:1}],path:<|"|>x<|"|>}`,
			expected: `{edits:[{a:1}],path:<|"|>x<|"|>}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := repairGemma4UnclosedBrackets(tt.input); got != tt.expected {
				t.Fatalf("repairGemma4UnclosedBrackets(%q) = %q, want %q", tt.input, got, tt.expected)
			}
		})
	}
}

func TestParseGemma4ClosedToolCall_ClosesUnclosedBrackets(t *testing.T) {
	// The shapes of the 30 edit calls gemma4:e4b left unclosed in ollama/ollama#19273, its editRecord call with a
	// brace too many, and #17565's missing brace. The last case also leaves a string unterminated, which has to be
	// closed before the brackets can be.
	tests := []struct {
		name    string
		content string
		want    map[string]any
	}{
		{
			name:    "path_inside_the_item_list_unclosed",
			content: `call:edit{edits:[{newText:<|"|>a: 1<|"|>,oldText:<|"|>a: 0<|"|>,path:<|"|>x.yaml<|"|>}`,
			want: map[string]any{"edits": []any{
				map[string]any{"newText": "a: 1", "oldText": "a: 0", "path": "x.yaml"},
			}},
		},
		{
			name:    "list_unclosed_before_path",
			content: `call:edit{edits:[{newText:<|"|>a: 1<|"|>,oldText:<|"|>a: 0<|"|>},path:<|"|>x.yaml<|"|>}`,
			want: map[string]any{
				"edits": []any{map[string]any{"newText": "a: 1", "oldText": "a: 0"}},
				"path":  "x.yaml",
			},
		},
		{
			name:    "list_and_call_unclosed",
			content: `call:edit{edits:[{newText:<|"|>a: 1<|"|>,oldText:<|"|>a: 0<|"|>}`,
			want: map[string]any{"edits": []any{
				map[string]any{"newText": "a: 1", "oldText": "a: 0"},
			}},
		},
		{
			name:    "two_items_list_unclosed_before_path",
			content: `call:edit{edits:[{newText:<|"|>a<|"|>,oldText:<|"|>b<|"|>},{newText:<|"|>c<|"|>},path:<|"|>x.yaml<|"|>}`,
			want: map[string]any{
				"edits": []any{map[string]any{"newText": "a", "oldText": "b"}, map[string]any{"newText": "c"}},
				"path":  "x.yaml",
			},
		},
		{
			name:    "brace_too_many_before_list_close",
			content: `call:editRecord{record:<|"|>Hilda<|"|>,set:[{path:<|"|>Name<|"|>,value:<|"|>Hilda<|"|>},{path:<|"|>Level<|"|>,value:{LevelMult:1,CalcMinLevel:5}}}]}`,
			want: map[string]any{
				"record": "Hilda",
				"set": []any{
					map[string]any{"path": "Name", "value": "Hilda"},
					map[string]any{"path": "Level", "value": map[string]any{"LevelMult": 1.0, "CalcMinLevel": 5.0}},
				},
			},
		},
		{
			name:    "missing_final_brace",
			content: `call:read_files{files:[{path:<|"|>assets.json<|"|>,start_line:1}]`,
			want: map[string]any{"files": []any{
				map[string]any{"path": "assets.json", "start_line": 1.0},
			}},
		},
		{
			name:    "string_and_list_unclosed",
			content: `call:edit{edits:[{newText:<|"|>a: 1<|"|>,oldText:<|"|>a: 0}`,
			want: map[string]any{"edits": []any{
				map[string]any{"newText": "a: 1", "oldText": "a: 0"},
			}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := parseGemma4ToolCall(tt.content, nil); err == nil {
				t.Fatal("expected the call to fail without the repair")
			}
			got, err := parseGemma4ClosedToolCall(tt.content, nil)
			if err != nil {
				t.Fatalf("parseGemma4ClosedToolCall returned error: %v", err)
			}
			name := tt.content[len("call:"):strings.Index(tt.content, "{")]
			want := api.ToolCall{Function: api.ToolCallFunction{Name: name, Arguments: testArgs(tt.want)}}
			if diff := cmp.Diff(want, got, argsComparer); diff != "" {
				t.Fatalf("tool call mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestGemma4Parser_UnclosedListInFinishedToolCall(t *testing.T) {
	parser := &Gemma4Parser{hasThinkingSupport: true}
	parser.Init(nil, nil, &api.ThinkValue{Value: true})

	content, _, toolCalls, err := parser.Add(`<|tool_call>call:edit{edits:[{newText:<|"|>a<|"|>,oldText:<|"|>b<|"|>},path:<|"|>x.yaml<|"|>}<tool_call|>`, true)
	if err != nil {
		t.Fatalf("Add() error = %v", err)
	}
	if content != "" {
		t.Fatalf("expected no content, got %q", content)
	}
	want := []api.ToolCall{{Function: api.ToolCallFunction{Name: "edit", Arguments: testArgs(map[string]any{
		"edits": []any{map[string]any{"newText": "a", "oldText": "b"}},
		"path":  "x.yaml",
	})}}}
	if diff := cmp.Diff(want, toolCalls, argsComparer); diff != "" {
		t.Fatalf("tool calls mismatch (-want +got):\n%s", diff)
	}
}

func TestGemma4Parser_UnparsableToolCallIsContent(t *testing.T) {
	tests := []struct {
		name            string
		input           string
		expectedContent string
	}{
		{
			// Finished, and still not valid once its brackets are closed: the client sees it.
			name:            "finished_call",
			input:           `<|tool_call>call:edit{edits:[{newText:<|"|>a<|"|>,oldText}<tool_call|>`,
			expectedContent: `call:edit{edits:[{newText:<|"|>a<|"|>,oldText}`,
		},
		{
			// Flushed on done, with no closing tag, it may have been cut off: not repaired, and passed on.
			name:            "flushed_on_done",
			input:           `<|tool_call>call:edit{edits:[{newText:<|"|>a<|"|>,oldText:<|"|>b<|"|>}`,
			expectedContent: `call:edit{edits:[{newText:<|"|>a<|"|>,oldText:<|"|>b<|"|>}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := &Gemma4Parser{hasThinkingSupport: true}
			parser.Init(nil, nil, &api.ThinkValue{Value: true})

			content, _, toolCalls, err := parser.Add(tt.input, true)
			if err != nil {
				t.Fatalf("Add() error = %v", err)
			}
			if len(toolCalls) != 0 {
				t.Fatalf("expected no tool calls, got %v", toolCalls)
			}
			if content != tt.expectedContent {
				t.Fatalf("content = %q, want %q", content, tt.expectedContent)
			}
		})
	}
}
