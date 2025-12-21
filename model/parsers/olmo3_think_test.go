package parsers

import (
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
)

func TestOlmo3ThinkParser(t *testing.T) {
	tests := []struct {
		name             string
		input            string
		expectedContent  string
		expectedThinking string
		lastMessage      *api.Message
	}{
		{
			name:             "thinking_only",
			input:            "I need to think about this.</think>Here is my response.",
			expectedContent:  "Here is my response.",
			expectedThinking: "I need to think about this.",
		},
		{
			name:             "thinking_with_newlines",
			input:            "Let me think step by step.\n\n1. First point\n2. Second point</think>The answer is 42.",
			expectedContent:  "The answer is 42.",
			expectedThinking: "Let me think step by step.\n\n1. First point\n2. Second point",
		},
		{
			name:             "thinking_then_content",
			input:            "Deep thinking here.</think>Here is my detailed response with multiple sentences. I have thought carefully.",
			expectedContent:  "Here is my detailed response with multiple sentences. I have thought carefully.",
			expectedThinking: "Deep thinking here.",
		},
		{
			name:             "empty_thinking",
			input:            "</think>Just content here.",
			expectedContent:  "Just content here.",
			expectedThinking: "",
		},
		{
			name:            "prefill_skips_thinking",
			input:           "Continuing from previous content.",
			expectedContent: "Continuing from previous content.",
			lastMessage: &api.Message{
				Role:    "assistant",
				Content: "Previous content",
			},
		},
		{
			name:             "thinking_with_whitespace",
			input:            "  Some thinking  </think>  Content here  ",
			expectedContent:  "Content here  ",
			expectedThinking: "  Some thinking",
		},
		{
			name:             "real_model_output_with_newlines",
			input:            "Yes, that should work. Let me go with that response.\n\n</think>\n\nHi! I'm all set and ready to assist. How about you? How are you today? 😊",
			expectedThinking: "Yes, that should work. Let me go with that response.",
			expectedContent:  "Hi! I'm all set and ready to assist. How about you? How are you today? 😊",
		},
		// Edge cases
		{
			name:             "nested_think_tags_in_thinking",
			input:            "I'm thinking <think>nested</think> more thinking</think>Final content.",
			expectedContent:  "more thinking</think>Final content.",
			expectedThinking: "I'm thinking <think>nested",
		},
		{
			name:             "multiple_think_close_tags",
			input:            "First thinking</think>Content</think>More content.",
			expectedContent:  "Content</think>More content.",
			expectedThinking: "First thinking",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := &Olmo3ThinkParser{}
			parser.Init(nil, tt.lastMessage, nil)

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

			// No tool calls expected
			if len(toolCalls) > 0 {
				t.Errorf("expected no tool calls, got %d", len(toolCalls))
			}
		})
	}
}

func TestOlmo3ThinkParser_Streaming(t *testing.T) {
	parser := &Olmo3ThinkParser{}
	parser.Init(nil, nil, nil)

	chunks := []string{
		"I am ",
		"thinking about",
		" this.</think>Here ",
		"is the response.",
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

	expectedContent := "Here is the response."
	expectedThinking := "I am thinking about this."

	if finalContent.String() != expectedContent {
		t.Errorf("expected content %q, got %q", expectedContent, finalContent.String())
	}

	if finalThinking.String() != expectedThinking {
		t.Errorf("expected thinking %q, got %q", expectedThinking, finalThinking.String())
	}
}

func TestOlmo3ThinkParser_StreamingEdgeCases(t *testing.T) {
	tests := []struct {
		name             string
		chunks           []string
		expectedContent  string
		expectedThinking string
	}{
		{
			name: "thinking_tag_split_across_chunks",
			chunks: []string{
				"This is thinking content",
				"</think>",
				"This is content.",
			},
			expectedContent:  "This is content.",
			expectedThinking: "This is thinking content",
		},
		{
			name: "thinking_tag_split_mid_token",
			chunks: []string{
				"Thinking?</",
				"think>",
				"Content here.",
			},
			expectedContent:  "Content here.",
			expectedThinking: "Thinking?",
		},
		{
			name: "thinking_tag_split_at_angle_bracket",
			chunks: []string{
				"Thinking<",
				"/think>",
				"Content.",
			},
			expectedContent:  "Content.",
			expectedThinking: "Thinking",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := &Olmo3ThinkParser{}
			parser.Init(nil, nil, nil)

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

// TestOlmo3ThinkParser_ThinkBoundary tests streaming thinking content
// where thinking chunks come in succession before the </think> tag
func TestOlmo3ThinkParser_ThinkBoundary(t *testing.T) {
	tests := []struct {
		name             string
		chunks           []string
		expectedThinking string
		expectedContent  string
	}{
		{
			name: "multiple_thinking_chunks",
			chunks: []string{
				"First part of thinking. ",
				"Second part of thinking. ",
				"Third part.</think>",
				"Content here.",
			},
			expectedThinking: "First part of thinking. Second part of thinking. Third part.",
			expectedContent:  "Content here.",
		},
		{
			name: "thinking_chunks_with_newlines",
			chunks: []string{
				"Step 1: Analyze the problem.\n",
				"Step 2: Consider options.\n",
				"Step 3: Make decision.</think>",
				"Here is my answer.",
			},
			expectedThinking: "Step 1: Analyze the problem.\nStep 2: Consider options.\nStep 3: Make decision.",
			expectedContent:  "Here is my answer.",
		},
		{
			name: "single_char_thinking_chunks",
			chunks: []string{
				"H", "e", "l", "l", "o", "</think>", "World",
			},
			expectedThinking: "Hello",
			expectedContent:  "World",
		},
		{
			name: "thinking_with_special_chars",
			chunks: []string{
				"Let me think... ",
				"Option A: $100 ",
				"Option B: €200</think>",
				"I recommend Option A.",
			},
			expectedThinking: "Let me think... Option A: $100 Option B: €200",
			expectedContent:  "I recommend Option A.",
		},
		{
			name: "long_thinking_multiple_chunks",
			chunks: []string{
				"This is a very long thinking process. ",
				"I need to consider many factors. ",
				"First, let me look at the data. ",
				"The numbers show interesting patterns. ",
				"Based on my analysis, ",
				"I can conclude that...</think>",
				"The answer is 42.",
			},
			expectedThinking: "This is a very long thinking process. I need to consider many factors. First, let me look at the data. The numbers show interesting patterns. Based on my analysis, I can conclude that...",
			expectedContent:  "The answer is 42.",
		},
		{
			name: "thinking_ends_exactly_at_chunk_boundary",
			chunks: []string{
				"Thinking content",
				"</think>",
				"Content",
			},
			expectedThinking: "Thinking content",
			expectedContent:  "Content",
		},
		{
			name: "empty_chunks_between_thinking",
			chunks: []string{
				"Start thinking",
				"",
				" middle ",
				"",
				"end</think>",
				"Content",
			},
			expectedThinking: "Start thinking middle end",
			expectedContent:  "Content",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := &Olmo3ThinkParser{}
			parser.Init(nil, nil, nil)

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

			if finalThinking.String() != tt.expectedThinking {
				t.Errorf("thinking mismatch:\nexpected: %q\ngot:      %q", tt.expectedThinking, finalThinking.String())
			}

			if finalContent.String() != tt.expectedContent {
				t.Errorf("content mismatch:\nexpected: %q\ngot:      %q", tt.expectedContent, finalContent.String())
			}
		})
	}
}

// TestOlmo3ThinkParser_StateTransitions tests that state transitions work correctly
func TestOlmo3ThinkParser_StateTransitions(t *testing.T) {
	t.Run("thinking_to_content", func(t *testing.T) {
		parser := &Olmo3ThinkParser{}
		parser.Init(nil, nil, nil)

		if parser.state != olmo3CollectingThink {
			t.Errorf("initial state should be olmo3CollectingThink, got %v", parser.state)
		}

		parser.Add("thinking</think>content", true)

		if parser.state != olmo3CollectingContent {
			t.Errorf("state after </think> should be olmo3CollectingContent, got %v", parser.state)
		}
	})
}

func TestOlmo3ThinkParser_HasToolSupport(t *testing.T) {
	parser := &Olmo3ThinkParser{}
	if parser.HasToolSupport() {
		t.Error("Olmo3ThinkParser should NOT support tools")
	}
}

func TestOlmo3ThinkParser_HasThinkingSupport(t *testing.T) {
	parser := &Olmo3ThinkParser{}
	if !parser.HasThinkingSupport() {
		t.Error("Olmo3ThinkParser should support thinking")
	}
}

func TestOlmo3ThinkParser_Init(t *testing.T) {
	parser := &Olmo3ThinkParser{}

	tools := []api.Tool{
		{Function: api.ToolFunction{Name: "test_tool"}},
	}

	lastMessage := &api.Message{Role: "assistant", Content: "previous"}

	returnedTools := parser.Init(tools, lastMessage, nil)

	if len(returnedTools) != len(tools) {
		t.Errorf("expected %d tools returned, got %d", len(tools), len(returnedTools))
	}

	// Should be in content collection mode due to prefill
	if parser.state != olmo3CollectingContent {
		t.Errorf("expected state olmo3CollectingContent, got %v", parser.state)
	}
}

func TestOlmo3ThinkParser_InitWithoutPrefill(t *testing.T) {
	parser := &Olmo3ThinkParser{}

	parser.Init(nil, nil, nil)

	// Should be in thinking collection mode (model always thinks first)
	if parser.state != olmo3CollectingThink {
		t.Errorf("expected state olmo3CollectingThink, got %v", parser.state)
	}
}
