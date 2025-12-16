package parsers

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
)

func TestNemotron3NanoParser(t *testing.T) {
	tests := []struct {
		name             string
		input            string
		thinkValue       *api.ThinkValue
		expectedContent  string
		expectedThinking string
		expectedCalls    []api.ToolCall
	}{
		{
			name:            "simple content - no thinking",
			input:           "Hello, how can I help you?",
			thinkValue:      nil,
			expectedContent: "Hello, how can I help you?",
		},
		{
			name:            "simple content - thinking disabled",
			input:           "Hello, how can I help you?",
			thinkValue:      &api.ThinkValue{Value: false},
			expectedContent: "Hello, how can I help you?",
		},
		{
			name:             "thinking then content",
			input:            "Let me think about this...</think>\nHere is my answer.",
			thinkValue:       &api.ThinkValue{Value: true},
			expectedThinking: "Let me think about this...",
			expectedContent:  "Here is my answer.",
		},
		{
			name:             "thinking with newlines",
			input:            "Step 1: Analyze\nStep 2: Process\nStep 3: Conclude</think>\nThe answer is 42.",
			thinkValue:       &api.ThinkValue{Value: true},
			expectedThinking: "Step 1: Analyze\nStep 2: Process\nStep 3: Conclude",
			expectedContent:  "The answer is 42.",
		},
		{
			name:       "simple tool call",
			input:      "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
			thinkValue: nil,
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "get_weather",
						Arguments: map[string]any{"city": "Paris"},
					},
				},
			},
		},
		{
			name:            "content then tool call",
			input:           "Let me check the weather.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nNYC\n</parameter>\n</function>\n</tool_call>",
			thinkValue:      nil,
			expectedContent: "Let me check the weather.",
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "get_weather",
						Arguments: map[string]any{"city": "NYC"},
					},
				},
			},
		},
		{
			name:       "tool call with multiple parameters",
			input:      "<tool_call>\n<function=book_flight>\n<parameter=from>\nSFO\n</parameter>\n<parameter=to>\nNYC\n</parameter>\n</function>\n</tool_call>",
			thinkValue: nil,
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "book_flight",
						Arguments: map[string]any{
							"from": "SFO",
							"to":   "NYC",
						},
					},
				},
			},
		},
		{
			name: "multiple tool calls",
			input: "<tool_call>\n<function=get_weather>\n<parameter=city>\nSan Francisco\n</parameter>\n</function>\n</tool_call>\n" +
				"<tool_call>\n<function=get_weather>\n<parameter=city>\nNew York\n</parameter>\n</function>\n</tool_call>",
			thinkValue: nil,
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "get_weather",
						Arguments: map[string]any{"city": "San Francisco"},
					},
				},
				{
					Function: api.ToolCallFunction{
						Name:      "get_weather",
						Arguments: map[string]any{"city": "New York"},
					},
				},
			},
		},
		{
			name:             "thinking then tool call",
			input:            "I should check the weather...</think>\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
			thinkValue:       &api.ThinkValue{Value: true},
			expectedThinking: "I should check the weather...",
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "get_weather",
						Arguments: map[string]any{"city": "Paris"},
					},
				},
			},
		},
		{
			name:             "thinking content then tool call",
			input:            "Let me think...</think>\nI'll check for you.\n<tool_call>\n<function=search>\n<parameter=query>\ntest\n</parameter>\n</function>\n</tool_call>",
			thinkValue:       &api.ThinkValue{Value: true},
			expectedThinking: "Let me think...",
			expectedContent:  "I'll check for you.",
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "search",
						Arguments: map[string]any{"query": "test"},
					},
				},
			},
		},
		{
			name:       "tool call with multiline parameter value",
			input:      "<tool_call>\n<function=create_note>\n<parameter=content>\nLine 1\nLine 2\nLine 3\n</parameter>\n</function>\n</tool_call>",
			thinkValue: nil,
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "create_note",
						Arguments: map[string]any{"content": "Line 1\nLine 2\nLine 3"},
					},
				},
			},
		},
		{
			name:             "empty thinking block - immediate close",
			input:            "</think>\nHere is my answer.",
			thinkValue:       &api.ThinkValue{Value: true},
			expectedThinking: "",
			expectedContent:  "Here is my answer.",
		},
		{
			name:            "thinking disabled but model outputs think close anyway",
			input:           "</think>\nSome content after spurious tag.",
			thinkValue:      &api.ThinkValue{Value: false},
			expectedContent: "</think>\nSome content after spurious tag.",
		},
		{
			name:          "tool call with no function name - returns empty tool call",
			input:         "<tool_call>\n<function=>\n</function>\n</tool_call>",
			thinkValue:    nil,
			expectedCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "", Arguments: nil}}},
		},
		{
			name:            "content with newlines preserved",
			input:           "Line 1\n\nLine 2\n\n\nLine 3",
			thinkValue:      nil,
			expectedContent: "Line 1\n\nLine 2\n\n\nLine 3",
		},
		{
			name:             "thinking with only whitespace after close tag",
			input:            "My thoughts...</think>   \n\t\n   Content here.",
			thinkValue:       &api.ThinkValue{Value: true},
			expectedThinking: "My thoughts...",
			expectedContent:  "Content here.",
		},
		{
			name:            "unicode content",
			input:           "Hello 世界! 🌍 Ñoño",
			thinkValue:      nil,
			expectedContent: "Hello 世界! 🌍 Ñoño",
		},
		{
			name:       "tool call with numeric parameter",
			input:      "<tool_call>\n<function=set_temp>\n<parameter=value>\n42\n</parameter>\n</function>\n</tool_call>",
			thinkValue: nil,
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "set_temp",
						Arguments: map[string]any{"value": "42"},
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &Nemotron3NanoParser{}
			p.Init(nil, nil, tt.thinkValue)

			content, thinking, calls, err := p.Add(tt.input, false)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			// Drain remaining content
			finalContent, finalThinking, finalCalls, err := p.Add("", true)
			if err != nil {
				t.Fatalf("unexpected error on done: %v", err)
			}
			content += finalContent
			thinking += finalThinking
			calls = append(calls, finalCalls...)

			if diff := cmp.Diff(content, tt.expectedContent); diff != "" {
				t.Errorf("content mismatch (-got +want):\n%s", diff)
			}
			if diff := cmp.Diff(thinking, tt.expectedThinking); diff != "" {
				t.Errorf("thinking mismatch (-got +want):\n%s", diff)
			}
			if diff := cmp.Diff(calls, tt.expectedCalls); diff != "" {
				t.Errorf("calls mismatch (-got +want):\n%s", diff)
			}
		})
	}
}

func TestNemotron3NanoParser_Streaming(t *testing.T) {
	tests := []struct {
		name             string
		chunks           []string
		thinkValue       *api.ThinkValue
		expectedContent  string
		expectedThinking string
		expectedCalls    []api.ToolCall
	}{
		{
			name:            "streaming content character by character",
			chunks:          []string{"H", "e", "l", "l", "o", ",", " ", "w", "o", "r", "l", "d", "!"},
			thinkValue:      nil,
			expectedContent: "Hello, world!",
		},
		{
			name:            "streaming content small tokens",
			chunks:          []string{"Hel", "lo", ", ", "how ", "can", " I", " help", " you", " today", "?"},
			thinkValue:      nil,
			expectedContent: "Hello, how can I help you today?",
		},
		{
			name:             "streaming thinking then content - granular",
			chunks:           []string{"Let", " me", " th", "ink", " about", " this", "...", "<", "/", "think", ">", "\n", "Here", " is", " my", " answer", "."},
			thinkValue:       &api.ThinkValue{Value: true},
			expectedThinking: "Let me think about this...",
			expectedContent:  "Here is my answer.",
		},
		{
			name:             "streaming thinking with newlines - granular",
			chunks:           []string{"Step", " 1", ":", " Ana", "lyze\n", "Step", " 2", ":", " Pro", "cess", "</", "thi", "nk>", "\n", "The", " ans", "wer."},
			thinkValue:       &api.ThinkValue{Value: true},
			expectedThinking: "Step 1: Analyze\nStep 2: Process",
			expectedContent:  "The answer.",
		},
		{
			name:       "streaming tool call - highly granular",
			chunks:     []string{"<", "tool", "_", "call", ">", "\n", "<", "func", "tion", "=", "get", "_", "weather", ">", "\n", "<", "param", "eter", "=", "city", ">", "\n", "Par", "is", "\n", "</", "param", "eter", ">", "\n", "</", "func", "tion", ">", "\n", "</", "tool", "_", "call", ">"},
			thinkValue: nil,
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "get_weather",
						Arguments: map[string]any{"city": "Paris"},
					},
				},
			},
		},
		{
			name:            "streaming content then tool call - granular",
			chunks:          []string{"Let", " me", " check", " the", " weather", ".", "\n<", "tool_call", ">", "\n", "<function=", "get_weather", ">", "\n", "<parameter=", "city", ">", "\n", "NYC", "\n", "</parameter>", "\n", "</function>", "\n", "</tool_call>"},
			thinkValue:      nil,
			expectedContent: "Let me check the weather.",
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "get_weather",
						Arguments: map[string]any{"city": "NYC"},
					},
				},
			},
		},
		{
			name:   "tool call tag split character by character",
			chunks: []string{"<", "t", "o", "o", "l", "_", "c", "a", "l", "l", ">", "\n", "<", "f", "u", "n", "c", "t", "i", "o", "n", "=", "t", "e", "s", "t", ">", "\n", "<", "/", "f", "u", "n", "c", "t", "i", "o", "n", ">", "\n", "<", "/", "t", "o", "o", "l", "_", "c", "a", "l", "l", ">"},
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "test",
						Arguments: map[string]any{},
					},
				},
			},
		},
		{
			name:             "thinking close tag split character by character",
			chunks:           []string{"I", "'", "m", " ", "t", "h", "i", "n", "k", "i", "n", "g", ".", ".", ".", "<", "/", "t", "h", "i", "n", "k", ">", "\n", "D", "o", "n", "e", "!"},
			thinkValue:       &api.ThinkValue{Value: true},
			expectedThinking: "I'm thinking...",
			expectedContent:  "Done!",
		},
		{
			name:             "multiple whitespace after think tag - separate chunks",
			chunks:           []string{"Thinking...", "</think>", "\n", "\n", " ", "Content here."},
			thinkValue:       &api.ThinkValue{Value: true},
			expectedThinking: "Thinking...",
			expectedContent:  "Content here.",
		},
		{
			name:       "tool call with multiple parameters - streaming",
			chunks:     []string{"<tool_", "call>\n", "<function", "=book_", "flight>", "\n<para", "meter=", "from>\n", "SFO\n", "</param", "eter>", "\n<param", "eter=to", ">\nNYC", "\n</para", "meter>", "\n</func", "tion>\n", "</tool_", "call>"},
			thinkValue: nil,
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name: "book_flight",
						Arguments: map[string]any{
							"from": "SFO",
							"to":   "NYC",
						},
					},
				},
			},
		},
		{
			name:             "thinking then content then tool call - streaming",
			chunks:           []string{"Ana", "lyzing", " your", " request", "...", "</", "think", ">\n", "I'll", " check", " that", " for", " you", ".", "\n", "<tool", "_call", ">\n", "<function", "=search", ">\n", "<parameter", "=query", ">\n", "test", " query", "\n</", "parameter", ">\n", "</function", ">\n", "</tool", "_call", ">"},
			thinkValue:       &api.ThinkValue{Value: true},
			expectedThinking: "Analyzing your request...",
			expectedContent:  "I'll check that for you.",
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "search",
						Arguments: map[string]any{"query": "test query"},
					},
				},
			},
		},
		{
			name: "multiple tool calls - streaming",
			chunks: []string{
				"<tool_call>", "\n", "<function=", "get_weather>", "\n",
				"<parameter=", "city>\n", "San Fran", "cisco\n", "</parameter>", "\n",
				"</function>", "\n", "</tool_call>", "\n",
				"<tool_", "call>\n", "<function", "=get_weather", ">\n",
				"<param", "eter=city", ">\nNew", " York\n", "</parameter>\n",
				"</function>\n", "</tool_call>",
			},
			thinkValue: nil,
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "get_weather",
						Arguments: map[string]any{"city": "San Francisco"},
					},
				},
				{
					Function: api.ToolCallFunction{
						Name:      "get_weather",
						Arguments: map[string]any{"city": "New York"},
					},
				},
			},
		},
		{
			name:       "tool call with multiline parameter - streaming",
			chunks:     []string{"<tool_call>\n", "<function=", "create_note>\n", "<parameter=", "content>\n", "Line 1", "\nLine", " 2\n", "Line 3", "\n</parameter>\n", "</function>\n", "</tool_call>"},
			thinkValue: nil,
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "create_note",
						Arguments: map[string]any{"content": "Line 1\nLine 2\nLine 3"},
					},
				},
			},
		},
		{
			name:             "empty thinking block",
			chunks:           []string{"</think>", "\n", "Just content."},
			thinkValue:       &api.ThinkValue{Value: true},
			expectedThinking: "",
			expectedContent:  "Just content.",
		},
		{
			name:            "empty input chunks interspersed",
			chunks:          []string{"Hello", "", " ", "", "world", "", "!"},
			thinkValue:      nil,
			expectedContent: "Hello world!",
		},
		{
			name:             "tool call immediately after think close - no content",
			chunks:           []string{"Analyzing...", "</think>", "\n", "<tool_call>", "\n<function=test>\n</function>\n", "</tool_call>"},
			thinkValue:       &api.ThinkValue{Value: true},
			expectedThinking: "Analyzing...",
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "test",
						Arguments: map[string]any{},
					},
				},
			},
		},
		{
			name:       "tool call with empty parameter value",
			chunks:     []string{"<tool_call>\n<function=test>\n<parameter=name>\n", "\n</parameter>\n</function>\n</tool_call>"},
			thinkValue: nil,
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "test",
						Arguments: map[string]any{"name": ""},
					},
				},
			},
		},
		{
			name:            "partial tool call tag at end - buffered",
			chunks:          []string{"Here's some content", "<tool"},
			thinkValue:      nil,
			expectedContent: "Here's some content",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &Nemotron3NanoParser{}
			p.Init(nil, nil, tt.thinkValue)

			var allContent string
			var allThinking string
			var allCalls []api.ToolCall

			for _, chunk := range tt.chunks {
				content, thinking, calls, err := p.Add(chunk, false)
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				allContent += content
				allThinking += thinking
				allCalls = append(allCalls, calls...)
			}

			// Drain
			content, thinking, calls, err := p.Add("", true)
			if err != nil {
				t.Fatalf("unexpected error on done: %v", err)
			}
			allContent += content
			allThinking += thinking
			allCalls = append(allCalls, calls...)

			if diff := cmp.Diff(allContent, tt.expectedContent); diff != "" {
				t.Errorf("content mismatch (-got +want):\n%s", diff)
			}
			if diff := cmp.Diff(allThinking, tt.expectedThinking); diff != "" {
				t.Errorf("thinking mismatch (-got +want):\n%s", diff)
			}
			if diff := cmp.Diff(allCalls, tt.expectedCalls); diff != "" {
				t.Errorf("calls mismatch (-got +want):\n%s", diff)
			}
		})
	}
}

func TestNemotron3NanoParser_HasToolSupport(t *testing.T) {
	p := &Nemotron3NanoParser{}
	if !p.HasToolSupport() {
		t.Error("expected HasToolSupport to return true")
	}
}

func TestNemotron3NanoParser_HasThinkingSupport(t *testing.T) {
	p := &Nemotron3NanoParser{}
	if !p.HasThinkingSupport() {
		t.Error("expected HasThinkingSupport to return true")
	}
}

func TestNemotron3NanoParser_Init(t *testing.T) {
	t.Run("starts in thinking state when enabled", func(t *testing.T) {
		p := &Nemotron3NanoParser{}
		p.Init(nil, nil, &api.ThinkValue{Value: true})
		if p.state != Nemotron3NanoCollectingThinking {
			t.Errorf("expected state Nemotron3NanoCollectingThinking, got %v", p.state)
		}
	})

	t.Run("starts in content state when thinking disabled", func(t *testing.T) {
		p := &Nemotron3NanoParser{}
		p.Init(nil, nil, &api.ThinkValue{Value: false})
		if p.state != Nemotron3NanoCollectingContent {
			t.Errorf("expected state Nemotron3NanoCollectingContent, got %v", p.state)
		}
	})

	t.Run("starts in content state when nil thinkValue", func(t *testing.T) {
		p := &Nemotron3NanoParser{}
		p.Init(nil, nil, nil)
		if p.state != Nemotron3NanoCollectingContent {
			t.Errorf("expected state Nemotron3NanoCollectingContent, got %v", p.state)
		}
	})

	t.Run("starts in content state with assistant prefill", func(t *testing.T) {
		p := &Nemotron3NanoParser{}
		prefill := &api.Message{Role: "assistant", Content: "Starting..."}
		p.Init(nil, prefill, &api.ThinkValue{Value: true})
		if p.state != Nemotron3NanoCollectingContent {
			t.Errorf("expected state Nemotron3NanoCollectingContent, got %v", p.state)
		}
	})
}

func TestNemotron3NanoParser_WithTools(t *testing.T) {
	tools := []api.Tool{
		{
			Type: "function",
			Function: api.ToolFunction{
				Name: "get_weather",
				Parameters: api.ToolFunctionParameters{
					Type: "object",
					Properties: map[string]api.ToolProperty{
						"city": {Type: api.PropertyType{"string"}},
					},
				},
			},
		},
	}

	p := &Nemotron3NanoParser{}
	returnedTools := p.Init(tools, nil, nil)

	if diff := cmp.Diff(returnedTools, tools); diff != "" {
		t.Errorf("tools mismatch (-got +want):\n%s", diff)
	}

	// Parse a tool call
	input := "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"
	_, _, calls, err := p.Add(input, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expectedCalls := []api.ToolCall{
		{
			Function: api.ToolCallFunction{
				Name:      "get_weather",
				Arguments: map[string]any{"city": "Paris"},
			},
		},
	}

	if diff := cmp.Diff(calls, expectedCalls); diff != "" {
		t.Errorf("calls mismatch (-got +want):\n%s", diff)
	}
}
