package parsers

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
)

// TestNemotron3NanoParser tests Nemotron-specific behavior (thinking support).
// Tool call parsing is tested in qwen3coder_test.go since Nemotron delegates to Qwen3CoderParser.
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
			name:             "thinking then tool call",
			input:            "I should check the weather...</think>\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
			thinkValue:       &api.ThinkValue{Value: true},
			expectedThinking: "I should check the weather...",
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "get_weather",
						Arguments: testArgs(map[string]any{"city": "Paris"}),
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
						Arguments: testArgs(map[string]any{"query": "test"}),
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
			name:             "thinking with only whitespace after close tag",
			input:            "My thoughts...</think>   \n\t\n   Content here.",
			thinkValue:       &api.ThinkValue{Value: true},
			expectedThinking: "My thoughts...",
			expectedContent:  "Content here.",
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
			if diff := cmp.Diff(calls, tt.expectedCalls, argsComparer); diff != "" {
				t.Errorf("calls mismatch (-got +want):\n%s", diff)
			}
		})
	}
}

// TestNemotron3NanoParser_Streaming tests streaming behavior for thinking support.
// Tool call streaming is tested in qwen3coder_test.go.
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
			name:             "thinking then content then tool call - streaming",
			chunks:           []string{"Ana", "lyzing", " your", " request", "...", "</", "think", ">\n", "I'll", " check", " that", " for", " you", ".", "\n", "<tool", "_call", ">\n", "<function", "=search", ">\n", "<parameter", "=query", ">\n", "test", " query", "\n</", "parameter", ">\n", "</function", ">\n", "</tool", "_call", ">"},
			thinkValue:       &api.ThinkValue{Value: true},
			expectedThinking: "Analyzing your request...",
			expectedContent:  "I'll check that for you.",
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "search",
						Arguments: testArgs(map[string]any{"query": "test query"}),
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
			name:             "tool call immediately after think close - no content",
			chunks:           []string{"Analyzing...", "</think>", "\n", "<tool_call>", "\n<function=test>\n</function>\n", "</tool_call>"},
			thinkValue:       &api.ThinkValue{Value: true},
			expectedThinking: "Analyzing...",
			expectedCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "test",
						Arguments: api.NewToolCallFunctionArguments(),
					},
				},
			},
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
			if diff := cmp.Diff(allCalls, tt.expectedCalls, argsComparer); diff != "" {
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
					Properties: testPropsMap(map[string]api.ToolProperty{
						"city": {Type: api.PropertyType{"string"}},
					}),
				},
			},
		},
	}

	p := &Nemotron3NanoParser{}
	returnedTools := p.Init(tools, nil, nil)

	if diff := cmp.Diff(returnedTools, tools, toolsComparer); diff != "" {
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
				Arguments: testArgs(map[string]any{"city": "Paris"}),
			},
		},
	}

	if diff := cmp.Diff(calls, expectedCalls, argsComparer); diff != "" {
		t.Errorf("calls mismatch (-got +want):\n%s", diff)
	}
}

// TestNemotron3NanoParser_ToolCallWithoutThinkClose tests the case where thinking is enabled
// but the model outputs content + tool call WITHOUT the </think> tag.
// The parser should still parse the tool call (content before is treated as thinking).
func TestNemotron3NanoParser_ToolCallWithoutThinkClose(t *testing.T) {
	chunks := []string{
		"Let", " me", " analyze", " this", ".", "\n",
		"<tool_call>", "\n",
		"<function=get_weather>", "\n",
		"<parameter=city>", "Paris", "</parameter>", "\n",
		"</function>", "\n",
		"</tool_call>",
	}

	p := &Nemotron3NanoParser{}
	p.Init(nil, nil, &api.ThinkValue{Value: true}) // thinking ENABLED but model doesn't output </think>

	var allContent string
	var allThinking string
	var allCalls []api.ToolCall

	for _, chunk := range chunks {
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

	// The parser was in thinking mode, so text before <tool_call> is emitted as thinking.
	expectedThinking := "Let me analyze this."

	expectedCalls := []api.ToolCall{
		{
			Function: api.ToolCallFunction{
				Name:      "get_weather",
				Arguments: testArgs(map[string]any{"city": "Paris"}),
			},
		},
	}

	if allContent != "" {
		t.Errorf("expected no content (text was streamed as thinking), got: %q", allContent)
	}
	if diff := cmp.Diff(allThinking, expectedThinking); diff != "" {
		t.Errorf("thinking mismatch (-got +want):\n%s", diff)
	}
	if diff := cmp.Diff(allCalls, expectedCalls, argsComparer); diff != "" {
		t.Errorf("calls mismatch (-got +want):\n%s", diff)
	}
}
