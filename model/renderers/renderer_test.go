package renderers

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

type mockRenderer struct{}

func (m *mockRenderer) Render(msgs []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
	return "mock-output", nil
}

func (m *mockRenderer) LeadingBOS() string {
	return ""
}

func TestRegisterCustomRenderer(t *testing.T) {
	// Register a custom renderer
	Register("custom-renderer", func() Renderer {
		return &mockRenderer{}
	})

	// Retrieve and use it
	result, err := RenderWithRenderer("custom-renderer", nil, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "mock-output" {
		t.Errorf("expected 'mock-output', got %q", result)
	}
}

func TestBuiltInRendererStillWorks(t *testing.T) {
	tests := []struct {
		name string
	}{
		{name: "qwen3-coder"},
		{name: "qwen3.5"},
	}

	messages := []api.Message{
		{Role: "user", Content: "Hello"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := RenderWithRenderer(tt.name, messages, nil, nil)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if result == "" {
				t.Fatalf("expected non-empty result from %s renderer", tt.name)
			}
		})
	}
}

func TestLeadingBOSForRenderer(t *testing.T) {
	tests := []struct {
		name string
		want string
	}{
		{name: "gemma4", want: "<bos>"},
		{name: "gemma4-small", want: "<bos>"},
		{name: "gemma4-large", want: "<bos>"},
		{name: "functiongemma", want: "<bos>"},
		{name: "lfm2", want: "<|startoftext|>"},
		{name: "lfm2-thinking", want: "<|startoftext|>"},
		{name: "laguna", want: "〈|EOS|〉"},
		{name: "poolside-v1", want: "〈|EOS|〉"},
		{name: "deepseek3.1", want: "<｜begin▁of▁sentence｜>"},
		{name: "cogito", want: "<｜begin▁of▁sentence｜>"},
		{name: "qwen3-coder", want: ""},
		{name: "unknown", want: ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := LeadingBOSForRenderer(tt.name); got != tt.want {
				t.Fatalf("LeadingBOSForRenderer(%q) = %q, want %q", tt.name, got, tt.want)
			}
		})
	}
}

func TestOverrideBuiltInRenderer(t *testing.T) {
	// Override the built-in renderer
	Register("qwen3-coder", func() Renderer {
		return &mockRenderer{}
	})

	// Should get the override
	result, err := RenderWithRenderer("qwen3-coder", nil, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "mock-output" {
		t.Errorf("expected 'mock-output' from override, got %q", result)
	}
}

func TestUnknownRendererReturnsError(t *testing.T) {
	_, err := RenderWithRenderer("nonexistent-renderer", nil, nil, nil)
	if err == nil {
		t.Error("expected error for unknown renderer")
	}
}

// TestThinkingPromptPrefill pins whether each thinking renderer primes the
// thinking block by ending the prompt with the opening tag. A thinking-token
// budget is enforced by a sampler that watches for that tag: when the prompt
// already contains it the tag has to be replayed into the sampler, and when it
// does not the model emits it and the sampler sees it directly. Either way the
// budget engages — but only as long as a renderer does not silently change
// which of the two it does.
func TestThinkingPromptPrefill(t *testing.T) {
	tests := []struct {
		renderer string
		openTag  string
		closeTag string
		// primes records whether the prompt is left inside a thinking block
		// once thinking is on, so the model never emits the opening tag
		primes bool
		// primesAfterToolResponse is the same for a turn that resumes after a
		// tool response, which is where gemma4 differs from the rest and where
		// an agent loop spends nearly every request
		primesAfterToolResponse bool
		// primesWithThinkingOff marks renderers that prime the block even when
		// thinking is off, because the model always reasons
		primesWithThinkingOff bool
	}{
		{renderer: "gemma4", openTag: "<|channel>", closeTag: "<channel|>", primesAfterToolResponse: true},
		{renderer: "qwen3.5", openTag: "<think>", closeTag: "</think>", primes: true, primesAfterToolResponse: true},
		{renderer: "qwen3-vl-thinking", openTag: "<think>", closeTag: "</think>", primes: true, primesAfterToolResponse: true},
		{renderer: "qwen3-vl-instruct", openTag: "<think>", closeTag: "</think>", primes: true, primesAfterToolResponse: true},
		{renderer: "deepseek3.1", openTag: "<think>", closeTag: "</think>", primes: true},
		{renderer: "cogito", openTag: "<think>", closeTag: "</think>", primes: true, primesAfterToolResponse: true},
		{renderer: "glm-4.7", openTag: "<think>", closeTag: "</think>", primes: true, primesAfterToolResponse: true},
		{renderer: "laguna", openTag: "<think>", closeTag: "</think>", primes: true, primesAfterToolResponse: true},
		{renderer: "poolside-v1", openTag: "<think>", closeTag: "</think>", primes: true, primesAfterToolResponse: true},
		{renderer: "ornith", openTag: "<think>", closeTag: "</think>", primes: true, primesAfterToolResponse: true},
		{renderer: "nemotron-3-nano", openTag: "<think>", closeTag: "</think>", primes: true, primesAfterToolResponse: true},
		{renderer: "cohere", openTag: "<|START_THINKING|>", closeTag: "<|END_THINKING|>", primes: true, primesAfterToolResponse: true},
		{renderer: "olmo3-think", openTag: "<think>", closeTag: "</think>", primes: true, primesAfterToolResponse: true, primesWithThinkingOff: true},
		{renderer: "olmo3-32b-think", openTag: "<think>", closeTag: "</think>", primes: true, primesAfterToolResponse: true, primesWithThinkingOff: true},
		{renderer: "lfm2-thinking", openTag: "<think>", closeTag: "</think>", primes: false},
	}

	// primesThinking mirrors how the runner decides to replay a primed block
	// into the sampler; llm.thinkingGenerationPrompt is the authority
	primesThinking := func(prompt, openTag, closeTag string) bool {
		i := strings.LastIndex(prompt, openTag)
		if i == -1 {
			return false
		}
		rest := prompt[i:]
		return !strings.Contains(rest[len(openTag):], closeTag) && len(rest) <= 64
	}

	msgs := []api.Message{{Role: "user", Content: "hi"}}
	tools := []api.Tool{{Type: "function", Function: api.ToolFunction{Name: "get_time", Description: "the time"}}}
	afterToolResponse := []api.Message{
		{Role: "user", Content: "what time is it"},
		{Role: "assistant", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "get_time", Arguments: api.ToolCallFunctionArguments{}}}}},
		{Role: "tool", Content: "12:00", ToolName: "get_time"},
	}

	for _, tt := range tests {
		t.Run(tt.renderer, func(t *testing.T) {
			for _, shape := range []struct {
				name     string
				msgs     []api.Message
				tools    []api.Tool
				thinking bool
				want     bool
			}{
				{"thinking on", msgs, nil, true, tt.primes},
				{"thinking on, after a tool response", afterToolResponse, tools, true, tt.primesAfterToolResponse},
				// With thinking off the tag must not be left dangling, or the
				// budget would engage on a block the model never opens. The
				// exception is renderers whose model always reasons.
				{"thinking off", msgs, nil, false, tt.primesWithThinkingOff},
				{"thinking off, after a tool response", afterToolResponse, tools, false, tt.primesWithThinkingOff},
			} {
				prompt, err := RenderWithRenderer(tt.renderer, shape.msgs, shape.tools, &api.ThinkValue{Value: shape.thinking})
				if err != nil {
					t.Fatalf("%s: %v", shape.name, err)
				}
				if got := primesThinking(prompt, tt.openTag, tt.closeTag); got != shape.want {
					t.Errorf("%s: prompt primes %q = %v, want %v\nprompt tail: %q",
						shape.name, tt.openTag, got, shape.want, tail(prompt))
				}
			}
		})
	}
}

func tail(s string) string {
	if len(s) > 48 {
		return s[len(s)-48:]
	}
	return s
}
