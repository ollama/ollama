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
		{name: "qwen3.8"},
		{name: "nemotron-3-nano"},
		{name: "nemotron-3.5-nano"},
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
		// prefilled records whether the prompt ends with openTag once thinking
		// is on, so the model never emits it
		prefilled bool
		// alwaysPrefilled marks renderers that prime the block even when
		// thinking is off, because the model always reasons
		alwaysPrefilled bool
	}{
		{renderer: "gemma4", openTag: "<|channel>", prefilled: false},
		{renderer: "qwen3.5", openTag: "<think>", prefilled: true},
		{renderer: "qwen3-vl-thinking", openTag: "<think>", prefilled: true},
		{renderer: "deepseek3.1", openTag: "<think>", prefilled: true},
		{renderer: "cogito", openTag: "<think>", prefilled: true},
		{renderer: "glm-4.7", openTag: "<think>", prefilled: true},
		{renderer: "laguna", openTag: "<think>", prefilled: true},
		{renderer: "nemotron-3-nano", openTag: "<think>", prefilled: true},
		{renderer: "olmo3-think", openTag: "<think>", prefilled: true, alwaysPrefilled: true},
		{renderer: "olmo3-32b-think", openTag: "<think>", prefilled: true, alwaysPrefilled: true},
		{renderer: "lfm2-thinking", openTag: "<think>", prefilled: false},
	}

	// primesThinking mirrors how a runner decides to replay the opening tag
	primesThinking := func(prompt, openTag string) bool {
		return strings.HasSuffix(strings.TrimRight(prompt, " \t\r\n"), openTag)
	}

	msgs := []api.Message{{Role: "user", Content: "hi"}}

	for _, test := range tests {
		t.Run(test.renderer, func(t *testing.T) {
			thinking, err := RenderWithRenderer(test.renderer, msgs, nil, &api.ThinkValue{Value: true})
			if err != nil {
				t.Fatalf("rendering with thinking on: %v", err)
			}
			if got := primesThinking(thinking, test.openTag); got != test.prefilled {
				t.Errorf("thinking on: prompt primes %q = %v, want %v\nprompt tail: %q",
					test.openTag, got, test.prefilled, tail(thinking))
			}

			// With thinking off the tag must not be left dangling, or the
			// budget would engage on a block the model never opens. The
			// exception is renderers whose model always reasons.
			notThinking, err := RenderWithRenderer(test.renderer, msgs, nil, &api.ThinkValue{Value: false})
			if err != nil {
				t.Fatalf("rendering with thinking off: %v", err)
			}
			if got := primesThinking(notThinking, test.openTag); got != test.alwaysPrefilled {
				t.Errorf("thinking off: prompt primes %q = %v, want %v\nprompt tail: %q",
					test.openTag, got, test.alwaysPrefilled, tail(notThinking))
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
