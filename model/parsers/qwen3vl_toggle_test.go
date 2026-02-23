package parsers

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func TestQwen3VLParserThinkToggle(t *testing.T) {
	t.Run("thinking enabled by default when configured", func(t *testing.T) {
		p := &Qwen3VLParser{hasThinkingSupport: true, defaultThinking: true}
		p.Init(nil, nil, nil)
		if p.state != CollectingThinkingContent {
			t.Fatalf("state = %v, want %v", p.state, CollectingThinkingContent)
		}
	})

	t.Run("thinking disabled by default", func(t *testing.T) {
		p := &Qwen3VLParser{hasThinkingSupport: true, defaultThinking: false}
		p.Init(nil, nil, nil)
		if p.state != CollectingContent {
			t.Fatalf("state = %v, want %v", p.state, CollectingContent)
		}
	})

	t.Run("thinking disabled at runtime", func(t *testing.T) {
		p := &Qwen3VLParser{hasThinkingSupport: true, defaultThinking: true}
		p.Init(nil, nil, &api.ThinkValue{Value: false})
		if p.state != CollectingContent {
			t.Fatalf("state = %v, want %v", p.state, CollectingContent)
		}

		content, thinking, calls, err := p.Add("plan</think>answer", true)
		if err != nil {
			t.Fatal(err)
		}
		if content != "plan</think>answer" {
			t.Fatalf("content = %q, want %q", content, "plan</think>answer")
		}
		if thinking != "" {
			t.Fatalf("thinking = %q, want empty", thinking)
		}
		if len(calls) != 0 {
			t.Fatalf("calls = %d, want 0", len(calls))
		}
	})
}
